// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
}

namespace VideoCore {

using BufferId = Common::SlotId;

class TextureCache;
class MemoryTracker;
class PageManager;

// [PAGETRACE] SHADPS4_PAGETRACE=<hex base>:<hex size>[,...]: per-16 KiB-page access counters
// for the listed guest ranges (job heaps), fed by the existing host hooks: GPU first-touch
// faults (FT), GPU raw writes from the page journal (GW), dropped writes landed (DL), CPU write
// faults (CW), CPU read faults on GPU-marked pages (CR) and those served fast (CRS), CPU-dirty
// uploads into VRAM owners (UP), SpinRescue snoop hits (SPA/SPB). Dumped every 5 s with a class
// per page (P polled, M mixed, R result, S scratch, I input, U untouched) and the current home.
namespace PageTrace {
// CRA = a readback delivery refused because the CPU owns bytes on the page (cpu_authority or
// CPU-modified): the mixed page whose GPU results never reach the CPU through readbacks.
// PJ = [POLLJ] shader-side presence: a census poll loop read the page through a live load this
// batch - cached (VRAM-owned) pages included, which the fault bitmap never shows.
enum Field : u32 { FT = 0, GW, DL, CW, CR, CRS, UP, SPA, SPB, CRA, PJ, NUM_FIELDS };
bool Enabled();
void Note(VAddr addr, Field field);
void NoteBitmap(const u32* words, size_t num_words, Field field);
void NoteStall(); // SpinRescue stall incident (SPINWATCH): per-window context for the P class
std::vector<std::pair<VAddr, u64>> Ranges(); // traced ranges, page-aligned; empty when off
} // namespace PageTrace

// [HOMEPOLICY] SHADPS4_HOMEPOLICY_IMPORT=<hex base>:<hex size>[,...]: guest pages whose home
// must be guest memory (an import-backed owner over the shared backing, force mode) instead of
// VRAM - pages the CPU polls or reads after the GPU wrote them (classes P/F/R/M of the PAGETRACE
// map). Sticky for the session; a guest unmap trims the range. World A only: no VRAM+writeback
// home and no import->VRAM demotion.
// [HOMEPOLICY] who binds: Rasterizer::BindBuffers publishes the current shader hash so the
// union log in CreateBuffer can name the descriptor that dragged a policy page into a union.
extern thread_local u64 g_bind_pgm_hash;

namespace HomePolicy {
bool Enabled();
bool WantsImport(VAddr addr, u64 size);   // any listed page inside the range
bool WantsPt(VAddr addr, u64 size);       // [HOMEPOLICY-PT] any page-table-redirect page inside
bool StaticWantsPt(VAddr addr, u64 size); // [POLLMAP v3.2] static list only
u64 CoveredBytes(VAddr addr, u64 size);   // listed bytes inside the range (union detection)
std::vector<std::pair<VAddr, u64>> Ranges();
void Subtract(VAddr addr, u64 size); // guest unmap
} // namespace HomePolicy

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
    /// [DUALHOME] Second-home window appended to the page table: one device address per 16 KiB
    /// page of guest VA [0x1000000000, 0x1400000000) plus a clamp entry; never zero (sink page
    /// by default, an import piece while armed). Indexed by CACHING_NUMPAGES + (page - base).
    static constexpr u64 DUAL_WINDOW_BASE = 0x1000000000ULL;
    static constexpr u64 DUAL_WINDOW_BASE_PAGE = DUAL_WINDOW_BASE >> CACHING_PAGEBITS;
    static constexpr u64 DUAL_WINDOW_PAGES = u64{1} << 20;
    static constexpr u64 DUAL_TABLE_BYTES = (DUAL_WINDOW_PAGES + 1) * sizeof(vk::DeviceAddress);
    static constexpr u64 DUAL_SINK_BYTES = u64{64} * CACHING_PAGESIZE;
    /// Writes a shader performs through a raw device address are invisible to the host, so
    /// each one appends {address, value} here and the host replays them into guest memory.
    /// The log lives right after the fault bitmap inside the same buffer, so no shader needs
    /// an extra descriptor; entries beyond the capacity are dropped and counted.
    static constexpr u32 BDA_WRITE_LOG_ENTRIES = 65536;
    static constexpr u32 BDA_WRITE_LOG_WORDS_PER_ENTRY = 4;
    static constexpr u64 BDA_WRITE_LOG_BYTES =
        u64{4} + u64{BDA_WRITE_LOG_ENTRIES} * BDA_WRITE_LOG_WORDS_PER_ENTRY * 4;
    /// [DROPLOG] A raw-address write to a page with no live page-table entry is dropped by the
    /// GPU (the walk only marks the fault). The first write of a GPU job into a fresh page is
    /// exactly that write; its slot then keeps the game's 0x7FFFFFFF filler and the job-result
    /// validator asserts (line 3543). A second, small log after the write log carries those
    /// {address, value} pairs so the host lands them in guest memory. Same entry layout.
    static constexpr u32 BDA_DROP_LOG_ENTRIES = 65536;
    static constexpr u64 BDA_DROP_LOG_BYTES =
        u64{4} + u64{BDA_DROP_LOG_ENTRIES} * BDA_WRITE_LOG_WORDS_PER_ENTRY * 4;
    static constexpr u64 BDA_LOGS_BYTES = BDA_WRITE_LOG_BYTES + BDA_DROP_LOG_BYTES;
    /// [PAGEJOURNAL] Instead of one journal entry per dword (2.7M per frame, 97% lost in the
    /// overflow, so the CPU never sees most GPU results in VRAM pages), the shader sets the
    /// page's bit in a bitmap and appends the page index once per batch. The host marks those
    /// pages GPU-modified (readback protection + download on CPU read), skipping import-backed
    /// pages whose stores already landed in guest memory. Bitmap: CACHING_NUMPAGES bits; list:
    /// count + indices. Both sit after the dword logs in the fault buffer.
    static constexpr u64 PAGEJ_BITMAP_BYTES = CACHING_NUMPAGES / 8;
    static constexpr u32 PAGEJ_LIST_ENTRIES = 65536;
    static constexpr u64 PAGEJ_LIST_BYTES = u64{4} + u64{PAGEJ_LIST_ENTRIES} * 4;
    static constexpr u64 PAGEJ_BYTES = PAGEJ_BITMAP_BYTES + PAGEJ_LIST_BYTES;
    /// [POLLJ] SHADPS4_PAGETRACE_POLLJ=1 (with SHADPS4_PAGETRACE): presence window right after
    /// the page journal, one u32 per 16 KiB page of the DUAL window [0x1000000000, 0x1400000000)
    /// plus a clamp slot for everything outside it. Every scalar live read issued from a census
    /// poll loop stores its shader hash (low 32 bits | 1) into the slot of the page it read, so
    /// polled pages are visible even while they are cached in VRAM (the fault bitmap only ever
    /// shows pages without a page-table entry). The host copies the PAGETRACE-covered slices out
    /// per batch, clears them, and folds them into the per-page counters (field PJ).
    /// Shader base, allocation, copy-out and fold all derive from these constants.
    static constexpr u64 POLLJ_WINDOW_PAGES = DUAL_WINDOW_PAGES;
    static constexpr u64 POLLJ_BYTES = (POLLJ_WINDOW_PAGES + 1) * 4;
    static constexpr u64 POLLJ_BASE_DWORDS =
        CACHING_NUMPAGES / 32 + BDA_LOGS_BYTES / 4 + PAGEJ_BYTES / 4;
    static constexpr u64 POLLJ_BASE_BYTES = POLLJ_BASE_DWORDS * 4;
    /// Read once. Codegen-affecting: a SPIR-V/pipeline cache written with it on embeds the
    /// presence store, and with it off the fault buffer is POLLJ_BYTES shorter, so the cache
    /// must be purged when toggling (the stand runs SHADPS4_PCACHE=0).
    static bool PollTraceEnabled() {
        static const bool on = [] {
            const char* e = std::getenv("SHADPS4_PAGETRACE_POLLJ");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        return on;
    }
    /// [POLLMAP] SHADPS4_POLLMAP=1: dynamic page-table redirect for the pages the GPU polls.
    /// The shader side of read_const_live_poll sets, once per page, a bit in a host-visible
    /// bitmap (one bit per 16 KiB page of the whole 40-bit guest space, 8 MiB) and a bit in a
    /// summary (one bit per 1024-page region, 8 KiB): [fine 8 MiB][summary 8 KiB]. The host
    /// (PollMapStep, GPU thread) folds the summary, adds every new page to a sticky set that
    /// HomePolicy::WantsPt consults lock-free, and rewrites the page's table entry to point at
    /// guest RAM (PolicyPtBda) - the same redirect SHADPS4_HOMEPOLICY_PT applies from a static
    /// list, without the list. The shader learns the bitmap's device address from a slot
    /// appended to the fault buffer (POLLMAP_SLOT_*); a zero slot disables the marking.
    /// Codegen-affecting like POLLJ: purge the pipeline cache when toggling.
    static constexpr u64 POLLMAP_FINE_WORDS = CACHING_NUMPAGES / 32;
    static constexpr u64 POLLMAP_FINE_BYTES = POLLMAP_FINE_WORDS * 4;
    static constexpr u64 POLLMAP_REGION_PAGES = 1024;
    static constexpr u64 POLLMAP_REGION_WORDS = POLLMAP_REGION_PAGES / 32;
    static constexpr u64 POLLMAP_REGIONS = CACHING_NUMPAGES / POLLMAP_REGION_PAGES;
    static constexpr u64 POLLMAP_SUMMARY_WORDS = POLLMAP_REGIONS / 32;
    static constexpr u64 POLLMAP_SUMMARY_BYTES = POLLMAP_SUMMARY_WORDS * 4;
    static constexpr u64 POLLMAP_BYTES = POLLMAP_FINE_BYTES + POLLMAP_SUMMARY_BYTES;
    /// Address slot inside the fault buffer: {lo, hi} of the bitmap's device address, right
    /// after the presence window (or after the page journal when POLLJ is off). 16 bytes so the
    /// fault buffer keeps a 16-byte multiple; only the first 8 are read.
    static constexpr u64 POLLMAP_SLOT_BYTES = 16;
    static bool PollMapEnabled() {
        static const bool on = [] {
            const char* e = std::getenv("SHADPS4_POLLMAP");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        return on;
    }
    static u64 PollMapSlotOffset() {
        return POLLJ_BASE_BYTES + (PollTraceEnabled() ? POLLJ_BYTES : 0);
    }
    static u32 PollMapSlotDwords() {
        return static_cast<u32>(PollMapSlotOffset() / 4);
    }

    /// Buffers are allocated one dedicated VkDeviceMemory each (they all request a device
    /// address), so the count matters as much as the byte total - drivers cap the number of
    /// allocations, typically at a few thousand.
    static constexpr u64 BufferCountSoftLimit = 1024;
    static constexpr u64 BufferCountHardLimit = 2048;

    // Default values for garbage collection
    static constexpr s64 DEFAULT_TRIGGER_GC_MEMORY = 1_GB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct PageData {
        BufferId buffer_id{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// [LISTCOUNT] Which home a guest page currently has for the GPU: 0 = no buffer,
    /// 1 = VRAM copy, 2 = import-backed (GPU reads guest RAM directly), 3 = backing-backed.
    /// Diagnostic only; takes buffers_mutex.
    u32 PageHomeKind(VAddr addr);

    /// [LISTCOUNT] Read one guest dword without ever taking the fault path (a protected,
    /// GPU-owned page answers "not readable" instead of triggering a readback). Returns 1 when
    /// the value was read; fills the page's home kind either way.
    u32 PeekGuestDword(VAddr addr, u32* out_value, u32* out_home);

    /// [SKINVRAM] SHADPS4_SKINVRAM=<hex hash>[,...]: serve the ND skinning worker's raw-pointer
    /// inputs (count dword, 128-byte descriptor block, per-vertex headers, influence list, bone
    /// palette) from a DEVICE_LOCAL ring instead of guest memory over PCIe. Wanted() is a
    /// static-list check (zero cost while the env is unset). Redirect() runs on the GPU thread
    /// between Rasterizer::BindResources and Pipeline::BindResources: it copies the five ranges
    /// into a ring slot inside the current command buffer, patches the block COPY so its V#s
    /// name the slot copies, and rewrites ud_0..3 in the push constants so the unchanged shader
    /// walks the page table into VRAM. Stamp() re-stamps the slot with the dispatch's tick.
    /// The ring is host-visible DEVICE_LOCAL (ReBAR) whenever the driver gives such memory: the
    /// slot is then filled by a plain CPU store and needs neither a staging copy nor a barrier.
    /// Only the fallback (no ReBAR) records a transfer, and it APPENDS its barrier to `barriers`
    /// so Pipeline::BindResources folds it into the one pipelineBarrier2 it issues anyway -
    /// never a standalone global barrier in the dispatch hot loop.
    static bool SkinVramWanted(u64 pgm_hash);
    void SkinVramRedirect(u64 pgm_hash, u32 ud_mask, u32* ud_regs,
                          boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);
    void SkinVramStamp(u64 tick);
    /// [SKINVRAM] Called by the fault parser for a page it refused to cache: true means the page
    /// belongs to the synthetic window, i.e. the shader dereferenced an entry that never landed.
    /// That is the signature failure of the feature, so it must not hide in a Warning line.
    bool SkinVramNoteFaultPage(VAddr page);

    /// [SKINDIRECT] SHADPS4_SKINDIRECT=<hex hash>[,...]: for the listed shaders the recompiler
    /// DECLINES the BDA rewrite for reads whose V# is one of the ND skinning worker's bounded
    /// input descriptors (the V# the shader loads from block dword 16/20/24 behind ud_0|ud_1 -
    /// the same three ranges DESCDUMP decodes and SKINVRAM copies), so those reads keep the
    /// ordinary buffer binding instead: one OpAccessChain+OpLoad on a bounds-checked SSBO
    /// instead of {dependent page-table load + branch + volatile PhysicalStorageBuffer load}.
    /// Nothing here changes behaviour - the recompiler asks Hashes()/Wanted(), records what it
    /// converted through Register(), and the rasterizer reports how each converted descriptor
    /// resolved per dispatch through Note(). Empty env -> Hashes() is empty and every entry
    /// point below is one empty() test.
    static const std::vector<u64>& SkinDirectHashes();
    static bool SkinDirectWanted(u64 pgm_hash);
    /// Compile time: remember that (pgm_hash, sharp_idx) was converted. True the first time a
    /// pair is seen, so the caller logs the conversion once per shader+word instead of per read.
    static bool SkinDirectRegister(u64 pgm_hash, u32 sharp_idx, u32 word);
    /// Compile time: a listed shader whose read matched a selected block dword but whose V# was
    /// not bindable (unbounded, zero, swizzled, wrong stride) and kept the BDA path.
    static void SkinDirectNoteDecline();
    /// [SKINDIRECT-STORES] env accessors, parsed once in buffer_cache.cpp (same discipline as
    /// Hashes()): the recompiler and the report read the same parsed values.
    static bool SkinDirectStoresEnabled();
    static const std::vector<u32>& SkinDirectStoreWords();
    static bool SkinDirectStoreImgInv();
    static bool SkinDirectStoreVerify();
    /// Compile time: an output store of a listed shader kept a WRITTEN binding. True the first
    /// time (pgm_hash, sharp_idx) is seen. `needs_lowering` = StoreBufferFormatF32 (counted as
    /// lowered by SkinDirectNoteStoreLowered), raw StoreBufferU32/F32xN count as lowered here.
    static bool SkinDirectRegisterStore(u64 pgm_hash, u32 sharp_idx, u32 word, bool shape_odd,
                                        bool needs_lowering);
    /// Compile time, from LowerBufferFormatToRaw: the typed store reached raw OpStores (true) or
    /// was dropped by the FMT0 guard (false). Ignored for (hash, sharp_idx) not in the registry.
    static void SkinDirectNoteStoreLowered(u64 pgm_hash, u32 sharp_idx, bool emitted);
    /// GPU thread, from IsComputeMetaClear: a listed shader's dispatch was skipped as a meta clear.
    static void SkinDirectNoteMetaSkip(u64 pgm_hash, VAddr base);
    /// Bind time, FIRST pass of BindBuffers (before ObtainBuffer). Returns 0 = not a converted
    /// descriptor, 1 = converted read (classified here as before), 2 = converted store (only the
    /// null case is counted here; the rest in SkinDirectNoteStoreBind after ObtainBuffer).
    /// Instrumentation only.
    u8 SkinDirectNote(u64 pgm_hash, u32 sharp_idx, VAddr base, u64 size, bool is_written);
    /// Bind time, SECOND pass, after ObtainBuffer: where the bound store lands. Also records the
    /// range in skind.out_ranges (see SkinDirectOutputIntersects) unless it is unbounded/oversize.
    void SkinDirectNoteStoreBind(VAddr base, u64 size, const Buffer& owner, bool unbounded,
                                 bool barrier_emitted);
    /// A range the skinning worker's converted OUTPUT store was bound over (any home). An
    /// import-backed owner gets no tracker mark (SynchronizeBuffer returns before
    /// ForEachUploadRange), so IsRegionGpuModified stays false there and the consumer would (a)
    /// take the record-time stream copy for <= 16 KiB ranges - a snapshot taken BEFORE the
    /// dispatch executes - and (b) request no compute->vertex barrier. ObtainBuffer and
    /// BindVertexBuffers ask this instead; one bool test while no output was ever bound.
    /// GPU thread only (BindBuffers writes it, ObtainBuffer/BindVertexBuffers read it).
    [[nodiscard]] bool SkinDirectOutputIntersects(VAddr addr, u64 size) const {
        return skind.out_armed && skind.out_ranges.Intersects(addr, size);
    }
    /// [CSBATCH] Mirrors ObtainBuffer's decision: true when a bind of this range would go through
    /// FindBuffer and NOT find a live owner covering it (i.e. would CreateBuffer -> JoinOverlap ->
    /// DeleteBuffer of every overlapped owner). Stream-path ranges (read-only <= 16 KiB, not GPU
    /// modified, no label import, not a converted output) never create anything. GPU thread.
    [[nodiscard]] bool WouldCreate(VAddr addr, u32 size, bool is_written) {
        if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(addr, size) &&
            !IntersectsLabelImport(addr, size) && !SkinDirectOutputIntersects(addr, size)) {
            return false;
        }
        const BufferId id = page_table[addr >> CACHING_PAGEBITS].buffer_id;
        return !id || slot_buffers[id].is_deleted || !slot_buffers[id].IsInBounds(addr, size);
    }
    /// [CSBATCH] Mirrors what ObtainBuffer would record for this range: true when its cache-path
    /// owner would take an upload (SynchronizeBuffer over CPU-dirty pages) into the current
    /// cmdbuf. A stream-path range snapshots into a private region instead, an import-backed
    /// owner is guest RAM read at execution time and never uploads. Call after WouldCreate
    /// answered false (the owner exists). GPU thread.
    [[nodiscard]] bool WouldUpload(VAddr addr, u32 size, bool is_written) {
        if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(addr, size) &&
            !IntersectsLabelImport(addr, size) && !SkinDirectOutputIntersects(addr, size)) {
            return false;
        }
        const BufferId id = page_table[addr >> CACHING_PAGEBITS].buffer_id;
        if (id && !slot_buffers[id].is_deleted && slot_buffers[id].is_import_backed) {
            return false;
        }
        return IsRegionCpuModified(addr, size);
    }
    /// [CSBATCH] consumer guard for a pointer-written output: the tail of SkinDirectNoteStoreBind
    /// without the STORES census. BindVertexBuffers then asks for the compute->vertex barrier
    /// and ObtainBuffer refuses the record-time stream snapshot for the range. GPU thread.
    void CsBatchNoteOutput(VAddr base, u64 size) {
        skind.out_ranges.Add(base, size);
        skind.out_armed = true;
    }
    /// [CSBATCH] The R1 loop of SkinDirectNoteStoreBind as a predicate: a VRAM owner under a
    /// PT-redirected page would take the batch's store into VRAM while PT readers and the CPU
    /// keep guest RAM (home-4 divergence). WantsPt && PolicyPtBda(p) != 0 per 16 KiB page.
    [[nodiscard]] bool PtRedirectedInside(VAddr base, u64 size);
    /// [CSBATCH] Bulk guest read that never takes the fault path (PeekGuestDword's rule): false
    /// when the range is unreadable. The shadow compare reads the guest's output through it.
    [[nodiscard]] bool PeekGuestBytes(VAddr addr, void* dst, size_t n);
    /// [CSBATCH] True while the slot still holds a live (not deleted, not erased) buffer: the
    /// batch re-arms its output owners by id at flush time. GPU thread.
    [[nodiscard]] bool IsBufferLive(BufferId id) const {
        return id && slot_buffers.is_allocated(id) && !slot_buffers[id].is_deleted;
    }
    /// Bind time: SHADPS4_SKINDIRECT_STORE_IMGINV=0 skipped the image invalidation of an output.
    void SkinDirectNoteImgInvSkipped();
    /// [VERIFY] run-time detectors, on by default with STORES=1 (SHADPS4_SKINDIRECT_STORE_VERIFY=0
    /// turns them off). `barrier` = a compute->vertex barrier was requested for the consumer.
    void SkinDirectVerifyUpload(VAddr addr, u64 size); // SynchronizeBuffer upload lambda
    void SkinDirectVerifyConsumer(VAddr addr, u64 size, bool stream,
                                  bool barrier); // BindVertexBuffers

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size);

    /// Records that the GPU wrote this region through a raw device address, so a later
    /// CPU read faults and pulls the data back instead of seeing stale guest memory.
    void MarkGpuWrittenRegion(VAddr device_addr, u64 size);

    /// [debug] Compare the GPU copy of a range against guest memory.
    void DebugCompareWatched(VAddr device_addr);
    VAddr debug_watched_addr{0};

    /// [debug] Name the owner of a host device address reported by VK_EXT_device_fault:
    /// a live buffer, or one whose memory was already released (use-after-free).
    static std::string DescribeHostAddress(u64 host_addr);
    static std::string DescribeHostRange(u64 base, u64 size); // [ATTRIB]

    /// [debug] Compare a sample of the device page table against the CPU mirror.
    void DebugVerifyPageTable();

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false);

    /// Services a CPU readback fault entirely on the faulting thread via the
    /// dedicated readback queue. Returns false when the slow path must run
    /// (no queue, page not backed by an existing buffer, writing tick not yet
    /// submitted, nothing to download, or the buffer changed underneath).
    bool TryServeFaultFast(VAddr device_addr, u64 size, bool is_write);
    bool WaitForBacking(VAddr device_addr);

    /// LABELUMA: promotes pending label regions to import-backed buffers
    /// (called unconditionally from Rasterizer::OnSubmit on the GPU thread).
    void PromoteLabelRegions();
    void RequestLabelPromotion(VAddr addr, u64 size);

    /// [BACKIMP] One bounded prefetch pass over SHADPS4_BACKIMP_PREFETCH ranges: registers
    /// backing buffers through the ordinary CreateBuffer path so the device BDA table has no
    /// holes there. Called from Rasterizer::OnSubmit (GPU thread); self-throttled, repeats
    /// so parts the guest maps later get picked up.
    void BackimpPrefetchStep();
    /// [VRAMPREFETCH] SHADPS4_VRAM_PREFETCH ranges get VRAM buffers before the GPU's first touch.
    void VramPrefetchStep();
    /// [PAGETRACE] 5 s dump of the per-page counters with the current home of each page
    void PageTraceStep();
    /// [HOMEPOLICY] every 3 s: give owner-less, mapped policy pages an import owner BEFORE the
    /// GPU's first touch (a spinning wave never lets its parse tick retire, so the parser is too
    /// late for those pages). GPU thread only.
    void PolicyPrefetchStep();
    /// [POLLMAP] once per submit (GPU thread): fold the shader-side poll bitmap into the sticky
    /// set, rewrite the page-table entries of the pages it found and, with a host-mapped table
    /// (SHADPS4_PTHOST=1), store a slice of the redirected entries again (see POLLMAP_* above).
    void PollMapStep();
    /// [DUALHOME] SHADPS4_DUALHOME=<hex base>:<hex size>[,...]: the GPU reads the VRAM owner of
    /// these ranges, and every raw-address store also lands in guest memory through an import
    /// piece, so the CPU sees results without journal, replay or readback protection.
    static bool DualHomeEnabled();
    [[nodiscard]] bool IsDualRange(VAddr addr, u64 size) const;
    void DualArmStep();
    void ArmDualRangesFor(VAddr addr, u64 size);
    /// [VRAMPREFETCH-AUTO] a real (guest-range) page the GPU touched before anyone cached it:
    /// with SHADPS4_VRAM_PREFETCH_AUTO=<MB> its <MB>-aligned block gets VRAM buffers next pass.
    void NoteGpuFaultPage(VAddr page);
    /// [PAGEJOURNAL] a page the GPU wrote through a raw address this batch (from the fault parser)
    void NoteGpuWrittenPage(VAddr page);
    /// [PAGEJOURNAL=2] re-arm readback protection for every page the GPU has ever written
    /// through a raw address, at each submit boundary - BEFORE the GPU runs, the way bound
    /// buffers are marked. Closes the window between the GPU store and the host mark.
    void PageJournalStickyStep();
    /// [BYWRITE] Deferred correction for heuristic misses: a buffer that was placed in VRAM and
    /// then turned out to be GPU-written is demoted back to guest memory here - at a submit
    /// boundary, never in the middle of a draw. Called from Rasterizer::OnSubmit (GPU thread).
    void BywriteCorrectionStep();
    /// [SNAPRO] Refreshes the VRAM snapshots of read-only guest ranges. Runs once per frame from
    /// Rasterizer::OnSubmit, BEFORE any drawing: a copy issued at bind time would have to break
    /// the render pass, and there are ~70 first-touches per frame. Measured on the gameplay
    /// segment: the GPU reads 127 MB/frame but only 21-44 MB of it is distinct, so one
    /// sequential copy per frame replaces thousands of scattered reads over PCIe.
    void SnaproRefreshStep();
    void ReleaseLabelRegion(VAddr addr, u64 size);
    /// Arms the label seed against a fresh guest GPU mapping.
    void OnGuestMap(VAddr addr, u64 size);
    bool IntersectsLabelImport(VAddr addr, u64 size);

    /// Binds host vertex buffers for the current draw.
    void BindVertexBuffers(const Vulkan::GraphicsPipeline& pipeline,
                           boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Bind host index buffer for the current draw.
    void BindIndexBuffer(u32 index_offset,
                         boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false,
                                                       BufferId buffer_id = {}, u32 sync_limit = 0);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// True when a buffer actually owns the page holding this address. Unlike
    /// IsRegionRegistered (which asks a merged range set and answers yes for gaps between
    /// buffers), this looks the page up directly.
    [[nodiscard]] bool HasBufferAt(VAddr addr);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Installs the callback that decides whether a faulted GPU address range
    /// may be cached (it must correspond to mapped guest memory).
    void SetFaultRegionValidator(std::function<bool(VAddr, u64)> fn) {
        fault_region_valid = std::move(fn);
    }

    /// Returns true when a faulted range points at mapped guest memory.
    [[nodiscard]] bool IsFaultRegionValid(VAddr addr, u64 size) const {
        return !fault_region_valid || fault_region_valid(addr, size);
    }

    /// UMA: imports a freshly GPU-mapped guest region as host memory so BDA
    /// page table entries can fall back to live host pages (guest VA == host
    /// VA). Called from guest threads; page table writes are queued and
    /// applied on the GPU thread. Gated by SHADPS4_UMA=1.
    void ImportHostRegion(VAddr addr, u64 size);

    /// UMA: releases the host import backing an unmapped guest region.
    void ReleaseHostRegion(VAddr addr, u64 size);

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size);

    /// Synchronizes all buffers neede for DMA.
    void SynchronizeDmaBuffers();

    /// Runs the garbage collector.
    void RunGarbageCollector();

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        buffer_ranges.ForEachInRange(device_addr, size,
                                     [&](u64 page_start, u64 page_end, BufferId id) {
                                         Buffer& buffer = slot_buffers[id];
                                         func(id, buffer);
                                     });
    }

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    template <bool async>
    void DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size);

    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);

    /// Releases the buffer slot once the GPU has passed the current tick, optionally holding it
    /// for a few more ticks (SHADPS4_DELAYFREE) while the premature-release bug is investigated.
    void DeferErase(BufferId buffer_id, u32 extra_ticks);

    // [TDR SPINWATCH/SPINSNAP] Spin-rescue diagnostic contour. Never takes serve_mutex, never
    // touches rb_pool, never calls Register/CreateBuffer/FindBuffer/texture_cache/SendCommand/
    // Refresh. Own command pool/fence/staging + a private probe compute pipeline.
    void SpinRescueInit();                                 // build resources + start thread (ctor)
    void SpinRescueLoop(std::stop_token stop);             // 10ms watchdog on the semaphore counter
    void SpinRescueSnapshot(u64 gpu_tick, u64 stalled_ms); // snoop: cross-queue fault-bitmap snap
    bool SpinSubmitWait(u64 timeout_ns); // submit spin_cmdbuf to rescue queue+wait
    // [TDR ETAP4/5] import (no publish) + pt_patch publish. Called only from the rescue thread.
    void SpinImportAndPublish(std::span<const u64> pages, u64 gpu_tick);
    bool SpinReadDeviceEntry(u64 page, vk::DeviceAddress& out); // copy 8B table entry via rescue q
    bool SpinPatchEntries(
        const boost::container::small_vector<std::pair<u64, vk::DeviceAddress>, 32>&
            entries); // pt_patch atomicStore, rescue q

public:
    /// [TDR ETAP5] Live rescue-import BDA for a guest page (bda+offset), or 0 if none. Lifecycle
    /// hooks (fault parser, ChangeRegister, unmap) consult this. Takes spin_import_mutex (leaf).
    [[nodiscard]] vk::DeviceAddress RescueBdaFor(VAddr addr);
    /// [LIVEREPLAY] True when a raw-address (live BDA) GPU write to this guest address has already
    /// landed in guest memory on the GPU side: the page is served by an import of the backing
    /// mirror (import-backed owner, UMA host import or spin rescue import), which shares physical
    /// pages with the guest mapping. Replaying the write log over such a page is redundant and can
    /// clobber a newer CPU write with the stale GPU value. Takes buffers_mutex (then the leaf
    /// spin_import_mutex); call it OUTSIDE any memory-manager lock.
    [[nodiscard]] bool LiveWriteLandsInGuest(VAddr addr);
    /// [TDR ETAP5] Guest unmap: drop rescue imports in range and zero their table entries.
    void SpinReleaseImports(VAddr addr, u64 size);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::function<bool(VAddr, u64)> fault_region_valid;

    // UMA host imports: guest region -> imported VkDeviceMemory/VkBuffer/BDA.
    struct HostImport {
        VAddr base;
        u64 size;
        vk::Buffer buffer;
        vk::DeviceMemory memory;
        vk::DeviceAddress bda;
    };
    struct PendingHostFill {
        VAddr addr;
        u64 size;
    };
    void FlushPendingHostImports();
    [[nodiscard]] vk::DeviceAddress HostBdaFor(VAddr addr);
    /// [HOMEPOLICY-PT] guest-RAM device address of one page through the backing (0 = none)
    [[nodiscard]] vk::DeviceAddress PolicyPtBda(VAddr page_addr);
    void WritePageTableRange(u64 page_begin, u64 size_pages);
    bool uma_enabled{};
    std::map<VAddr, HostImport> host_imports;
    std::mutex host_import_mutex;
    std::vector<PendingHostFill> pending_host_fills;

    std::unique_ptr<MemoryTracker> memory_tracker;
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    u64 unbounded_sync_stamp = ~0ull;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    Common::SlotVector<Buffer> slot_buffers;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    // Range -> tick of the LAST submitted-or-recorded GPU write into it. inplace_max
    // keeps re-marks monotonic. Guarded by gpu_modified_mutex; lock order is
    // gpu_modified_mutex -> RegionManager::lock (never the other way).
    SplitRangeMap<u64, boost::icl::inplace_max> gpu_modified_ranges;
    std::mutex gpu_modified_mutex;
    // LABELUMA: regions whose owner is an import-backed buffer (guest RAM is
    // the single source of truth for them). Guarded by gpu_modified_mutex.
    bool label_uma_enabled{};
    RangeSet label_import_ranges;
    RangeSet label_failed_ranges;
    std::vector<std::pair<VAddr, u64>> pending_label_promotions;
    std::vector<std::pair<VAddr, u64>> label_seeds;
    // Set only around the CreateBuffer call inside PromoteLabelRegions
    // (GPU thread), so ordinary buffer creation never imports.
    bool label_promote_intent{};
    // Demotion -> re-promotion ratchet counter per 16MB arena; a range that
    // keeps ping-ponging goes to the failed set instead of thrashing.
    std::unordered_map<u64, u32> label_repromote_counts;
    // OOM retries: the demoted buffer's import frees only at tick retire, so an
    // immediate union re-import can exceed the driver's imported-memory cap.
    struct LabelRetry {
        VAddr base;
        u64 size;
        u64 not_before_pass;
        u32 attempts;
    };
    std::vector<LabelRetry> label_retries;
    u64 label_promote_passes = 0;
    u64 label_budget = 32_MB;
    // Sticky CPU-authority: pages the guest CPU has EVER written through a
    // write fault. The per-frame sweep clears the tracker's CPU-dirty bit, but
    // ownership semantics survive it: readback delivery must never clobber
    // these. Guarded by gpu_modified_mutex.
    RangeSet cpu_authority_ranges;
    // Guards page_table/buffer_ranges mutation and buffer deletion against the
    // fault fast path, plus the pin side-table (SlotVector reallocates, so pins
    // cannot live inside Buffer).
    std::mutex buffers_mutex;
    std::unordered_map<u32, u32> buffer_pins;
    // Readback engine: services CPU faults on the dedicated queue without the
    // GPU thread. serve_mutex serialises fault service (fast AND slow paths);
    // rb_submit_mutex covers vkQueueSubmit on the readback queue only.
    struct ReadbackSlot {
        vk::UniqueCommandBuffer cmdbuf;
        vk::UniqueFence fence;
        std::unique_ptr<Buffer> staging;
    };
    vk::UniqueCommandPool rb_pool;
    std::array<ReadbackSlot, 2> rb_slots;
    u32 rb_slot_index = 0;
    std::mutex serve_mutex;
    std::mutex rb_submit_mutex;
    std::atomic<bool> rb_flush_pending{false};
    bool rb_ready = false;
    SplitRangeMap<BufferId> buffer_ranges;
    PageTable page_table;
    // [CONSTCACHE] Exact (address,size) -> stream offset, valid only within one frame and one
    // stream-ring generation. GPU-thread only.
    u64 ConstCacheCopy(VAddr device_addr, u32 size);
    struct ConstKeyHash {
        size_t operator()(const std::pair<VAddr, u32>& k) const noexcept {
            return std::hash<u64>{}(k.first) ^ (std::hash<u32>{}(k.second) << 1);
        }
    };
    // [CONSTCACHE][VERIFY] SHADPS4_CONSTCACHE_VERIFY=1 keeps a checksum of the bytes that were
    // copied, and re-checks it on every hit. The cache trusts IsRegionCpuModified to say whether
    // the range changed; if the guest updates it any other way, the hit hands the shader last
    // frame's contents - for the skinning matrices that is literally the previous pose. A cache
    // in a hot path has to be able to prove it is right, not argue it (11.09 lesson).
    struct ConstEntry {
        u64 offset;
        u64 sum;
    };
    std::unordered_map<std::pair<VAddr, u32>, ConstEntry, ConstKeyHash> const_cache;
    u64 const_cache_frame = ~0ull;
    u64 const_cache_wrap = ~0ull;

    // [TDR SPINWATCH/SPINSNAP] Rescue contour state. Own resources only - never rb_pool.
    bool spin_enabled = false; // detect+ (watchdog running)
    bool spin_snoop = false;   // snoop+ (needs the dedicated rescue queue)
    u32 spin_stall_ms = 300;   // SHADPS4_SPINRESCUE_MS
    vk::UniqueCommandPool spin_cmd_pool;
    vk::UniqueCommandBuffer spin_cmdbuf;
    vk::UniqueFence spin_fence;
    std::unique_ptr<Buffer> spin_staging;   // Download: fault_buffer_size + 4KB
    std::unique_ptr<Buffer> spin_probe_buf; // tiny host-visible SSBO for the probe atomicAdd
    vk::UniqueDescriptorSetLayout spin_probe_layout;
    vk::UniquePipelineLayout spin_probe_pl_layout;
    vk::UniquePipeline spin_probe_pipeline;
    // [TDR ETAP4] import registry + budget/ratchet. Leaf lock order:
    // buffers_mutex -> gpu_modified_mutex -> spin_import_mutex.
    bool spin_import_mode = false;        // import+ (build host imports)
    bool spin_full = false;               // full: publish via pt_patch (int64 atomics present)
    u64 spin_budget_bytes = 8ull << 20;   // SHADPS4_SPINRESCUE_BUDGET_MB
    u64 spin_maxrun_bytes = 256ull << 10; // SHADPS4_SPINRESCUE_MAXRUN_KB
    u64 spin_imported_bytes = 0;          // live imported bytes (session budget)
    u32 spin_incident_imports = 0;        // imports this stall incident (<=16)
    std::mutex spin_import_mutex;
    struct SpinImport {
        VAddr base;
        u64 size;
        vk::DeviceAddress bda;
        // [BACKIMP] Bound over the shared backing import: costs no pinning budget, so it is
        // neither counted into spin_imported_bytes nor subtracted on release.
        bool backing;
    };
    std::map<VAddr, SpinImport> spin_imports;              // keyed by aligned base
    std::vector<std::unique_ptr<Buffer>> spin_import_bufs; // keep-alive (bounded by budget)
    std::unordered_map<u64, u32> spin_ratchet;             // page -> heals without tick progress
    // [TDR ETAP5] pt_patch publication pipeline (64-bit atomicStore into bda_pagetable_buffer).
    std::unique_ptr<Buffer> spin_params_buf; // Upload: u32 count + {page_idx, lo, hi}[256]
    vk::UniqueDescriptorSetLayout spin_patch_layout;
    vk::UniquePipelineLayout spin_patch_pl_layout;
    vk::UniquePipeline spin_patch_pipeline;
    // [BACKIMP] Prefetch state (GPU thread only, no locking needed beyond what the step takes).
    // [BYWRITE] Deferred correction state. GPU thread only: ObtainBuffer fills the queue,
    // BywriteCorrectionStep drains it, both run on the single GPU thread.
    std::vector<std::pair<VAddr, u64>> bywrite_requeue;
    // [SNAPRO] VRAM snapshots of read-only guest ranges, keyed by guest address (NOT BufferId:
    // buffers are recreated and SlotId has no generation counter - that bit us before).
    struct SnapShadow {
        std::unique_ptr<Buffer> buf;
        u64 size = 0;
        u64 used_frame = 0;  // last frame a shader asked for this range
        u64 fresh_frame = 0; // frame the copy was refreshed in
        u32 binds_now = 0;   // binds seen in the frame being recorded
        u32 binds_prev = 0;  // binds in the previous frame - decides whether a copy pays off
    };
    std::unordered_map<VAddr, SnapShadow> snap_shadows;
    u64 snap_last_refresh_frame = 0;
    u64 snap_copied_bytes = 0;
    u64 snap_copies = 0;
    u64 snap_hits = 0;
    u64 snap_misses = 0;
    std::unordered_map<u64, u32> bywrite_fix_counts;
    u64 bywrite_seen = 0;
    u64 bywrite_fixed = 0;
    u64 bywrite_skipped_big = 0;
    u64 bywrite_gaveup = 0;
    u64 bywrite_fix_us = 0;
    std::vector<std::pair<VAddr, u64>> backimp_prefetch_ranges;
    std::chrono::steady_clock::time_point backimp_prefetch_next{};
    u64 backimp_prefetch_buffers = 0;
    u64 backimp_prefetch_covered = 0;
    u64 backimp_prefetch_logged = 0;
    std::vector<std::pair<VAddr, u64>> vram_prefetch_ranges;
    std::chrono::steady_clock::time_point vram_prefetch_next{};
    u64 vram_prefetch_buffers = 0;
    u64 vram_prefetch_bytes = 0;
    std::mutex vram_auto_mutex;
    std::vector<VAddr> vram_auto_blocks;      // pending, from the fault-parse thread
    std::vector<VAddr> vram_auto_done_blocks; // already turned into ranges
    std::mutex pagej_mutex;
    std::vector<VAddr> pagej_sticky; // pages seen in the page journal (VRAM-owned), deduped
    std::vector<VAddr> pagej_new;    // noted since the last sticky step
    u64 pagej_sticky_marks = 0;
    struct DualPiece {
        VAddr base;
        u64 size;
        std::unique_ptr<Buffer> buffer;
    };
    [[nodiscard]] const DualPiece* DualPieceFor(VAddr addr) const;
    void PublishDualEntries(const DualPiece& piece);
    std::vector<std::pair<VAddr, u64>> dual_ranges;
    std::vector<std::unique_ptr<DualPiece>> dual_pieces;
    std::unique_ptr<Buffer> dual_sink;
    std::chrono::steady_clock::time_point dual_next{};
    std::chrono::steady_clock::time_point dual_report_next{};
    u64 dual_armed_bytes = 0;
    u64 dual_piece_copies = 0;
    u64 dual_staging_bytes = 0;
    // [SKINVRAM] see SkinVramRedirect. Config is parsed once in the constructor; everything else
    // is GPU-thread only except `clobbered`, which OnGuestMap (guest threads) may also set.
    // Synthetic mapping is fixed and linear: synthetic VA v <-> ring byte (v - synth_base); slot
    // s occupies ring bytes [s*slot_bytes, (s+1)*slot_bytes). The ring carries one extra 16 KiB
    // guard page after the last slot so a 16-byte over-read at the end of the last slot stays
    // inside the buffer.
    struct SkinVram {
        std::vector<u64> hashes;            // SHADPS4_SKINVRAM
        VAddr synth_base = 0xFF00000000ULL; // SHADPS4_SKINVRAM_BASE (hex), 16 KiB aligned
        u64 ring_bytes = u64{128} << 20;    // SHADPS4_SKINVRAM_MB (allocatable part)
        u64 slot_bytes = u64{128} << 10;    // SHADPS4_SKINVRAM_SLOT_KB (multiple of 16 KiB)
        u32 num_slots = 1024;               // ring_bytes / slot_bytes
        bool verbose = false;               // SHADPS4_SKINVRAM_LOG=1
        bool verify = false;                // SHADPS4_SKINVRAM_VERIFY=1 (staleness self-check)
        bool gpumod_allow = false;          // SHADPS4_SKINVRAM_GPUMOD=allow (count, don't skip)
        bool config_ok = false;             // geometry + refusals checked in the ctor
        bool init_tried = false;
        bool ok = false;
        std::atomic<bool> clobbered{false}; // guest mapped / buffer registered over the window
        std::unique_ptr<Buffer> ring;       // ring_bytes + CACHING_PAGESIZE, created in the ctor
        bool host_ring = false;             // ring is DEVICE_LOCAL *and* host-mapped (ReBAR)
        vk::DeviceAddress ring_bda = 0;
        std::vector<u64> slot_tick; // per slot: tick of the cmdbuf reading it (0 = never)
        u32 next_slot = 0;
        s32 pending_slot = -1;   // slot of the dispatch being recorded
        std::vector<u8> scratch; // CPU image of the slot prefix
        u64 wraps = 0;
        enum Skip : u32 {
            UdMask = 0,
            Unread,
            Decode,
            TooBig,
            RingBusy,
            Staging,
            Clobbered,
            GpuAuth,  // a range's authority is the GPU's copy, guest RAM would be stale
            ShortHdr, // vertex count exceeds the header array the V# declares
            NumSkips
        };
        u64 redirected = 0;
        u64 bytes = 0;
        u64 skipped[NumSkips]{};
        // Synthetic pages the GPU touched with no entry; bumped by the fault parser thread.
        std::atomic<u64> fault_pages{0};
        std::chrono::steady_clock::time_point report_next{};
        bool skip_logged[NumSkips]{};
        // [SKINVRAM_VERIFY] CPU image of the previous redirect's three inputs, re-read from
        // guest memory at the next redirect: a mismatch means the bytes changed after the
        // snapshot was taken, i.e. the record-time copy can be stale (GPU-produced input).
        std::vector<u8> verify_image;
        std::vector<u8> verify_tmp;
        struct VerifyRange {
            VAddr guest;
            u64 off;
            u64 size;
        } verify_ranges[3]{};
        bool verify_armed = false;
        u64 verify_checked = 0;
        u64 verify_mismatch = 0;
    } skin;
    // [SKINDIRECT] bind-time census of the converted descriptors. GPU-thread only (BindBuffers),
    // like everything else in SkinVram. Printed by SkinVramReport under the same 5 s timer, so
    // the feature never brings a timer of its own.
    struct SkinDirect {
        u64 bound = 0;     // V# resolved to a real buffer - the normal, wanted outcome
        u64 null_desc = 0; // base 0 / size 0 -> null descriptor -> robust zero reads
        u64 gpumod = 0;    // range is GPU-modified: guest RAM is not the authority
        u64 oversize = 0;  // declared range > 64 MiB: the V# is not the bounded one we matched
        bool logged_null = false;
        bool logged_gpumod = false;
        bool logged_oversize = false;
        bool gpumod_probe = true; // SHADPS4_SKINDIRECT_GPUMOD=0 turns the probe off
        bool probe_read = false;
        // [SKINDIRECT-STORES] bind-time census of converted OUTPUT descriptors (GPU thread only)
        u64 store_bound = 0;     // resolved to a cache-owned VkBuffer (second pass reached)
        u64 store_null = 0;      // base 0 / size 0 -> null descriptor -> stores DISCARDED
        u64 store_oversize = 0;  // > 64 MiB
        u64 store_unbounded = 0; // num_records==0xffffffff && >=1 GiB: is_written suppressed
        u64 store_home[4] = {};  // [1] VRAM owner, [2] import-backed, [3] backing-backed
        u64 store_pt = 0;        // VRAM owner AND a page of the range is PT-redirected (home 4)
        u64 store_barriers = 0;  // GetBarrier emitted on the write bind (read<->write flips)
        u64 store_imginv_skipped = 0; // SHADPS4_SKINDIRECT_STORE_IMGINV=0 skipped the lookup
        bool logged_store_null = false;
        bool logged_store_pt = false;
        bool logged_store_oversize = false;
        bool logged_store_unbounded = false;
        bool logged_store_home1_noreadback = false; // VRAM owner with readbacks Disabled
        // Ranges a converted output was bound over (SkinDirectOutputIntersects). Bounded: only
        // bounded (< 64 MiB) binds are added, adjacent slots merge, and a slot's address is
        // reused across frames, so the set converges to the worker's output arena.
        RangeSet out_ranges;
        bool out_armed = false;
        u64 out_stream_skips = 0; // ObtainBuffer: cache path instead of the stream copy
        u64 out_snapro_skips = 0; // ObtainBuffer: SNAPRO shadow refused for an output
        u64 out_vtx_barriers = 0; // BindVertexBuffers: barrier requested by the set alone
        // [VERIFY] ring of the last converted output binds, GPU thread only
        struct Out {
            VAddr base;
            u64 size;
            u64 tick;
            u8 home;
        };
        std::array<Out, 64> ring{};
        u32 ring_n = 0;
        VAddr ring_lo = ~VAddr{0};
        VAddr ring_hi = 0;
        u64 uploads_in_output = 0;  // guest-RAM upload landed inside an output bound earlier
        u64 consumer_hits = 0;      // vertex-buffer bind overlapping a converted output
        u64 consumer_stream = 0;    // ...served from the stream buffer (record-time snapshot)
        u64 consumer_nobarrier = 0; // ...with no GPU-modified mark, i.e. no barrier requested
        u32 logged_uploads = 0;
    } skind;
    void SkinVramCreateRing();
    void SkinVramInit();
    bool PublishSkinEntries(bool allow_wait);
    void SkinVramReport();
    void SkinVramVerifyPrev();
    [[nodiscard]] bool SkinVramPageOwned(VAddr addr);
    // [POLLMAP] host-visible|host-coherent poll bitmap with a device address (null when the env
    // is off or the allocation could not be mapped). Everything below is GPU-thread only: the
    // fold, the drain and the PTHOST re-apply all run from PollMapStep (Rasterizer::OnSubmit).
    std::unique_ptr<Buffer> pollmap_buffer;
    u8* pollmap_host = nullptr;
    bool pollmap_pt_host = false; // page table host-mapped (SHADPS4_PTHOST=1): stores land now
    bool pollmap_cached = false;  // bitmap memory is HOST_CACHED (sweep slices can be larger)
    struct PollMapPending {
        u64 page;     // 16 KiB page index
        u32 attempts; // passes in which the backing had no address for it yet
    };
    std::vector<PollMapPending> pollmap_pending;
    // Pages whose entry was redirected at least once, with the entry written (backing address,
    // landed bit included). With PTHOST the priority thread stores into the table too (the
    // deferred register publish), so PollMapStep walks this list in slices and stores the
    // entries again: a store that slipped past the publish-time re-check cannot undo a redirect
    // for more than a few passes. A page's guest-RAM address never changes within a session
    // (backing chunks are freed at shutdown only), so the entry is replayed, not recomputed.
    struct PollMapApplied {
        u64 page;
        vk::DeviceAddress entry;
    };
    std::vector<PollMapApplied> pollmap_applied;
    size_t pollmap_reapply_cursor = 0;
    std::vector<VAddr> pollmap_report_pages; // marked since the last 5 s report (capped)
    // Incremental sweep of the fine bitmap (lost-summary safety net): one slice of regions per
    // pass, a full circle at most once per 2 s.
    u64 pollmap_sweep_cursor = 0;           // next region of the running circle (0 = idle)
    u64 pollmap_sweep_regions_per_pass = 0; // chosen at arming from the memory type
    std::chrono::steady_clock::time_point pollmap_sweep_next{};    // earliest start of a circle
    std::chrono::steady_clock::time_point pollmap_sweep_started{}; // start of the running one
    std::chrono::steady_clock::time_point pollmap_report_next{};
    u64 pollmap_marked = 0;        // pages ever taken from the bitmap into the sticky set
    u64 pollmap_redirected = 0;    // page-table entries rewritten to guest RAM
    u64 pollmap_reapplied = 0;     // entries stored again by the PTHOST re-apply
    u64 pollmap_dropped = 0;       // pending pages given up on (no backing after many passes)
    u64 pollmap_sweeps = 0;        // completed circles of the fine bitmap
    u64 pollmap_sweep_work_us = 0; // CPU time of the running circle so far
    u64 pollmap_sweep_us = 0;      // CPU time of the last completed circle
    u64 pollmap_sweep_span_ms = 0; // wall time of the last completed circle
    u64 pollmap_slice_max_us = 0;  // worst single slice so far (the hiccup meter)
    u64 pollmap_folds = 0;         // PollMapStep passes
    // [BACKIMP] Optional deferred backing import (SHADPS4_BACKIMP_DELAY_S). Joined explicitly in
    // ~BufferCache; touches only the Vulkan device, which outlives the buffer cache.
    std::jthread backimp_thread;
    // MUST stay the LAST member: destroyed first (reverse order) so the loop stops before any
    // resource it touches is gone. ~BufferCache also request_stop()+join()s it as its first act.
    std::jthread spin_rescue_thread;
};

} // namespace VideoCore
