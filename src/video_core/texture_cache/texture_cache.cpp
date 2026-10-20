// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <chrono>
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/scope_exit.h"
#include "common/postmortem.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

// [FREEIMG-RING] lock-free ring of image deletions, read once at DEVFAULT (vk_presenter.cpp).
namespace VideoCore::FreeImgRing {
struct Entry {
    u64 handle;
    u64 guest_addr;
    u64 guest_size;
    u32 width;
    u32 height;
    u32 levels;
    u64 tick;
    u64 last_used;
};
constexpr u32 kSize = 256;
static std::array<Entry, kSize> ring{};
static std::atomic<u32> head{0};
void Note(u64 handle, u64 guest_addr, u64 guest_size, u32 width, u32 height, u32 levels, u64 tick,
          u64 last_used) {
    const u32 i = head.fetch_add(1, std::memory_order_relaxed) & (kSize - 1);
    ring[i] = Entry{handle, guest_addr, guest_size, width, height, levels, tick, last_used};
}
void Dump(u64 want_handle) {
    const u32 h = head.load(std::memory_order_relaxed);
    const u32 total = std::min<u32>(h, kSize);
    u32 printed = 0;
    for (u32 n = 1; n <= total && printed < 40; ++n) {
        const Entry& e = ring[(h - n) & (kSize - 1)];
        if (e.handle == 0) {
            continue;
        }
        const bool hit = want_handle != 0 && e.handle == want_handle;
        if (!hit && n > 24) {
            continue; // the newest 24 always, older ones only when they match
        }
        LOG_CRITICAL(Render_Vulkan,
                     "FREEIMG-RING[{}]{}: handle={:#x} guest={:#x}+{:#x} {}x{} lvls={} tick={} "
                     "last_used={}",
                     n, hit ? " <== FAULT IMAGE" : "", e.handle, e.guest_addr, e.guest_size,
                     e.width, e.height, e.levels, e.tick, e.last_used);
        ++printed;
    }
    if (h == 0) {
        LOG_CRITICAL(Render_Vulkan, "FREEIMG-RING: empty");
    }
}
} // namespace VideoCore::FreeImgRing

// [FINDLOG] ring of FindImage resolutions for the watched shader (see TextureCache::FindImage).
namespace VideoCore::FindLog {
struct Entry {
    u64 tag;
    u64 want_addr;
    u64 want_size;
    u64 got_addr;
    u64 got_size;
    u32 slot;
    u32 want_w;
    u32 want_h;
    u32 want_lvls;
    u32 got_w;
    u32 got_h;
    u32 got_lvls;
    u32 view_lvl;
    u32 view_lvls;
    int view_mip;
    int view_slice;
    u32 fmt;
};
static constexpr size_t kSize = 64;
static Entry ring[kSize];
static size_t head = 0;
static u64 total = 0;
void Note(const Entry& e) {
    ring[head] = e;
    head = (head + 1) % kSize;
    ++total;
}
void Dump() {
    const size_t n = std::min<u64>(total, kSize);
    LOG_CRITICAL(Render_Vulkan, "FINDLOG: total={} showing last {}", total, n);
    for (size_t k = 0; k < n; k++) {
        const auto& e = ring[(head + kSize - n + k) % kSize];
        LOG_CRITICAL(Render_Vulkan,
                     "FINDLOG[{}]: tag={:#x} want={:#x}+{:#x} {}x{} lvls={} -> slot={} got={:#x}+{:#x} "
                     "{}x{} lvls={} fmt={} view(lvl={} lvls={}) mip={} slice={}",
                     k, e.tag, e.want_addr, e.want_size, e.want_w, e.want_h, e.want_lvls, e.slot,
                     e.got_addr, e.got_size, e.got_w, e.got_h, e.got_lvls, e.fmt, e.view_lvl,
                     e.view_lvls, e.view_mip, e.view_slice);
    }
}
} // namespace VideoCore::FindLog

namespace VideoCore {

// [debug] SHADPS4_TEXSTAT=1: health of the texture path once every five seconds. The visible
// symptom is textures that do not appear at the start of gameplay, so count what actually
// happens: how many images are created and freed, how often an overlap cannot be resolved (the
// image is then thrown away and made anew), and - the important one - how many refreshes upload
// nothing because every mip hash matched, which is exactly how a texture stays empty.
namespace texstat {
std::atomic<u64> img_new{0}, img_free{0}, overlap_fail{0};
// [postmortem] Video-ring 960x540 (guest_size 0x240000, levels=1) bookkeeping.
// Uncapped and cumulative for the whole run (never exchanged): ring_hits counts
// the ring signature passing the right-overlap branch of ResolveOverlap,
// ring_mipfail counts the same signature failing MipOf (the capped MIPFAIL log
// stops printing long before the fatal window), ring_migrate counts RINGMIG
// migrations (SHADPS4_RINGMIG=1).
std::atomic<u64> ring_hits{0}, ring_mipfail{0}, ring_migrate{0};
// [Stage1] ring signature also arriving through the LEFT overlap branch (only counted,
// never migrated here); ring_migrate_left reserved for Stage 6 left-branch migration.
std::atomic<u64> ring_hits_left{0}, ring_migrate_left{0};
// [R6 D5] equal-address recreators (the uncounted insert+FreeImage pairs that explain
// new=6/free=204): depth<->color ping-pong recreate in ResolveDepthOverlap and the stencil
// insert in FindDepthTarget. Cumulative, printed in TEXSTAT and POSTMORTEM-TEX.
std::atomic<u64> depth_recreate{0}, stencil_new{0};
// [R9 RINGPOOL] pool telemetry (all cumulative): perfect-match hits on members, adoptions /
// rejected adoptions, evictions, tail-owner demotions, CpuDirty writeback skips (each = a saved
// decoder frame), membership carve-outs on FreeImage, and writebacks skipped on a freed slot.
std::atomic<u64> pool_hits{0}, pool_adopt{0}, pool_adopt_reject{0}, pool_evict{0};
std::atomic<u64> pool_demote{0}, pool_cpuskip{0}, pool_carveout{0}, pool_wb_dead{0};
// [R10 RINGPOOL_DEPTH] depth-half: recreate-avoidance adoptions in ResolveDepthOverlap and
// perfect-match hits on depth members (aspect twins living beside colour members).
std::atomic<u64> pool_depth_adopt{0}, pool_depth_hit{0};
// [Stage1 TEXTIME] SHADPS4_TEXTIME=1: parser-thread microsecond buckets of the texture
// path. Cumulative; exchanged in the 5s TEXSTAT window, loaded in POSTMORTEM-TEX.
std::atomic<u64> hash_us{0}, copy_us{0}, detile_us{0}, create_us{0}, overlap_us{0};
inline bool TexTimeEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_TEXTIME");
        return e && e[0] == '1';
    }();
    return on;
}
// Behaviour-neutral when the flag is off: the ctor only reads a cached bool and takes
// no clock, so no atomic is touched (bit-identical to the reference build).
struct TexTimer {
    std::atomic<u64>& bucket;
    std::chrono::steady_clock::time_point t0;
    bool on;
    explicit TexTimer(std::atomic<u64>& b) : bucket(b), on(TexTimeEnabled()) {
        if (on) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    ~TexTimer() {
        if (on) {
            bucket.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count(),
                             std::memory_order_relaxed);
        }
    }
};
// Recreations: an image whose resources turned out too small is rebuilt from scratch and the old
// one copied into it. Done a few times it is normal; done hundreds of times a second it means
// textures are constantly being thrown away and made anew - which on screen looks exactly like
// textures failing to load.
std::atomic<u64> expand{0}, expand_bytes{0}, depth_overlap{0}, free_chance{0};
// Evictions by the collector, plus the memory numbers that drive it. A texture thrown out here
// has to be created and uploaded all over again the moment it is used - which is what "textures
// keep failing to load" looks like from the outside.
std::atomic<u64> gc_free{0}, gc_runs{0};
// Where exactly images get thrown away during overlap resolution. Raising the collector
// threshold stopped the collector but not the churn, so the remaining source must be here.
std::atomic<u64> f_blockdim{0}, f_tile{0}, f_left_target{0}, f_copymip{0}, f_unreach{0};
// Sergey's observation: after Drake dies the second attempt shows textures perfectly. So the
// churn only happens while the game is streaming texture data in for the first time. Measure
// the re-upload cost directly: bytes pushed to the GPU, and how often the SAME image is
// uploaded more than once inside one reporting window.
std::atomic<u64> upload_bytes{0}, reupload{0};
std::mutex seen_mutex;
std::unordered_map<u64, u32> seen_uploads;
std::atomic<u64> mem_used{0}, mem_trigger{0}, mem_pressure{0}, mem_critical{0};
std::atomic<u64> refresh_total{0}, refresh_empty{0}, refresh_upload{0}, refresh_skip_gpu{0};

bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_TEXSTAT");
        return e && e[0] == '1';
    }();
    return on;
}

// [R6 D5] SHADPS4_RINGSENSE=1: gate for the pure-telemetry sensor logs (RINGREC, WRMSTUCK,
// FENCEWAIT, parser heartbeat). Independent of SHADPS4_STUCKLOG by design (that env kept the
// WaitRegMem sensor dead in the field). Counters themselves increment unconditionally.
bool RingSenseEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_RINGSENSE");
        return e && e[0] == '1';
    }();
    return on;
}

void Report() {
    if (!Enabled()) {
        return;
    }
    static std::atomic<s64> next_ms{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    s64 expected = next_ms.load(std::memory_order_relaxed);
    if (now < expected || !next_ms.compare_exchange_strong(expected, now + 5000)) {
        return;
    }
    if (expected == 0) {
        return;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "TEXSTAT new={} free={} ovl_fail={} | refresh={} upload={} empty={} "
                 "skip={} | EXPAND={} expand_MB={} chance_free={} depth_ovl={} | GC_runs={} "
                 "GC_free={} vram_MB={} trigger_MB={} | FREE blockdim={} tile={} "
                 "left_target={} copymip={} | uploaded_MB={} reuploads={} | ring_hits={} "
                 "ring_hits_left={} ring_mipfail={} ring_migrate={} ring_migrate_left={} | "
                 "depth_recreate={} stencil_new={} | "
                 "pool hit={} adopt={}/{} evict={} demote={} cpuskip={} carve={} wbdead={} depth_adopt={} depth_hit={} | "
                 "TEXTIME hash_us={} copy_us={} detile_us={} create_us={} overlap_us={}",
                 img_new.exchange(0), img_free.exchange(0), overlap_fail.exchange(0),
                 refresh_total.exchange(0), refresh_upload.exchange(0), refresh_empty.exchange(0),
                 refresh_skip_gpu.exchange(0), expand.exchange(0),
                 expand_bytes.exchange(0) >> 20, free_chance.exchange(0),
                 depth_overlap.exchange(0), gc_runs.exchange(0), gc_free.exchange(0),
                 mem_used.load() >> 20, mem_trigger.load() >> 20, f_blockdim.exchange(0),
                 f_tile.exchange(0), f_left_target.exchange(0), f_copymip.exchange(0),
                 upload_bytes.exchange(0) >> 20,
                 [] {
                     std::scoped_lock lk{seen_mutex};
                     u64 n = 0;
                     for (const auto& [addr, count] : seen_uploads) {
                         if (count > 1) {
                             n += count - 1;
                         }
                     }
                     seen_uploads.clear();
                     return n;
                 }(),
                 ring_hits.load(std::memory_order_relaxed),
                 ring_hits_left.load(std::memory_order_relaxed),
                 ring_mipfail.load(std::memory_order_relaxed),
                 ring_migrate.load(std::memory_order_relaxed),
                 ring_migrate_left.load(std::memory_order_relaxed),
                 depth_recreate.load(std::memory_order_relaxed),
                 stencil_new.load(std::memory_order_relaxed), pool_hits.load(),
                 pool_adopt.load(), pool_adopt_reject.load(), pool_evict.load(),
                 pool_demote.load(), pool_cpuskip.load(), pool_carveout.load(),
                 pool_wb_dead.load(), pool_depth_adopt.load(), pool_depth_hit.load(), hash_us.exchange(0),
                 copy_us.exchange(0), detile_us.exchange(0), create_us.exchange(0),
                 overlap_us.exchange(0));
}

// [postmortem] Partial-window TEXSTAT for the crash handler: loads only, no
// exchange (the periodic Report keeps its own windows), no locks, no allocations.
void DumpPostmortem() {
    LOG_CRITICAL(Render_Vulkan,
                 "POSTMORTEM-TEX: new={} free={} chance_free={} upload_MB={} ring_hits={} "
                 "ring_hits_left={} ring_mipfail={} ring_migrate={} ring_migrate_left={} | "
                 "depth_recreate={} stencil_new={} | parser_op={:#x} op_seq={} | "
                 "pool hit={} adopt={}/{} evict={} demote={} cpuskip={} carve={} wbdead={} depth_adopt={} depth_hit={} | "
                 "TEXTIME hash_us={} copy_us={} detile_us={} create_us={} overlap_us={}",
                 img_new.load(std::memory_order_relaxed), img_free.load(std::memory_order_relaxed),
                 free_chance.load(std::memory_order_relaxed),
                 upload_bytes.load(std::memory_order_relaxed) >> 20,
                 ring_hits.load(std::memory_order_relaxed),
                 ring_hits_left.load(std::memory_order_relaxed),
                 ring_mipfail.load(std::memory_order_relaxed),
                 ring_migrate.load(std::memory_order_relaxed),
                 ring_migrate_left.load(std::memory_order_relaxed),
                 depth_recreate.load(std::memory_order_relaxed),
                 stencil_new.load(std::memory_order_relaxed),
                 Postmortem::g_gfx_cur_op.load(std::memory_order_relaxed),
                 Postmortem::g_gfx_op_seq.load(std::memory_order_relaxed), pool_hits.load(),
                 pool_adopt.load(), pool_adopt_reject.load(), pool_evict.load(),
                 pool_demote.load(), pool_cpuskip.load(), pool_carveout.load(),
                 pool_wb_dead.load(), pool_depth_adopt.load(), pool_depth_hit.load(),
                 hash_us.load(std::memory_order_relaxed), copy_us.load(std::memory_order_relaxed),
                 detile_us.load(std::memory_order_relaxed),
                 create_us.load(std::memory_order_relaxed),
                 overlap_us.load(std::memory_order_relaxed));
}
} // namespace texstat

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

// [debug] SHADPS4_IMGWATCH=<hex guest base>: log the life of every image created inside the
// 1 MiB window starting there. Used to find out why the exposure pyramid still ends up with
// standalone per-level images instead of mips of one parent.
static bool ImgWatched(VAddr addr) {
    static const VAddr watch_base = [] {
        const char* e = std::getenv("SHADPS4_IMGWATCH");
        return e ? static_cast<VAddr>(std::strtoull(e, nullptr, 16)) : VAddr{0};
    }();
    return watch_base != 0 && addr >= watch_base && addr < watch_base + 0x100000;
}

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           AmdGpu::Liverpool* liverpool_, BufferCache& buffer_cache_,
                           PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // [postmortem] Register the texture partial-window dump for the crash handler.
    // Before any early return below - the hook must exist on every configuration.
    Postmortem::RegisterPostmortemDump(&texstat::DumpPostmortem);

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    pressure_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_PRESSURE_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    // [experiment] SHADPS4_TEXGC_TRIGGER_MB overrides where collection starts.
    // Measured default on a 14 GB budget: collection begins at 3.1 GB while its own "pressure"
    // and "critical" marks sit at 9.5 and 12.7 GB. So between 3 and 9 GB the mildest tier runs
    // on every submit - 151 runs and up to 1500 evictions per five seconds - throwing out
    // textures untouched for a mere 16 ticks and forcing 300 re-uploads a second. With eleven
    // gigabytes still free that work buys nothing and shows up on screen as textures that keep
    // popping in and out.
    if (const char* e = std::getenv("SHADPS4_TEXGC_TRIGGER_MB")) {
        const u64 mb = std::strtoull(e, nullptr, 10);
        if (mb > 0) {
            trigger_gc_memory = mb * 1024ULL * 1024ULL;
            LOG_CRITICAL(Render_Vulkan,
                         "TEXGC: порог сборки текстур поднят до {} МБ (было {} МБ, давление {} МБ)",
                         mb, static_cast<u64>((device_local_memory - mem_threshold) / 2) >> 20,
                         pressure_gc_memory >> 20);
        }
    }
}

TextureCache::~TextureCache() = default;

void TextureCache::SetGpuModified(Image& image) {
    // [D1] Definition lives here (not the header) so texture_cache.h needs only a forward
    // declaration of Scheduler; CurrentTick() requires the complete type.
    image.flags |= ImageFlagBits::GpuModified;
    image.modified_tick = scheduler.CurrentTick();
}

bool TextureCache::RingPoolEnabled() {
    // [R9] SHADPS4_RINGPOOL=1, but only meaningful with the batch drain (ASC ordering + single
    // Finish live there); refuse otherwise so the pool never runs on the per-image path.
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_RINGPOOL");
        if (!(e && e[0] == '1')) {
            return false;
        }
        const char* b = std::getenv("SHADPS4_DLIMG_BATCH");
        if (!(b && b[0] == '1')) {
            LOG_CRITICAL(Render_Vulkan, "RINGPOOL: requires DLIMG_BATCH=1, pool disabled");
            return false;
        }
        return true;
    }();
    return on;
}

bool TextureCache::RingPoolDepthEnabled() {
    // [R10] Depth adoption rides on top of the colour pool: only meaningful with RINGPOOL on.
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_RINGPOOL_DEPTH");
        return e && e[0] == '1' && RingPoolEnabled();
    }();
    return on;
}

void TextureCache::AdoptIntoRingPool(ImageId image_id) { // called under `mutex` (FindImage)
    if (ring_pool.size() >= kRingPoolCap) {
        // Age eviction, ONLY a member whose backing is proven current (down>=mod || !GpuModified).
        ImageId victim{};
        u64 oldest = std::numeric_limits<u64>::max();
        for (const ImageId id : ring_pool) {
            Image& im = slot_images[id];
            const bool evictable = False(im.flags & ImageFlagBits::GpuModified) ||
                                   im.downloaded_tick >= im.modified_tick;
            if (evictable && im.tick_accessed_last < oldest) {
                oldest = im.tick_accessed_last;
                victim = id;
            }
        }
        if (!victim) {
            // No safe victim -> never inline-Finish just to evict: fall through to old behaviour.
            texstat::pool_adopt_reject.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        FreeImage(victim); // membership removed by DeleteImage carve-out
        const u64 n = texstat::pool_evict.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 8 || n % 256 == 0) {
            LOG_CRITICAL(Render_Vulkan, "RINGPOOL evict #{}: id={} pool={}", n, victim.index,
                         ring_pool.size());
        }
    }
    Image& im = slot_images[image_id];
    im.flags |= ImageFlagBits::RingPooled;
    ring_pool.push_back(image_id);
    const u64 n = texstat::pool_adopt.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 16 || n % 256 == 0) {
        LOG_CRITICAL(Render_Vulkan, "RINGPOOL adopt #{}: id={} addr={:#x} pool={}", n,
                     image_id.index, im.info.guest_address, ring_pool.size());
    }
}

// [DLIMG-TIME] where ProcessDownloadImages spends its time (FRAMECHAIN: 22.5 ms/frame even with
// readbackLinearImages=false, liveloop182): lock wait vs Finish vs writebacks, printed every 5 s.
namespace {
struct DlimgTime {
    std::atomic<u64> calls{0}, lock_ns{0}, finish_ns{0}, wb_ns{0}, total_ns{0}, images{0};
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
    void MaybeReport() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report < std::chrono::seconds(5)) {
            return;
        }
        last_report = now;
        const u64 c = calls.exchange(0);
        LOG_CRITICAL(Render_Vulkan,
                     "DLIMG-TIME: calls={} images={} lock={}us finish={}us wb={}us total={}us (per 5 s)",
                     c, images.exchange(0), lock_ns.exchange(0) / 1000, finish_ns.exchange(0) / 1000,
                     wb_ns.exchange(0) / 1000, total_ns.exchange(0) / 1000);
    }
} g_dlimg_time;
} // namespace

void TextureCache::ProcessDownloadImages() {
    const auto dl_t0 = std::chrono::steady_clock::now();
    // [DLIMG-GAP] SHADPS4_DLIMG_GAP_MS=N: at most one synchronous batch per N ms from THIS entry
    // point (OnSubmit calls it ~4600 times/s; DLIMG_MINGAP_MS only gates the EOP path, 194).
    // Between batches the CPU reads the images through the readback fault path as usual.
    static const u64 gap_ms = [] {
        const char* e = std::getenv("SHADPS4_DLIMG_GAP_MS");
        return e ? static_cast<u64>(std::strtoull(e, nullptr, 10)) : u64{0};
    }();
    static std::chrono::steady_clock::time_point last_batch{};
    if (gap_ms != 0 && dl_t0 - last_batch < std::chrono::milliseconds(gap_ms)) {
        return;
    }
    last_batch = dl_t0;
    struct DlTotal {
        std::chrono::steady_clock::time_point t0;
        ~DlTotal() {
            g_dlimg_time.total_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count(),
                std::memory_order_relaxed);
            g_dlimg_time.calls.fetch_add(1, std::memory_order_relaxed);
            g_dlimg_time.MaybeReport();
        }
    } dl_total{dl_t0};
    // [A-BATCH] k images x (Finish + writeback) -> record ALL copies, ONE Finish, then all
    // writebacks. Bit-identical guest semantics: every writeback still happens strictly
    // after its copy retired on GPU. Chunked by staging budget: the 32MB Download buffer
    // is shared with Precise buffer readbacks; a wrap would hand out the region of an
    // earlier copy before its memcpy (silent v1 data loss), so flush the chunk early.
    static const bool batch = [] {
        const char* e = std::getenv("SHADPS4_DLIMG_BATCH");
        return e && e[0] == '1';
    }();
    // [R9] Lock order strictly mutex -> download_images_mutex (as in the FindImage branches).
    // Only taken under RINGPOOL - otherwise behaviour is bit-identical to before.
    std::unique_lock lk_tc{mutex, std::defer_lock};
    if (RingPoolEnabled()) {
        lk_tc.lock();
    }
    std::unique_lock lk{download_images_mutex};
    g_dlimg_time.lock_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - dl_t0)
                                       .count(),
                                   std::memory_order_relaxed);
    g_dlimg_time.images.fetch_add(download_images.size(), std::memory_order_relaxed);
    if (!batch) {
        for (const ImageId image_id : download_images) {
            DownloadImageMemory(image_id, true);
        }
        download_images.clear();
        return;
    }
    struct PendingWb {
        VAddr addr;
        u8* src;
        u32 size;
        ImageId id;
        u64 tick; // [D1->T_d] CurrentTick at copy record time of this chunk
    };
    boost::container::small_vector<PendingWb, 32> wbs;
    u64 chunk_bytes = 0;
    constexpr u64 kChunkLimit = 8u << 20; // 8 MB, well under the shared 32MB Download buffer
    const auto flush_chunk = [&] {
        if (wbs.empty()) {
            return;
        }
        const auto f0 = std::chrono::steady_clock::now();
        scheduler.Finish();
        g_dlimg_time.finish_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - f0)
                                             .count(),
                                         std::memory_order_relaxed);
        const auto w0 = std::chrono::steady_clock::now();
        struct WbTime {
            std::chrono::steady_clock::time_point t;
            ~WbTime() {
                g_dlimg_time.wb_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - t)
                                                 .count(),
                                             std::memory_order_relaxed);
            }
        } wb_time{w0};
        for (const auto& wb : wbs) {
            static const bool clearmod = [] {
                const char* e = std::getenv("SHADPS4_DLIMG_CLEARMOD");
                return e && e[0] == '1';
            }();
            // [R9] Finish executes deferred DeleteImage ops: the slot may be freed now. Writing
            // a freed slot's pages (reused) or its flags is a use-after-free.
            if (!slot_images.is_allocated(wb.id)) {
                texstat::pool_wb_dead.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            Image& img = slot_images[wb.id];
            // [R9] CpuDirty-skip: the decoder wrote after the copy was recorded, so this download
            // is stale - writing it back would clobber the fresh decoder frame (v1). Same v1
            // precedent as RINGMIG_DL, which also drops GpuModified and skips.
            if (RingPoolEnabled() && True(img.flags & ImageFlagBits::RingPooled) &&
                True(img.flags & (ImageFlagBits::CpuDirty | ImageFlagBits::MaybeCpuDirty))) {
                texstat::pool_cpuskip.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const bool wrote = Core::Memory::Instance()->TryWriteBacking(
                std::bit_cast<u8*>(wb.addr), wb.src, wb.size);
            if (wrote) {
                // [D1->T_d] tick of THIS chunk's copy record: the cmdbuf is closed by Finish, its
                // content includes every bind up to T_d; a post-drain bind gets mod >= T_d+1.
                img.downloaded_tick = wb.tick;
            }
            if (clearmod && wrote && img.info.resources.levels == 1) {
                img.flags &= ~ImageFlagBits::GpuModified;
            }
        }
        wbs.clear();
        chunk_bytes = 0;
    };
    // [DLIMG-ASYNC] pool slot for this batch; -1 = synchronous path (env off or pool exhausted).
    static const bool dlimg_async = [] {
        const char* e = std::getenv("SHADPS4_DLIMG_ASYNC");
        return e && e[0] == '1';
    }();
    constexpr u64 kAsyncSlotBytes = 8u << 20;
    constexpr size_t kAsyncSlots = 4;
    int async_slot = -1;
    u64 async_used = 0;
    static u64 async_batches = 0, async_fallbacks = 0, async_deferred_wbs = 0;
    const auto async_acquire = [&]() -> int {
        if (!dlimg_async) {
            return -1;
        }
        if (dlimg_pool.empty()) {
            dlimg_pool.resize(kAsyncSlots);
        }
        for (size_t i = 0; i < dlimg_pool.size(); ++i) {
            if (dlimg_busy_mask.load(std::memory_order_acquire) & (1u << i)) {
                continue;
            }
            if (!dlimg_pool[i].buf) {
                dlimg_pool[i].buf = std::make_unique<Buffer>(instance, scheduler,
                                                             MemoryUsage::Download, 0, AllFlags,
                                                             kAsyncSlotBytes);
            }
            dlimg_busy_mask.fetch_or(1u << i, std::memory_order_acq_rel);
            return static_cast<int>(i);
        }
        ++async_fallbacks;
        if (async_fallbacks <= 8 || async_fallbacks % 1024 == 0) {
            LOG_CRITICAL(Render_Vulkan, "DLIMG-ASYNC: pool exhausted ({} times) - synchronous batch",
                         async_fallbacks);
        }
        return -1;
    };
    // Deferred writeback of an async slot: runs when the copies' tick retired (parser thread,
    // PopPendingOperations), same thread and same assumptions as DeleteImage's deferred op.
    const auto async_flush = [&] {
        if (async_slot < 0 || wbs.empty()) {
            return;
        }
        static const bool clearmod = [] {
            const char* e = std::getenv("SHADPS4_DLIMG_CLEARMOD");
            return e && e[0] == '1';
        }();
        ++async_batches;
        async_deferred_wbs += wbs.size();
        if (async_batches <= 4 || async_batches % 2048 == 0) {
            LOG_CRITICAL(Render_Vulkan,
                         "DLIMG-ASYNC: batch #{} slot {} images={} bytes={} deferred to tick {} "
                         "(fallbacks {})",
                         async_batches, async_slot, wbs.size(), async_used,
                         scheduler.CurrentTick(), async_fallbacks);
        }
        std::vector<PendingWb> deferred(wbs.begin(), wbs.end());
        const int slot = async_slot;
        const bool clearmod_v = clearmod; // function-static cannot be captured directly
        // [EOPRETIRE] SHADPS4_EOPRETIRE=1: the EOP/ReleaseMem labels are deferred to the same
        // priority queue AFTER this op, so the guest sees the label only once the data is in
        // place (hardware order) while the parser never blocks. The op runs on the priority
        // thread: it touches guest memory only - no slot_images access from there.
        static const bool eop_retire = [] {
            const char* e = std::getenv("SHADPS4_EOPRETIRE");
            return e && e[0] != '\0' && e[0] != '0'; // any level (1 or 2)
        }();
        if (eop_retire) {
            for (const auto& wb : deferred) {
                if (slot_images.is_allocated(wb.id)) {
                    // Nobody can observe the guest bytes before the label; mark now (parser
                    // thread) so the dedup at the next drain does not re-download.
                    slot_images[wb.id].downloaded_tick = wb.tick;
                }
            }
            scheduler.DeferPriorityOperation([this, slot, deferred = std::move(deferred)] {
                for (const auto& wb : deferred) {
                    Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(wb.addr), wb.src,
                                                              wb.size);
                }
                dlimg_busy_mask.fetch_and(~(1u << slot), std::memory_order_acq_rel);
            });
            wbs.clear();
            chunk_bytes = 0;
            async_slot = -1;
            async_used = 0;
            return;
        }
        scheduler.DeferOperation([this, slot, deferred = std::move(deferred), clearmod_v] {
            for (const auto& wb : deferred) {
                if (!slot_images.is_allocated(wb.id)) {
                    texstat::pool_wb_dead.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                Image& img = slot_images[wb.id];
                if (RingPoolEnabled() && True(img.flags & ImageFlagBits::RingPooled) &&
                    True(img.flags & (ImageFlagBits::CpuDirty | ImageFlagBits::MaybeCpuDirty))) {
                    texstat::pool_cpuskip.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                const bool wrote = Core::Memory::Instance()->TryWriteBacking(
                    std::bit_cast<u8*>(wb.addr), wb.src, wb.size);
                if (wrote) {
                    img.downloaded_tick = wb.tick;
                }
                if (clearmod_v && wrote && img.info.resources.levels == 1) {
                    img.flags &= ~ImageFlagBits::GpuModified;
                }
            }
            dlimg_busy_mask.fetch_and(~(1u << slot), std::memory_order_acq_rel);
        });
        wbs.clear();
        chunk_bytes = 0;
        async_slot = -1;
        async_used = 0;
    };
    // [R9] ASC by modified_tick over the WHOLE queue before chunking: overlapping members are
    // written oldest-first so the overlap zone always ends with the freshest data (owner last).
    // Chunks are written in order, so the between-chunk order is preserved for free.
    boost::container::small_vector<ImageId, 32> ordered(download_images.begin(),
                                                        download_images.end());
    if (RingPoolEnabled()) {
        std::ranges::sort(ordered, {}, [&](ImageId id) { return slot_images[id].modified_tick; });
    }
    for (const ImageId image_id : ordered) {
        Image& image = slot_images[image_id];
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            continue;
        }
        const u32 download_size = image.info.pitch * image.info.size.height *
                                  image.info.size.depth * image.info.resources.layers *
                                  (image.info.num_bits / 8);
        ASSERT(download_size <= image.info.guest_size);
        if (async_slot < 0 && dlimg_async && wbs.empty()) {
            async_slot = async_acquire();
        }
        if (async_slot >= 0) {
            if (async_used + download_size > kAsyncSlotBytes) {
                async_flush();
                async_slot = async_acquire();
            }
        } else if (chunk_bytes + download_size > kChunkLimit) {
            flush_chunk();
        }
        u8* download = nullptr;
        u64 offset = 0;
        vk::Buffer staging_handle{};
        if (async_slot >= 0 && download_size <= kAsyncSlotBytes) {
            Buffer& pool_buf = *dlimg_pool[async_slot].buf;
            offset = async_used;
            download = pool_buf.mapped_data.data() + offset;
            staging_handle = pool_buf.Handle();
            async_used += (download_size + 255) & ~u64{255};
        } else {
            if (async_slot >= 0) {
                // an image larger than a slot: finish the async batch, take the sync ring
                async_flush();
            }
            auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
            const auto mapped = download_buffer.Map(download_size);
            download_buffer.Commit();
            download = mapped.first;
            offset = mapped.second;
            staging_handle = download_buffer.Handle();
        }
        const vk::BufferImageCopy image_download = {
            .bufferOffset = offset,
            .bufferRowLength = image.info.pitch,
            .bufferImageHeight = image.info.size.height,
            .imageSubresource =
                {
                    .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                            : vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = image.info.resources.layers,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {image.info.size.width, image.info.size.height,
                            image.info.size.depth},
        };
        // Re-fetch per image: Finish() inside flush_chunk swaps the worker cmdbuf.
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      {});
        cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 staging_handle, image_download);
        wbs.push_back({image.info.guest_address, download, download_size, image_id,
                       scheduler.CurrentTick()});
        chunk_bytes += download_size;
    }
    if (async_slot >= 0) {
        async_flush();
    }
    flush_chunk();
    // [R9] Single-owner invariant: after the full ASC writeback, keep GpuModified only on the
    // tail (max modified_tick); demote the rest so following drains carry the single owner in the
    // MINGAP rhythm. Correctness rests on the ASC order + CpuDirty-skip above, not this - this is
    // the optimization. Done here (under mutex), NEVER in SetGpuModified (no lock there).
    if (RingPoolEnabled()) {
        ImageId owner{};
        u64 best = 0;
        for (const ImageId id : ordered) {
            if (!slot_images.is_allocated(id)) {
                continue;
            }
            Image& im = slot_images[id];
            if (True(im.flags & ImageFlagBits::RingPooled) &&
                True(im.flags & ImageFlagBits::GpuModified) && im.modified_tick >= best) {
                best = im.modified_tick;
                owner = id;
            }
        }
        for (const ImageId id : ordered) {
            if (!slot_images.is_allocated(id) || id == owner) {
                continue;
            }
            Image& im = slot_images[id];
            if (True(im.flags & ImageFlagBits::RingPooled) &&
                True(im.flags & ImageFlagBits::GpuModified)) {
                im.flags &= ~ImageFlagBits::GpuModified; // content already written in ASC order
                texstat::pool_demote.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    const auto [download, offset] = download_buffer.Map(download_size);
    download_buffer.Commit();
    const vk::BufferImageCopy image_download = {
        .bufferOffset = offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             download_buffer.Handle(), image_download);

    if (sync) {
        scheduler.Finish();
        const bool wrote = Core::Memory::Instance()->TryWriteBacking(
            std::bit_cast<u8*>(image.info.guest_address), download, download_size);
        // [C1 DLIMG-CLEARMOD] After a successful synchronous download the backing IS the GPU
        // content: clear GpuModified so the ring-migration guard (b8) passes and GC stops
        // re-downloading the same image. mip0-only download => only for levels==1 (multi-mip
        // storage pyramids would falsely claim mips>0 as flushed).
        if (wrote) {
            // [D1] The CPU backing now holds GPU content as of modified_tick.
            image.downloaded_tick = image.modified_tick;
        }
        static const bool clearmod = [] {
            const char* e = std::getenv("SHADPS4_DLIMG_CLEARMOD");
            return e && e[0] == '1';
        }();
        if (clearmod && wrote && image.info.resources.levels == 1) {
            image.flags &= ~ImageFlagBits::GpuModified;
            static std::atomic<u64> n{0};
            const u64 k = n.fetch_add(1, std::memory_order_relaxed) + 1;
            if (k <= 4 || k % 1024 == 0) {
                LOG_CRITICAL(Render_Vulkan, "DLIMG-CLEARMOD #{}: addr={:#x} size={:#x}", k,
                             image.info.guest_address, download_size);
            }
        }
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr), download,
                                                          download_size);
            });
    }
}

namespace {
// [TEXHASH] XXH3 over GUEST memory is not a CPU-bound hash: the pages may be write-protected for
// GPU tracking, so every 4KB can cost a fault + readback. Report the achieved throughput so a
// pathological range is named instead of guessed. Measures the real range - a synthetic memcpy
// benchmark would fault the very pages it is trying to measure and destroy the evidence.
std::atomic<u64> g_texhash_slow_lines{0};
std::atomic<u64> g_texhash_slow_suppressed{0};
std::atomic<u64> g_hashcap_lines{0};

// [HASHCAP] SHADPS4_HASHCAP=<KB>, default 64, 0 = no cap (old behaviour).
// Hashing guest memory is not a CPU-bound operation: the pages are write-protected for GPU
// tracking, so every 4KB costs a fault plus a readback (measured ~0.44 ms/page - an 8 MB render
// target took 893 ms on the parser thread and froze the whole frame loop). Above the cap the
// honest answer is "assume it changed" - the same verdict the neighbouring invalidation branch
// reaches - which costs one upload instead of a second of dead parser.
u64 HashCapBytes() {
    static const u64 cap = [] {
        const char* e = std::getenv("SHADPS4_HASHCAP");
        const u64 kb = e != nullptr ? std::strtoull(e, nullptr, 10) : 64ull;
        return kb * 1024ull;
    }();
    return cap;
}
bool HashCapExceeded(u64 size) {
    const u64 cap = HashCapBytes();
    return cap != 0 && size > cap;
}
void NoteHashCapSkip(u64 addr, u64 size, const char* site) {
    Postmortem::g_hashcap_skipped.fetch_add(1, std::memory_order_relaxed);
    if (g_hashcap_lines.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_CRITICAL(Render_Vulkan, "HASHCAP: skipped hash site={} addr={:#x} size={}KB -> {}",
                     site, addr, size >> 10,
                     std::strcmp(site, "maybedirty") == 0 ? "CpuDirty" : "upload");
    }
}
u64 HashGuestRange(const u8* addr, u64 size, const char* site) {
    // Names the work in HOSTSTALL even between packets, where the phase clock is not armed.
    Postmortem::HostWaitScope host_scope{"TextureHash", reinterpret_cast<u64>(addr), size};
    Parserprof::PhaseProbe phase{Parserprof::PhHash};
    const auto t0 = std::chrono::steady_clock::now();
    const u64 hash = XXH3_64bits(addr, size);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    if (us > 5000) {
        if (g_texhash_slow_lines.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_CRITICAL(Render_Vulkan,
                         "TEXHASH: slow site={} addr={:#x} size={}KB took={}us => {}MB/s", site,
                         reinterpret_cast<u64>(addr), size >> 10, us,
                         us > 0 ? (size * 1000000ull / static_cast<u64>(us)) >> 20 : 0);
        } else {
            g_texhash_slow_suppressed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return hash;
}
} // Anonymous namespace

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // [HASHCAP] Too big to read here. Fall back to the exact semantics of the first branch
        // of InvalidateMemory ("the image was definitely touched"): flag CpuDirty and untrack.
        // CpuDirty is part of ImageFlagBits::Dirty, so RefreshImage cannot take its early-out,
        // and because CpuDirty is set it also skips the MaybeCpuDirty probe - the image is
        // re-uploaded in full. Strictly conservative: at worst one extra upload, never a stale
        // image. All three callers of this function (InvalidateMemory, UntrackImageHead,
        // UntrackImageTail) inherit that guarantee.
        if (HashCapExceeded(image.info.guest_size)) {
            NoteHashCapSkip(image.info.guest_address, image.info.guest_size, "maybedirty");
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
            return;
        }
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = HashGuestRange(addr, image.info.guest_size, "maybedirty");
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::MarkGpuDirtyIfBufferModified(ImageId image_id) {
    std::scoped_lock lock{mutex};
    Image& image = slot_images[image_id];
    const bool modified =
        buffer_cache.IsRegionGpuModified(image.info.guest_address, image.info.guest_size);
    if (modified) {
        image.flags |= ImageFlagBits::GpuDirty;
    }
    static u32 s_cnt = 0;
    if ((++s_cnt % 60) == 0) {
        LOG_WARNING(Render_Vulkan, "BRIDGE: addr={:#x} size={:#x} gpu_buf_modified={}",
                    image.info.guest_address, image.info.guest_size, modified);
    }
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        const auto new_image_id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;

        if (cache_image.info.num_samples == 1 && new_info.num_samples == 1) {
            // Perform depth<->color copy using the intermediate copy buffer.
            if (instance.IsMaintenance8Supported()) {
                new_image.CopyImage(cache_image);
            } else {
                const auto& copy_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
                new_image.CopyImageWithBuffer(cache_image, copy_buffer.Handle(), 0);
            }
        } else if (cache_image.info.num_samples == 1 && new_info.props.is_depth &&
                   new_info.num_samples > 1) {
            // Perform a rendering pass to transfer the channels of source as samples in dest.
            cache_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead, {});
            new_image.Transit(vk::ImageLayout::eDepthAttachmentOptimal,
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {});
            blit_helper.ReinterpretColorAsMsDepth(
                new_info.size.width, new_info.size.height, new_info.num_samples,
                cache_image.info.pixel_format, new_info.pixel_format, cache_image.GetImage(),
                new_image.GetImage());
        } else {
            texstat::depth_overlap.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
        }

        // [R6 D5] RINGREC: the only uncounted insert+FreeImage pair on the ring address - names
        // the equal-address recreator and the ping-pong direction (depth<->color). Counter always
        // (cheap, cumulative); capped LOG_CRITICAL under SHADPS4_RINGSENSE.
        if (requested_info.size.width == 960 && requested_info.size.height == 540 &&
            requested_info.guest_size == 0x240000) {
            texstat::depth_recreate.fetch_add(1, std::memory_order_relaxed);
            if (texstat::RingSenseEnabled()) {
                static std::atomic<u32> pm_n{0};
                const u32 k = pm_n.fetch_add(1, std::memory_order_relaxed) + 1;
                if (k <= 8 || k % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "RINGREC #{}: addr={:#x} binding={} req_depth={} cache_depth={} "
                                 "old_id={} new_id={}",
                                 k, requested_info.guest_address, static_cast<u32>(binding),
                                 static_cast<u32>(requested_info.props.is_depth),
                                 static_cast<u32>(cache_image.info.props.is_depth),
                                 cache_image_id.index, new_image_id.index);
                }
            }
        }
        // [R10 RINGPOOL_DEPTH] Depth-half adoption: the equal-address recreate ping-pongs the
        // ring between depth and colour aspects, freeing the twin each time (depth_recreate=236/
        // window = the flicker source). Instead of freeing the cache twin, adopt it into the pool
        // and let the new image register beside it - both aspects live, and the format-compatible
        // perfect-match in FindImage routes each aspect to its own twin (aspect is the pool key),
        // so after the first round this branch is never reached again (depth_recreate -> 0). The
        // aspect guard is mandatory: never merge depth/color members on one address (fatal flaw).
        if (RingPoolDepthEnabled() && cache_image.info.size.width == 960 &&
            cache_image.info.size.height == 540 && cache_image.info.guest_size == 0x240000 &&
            cache_image.info.resources.levels == 1 && cache_image.info.resources.layers == 1) {
            if (False(cache_image.flags & ImageFlagBits::RingPooled)) {
                AdoptIntoRingPool(cache_image_id);
                const u64 n = texstat::pool_depth_adopt.fetch_add(1, std::memory_order_relaxed) + 1;
                if (n <= 8 || n % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "RINGPOOL-DEPTH adopt #{}: id={} addr={:#x} cache_depth={} "
                                 "req_depth={}",
                                 n, cache_image_id.index, cache_image.info.guest_address,
                                 static_cast<u32>(cache_image.info.props.is_depth),
                                 static_cast<u32>(requested_info.props.is_depth));
                }
            }
            // Do NOT free the cache twin - it stays registered+pooled beside the new image.
            return new_image_id;
        }
        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // [R9] A pool member at a DIFFERENT address is untouchable for resolve: do not free, migrate
    // or adopt it - just skip the candidate (adopt-once creates the new image in FindImage). This
    // both prevents multi-adoption onto one address and hard-disables the DL/RINGMIG branches for
    // members (priority in code, not by env convention). Equal-address falls through and resolves
    // normally (carve-out: DeleteImage strips membership).
    if (RingPoolEnabled() && True(cache_image.flags & ImageFlagBits::RingPooled) &&
        cache_image.info.guest_address != image_info.guest_address) {
        return {merged_image_id, -1, -1};
    }

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                texstat::f_blockdim.fetch_add(1, std::memory_order_relaxed);
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                texstat::f_tile.fetch_add(1, std::memory_order_relaxed);
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        const auto dbg_mip = image_info.MipOf(cache_image.info);
        // [postmortem] Uncapped ring-signature counter: the 960x540 video ring sliding
        // through this branch is the last log line before death in deck2/stable1/stableC,
        // but the capped MIPFAIL/MIPOF logs go quiet long before the fatal window.
        const bool pm_ring_sig =
            image_info.size.width == 960 && image_info.size.height == 540 &&
            image_info.guest_size == 0x240000 && image_info.resources.levels == 1 &&
            cache_image.info.size == image_info.size &&
            cache_image.info.guest_size == image_info.guest_size;
        if (pm_ring_sig) {
            texstat::ring_hits.fetch_add(1, std::memory_order_relaxed);
        }
        // The exposure pyramid renders into the mip levels of one image; if this recognition
        // fails, each level becomes a separate image and the chain reads levels nobody wrote.
        static u32 dbg_reported = 0;
        if (dbg_reported < 20) {
            ++dbg_reported;
            LOG_CRITICAL(Render_Vulkan,
                         "MIPOF: want {:#x} {}x{} mips={} fmt={} tile={:#x} | cached {:#x} {}x{} "
                         "mips={} fmt={} tile={:#x} | offset={:#x} -> mip={}",
                         image_info.guest_address, image_info.size.width, image_info.size.height,
                         image_info.resources.levels, static_cast<u32>(image_info.pixel_format),
                         static_cast<u32>(image_info.tile_mode), cache_image.info.guest_address,
                         cache_image.info.size.width, cache_image.info.size.height,
                         cache_image.info.resources.levels,
                         static_cast<u32>(cache_image.info.pixel_format),
                         static_cast<u32>(cache_image.info.tile_mode),
                         image_info.guest_address - cache_image.info.guest_address, dbg_mip);
        }
        if (auto mip = dbg_mip; mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // [D1 SHADPS4_RINGMIG_DL=1] Tick-versioned migration (right branch). Replaces the b8
        // guard: b8 is set on EVERY bind, so the plain RINGMIG never fires. Here, if the CPU
        // backing is already current (downloaded_tick >= modified_tick) migrate for free;
        // otherwise, only when the GPU retire is fresh (CurrentTick - KnownGpuTick small), do a
        // ~0.4ms inline sync download and migrate; otherwise skip the slide (old behaviour).
        // MANDATORY on migrate: clear GpuModified + erase from download_images, else the next
        // MINGAP/EOP drain writes stale GPU content over the fresh decoder frame at the new
        // address (v1 loss).
        {
            static const bool ring_mig_dl = [] {
                const char* e = std::getenv("SHADPS4_RINGMIG_DL");
                return e && e[0] == '1';
            }();
            const u64 pm_dd = image_info.guest_address - cache_image.info.guest_address;
            if (ring_mig_dl && !RingPoolEnabled() && pm_ring_sig &&
                pm_dd < image_info.guest_size &&
                image_info.resources.layers == 1 && cache_image.info.resources.levels == 1 &&
                cache_image.info.resources.layers == 1 &&
                cache_image.info.tile_mode == image_info.tile_mode &&
                cache_image.info.pixel_format == image_info.pixel_format &&
                cache_image.info.pitch == image_info.pitch &&
                !cache_image.binding.is_target && !cache_image.binding.is_bound &&
                cache_image.tick_accessed_last < scheduler.CurrentTick()) {
                bool ok = False(cache_image.flags & ImageFlagBits::GpuModified);
                if (!ok && cache_image.downloaded_tick >= cache_image.modified_tick) {
                    ok = true; // fast path: CPU backing already current
                } else if (!ok && scheduler.CurrentTick() -
                                          scheduler.GetMasterSemaphore()->KnownGpuTick() <= 2) {
                    DownloadImageMemory(cache_image_id, true); // inline sync (~0.4ms), gated fresh
                    ok = true;
                }
                if (ok) {
                    cache_image.flags &= ~ImageFlagBits::GpuModified; // mandatory (v1)
                    {
                        std::unique_lock lk{download_images_mutex};
                        download_images.erase(cache_image_id);
                    }
                    UntrackImage(cache_image_id);
                    UnregisterImage(cache_image_id);
                    cache_image.info.guest_address = image_info.guest_address;
                    RegisterImage(cache_image_id);
                    TrackImage(cache_image_id);
                    cache_image.flags |= ImageFlagBits::CpuDirty;
                    const u64 n =
                        texstat::ring_migrate.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (n <= 8 || n % 256 == 0) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "RINGMIGDL #{}: migrated ring image {:#x} -> {:#x} (delta {:#x})",
                                     n, image_info.guest_address - pm_dd, image_info.guest_address,
                                     pm_dd);
                    }
                    return {cache_image_id, -1, -1};
                }
                // gate not passed -> fall through to old behaviour (RINGMIG / RINGREJ / free+new)
            }
        }

        // [fix] SHADPS4_RINGMIG=1: the sliding video ring 960x540 lands here every
        // slide and the old image is chance-freed while a new one is created - churn
        // that lengthens exactly the frames the GfxEop tick-writer is starving on.
        // Instead of free+new, migrate the idle cache image to the new address with
        // the same primitives FreeImage/ExpandImage use (Untrack/Unregister ->
        // rebase -> Register/Track) and mark it CpuDirty so it re-uploads from the
        // new address. Flag off = bit-identical old behaviour.
        {
            static const bool ring_mig = [] {
                const char* e = std::getenv("SHADPS4_RINGMIG");
                return e && e[0] == '1';
            }();
            const u64 pm_d = image_info.guest_address - cache_image.info.guest_address;
            if (ring_mig && pm_ring_sig && pm_d < image_info.guest_size &&
                image_info.resources.layers == 1 && cache_image.info.resources.levels == 1 &&
                cache_image.info.resources.layers == 1 &&
                cache_image.info.tile_mode == image_info.tile_mode &&
                cache_image.info.pixel_format == image_info.pixel_format &&
                cache_image.info.pitch == image_info.pitch &&
                !cache_image.binding.is_target && !cache_image.binding.is_bound &&
                False(cache_image.flags & ImageFlagBits::GpuModified) &&
                cache_image.tick_accessed_last < scheduler.CurrentTick()) {
                UntrackImage(cache_image_id);
                UnregisterImage(cache_image_id);
                cache_image.info.guest_address = image_info.guest_address;
                RegisterImage(cache_image_id);
                TrackImage(cache_image_id);
                cache_image.flags |= ImageFlagBits::CpuDirty;
                const u64 n = texstat::ring_migrate.fetch_add(1, std::memory_order_relaxed) + 1;
                if (n <= 8 || n % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "RINGMIG #{}: migrated ring image {:#x} -> {:#x} (delta {:#x})",
                                 n, image_info.guest_address - pm_d, image_info.guest_address,
                                 pm_d);
                }
                return {cache_image_id, -1, -1};
            }
        }

        // [Stage1 RINGREJ] The ring signature reached the right branch but migration did not
        // fire: log the bit-mask of guards that would block it, so the single right-hit's
        // rejection reason is known (not guessed). Unconditional, capped at 8.
        if (pm_ring_sig) {
            const u64 pm_d = image_info.guest_address - cache_image.info.guest_address;
            const u32 mask =
                (u32(!(pm_d > 0 && pm_d < image_info.guest_size)) << 0) |
                (u32(!(image_info.resources.layers == 1 &&
                       cache_image.info.resources.layers == 1))
                 << 1) |
                (u32(cache_image.info.resources.levels != 1) << 2) |
                (u32(cache_image.info.tile_mode != image_info.tile_mode) << 3) |
                (u32(cache_image.info.pixel_format != image_info.pixel_format) << 4) |
                (u32(cache_image.info.pitch != image_info.pitch) << 5) |
                (u32(cache_image.binding.is_target != 0) << 6) |
                (u32(cache_image.binding.is_bound != 0) << 7) |
                (u32(True(cache_image.flags & ImageFlagBits::GpuModified)) << 8) |
                (u32(!(cache_image.tick_accessed_last < scheduler.CurrentTick())) << 9) |
                (u32(cache_image.binding.needs_rebind != 0) << 10);
            static u32 ringrej_reported = 0;
            if (ringrej_reported < 8) {
                ++ringrej_reported;
                LOG_CRITICAL(Render_Vulkan,
                             "RINGREJ-right #{}: mask={:#x} (b0delta b1layers b2levels b3tile "
                             "b4fmt b5pitch b6target b7bound b8gpumod b9tick b10rebind) delta={:#x}",
                             ringrej_reported, mask, pm_d);
            }
        }

        // Image isn't a subresource but a chance overlap.
        // [R9] Never chance-free a pool member (a member on a foreign address already returned at
        // the top of ResolveOverlap; this is belt-and-suspenders against future branch reorders).
        if (safe_to_delete && !(RingPoolEnabled() &&
                                True(cache_image.flags & ImageFlagBits::RingPooled))) {
            texstat::free_chance.fetch_add(1, std::memory_order_relaxed);
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        // [Stage1] Ring signature can also slide in through the LEFT branch (cache image is a
        // subresource of the request). Count it and, on MipOf failure, log the same guard mask -
        // Stage 6 will migrate from here; Stage 1 only measures.
        const bool pm_ring_sig_left =
            image_info.size.width == 960 && image_info.size.height == 540 &&
            image_info.guest_size == 0x240000 && image_info.resources.levels == 1 &&
            cache_image.info.size == image_info.size &&
            cache_image.info.guest_size == image_info.guest_size;
        if (pm_ring_sig_left) {
            texstat::ring_hits_left.fetch_add(1, std::memory_order_relaxed);
        }
        const auto mip_left = cache_image.info.MipOf(image_info);
        if (pm_ring_sig_left && mip_left < 0) {
            const u64 pm_dl = cache_image.info.guest_address - image_info.guest_address;
            const u32 mask =
                (u32(!(pm_dl < image_info.guest_size)) << 0) |
                (u32(!(image_info.resources.layers == 1 &&
                       cache_image.info.resources.layers == 1))
                 << 1) |
                (u32(cache_image.info.resources.levels != 1) << 2) |
                (u32(cache_image.info.tile_mode != image_info.tile_mode) << 3) |
                (u32(cache_image.info.pixel_format != image_info.pixel_format) << 4) |
                (u32(cache_image.info.pitch != image_info.pitch) << 5) |
                (u32(cache_image.binding.is_target != 0) << 6) |
                (u32(cache_image.binding.is_bound != 0) << 7) |
                (u32(True(cache_image.flags & ImageFlagBits::GpuModified)) << 8) |
                (u32(!(cache_image.tick_accessed_last < scheduler.CurrentTick())) << 9) |
                (u32(cache_image.binding.needs_rebind != 0) << 10);
            static u32 ringrej_left_reported = 0;
            if (ringrej_left_reported < 8) {
                ++ringrej_left_reported;
                LOG_CRITICAL(Render_Vulkan,
                             "RINGREJ-left #{}: mask={:#x} (b0delta b1layers b2levels b3tile "
                             "b4fmt b5pitch b6target b7bound b8gpumod b9tick b10rebind) delta={:#x}",
                             ringrej_left_reported, mask, pm_dl);
            }
        }
        // [D1 SHADPS4_RINGMIG_DL=1] Tick-versioned migration (left branch), mirror of the right
        // DL path. Replaces the b8 guard with the downloaded_tick/modified_tick comparison plus
        // the retire-freshness gate; on migrate MANDATORY clear GpuModified + erase from
        // download_images (v1). Same left geometry and full guard set incl. b10 needs_rebind.
        {
            static const bool ring_mig_dl = [] {
                const char* e = std::getenv("SHADPS4_RINGMIG_DL");
                return e && e[0] == '1';
            }();
            const u64 pm_dl3 = cache_image.info.guest_address - image_info.guest_address;
            if (ring_mig_dl && !RingPoolEnabled() && pm_ring_sig_left && mip_left < 0 &&
                pm_dl3 < image_info.guest_size && image_info.resources.layers == 1 &&
                cache_image.info.resources.levels == 1 &&
                cache_image.info.resources.layers == 1 &&
                cache_image.info.tile_mode == image_info.tile_mode &&
                cache_image.info.pixel_format == image_info.pixel_format &&
                cache_image.info.pitch == image_info.pitch &&
                !cache_image.binding.is_target && !cache_image.binding.is_bound &&
                !cache_image.binding.needs_rebind &&
                cache_image.tick_accessed_last < scheduler.CurrentTick()) {
                bool ok = False(cache_image.flags & ImageFlagBits::GpuModified);
                if (!ok && cache_image.downloaded_tick >= cache_image.modified_tick) {
                    ok = true;
                } else if (!ok && scheduler.CurrentTick() -
                                          scheduler.GetMasterSemaphore()->KnownGpuTick() <= 2) {
                    DownloadImageMemory(cache_image_id, true);
                    ok = true;
                }
                if (ok) {
                    cache_image.flags &= ~ImageFlagBits::GpuModified; // mandatory (v1)
                    {
                        std::unique_lock lk{download_images_mutex};
                        download_images.erase(cache_image_id);
                    }
                    UntrackImage(cache_image_id);
                    UnregisterImage(cache_image_id);
                    cache_image.info.guest_address = image_info.guest_address;
                    RegisterImage(cache_image_id);
                    TrackImage(cache_image_id);
                    cache_image.flags |= ImageFlagBits::CpuDirty;
                    const u64 n =
                        texstat::ring_migrate_left.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (n <= 8 || n % 256 == 0) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "RINGMIGDL-LEFT #{}: migrated ring image {:#x} -> {:#x} "
                                     "(delta {:#x})",
                                     n, image_info.guest_address + pm_dl3, image_info.guest_address,
                                     pm_dl3);
                    }
                    return {cache_image_id, -1, -1};
                }
                // gate not passed -> fall through to old behaviour
            }
        }
        // [C2 RINGMIG-LEFT] Mirror of the right-branch migration: the sliding video ring also
        // enters through the left branch (cache image above the requested address). Same
        // primitives, full guard set incl. b10 needs_rebind; FindImage's mutex is already held.
        {
            static const bool ring_mig_left = [] {
                const char* e = std::getenv("SHADPS4_RINGMIG_LEFT");
                return e && e[0] == '1';
            }();
            const u64 pm_dl2 = cache_image.info.guest_address - image_info.guest_address;
            if (ring_mig_left && pm_ring_sig_left && mip_left < 0 &&
                pm_dl2 < image_info.guest_size && image_info.resources.layers == 1 &&
                cache_image.info.resources.levels == 1 &&
                cache_image.info.resources.layers == 1 &&
                cache_image.info.tile_mode == image_info.tile_mode &&
                cache_image.info.pixel_format == image_info.pixel_format &&
                cache_image.info.pitch == image_info.pitch &&
                !cache_image.binding.is_target && !cache_image.binding.is_bound &&
                !cache_image.binding.needs_rebind &&
                False(cache_image.flags & ImageFlagBits::GpuModified) &&
                cache_image.tick_accessed_last < scheduler.CurrentTick()) {
                UntrackImage(cache_image_id);
                UnregisterImage(cache_image_id);
                cache_image.info.guest_address = image_info.guest_address;
                RegisterImage(cache_image_id);
                TrackImage(cache_image_id);
                cache_image.flags |= ImageFlagBits::CpuDirty;
                const u64 n =
                    texstat::ring_migrate_left.fetch_add(1, std::memory_order_relaxed) + 1;
                if (n <= 8 || n % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "RINGMIG-LEFT #{}: migrated ring image {:#x} -> {:#x} "
                                 "(delta {:#x})",
                                 n, image_info.guest_address + pm_dl2, image_info.guest_address,
                                 pm_dl2);
                }
                return {cache_image_id, -1, -1};
            }
        }
        if (auto mip = mip_left; mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    texstat::f_left_target.fetch_add(1, std::memory_order_relaxed);
                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    merged_image.CopyMip(cache_image, mip, slice);
                    texstat::f_copymip.fetch_add(1, std::memory_order_relaxed);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    texstat::expand.fetch_add(1, std::memory_order_relaxed);
    texstat::expand_bytes.fetch_add(info.guest_size, std::memory_order_relaxed);
    const auto new_image_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    new_image.CopyImage(src_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    // [SLOWPKT] tex phase (lookup/overlap/create).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhTex};
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};
    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
        // [R9] Perfect-match on a pool member: the whole point of the pool - the slide is caught
        // here and never reaches the resolve cycle. [R10] depth twins counted separately.
        if (RingPoolEnabled() && True(cache_image.flags & ImageFlagBits::RingPooled)) {
            if (cache_image.info.props.is_depth) {
                texstat::pool_depth_hit.fetch_add(1, std::memory_order_relaxed);
            } else {
                texstat::pool_hits.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        texstat::TexTimer _tov{texstat::overlap_us};
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            // [R9] Never free a pool member here (it must survive for the perfect-match path).
            if (!(RingPoolEnabled() && True(image_resolved.flags & ImageFlagBits::RingPooled))) {
                // [REBIND] A render target already resolved by PrepareRenderState would keep
                // being drawn into after FreeImage (BeginRendering does not re-resolve without
                // needs_rebind) and the frame's result would be lost - same guard ExpandImage uses.
                image_resolved.binding.needs_rebind =
                    image_resolved.binding.is_target || image_resolved.binding.is_bound;
                FreeImage(image_id);
            }
            image_id = {};
            texstat::overlap_fail.fetch_add(1, std::memory_order_relaxed);
            LOG_CRITICAL(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        {
            texstat::TexTimer _tcr{texstat::create_us};
            image_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            texstat::img_new.fetch_add(1, std::memory_order_relaxed);
            RegisterImage(image_id);
        }
        // [R9] adopt-once: the single entry point into the pool - a normal insert of a new image
        // on a new ring address joins the pool (mutex held here in FindImage).
        if (RingPoolEnabled() && info.size.width == 960 && info.size.height == 540 &&
            info.guest_size == 0x240000 && info.resources.levels == 1 &&
            info.resources.layers == 1) {
            // [RINGPOOL-NOOVL] SHADPS4_RINGPOOL_NOOVL=1: two 960x540 targets on overlapping guest
            // addresses (0x114b290000 / 0x114b2c0000, delta 0x30000) both became pool members,
            // and a member at a foreign address is skipped by overlap resolve - so the game's
            // writer landed in one image while the reader sampled the other (flat near water on
            // the deck, 06.09). Refuse to adopt an image that overlaps a live member: it then
            // takes the ordinary overlap path and aliases the member's bytes as intended.
            static const bool noovl = [] {
                const char* e = std::getenv("SHADPS4_RINGPOOL_NOOVL");
                return e && e[0] == '1';
            }();
            bool overlaps_member = false;
            VAddr member_addr = 0;
            if (noovl) {
                ForEachImageInRegion(info.guest_address, info.guest_size,
                                     [&](ImageId other_id, Image& other) {
                                         if (other_id != image_id &&
                                             True(other.flags & ImageFlagBits::RingPooled)) {
                                             overlaps_member = true;
                                             member_addr = other.info.guest_address;
                                         }
                                     });
            }
            if (overlaps_member) {
                static std::atomic<u32> s_skip{0};
                const u32 k = s_skip.fetch_add(1, std::memory_order_relaxed) + 1;
                if (k <= 16 || k % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "RINGPOOL-NOOVL#{}: not adopting {:#x} (960x540) - overlaps pool "
                                 "member {:#x}",
                                 k, info.guest_address, member_addr);
                }
            } else {
                AdoptIntoRingPool(image_id);
            }
        }
        if (ImgWatched(info.guest_address)) {
            std::string cands;
            for (const auto& cache_id : image_ids) {
                const auto& ci = slot_images[cache_id].info;
                cands += fmt::format(" [{}]{:#x} {}x{} lvl={}", cache_id.index, ci.guest_address,
                                     ci.size.width, ci.size.height, ci.resources.levels);
            }
            LOG_CRITICAL(Render_Vulkan,
                         "IMGNEW: id={} addr={:#x} {}x{} lvls={} lyrs={} fmt={} gsize={:#x} "
                         "type={} tile={:#x} | candidates({}):{}",
                         image_id.index, info.guest_address, info.size.width, info.size.height,
                         info.resources.levels, info.resources.layers,
                         static_cast<u32>(info.pixel_format), info.guest_size,
                         static_cast<u32>(desc.type), static_cast<u32>(info.tile_mode),
                         image_ids.size(), cands);
        }
    } else if (ImgWatched(info.guest_address)) {
        const auto& ri = slot_images[image_id].info;
        LOG_CRITICAL(Render_Vulkan,
                     "IMGHIT: want {:#x} {}x{} lvls={} -> id={} {:#x} {}x{} lvls={} view_mip={}",
                     info.guest_address, info.size.width, info.size.height, info.resources.levels,
                     image_id.index, ri.guest_address, ri.size.width, ri.size.height,
                     ri.resources.levels, view_mip);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    // The base shifts into the cached image, so the extent has to shrink with it, and any
    // mip offset the caller already applied must be kept rather than overwritten.
    if (view_mip > 0) {
        const u32 levels = image.info.resources.levels;
        desc.view_info.range.base.level += static_cast<u32>(view_mip);
        if (desc.view_info.range.base.level >= levels) {
            desc.view_info.range.base.level = levels - 1;
        }
        desc.view_info.range.extent.levels =
            std::min(desc.view_info.range.extent.levels, levels - desc.view_info.range.base.level);
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    // [FINDLOG] how the cache resolved this T# for a watched shader (tag set by the
    // rasterizer around BindTextures): the requested image vs the slot handed out, with the
    // view mip/slice chosen by the overlap path. Ring dumped at DEVFAULT.
    if (const u64 tag = bind_log_tag.load(std::memory_order_relaxed); tag != 0) {
        FindLog::Note(FindLog::Entry{
            .tag = tag,
            .want_addr = info.guest_address,
            .want_size = info.guest_size,
            .got_addr = image.info.guest_address,
            .got_size = image.info.guest_size,
            .slot = image_id.index,
            .want_w = info.size.width,
            .want_h = info.size.height,
            .want_lvls = info.resources.levels,
            .got_w = image.info.size.width,
            .got_h = image.info.size.height,
            .got_lvls = image.info.resources.levels,
            .view_lvl = desc.view_info.range.base.level,
            .view_lvls = desc.view_info.range.extent.levels,
            .view_mip = view_mip,
            .view_slice = view_slice,
            .fmt = static_cast<u32>(image.info.pixel_format),
        });
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    // [SLOWPKT] tex phase (view resolve, storage bookkeeping).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhTex};
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        SetGpuModified(image); // [D1] stamps modified_tick
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);

    // A view whose mip/layer range runs past the image is undefined behaviour:
    // sampling it reads unmapped device memory and takes the GPU down. The
    // range comes from the guest T# descriptor and can outlive an image that
    // was re-created smaller, so clamp it to what the image actually has.
    ImageViewInfo view_info = desc.view_info;
    const auto& res = image.info.resources;
    if (view_info.range.base.level + view_info.range.extent.levels > res.levels ||
        view_info.range.base.layer + view_info.range.extent.layers > res.layers) {
        LOG_CRITICAL(Render_Vulkan,
                     "VIEWOOB: addr={:#x} view lvl {}+{} of {}, lyr {}+{} of {}",
                    image.info.guest_address, view_info.range.base.level,
                    view_info.range.extent.levels, res.levels, view_info.range.base.layer,
                    view_info.range.extent.layers, res.layers);
        if (view_info.range.base.level >= res.levels) {
            view_info.range.base.level = res.levels - 1;
        }
        view_info.range.extent.levels =
            std::min(view_info.range.extent.levels, res.levels - view_info.range.base.level);
        if (view_info.range.base.layer >= res.layers) {
            view_info.range.base.layer = res.layers - 1;
        }
        view_info.range.extent.layers =
            std::min(view_info.range.extent.layers, res.layers - view_info.range.base.layer);
    }
    return image.FindView(view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    SetGpuModified(image); // [D1] stamps modified_tick
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    SetGpuModified(image); // [D1] stamps modified_tick
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        // [STENCILFIX] SHADPS4_STENCILFIX=1: only the stencil placeholder (created below, no
        // pixel format) may be linked. The guest reuses the half-res stencil memory for the
        // bloom mip chain (0x114b510000: 480x270 color); linking that color image redirected
        // every sample of it to the depth image (GetAssociatedDepth in BindTextures) -> the
        // black world and the DEVFAULT past the depth image's end (runs 196/198/203/206).
        static const bool stencil_fix = [] {
            const char* e = std::getenv("SHADPS4_STENCILFIX");
            return e && e[0] == '1';
        }();
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address != desc.info.stencil_addr) {
                                     return;
                                 }
                                 if (stencil_fix &&
                                     image.info.pixel_format != vk::Format::eUndefined) {
                                     static std::atomic<u32> s_n{0};
                                     const u32 k = s_n.fetch_add(1, std::memory_order_relaxed) + 1;
                                     if (k <= 8 || k % 65536 == 0) {
                                         LOG_CRITICAL(Render_Vulkan,
                                                      "STENCILFIX #{}: color image id={} {}x{} "
                                                      "fmt={} at stencil addr {:#x} not linked",
                                                      k, image_id.index, image.info.size.width,
                                                      image.info.size.height,
                                                      static_cast<u32>(image.info.pixel_format),
                                                      desc.info.stencil_addr);
                                     }
                                     return;
                                 }
                                 stencil_id = image_id;
                             });
        if (!stencil_id) {
            // [R6 D5] stencil insert: the other uncounted image creation on the ring address.
            texstat::stencil_new.fetch_add(1, std::memory_order_relaxed);
            if (texstat::RingSenseEnabled()) {
                static std::atomic<u32> pm_sn{0};
                const u32 k = pm_sn.fetch_add(1, std::memory_order_relaxed) + 1;
                if (k <= 8 || k % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "STENCILNEW #{}: stencil_addr={:#x} size={:#x} {}x{}", k,
                                 desc.info.stencil_addr, desc.info.stencil_size, desc.info.size.width,
                                 desc.info.size.height);
                }
            }
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id =
                slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    // [SLOWPKT] tex phase (upload: hash/copy/detile).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhTex};
    texstat::Report();
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
    texstat::refresh_total.fetch_add(1, std::memory_order_relaxed);

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        texstat::TexTimer _th{texstat::hash_us};
        const u64 hash = HashGuestRange(addr, size, "probe8x8");
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // [experiment] SHADPS4_TEXHASH=1 applies the same content check to EVERY image, not just
        // GPU-modified ones. Measured reason: while the game streams a level in, any CPU write to
        // a texture's memory marks the whole image dirty and every mip is pushed to the GPU
        // again - 1495 of 1510 uploads in a five-second window were repeats of an image already
        // uploaded, 7 GB of traffic. Hashing the source is far cheaper than re-uploading it, and
        // an unchanged mip needs no upload at all. Matches the observation that a second attempt
        // after Drake dies shows textures perfectly: nothing is streamed then, so nothing is
        // re-uploaded.
        static const bool hash_always = [] {
            const char* e = std::getenv("SHADPS4_TEXHASH");
            return e && e[0] == '1';
        }();
        // Protect GPU modified resources from accidental CPU reuploads.
        // [HASHCAP] A mip larger than the cap is exactly the case that cost 893 ms; skipping the
        // dedup check just uploads it, which is what happened before SHADPS4_TEXHASH existed.
        if (((is_gpu_modified && !is_gpu_dirty) || hash_always) && HashCapExceeded(mip_size)) {
            NoteHashCapSkip(image.info.guest_address + mip_offset, mip_size, "mip");
        } else if ((is_gpu_modified && !is_gpu_dirty) || hash_always) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const auto th0 = texstat::TexTimeEnabled() ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
            const u64 hash = HashGuestRange(addr + mip_offset, mip_size, "mip");
            if (texstat::TexTimeEnabled()) {
                texstat::hash_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now() - th0)
                                               .count(),
                                           std::memory_order_relaxed);
            }
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        texstat::refresh_empty.fetch_add(1, std::memory_order_relaxed);
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }
    if (image_copies.size() < image.info.resources.levels) {
        texstat::refresh_skip_gpu.fetch_add(1, std::memory_order_relaxed);
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    if (auto barrier = in_buffer->GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                             vk::PipelineStageFlagBits2::eTransfer)) {
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier.value(),
        });
    }

    const auto td0 = texstat::TexTimeEnabled() ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer->Handle(), in_offset, image.info);
    if (texstat::TexTimeEnabled()) {
        texstat::detile_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - td0)
                                         .count(),
                                     std::memory_order_relaxed);
    }
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    texstat::refresh_upload.fetch_add(1, std::memory_order_relaxed);
    {
        u64 bytes = 0;
        for (const auto& c : image_copies) {
            bytes += u64{c.imageExtent.width} * c.imageExtent.height * (image.info.num_bits / 8);
        }
        texstat::upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
        std::scoped_lock lk{texstat::seen_mutex};
        ++texstat::seen_uploads[image.info.guest_address];
    }
    {
        texstat::TexTimer _tc{texstat::copy_us};
        image.Upload(image_copies, buffer, offset);
    }
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base) {
    const u64 hash = XXH3_64bits(&sampler, sizeof(sampler));

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Postmortem::g_res_images.store(slot_images.size(), std::memory_order_relaxed);
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    texstat::img_free.fetch_add(1, std::memory_order_relaxed);
    Image& image = slot_images[image_id];
    if (ImgWatched(image.info.guest_address)) {
        LOG_CRITICAL(Render_Vulkan, "IMGFREE: id={} addr={:#x} {}x{} lvls={} target={} bound={}",
                     image_id.index, image.info.guest_address, image.info.size.width,
                     image.info.size.height, image.info.resources.levels,
                     u32(image.binding.is_target), u32(image.binding.is_bound));
    }
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    texstat::mem_used.store(total_used_memory, std::memory_order_relaxed);
    texstat::mem_trigger.store(trigger_gc_memory, std::memory_order_relaxed);
    texstat::mem_pressure.store(pressure_gc_memory, std::memory_order_relaxed);
    texstat::mem_critical.store(critical_gc_memory, std::memory_order_relaxed);
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    texstat::gc_runs.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        texstat::gc_free.fetch_add(1, std::memory_order_relaxed);
        auto& image = slot_images[image_id];
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const size_t lru_id = samplers.at(hash).lru_id;
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };

    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::TouchImage(const Image& image) {
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // [debug] журнал удалений: сопоставляется с BINDLOG при device lost
    LOG_WARNING(Render_Vulkan, "FREEIMG: image_id={} addr={:#x} size={:#x} tick={} last_used={}",
                image_id.index, image.info.guest_address, image.info.guest_size,
                scheduler.CurrentTick(), image.tick_accessed_last);
    // [FREEIMG-RING] 06.09: the LOG_WARNING above never reaches the log (Critical-only filter),
    // and the 161 device fault named an UNBOUND VkImage (ADDRBIND-COVER handle 0x6002197f12d0,
    // tick 27663) as the blur's read target. Keep the last 256 deletions with the Vulkan handle
    // so the fault dump can name the guest image and the ticks (delete vs last use vs fault).
    FreeImgRing::Note(reinterpret_cast<u64>(static_cast<VkImage>(image.GetImage())),
                      image.info.guest_address,
                      image.info.guest_size, image.info.size.width, image.info.size.height,
                      image.info.resources.levels, scheduler.CurrentTick(),
                      image.tick_accessed_last);

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    // [R9] carve-out: any regular FreeImage (UnmapMemory, depth-recreate, f_blockdim/f_tile,
    // eviction) strips pool membership here - so two images never live on one address.
    // DeleteImage is reached only from FreeImage under `mutex`, so ring_pool mutation is safe.
    if (True(image.flags & ImageFlagBits::RingPooled)) {
        image.flags &= ~ImageFlagBits::RingPooled;
        ring_pool.erase(std::remove(ring_pool.begin(), ring_pool.end(), image_id),
                        ring_pool.end());
        texstat::pool_carveout.fetch_add(1, std::memory_order_relaxed);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
