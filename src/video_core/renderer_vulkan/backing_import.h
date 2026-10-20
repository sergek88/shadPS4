// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

/**
 * BACKIMP - import the BACKING SECTION instead of guest virtual addresses.
 *
 * Host-pointer import on NVIDIA pins pages by pointer and wants a single contiguous
 * committed read-write VAD. Guest VA space is the opposite of that: dozens of separate
 * MapViewOfFile3 views (one VAD each), read-only stretches from the page manager, and
 * several guest addresses aliasing the very same physical pages. Measured on the deck
 * assert that gave 1489 MB imported out of 4560 MB, failing region by region.
 *
 * The backing section is the other side of the same memory: AddressSpace creates it with
 * CreateFileMapping2(SEC_COMMIT) and maps it WHOLE, once, at backing_base - one VAD, all
 * read-write, every physical page present exactly once. Guest views are just windows into
 * it at offset phys_addr. So we import the backing once at startup (the driver caches page
 * locks, repeated imports are what hurt), and any guest range becomes a plain VkBuffer
 * bound at an offset into memory that is already imported - no further vkAllocateMemory.
 */
class BackingImport {
public:
    /// SHADPS4_BACKIMP_MODE. Unset means Seeds; `off` keeps the import but binds nothing.
    enum class Mode {
        Off,
        Seeds,
        All,
        /// Shared memory for SMALL ranges only. Correctness needs the many small, per-frame
        /// buffers (constants, labels, job structures) to be coherent with the guest; speed
        /// needs the few large ones (vertices, indices, texture data) to stay in VRAM instead
        /// of being fetched over PCIe.
        Small,
        /// Shared memory only for ranges the guest actually WRITES; ranges it merely reads stay
        /// in VRAM. Measured split over 5 minutes of play: 1416 written buffers / 5327 MB
        /// against 3606 read-only buffers / 1329 MB.
        ByWrite,
    };

    struct Mapping {
        /// Buffer bound over the backing. Owned by the caller, free with DestroyBuffer.
        vk::Buffer buffer{};
        /// Bind offsets must respect the buffer's alignment, so the bind may start slightly
        /// before the requested offset. The requested data starts at `delta` inside the buffer.
        u64 delta{};
        /// CPU pointer to the FIRST BYTE OF THE BUFFER (i.e. delta bytes before the requested
        /// offset). It is the very same memory the GPU sees - one RW view of the section - so
        /// it is directly readable and writable, no mapping and no flushing needed.
        u8* host_ptr{};
    };

    static BackingImport& Get();

    /// Imports the whole backing section in chunks. Safe to call more than once (idempotent).
    bool Init(const Instance& instance, void* backing_base, u64 backing_size);

    /// Proof-of-life: builds a few buffers over the imported backing and logs their addresses.
    void SelfTest();

    /// Frees every imported chunk. Must run while the device is still alive.
    void Shutdown();

    /// Imports whatever is still only planned, off the critical path.
    void WarmInBackground();

    [[nodiscard]] bool IsAvailable() const {
        return available;
    }
    [[nodiscard]] u64 HostBase() const {
        return base;
    }
    [[nodiscard]] u64 ImportedSize() const {
        return total;
    }

    /// Buffer covering [backing_offset, backing_offset + size) of the backing section, or
    /// nullopt when the range is outside the import or straddles two chunks.
    std::optional<Mapping> GetBuffer(u64 backing_offset, u64 size);

    void DestroyBuffer(vk::Buffer buffer);

    [[nodiscard]] Mode GetMode() const {
        return mode;
    }

    /// True when the configured mode lets this guest range be served from the backing.
    /// Seeds mode also admits SHADPS4_BACKIMP_PREFETCH ranges.
    [[nodiscard]] bool ModeAllows(VAddr base, u64 size) const;
    /// [DUALHOME] Admit a guest range for CreateBackingBacked (like a prefetch range) without
    /// BufferCache prefetching it as an owner. Idempotent; GPU thread.
    void AllowRange(VAddr base, u64 size);

    /// Records that the guest wrote this range. Called once per buffer, on its first write.
    /// Returns true if this call taught the heuristic something new AFTER warm-up, i.e. a buffer
    /// may already be sitting in VRAM over an address the GPU writes - a heuristic miss.
    bool NoteGuestWrite(VAddr base, u64 size);
    /// Accounting for the bywrite split, called once per created buffer.
    void NoteBufferPlacement(bool backed, u64 size);

    /// True when [offset, offset+size) lies inside a single imported chunk - i.e. GetBuffer
    /// would not refuse it as cross-chunk. Creates nothing; for planning (prefetch).
    [[nodiscard]] bool CoversContiguously(u64 backing_offset, u64 size);
    /// [HOMEPOLICY-PT] Device address of one backing byte offset, through a buffer spanning the
    /// whole chunk (created once per chunk, cached). For page-table entries that must point at
    /// guest RAM even when no contiguous window exists for the bind ("split" refusals).
    std::optional<vk::DeviceAddress> PageBda(u64 backing_offset);

    /// Telemetry. Both rate-limit themselves: the first 32 events of each kind are printed in
    /// full, afterwards only the five-second summary moves.
    void NoteBind(VAddr guest, u64 size, u64 offset, u64 bda, u64 delta, const char* via);
    void NoteReject(VAddr guest, u64 size, const char* reason);
    void MaybeLogSummary();
    /// Unconditional variant for one-shot incident reports (bypasses the 5s throttle).
    void LogSummaryNow(const char* tag);

private:
    struct Chunk {
        u64 offset{};
        u64 size{};
        void* host{};
        vk::DeviceMemory memory{}; // null while the chunk is only planned (lazy import)
        u32 type_index{};
        bool failed{};                 // import attempted and refused - do not retry every frame
        vk::Buffer whole{};            // [HOMEPOLICY-PT] one buffer over the entire chunk
        vk::DeviceAddress whole_bda{}; // its device address (0 until created)
    };

    void FreeAllLocked();
    std::optional<Mapping> GetBufferLocked(u64 backing_offset, u64 size);
    /// Imports chunk `i` if it is still only planned. Caller holds `mutex`. Returns false when
    /// the import failed - the caller must then refuse rather than bind unbacked memory.
    bool EnsureChunkLocked(u32 index);

    std::mutex mutex;
    vk::Device device{};
    vk::PhysicalDevice pdev{};
    std::vector<Chunk> chunks;
    u64 base{};
    u64 total{};
    u64 import_align{0x1000};
    u64 bind_align{0x10};
    bool initialized{false};
    bool available{false};
    bool eager{false};
    u64 imported_bytes{0};
    std::jthread warm_thread;
    Mode mode{Mode::Seeds};
    /// SHADPS4_LABELUMA_RANGE parsed as base:size pairs; only consulted in Seeds mode.
    std::vector<std::pair<u64, u64>> seed_ranges;
    /// SHADPS4_BACKIMP_PREFETCH ranges; admitted by ModeAllows in Seeds mode too, so a
    /// prefetch configuration works without repeating itself in LABELUMA_RANGE.
    std::vector<std::pair<u64, u64>> prefetch_ranges;
    u64 small_max_bytes{64ull * 1024};
    // ModeAllows is a const query, but these are pure counters - mutable is exactly
    // what mutable is for.
    mutable std::atomic<u64> small_backed{0};
    mutable std::atomic<u64> large_vram{0};
    // [BYWRITE] Address granules (1 MB) the guest has ever written to. Both operations are rare -
    // insert runs once per buffer on its first write, lookup once per CreateBuffer - so a plainly
    // guarded set is cheaper in engineering risk than any lock-free scheme and trivial to reason
    // about. The mutex is never taken on a per-draw path.
    static constexpr u64 kWriteGranuleBits = 20;
    mutable std::mutex write_mutex;
    std::unordered_set<u64> write_granules;
    u64 warmup_frames{600};
    std::atomic<u64> vram_bufs{0};
    std::atomic<u64> vram_bytes{0};
    std::atomic<u64> backed_bufs{0};
    std::atomic<u64> backed_bytes{0};
    std::atomic<u64> late_writes{0};

    std::mutex stat_mutex;
    u64 bind_count{};
    u64 reject_count{};
    u64 bind_logged{};
    u64 reject_logged{};
    std::unordered_map<std::string_view, u64> reject_reasons;
    std::chrono::steady_clock::time_point next_summary{};
};

} // namespace Vulkan
