// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common/alignment.h"
#include "common/logging/log.h"
#include "common/postmortem.h"
#include "common/thread.h"
#include "core/address_space.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/backing_import.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

namespace {

constexpr u64 kMB = 1ull << 20;

// NVIDIA truncates an imported host allocation larger than 4GB to its low 32 bits, so every
// chunk stays well under that. 2GB also keeps the count of chunks small for a ~6GB backing.
// Default 256 MB: a lazily imported chunk is paid for on first touch, and 2 GB of synchronous
// page pinning inside a frame would just move the startup stall into gameplay. Still far below
// the 4 GB truncation limit. SHADPS4_BACKIMP_CHUNK_MB overrides.
u64 ChunkSize() {
    static const u64 sz = [] {
        const char* e = std::getenv("SHADPS4_BACKIMP_CHUNK_MB");
        const u64 mb = e != nullptr ? std::strtoull(e, nullptr, 10) : 256ull;
        return std::min<u64>(std::max<u64>(mb, 16ull) << 20, 2ull << 30);
    }();
    return sz;
}

constexpr vk::BufferUsageFlags kBufferUsage =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

vk::Buffer CreateExternalBuffer(vk::Device device, u64 size, vk::Result& err) {
    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_ci{
        vk::BufferCreateInfo{
            .size = size,
            .usage = kBufferUsage,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        vk::ExternalMemoryBufferCreateInfo{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        },
    };
    auto res = device.createBuffer(buffer_ci.get());
    err = res.result;
    return res.result == vk::Result::eSuccess ? res.value : vk::Buffer{};
}

} // Anonymous namespace

BackingImport& BackingImport::Get() {
    static BackingImport singleton;
    return singleton;
}

bool BackingImport::Init(const Instance& instance, void* backing_base, u64 backing_size) {
    std::scoped_lock lk{mutex};
    if (initialized) {
        return available;
    }
    initialized = true;

    // Mode first, so that even a disabled import reports what was asked for.
    if (const char* m = std::getenv("SHADPS4_BACKIMP_MODE")) {
        if (std::strcmp(m, "off") == 0) {
            mode = Mode::Off;
        } else if (std::strcmp(m, "all") == 0) {
            mode = Mode::All;
        } else if (std::strcmp(m, "small") == 0) {
            mode = Mode::Small;
        } else if (std::strcmp(m, "bywrite") == 0) {
            mode = Mode::ByWrite;
        } else {
            mode = Mode::Seeds;
        }
    }
    const auto parse_ranges = [](const char* r, std::vector<std::pair<u64, u64>>& out) {
        // Same syntax as LABELUMA: comma separated hexbase:hexsize.
        const char* cursor = r;
        while (*cursor != '\0') {
            u64 range_base = 0, range_size = 0;
            int consumed = 0;
            if (std::sscanf(cursor, "%llx:%llx%n", &range_base, &range_size, &consumed) == 2 &&
                range_base != 0 && range_size != 0) {
                out.emplace_back(range_base, range_size);
                cursor += consumed;
            } else {
                break;
            }
            if (*cursor == ',') {
                ++cursor;
            }
        }
    };
    if (const char* wf = std::getenv("SHADPS4_BACKIMP_WARMUP_FRAMES")) {
        const u64 v = std::strtoull(wf, nullptr, 10);
        if (v != 0) {
            warmup_frames = v;
        }
    }
    if (mode == Mode::Seeds || mode == Mode::Small || mode == Mode::ByWrite) {
        // Seeds carry the spin pollers; they stay shared in `small` too, whatever their size.
        if (const char* r = std::getenv("SHADPS4_LABELUMA_RANGE")) {
            parse_ranges(r, seed_ranges);
        }
    }
    if (const char* mk = std::getenv("SHADPS4_BACKIMP_MAX_KB")) {
        const u64 kb = std::strtoull(mk, nullptr, 10);
        if (kb != 0) {
            small_max_bytes = kb * 1024ull;
        }
    }
    if (const char* r = std::getenv("SHADPS4_BACKIMP_PREFETCH")) {
        parse_ranges(r, prefetch_ranges);
    }
    LOG_CRITICAL(Render_Vulkan, "BACKIMP: mode={} max_kb={} seeds={} prefetch_ranges={}",
                 mode == Mode::Off       ? "off"
                 : mode == Mode::All     ? "all"
                 : mode == Mode::Small   ? "small"
                 : mode == Mode::ByWrite ? "bywrite"
                                         : "seeds",
                 small_max_bytes >> 10, seed_ranges.size(), prefetch_ranges.size());

    if (!instance.IsExternalMemoryHostSupported()) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: VK_EXT_external_memory_host missing, disabled");
        return false;
    }
    if (backing_base == nullptr || backing_size == 0) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: no backing yet (base={:#x} size={}), disabled",
                     reinterpret_cast<u64>(backing_base), backing_size);
        return false;
    }

    device = instance.GetDevice();
    pdev = instance.GetPhysicalDevice();
    import_align = instance.GetMinImportedHostPointerAlignment();
    if (import_align == 0) {
        import_align = 0x1000;
    }
    base = reinterpret_cast<u64>(backing_base);
    if ((base & (import_align - 1)) != 0 || ChunkSize() < import_align) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: backing base {:#x} not aligned to {:#x}, disabled",
                     base, import_align);
        return false;
    }
    total = backing_size & ~(import_align - 1);
    const u64 chunk_size = ChunkSize() & ~(import_align - 1);
    eager = [] {
        const char* e = std::getenv("SHADPS4_BACKIMP_EAGER");
        return e != nullptr && e[0] == '1';
    }();

    // Plan only. Pinning 8.4 GB of host pages costs the driver seconds on the Main thread, and
    // doing it before the first frame is exactly the black screen before the Sony logo. Chunks
    // are now imported on first touch (and warmed in the background), so startup pays nothing.
    for (u64 offset = 0; offset < total; offset += chunk_size) {
        u64 size = std::min<u64>(chunk_size, total - offset);
        size &= ~(import_align - 1);
        if (size == 0) {
            break;
        }
        chunks.push_back(Chunk{offset, size, reinterpret_cast<void*>(base + offset), {}, 0, false});
    }
    if (chunks.empty()) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: nothing to plan, disabled");
        return false;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "BACKIMP: init base={:#x} size={} MB chunks={} chunk={} MB align={:#x} mode={}",
                 base, total / kMB, chunks.size(), chunk_size / kMB, import_align,
                 eager ? "eager" : "lazy");

    if (eager) {
        for (u32 i = 0; i < chunks.size(); ++i) {
            if (!EnsureChunkLocked(i)) {
                FreeAllLocked();
                LOG_CRITICAL(Render_Vulkan, "BACKIMP: disabled after failure");
                return false;
            }
        }
    } else if (!EnsureChunkLocked(0)) {
        // The very first chunk is imported up front: it proves the whole mechanism works while
        // costing only one chunk, and a later refusal in a hot path would be far worse to debug.
        FreeAllLocked();
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: disabled after failure");
        return false;
    }

    // Bind offsets have to respect the buffer's own alignment; it only depends on usage, so it
    // is measured once here with a throwaway buffer.
    {
        vk::Result err{};
        const vk::Buffer probe = CreateExternalBuffer(device, import_align, err);
        if (!probe) {
            LOG_CRITICAL(Render_Vulkan, "BACKIMP: probe buffer failed ({})", vk::to_string(err));
            FreeAllLocked();
            LOG_CRITICAL(Render_Vulkan, "BACKIMP: disabled after failure");
            return false;
        }
        const auto req = device.getBufferMemoryRequirements(probe);
        bind_align = req.alignment != 0 ? req.alignment : 0x10;
        device.destroyBuffer(probe);
    }

    available = true;
    LOG_CRITICAL(Render_Vulkan,
                 "BACKIMP: ready planned={} MB in {} chunks, imported={} MB, bind_align={:#x}",
                 total / kMB, chunks.size(), imported_bytes / kMB, bind_align);
    return true;
}

bool BackingImport::EnsureChunkLocked(u32 index) {
    if (index >= chunks.size()) {
        return false;
    }
    Chunk& c = chunks[index];
    if (c.memory) {
        return true;
    }
    if (c.failed) {
        return false; // do not retry a refusal on every access
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto hp = device.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, c.host);
    if (hp.result != vk::Result::eSuccess || hp.value.memoryTypeBits == 0) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: chunk[{}] off={:#x} -> FAIL(getprops {})", index,
                     c.offset, vk::to_string(hp.result));
        c.failed = true;
        return false;
    }
    const auto mem_props = pdev.getMemoryProperties();
    u32 type_index = std::numeric_limits<u32>::max();
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (!(hp.value.memoryTypeBits & (1u << i))) {
            continue;
        }
        const auto flags = mem_props.memoryTypes[i].propertyFlags;
        if (flags & vk::MemoryPropertyFlagBits::eHostVisible &&
            flags & vk::MemoryPropertyFlagBits::eHostCoherent) {
            type_index = i;
            break;
        }
        if (type_index == std::numeric_limits<u32>::max()) {
            type_index = i;
        }
    }
    if (type_index == std::numeric_limits<u32>::max()) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: chunk[{}] off={:#x} -> FAIL(no type)", index,
                     c.offset);
        c.failed = true;
        return false;
    }
    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT,
                             vk::MemoryAllocateFlagsInfo>
        alloc_ci{
            vk::MemoryAllocateInfo{
                .allocationSize = c.size,
                .memoryTypeIndex = type_index,
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = c.host,
            },
            vk::MemoryAllocateFlagsInfo{
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
            },
        };
    auto memory_res = device.allocateMemory(alloc_ci.get());
    if (memory_res.result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: chunk[{}] off={:#x} size={} MB -> FAIL({})", index,
                     c.offset, c.size / kMB, vk::to_string(memory_res.result));
        c.failed = true;
        return false;
    }
    c.memory = memory_res.value;
    c.type_index = type_index;
    imported_bytes += c.size;
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
            .count();
    LOG_CRITICAL(Render_Vulkan, "BACKIMP: chunk[{}] imported on demand ({}ms) total={}MB", index,
                 ms, imported_bytes / kMB);
    return true;
}

void BackingImport::WarmInBackground() {
    if (eager) {
        return;
    }
    // Only MODE=all can touch the whole section. In seeds mode a handful of ranges is bound
    // (measured: 93 backing buffers), so pinning all 33 chunks in the background is pure waste -
    // gigabytes of page locks nobody will ever read through.
    if (mode != Mode::All) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: warm skipped (mode is not all)");
        return;
    }
    warm_thread = std::jthread([this](std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:BackimpWarm");
        // Wait until the game is actually on screen. Started at init, the warmer competed with
        // module loading for the driver's allocator and put its whole cost (33 chunks x ~65ms)
        // squarely inside the black screen it was meant to avoid.
        static const u64 after_frames = [] {
            const char* e = std::getenv("SHADPS4_BACKIMP_WARM_AFTER_FRAMES");
            const u64 v = e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
            return e != nullptr ? v : 60ull;
        }();
        while (!stop.stop_requested() &&
               Postmortem::g_flips_total.load(std::memory_order_relaxed) < after_frames) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (u32 i = 0; i < chunks.size() && !stop.stop_requested(); ++i) {
            {
                std::scoped_lock lk{mutex};
                if (!available) {
                    return;
                }
                EnsureChunkLocked(i);
            }
            // Yield between chunks: this is a convenience, never a reason to compete with the
            // frame for the driver's allocator lock.
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::scoped_lock lk{mutex};
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: background warm done {}MB in {}ms",
                     imported_bytes / kMB,
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
    });
}

void BackingImport::FreeAllLocked() {
    for (auto& chunk : chunks) {
        if (chunk.whole) {
            device.destroyBuffer(chunk.whole);
            chunk.whole = vk::Buffer{};
        }
        if (chunk.memory) {
            device.freeMemory(chunk.memory);
        }
    }
    chunks.clear();
    available = false;
}

std::optional<vk::DeviceAddress> BackingImport::PageBda(u64 backing_offset) {
    std::scoped_lock lk{mutex};
    if (!available) {
        return std::nullopt;
    }
    Chunk* chunk = nullptr;
    for (auto& c : chunks) {
        if (backing_offset >= c.offset && backing_offset < c.offset + c.size) {
            chunk = &c;
            break;
        }
    }
    if (chunk == nullptr) {
        return std::nullopt;
    }
    const u32 index = static_cast<u32>(chunk - chunks.data());
    if (!chunk->memory && !EnsureChunkLocked(index)) {
        return std::nullopt;
    }
    if (!chunk->whole) {
        // One buffer over the entire chunk: every page of the chunk then has a device address
        // without a per-page VkBuffer. Bound at offset 0, so alignment is never a concern.
        vk::Result err{};
        const vk::Buffer buffer = CreateExternalBuffer(device, chunk->size, err);
        if (!buffer) {
            LOG_CRITICAL(Render_Vulkan, "BACKIMP: whole-chunk buffer for chunk {} failed ({})",
                         index, vk::to_string(err));
            return std::nullopt;
        }
        const auto req = device.getBufferMemoryRequirements(buffer);
        const bool type_ok = ((req.memoryTypeBits >> chunk->type_index) & 1u) != 0;
        if (!type_ok || req.size > chunk->size ||
            device.bindBufferMemory(buffer, chunk->memory, 0) != vk::Result::eSuccess) {
            LOG_CRITICAL(
                Render_Vulkan,
                "BACKIMP: whole-chunk buffer for chunk {} unbindable (type_ok={} req={:#x} "
                "chunk={:#x})",
                index, type_ok, static_cast<u64>(req.size), chunk->size);
            device.destroyBuffer(buffer);
            return std::nullopt;
        }
        chunk->whole = buffer;
        chunk->whole_bda = device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = buffer});
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: whole-chunk buffer for chunk {} ({} MB) bda={:#x}",
                     index, chunk->size >> 20, static_cast<u64>(chunk->whole_bda));
    }
    if (chunk->whole_bda == 0) {
        return std::nullopt;
    }
    return chunk->whole_bda + (backing_offset - chunk->offset);
}

void BackingImport::Shutdown() {
    // Stop the warmer BEFORE taking `mutex` - it takes the same lock per chunk.
    warm_thread.request_stop();
    if (warm_thread.joinable()) {
        warm_thread.join();
    }
    std::scoped_lock lk{mutex};
    if (chunks.empty()) {
        available = false;
        return;
    }
    LOG_CRITICAL(Render_Vulkan, "BACKIMP: releasing {} chunks", chunks.size());
    FreeAllLocked();
}

std::optional<BackingImport::Mapping> BackingImport::GetBuffer(u64 backing_offset, u64 size) {
    std::scoped_lock lk{mutex};
    return GetBufferLocked(backing_offset, size);
}

std::optional<BackingImport::Mapping> BackingImport::GetBufferLocked(u64 backing_offset, u64 size) {
    if (!available || size == 0) {
        return std::nullopt;
    }
    const Chunk* chunk = nullptr;
    for (const auto& c : chunks) {
        if (backing_offset >= c.offset && backing_offset < c.offset + c.size) {
            chunk = &c;
            break;
        }
    }
    if (chunk == nullptr) {
        return std::nullopt;
    }
    // Lazy import: the range is planned, so back it now rather than falling back to VRAM.
    if (!chunk->memory) {
        const u32 index = static_cast<u32>(chunk - chunks.data());
        if (!EnsureChunkLocked(index)) {
            return std::nullopt;
        }
    }
    const u64 in_chunk = backing_offset - chunk->offset;
    if (size > chunk->size - in_chunk) {
        // Straddles a chunk boundary - two allocations cannot back one buffer.
        return std::nullopt;
    }

    const u64 delta = in_chunk & (bind_align - 1);
    const u64 bind_offset = in_chunk - delta;

    vk::Result err{};
    const vk::Buffer buffer = CreateExternalBuffer(device, size + delta, err);
    if (!buffer) {
        return std::nullopt;
    }
    const auto req = device.getBufferMemoryRequirements(buffer);
    const bool type_ok = ((req.memoryTypeBits >> chunk->type_index) & 1u) != 0;
    const bool fits = req.size <= chunk->size - bind_offset;
    const bool aligned = req.alignment == 0 || (bind_offset % req.alignment) == 0;
    if (!type_ok || !fits || !aligned) {
        device.destroyBuffer(buffer);
        return std::nullopt;
    }
    if (device.bindBufferMemory(buffer, chunk->memory, bind_offset) != vk::Result::eSuccess) {
        device.destroyBuffer(buffer);
        return std::nullopt;
    }
    // Same bytes the GPU just got bound to: the section is mapped RW at `base` for the whole
    // process lifetime, so this pointer stays valid as long as the import does.
    u8* const host_ptr = reinterpret_cast<u8*>(base + chunk->offset + bind_offset);
    return Mapping{buffer, delta, host_ptr};
}

void BackingImport::DestroyBuffer(vk::Buffer buffer) {
    if (!buffer) {
        return;
    }
    std::scoped_lock lk{mutex};
    device.destroyBuffer(buffer);
}

bool BackingImport::ModeAllows(VAddr base, u64 size) const {
    switch (mode) {
    case Mode::Off:
        return false;
    case Mode::All:
        return true;
    case Mode::ByWrite: {
        // Warm-up: behave exactly like `all`. Nothing has been observed yet, and putting a written
        // range into VRAM by mistake is the failure mode that hangs the game, so the safe side of
        // the unknown is shared memory.
        if (Postmortem::g_flips_total.load(std::memory_order_relaxed) < warmup_frames) {
            return true;
        }
        for (const auto& [seed_base, seed_size] : seed_ranges) {
            if (base < seed_base + seed_size && seed_base < base + size) {
                return true;
            }
        }
        for (const auto& [pf_base, pf_size] : prefetch_ranges) {
            if (base < pf_base + pf_size && pf_base < base + size) {
                return true;
            }
        }
        const u64 first = base >> kWriteGranuleBits;
        const u64 last = (base + size - 1) >> kWriteGranuleBits;
        std::scoped_lock lk{write_mutex};
        for (u64 g = first; g <= last; ++g) {
            if (write_granules.contains(g)) {
                return true;
            }
        }
        return false;
    }
    case Mode::Small: {
        // Size decides, except for the explicitly listed ranges below, which are shared
        // unconditionally because they carry state the guest polls.
        if (size <= small_max_bytes) {
            small_backed.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        for (const auto& [seed_base, seed_size] : seed_ranges) {
            if (base < seed_base + seed_size && seed_base < base + size) {
                small_backed.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        for (const auto& [pf_base, pf_size] : prefetch_ranges) {
            if (base < pf_base + pf_size && pf_base < base + size) {
                small_backed.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        large_vram.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    case Mode::Seeds:
        for (const auto& [seed_base, seed_size] : seed_ranges) {
            if (base < seed_base + seed_size && seed_base < base + size) {
                return true;
            }
        }
        for (const auto& [pf_base, pf_size] : prefetch_ranges) {
            if (base < pf_base + pf_size && pf_base < base + size) {
                return true;
            }
        }
        return false;
    }
    return false;
}

void BackingImport::AllowRange(VAddr base, u64 size) {
    for (const auto& [b, s] : prefetch_ranges) {
        if (b == base && s == size) {
            return;
        }
    }
    prefetch_ranges.emplace_back(base, size);
}

bool BackingImport::NoteGuestWrite(VAddr base, u64 size) {
    if (mode != Mode::ByWrite || size == 0) {
        return false;
    }
    const u64 first = base >> kWriteGranuleBits;
    const u64 last = (base + size - 1) >> kWriteGranuleBits;
    const bool warm = Postmortem::g_flips_total.load(std::memory_order_relaxed) >= warmup_frames;
    bool late = false;
    std::scoped_lock lk{write_mutex};
    for (u64 g = first; g <= last; ++g) {
        if (write_granules.insert(g).second && warm) {
            // A granule learned after warm-up means some buffer may already sit in VRAM while the
            // guest writes to that address. Counting these tells us whether warm-up is long enough.
            late_writes.fetch_add(1, std::memory_order_relaxed);
            late = true;
        }
    }
    return late;
}

void BackingImport::NoteBufferPlacement(bool backed, u64 size) {
    if (mode != Mode::ByWrite) {
        return;
    }
    if (backed) {
        backed_bufs.fetch_add(1, std::memory_order_relaxed);
        backed_bytes.fetch_add(size, std::memory_order_relaxed);
    } else {
        vram_bufs.fetch_add(1, std::memory_order_relaxed);
        vram_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

bool BackingImport::CoversContiguously(u64 backing_offset, u64 size) {
    std::scoped_lock lk{mutex};
    if (!available || size == 0) {
        return false;
    }
    for (const auto& c : chunks) {
        if (backing_offset >= c.offset && backing_offset < c.offset + c.size) {
            return size <= c.size - (backing_offset - c.offset);
        }
    }
    return false;
}

void BackingImport::NoteBind(VAddr guest, u64 size, u64 offset, u64 bda, u64 delta,
                             const char* via) {
    std::scoped_lock lk{stat_mutex};
    ++bind_count;
    if (bind_logged < 32) {
        ++bind_logged;
        LOG_CRITICAL(Render_Vulkan,
                     "BACKIMP: bind guest={:#x} size={:#x} off={:#x} bda={:#x} delta={:#x} via={}",
                     guest, size, offset, bda, delta, via);
    }
}

void BackingImport::NoteReject(VAddr guest, u64 size, const char* reason) {
    std::scoped_lock lk{stat_mutex};
    ++reject_count;
    reject_reasons[std::string_view{reason}] += 1;
    if (reject_logged < 32) {
        ++reject_logged;
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: reject guest={:#x} size={:#x} reason={}", guest, size,
                     reason);
    }
}

void BackingImport::MaybeLogSummary() {
    std::scoped_lock lk{stat_mutex};
    if (bind_count == 0 && reject_count == 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (next_summary != std::chrono::steady_clock::time_point{} && now < next_summary) {
        return;
    }
    next_summary = now + std::chrono::seconds(5);
    std::string by_reason;
    for (const auto& [reason, count] : reject_reasons) {
        if (!by_reason.empty()) {
            by_reason += ", ";
        }
        by_reason += std::string{reason} + "=" + std::to_string(count);
    }
    LOG_CRITICAL(Render_Vulkan,
                 "BACKIMP: live binds={} rejects={} (by reason: {}) | mode={} max_kb={} "
                 "small_backed={} large_vram={}",
                 bind_count, reject_count, by_reason.empty() ? "-" : by_reason,
                 mode == Mode::Off       ? "off"
                 : mode == Mode::All     ? "all"
                 : mode == Mode::Small   ? "small"
                 : mode == Mode::ByWrite ? "bywrite"
                                         : "seeds",
                 small_max_bytes >> 10, small_backed.load(std::memory_order_relaxed),
                 large_vram.load(std::memory_order_relaxed));
    if (mode == Mode::ByWrite) {
        // Read every counter into a local BEFORE the log call: argument evaluation order is
        // unspecified, and mixing reads with mutating reads there has burned us once already.
        const u64 vb = vram_bufs.load(std::memory_order_relaxed);
        const u64 vmb = vram_bytes.load(std::memory_order_relaxed) >> 20;
        const u64 bb = backed_bufs.load(std::memory_order_relaxed);
        const u64 bmb = backed_bytes.load(std::memory_order_relaxed) >> 20;
        const u64 lw = late_writes.load(std::memory_order_relaxed);
        u64 granules = 0;
        {
            std::scoped_lock lk{write_mutex};
            granules = write_granules.size();
        }
        LOG_CRITICAL(Render_Vulkan,
                     "BACKIMP: mode=bywrite | vram_bufs={} ({} MB) backing_bufs={} ({} MB) "
                     "migrations={} | write_granules={} warmup_frames={}",
                     vb, vmb, bb, bmb, lw, granules, warmup_frames);
    }
}

void BackingImport::LogSummaryNow(const char* tag) {
    std::scoped_lock lk{stat_mutex};
    std::string by_reason;
    for (const auto& [reason, count] : reject_reasons) {
        if (!by_reason.empty()) {
            by_reason += ", ";
        }
        by_reason += std::string{reason} + "=" + std::to_string(count);
    }
    LOG_CRITICAL(Render_Vulkan, "BACKIMP: [{}] live binds={} rejects={} (by reason: {})", tag,
                 bind_count, reject_count, by_reason.empty() ? "-" : by_reason);
}

void BackingImport::SelfTest() {
    if (!IsAvailable()) {
        return;
    }
    constexpr u64 kTestSize = 64ull << 10;

    // Raw offsets inside the backing. The odd one is deliberately misaligned so the delta path
    // gets exercised too.
    std::vector<u64> offsets;
    offsets.push_back(0);
    if (total > 0x200000) {
        offsets.push_back(0x100004);
    }
    {
        std::scoped_lock lk{mutex};
        if (chunks.size() > 1) {
            offsets.push_back(chunks[1].offset);
        } else if (total > 4 * kMB) {
            offsets.push_back((total / 2) & ~0xFFFull);
        }
    }
    if (total > 2 * kMB) {
        offsets.push_back(total - 2 * kMB);
    }

    for (const u64 offset : offsets) {
        const auto mapping = GetBuffer(offset, kTestSize);
        if (!mapping) {
            LOG_CRITICAL(Render_Vulkan, "BACKIMP: selftest off={:#x} FAILED", offset);
            continue;
        }
        const vk::DeviceAddress bda =
            device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = mapping->buffer});
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: selftest off={:#x} bda={:#x} delta={:#x} ok", offset,
                     static_cast<u64>(bda) + mapping->delta, mapping->delta);
        DestroyBuffer(mapping->buffer);
    }

    // Negative control: a range crossing a chunk boundary has to be refused.
    {
        std::scoped_lock lk{mutex};
        if (chunks.size() > 1) {
            const u64 cross = chunks[0].size - 0x1000;
            const auto mapping = GetBufferLocked(cross, 0x4000);
            if (mapping) {
                LOG_CRITICAL(Render_Vulkan,
                             "BACKIMP: selftest cross-chunk off={:#x} UNEXPECTEDLY accepted",
                             cross);
                device.destroyBuffer(mapping->buffer);
            } else {
                LOG_CRITICAL(Render_Vulkan,
                             "BACKIMP: selftest cross-chunk off={:#x} rejected (expected)", cross);
            }
        }
    }

    // End to end: take live guest addresses, translate them through AddressSpace and build a
    // buffer over the already imported backing. This is the path the rest of the fix will use.
    const auto& address_space = Core::Memory::Instance()->GetAddressSpace();
    static constexpr u64 kBands[2][2] = {
        {0x200000000ull, 0x400000000ull},
        {0x1000000000ull, 0x1400000000ull},
    };
    u32 found = 0;
    for (const auto& band : kBands) {
        for (u64 addr = band[0]; addr < band[1] && found < 3; addr += 16 * kMB) {
            const auto backing_offset = address_space.GetBackingOffset(addr, kTestSize);
            if (!backing_offset) {
                continue;
            }
            const auto mapping = GetBuffer(*backing_offset, kTestSize);
            if (!mapping) {
                LOG_CRITICAL(Render_Vulkan, "BACKIMP: selftest guest={:#x} off={:#x} FAILED", addr,
                             *backing_offset);
                continue;
            }
            const vk::DeviceAddress bda =
                device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = mapping->buffer});
            LOG_CRITICAL(Render_Vulkan, "BACKIMP: selftest guest={:#x} off={:#x} bda={:#x} ok",
                         addr, *backing_offset, static_cast<u64>(bda) + mapping->delta);
            DestroyBuffer(mapping->buffer);
            ++found;
        }
        if (found >= 3) {
            break;
        }
    }
    if (found == 0) {
        LOG_CRITICAL(Render_Vulkan, "BACKIMP: selftest guest scan found no backing-backed range");
    }
}

} // namespace Vulkan
