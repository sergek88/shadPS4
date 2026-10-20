// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "common/postmortem.h"

#include "common/assert.h"
#include "common/debug.h"
#include "common/types.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

// [IMPORTPROBE] Manual WinAPI prototypes (no windows.h), same style as FastDieNow in
// vk_presenter.cpp. Only used by the SHADPS4_IMPORT_PROBE diagnostic path.
#ifdef _WIN32
extern "C"
    __declspec(dllimport) void* __stdcall VirtualAlloc(void*, size_t, unsigned long, unsigned long);
extern "C" __declspec(dllimport) int __stdcall VirtualFree(void*, size_t, unsigned long);
extern "C" __declspec(dllimport) int __stdcall TerminateProcess(void*, unsigned int);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
#endif

namespace {

#ifdef _WIN32
// [IMPORTPROBE] SHADPS4_IMPORT_PROBE=1/2: measure the NVIDIA driver's real imported-host-memory
// ceiling. Mirrors the live LABELUMA host-import path exactly (VirtualAlloc -> host-pointer props
// -> ImportMemoryHostPointerInfoEXT + external-memory VkBuffer + bind). Fully env-gated; every
// VkResult checked; no failure can abort init.
constexpr unsigned long PROBE_MEM_COMMIT = 0x00001000;
constexpr unsigned long PROBE_MEM_RESERVE = 0x00002000;
constexpr unsigned long PROBE_MEM_RELEASE = 0x00008000;
constexpr unsigned long PROBE_PAGE_READWRITE = 0x04;

struct ProbeChunk {
    void* host{};
    u64 size{};
    vk::Buffer buffer{};
    vk::DeviceMemory memory{};
};

// Import one chunk of `size` bytes (aligned up to `align`). Returns true and fills `out` on
// success; false with `err` set on any failure (all intermediate resources released).
bool ProbeImportChunk(vk::PhysicalDevice pdev, vk::Device dev, u64 size, u64 align, ProbeChunk& out,
                      vk::Result& err) {
    const u64 asize = (size + align - 1) & ~(align - 1);
    void* host =
        VirtualAlloc(nullptr, asize, PROBE_MEM_RESERVE | PROBE_MEM_COMMIT, PROBE_PAGE_READWRITE);
    if (host == nullptr) {
        err = vk::Result::eErrorOutOfHostMemory;
        return false;
    }
    const auto hp = dev.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, host);
    if (hp.result != vk::Result::eSuccess || hp.value.memoryTypeBits == 0) {
        err = hp.result;
        VirtualFree(host, 0, PROBE_MEM_RELEASE);
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
    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_ci{
        vk::BufferCreateInfo{
            .size = asize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        vk::ExternalMemoryBufferCreateInfo{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        },
    };
    auto buffer_res = dev.createBuffer(buffer_ci.get());
    if (buffer_res.result != vk::Result::eSuccess) {
        err = buffer_res.result;
        VirtualFree(host, 0, PROBE_MEM_RELEASE);
        return false;
    }
    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT,
                             vk::MemoryAllocateFlagsInfo>
        alloc_ci{
            vk::MemoryAllocateInfo{
                .allocationSize = asize,
                .memoryTypeIndex = type_index,
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = host,
            },
            vk::MemoryAllocateFlagsInfo{
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
            },
        };
    auto memory_res = dev.allocateMemory(alloc_ci.get());
    if (memory_res.result != vk::Result::eSuccess) {
        err = memory_res.result;
        dev.destroyBuffer(buffer_res.value);
        VirtualFree(host, 0, PROBE_MEM_RELEASE);
        return false;
    }
    if (dev.bindBufferMemory(buffer_res.value, memory_res.value, 0) != vk::Result::eSuccess) {
        err = vk::Result::eErrorMemoryMapFailed;
        dev.destroyBuffer(buffer_res.value);
        dev.freeMemory(memory_res.value);
        VirtualFree(host, 0, PROBE_MEM_RELEASE);
        return false;
    }
    out = ProbeChunk{host, asize, buffer_res.value, memory_res.value};
    return true;
}

void ProbeFree(vk::Device dev, ProbeChunk& c) {
    dev.destroyBuffer(c.buffer);
    dev.freeMemory(c.memory);
    VirtualFree(c.host, 0, PROBE_MEM_RELEASE);
    c = ProbeChunk{};
}

void RunImportProbe(vk::PhysicalDevice pdev, vk::Device dev, u64 align, bool has_ext,
                    bool has_budget, int mode, u64 cap_mb) {
    constexpr u64 MB = 1ull << 20;
    // Phase 0: memory budget per heap.
    if (has_budget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        vk::PhysicalDeviceMemoryProperties2 props2{.pNext = &budget};
        pdev.getMemoryProperties2(&props2);
        const auto& mp = props2.memoryProperties;
        for (u32 i = 0; i < mp.memoryHeapCount; ++i) {
            const bool dev_local =
                static_cast<bool>(mp.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
            LOG_CRITICAL(
                Render_Vulkan,
                "IMPORTPROBE: heap[{}] size={} MB budget={} MB usage={} MB device_local={}", i,
                mp.memoryHeaps[i].size / MB, budget.heapBudget[i] / MB, budget.heapUsage[i] / MB,
                dev_local ? 1 : 0);
        }
    } else {
        LOG_CRITICAL(Render_Vulkan,
                     "IMPORTPROBE: VK_EXT_memory_budget unsupported, skipping phase 0");
    }
    if (!has_ext || align == 0) {
        LOG_CRITICAL(Render_Vulkan,
                     "IMPORTPROBE: VK_EXT_external_memory_host unavailable (align={}), skipping",
                     align);
        if (mode == 2) {
            LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: exiting (probe mode 2)");
            TerminateProcess(GetCurrentProcess(), 0);
        }
        return;
    }

    std::vector<ProbeChunk> chunks;
    u64 total = 0;
    const u64 CAP = cap_mb * MB;
    vk::Result err = vk::Result::eSuccess;

    // Phase 1: coarse 16 MB steps until first failure (or cap).
    while (total + 16 * MB <= CAP) {
        ProbeChunk c{};
        if (!ProbeImportChunk(pdev, dev, 16 * MB, align, c, err)) {
            LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: COARSE ceiling={} MB err={}", total / MB,
                         vk::to_string(err));
            break;
        }
        chunks.push_back(c);
        total += c.size;
        LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: +16MB total={} MB", total / MB);
    }
    if (total >= CAP) {
        LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: reached cap {} MB without failure", total / MB);
    }

    // Phase 2: refine the boundary with progressively smaller steps.
    for (const u64 step : {4 * MB, 1 * MB, MB / 4}) {
        while (true) {
            ProbeChunk c{};
            if (!ProbeImportChunk(pdev, dev, step, align, c, err)) {
                break;
            }
            chunks.push_back(c);
            total += c.size;
        }
    }
    LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: CEILING precise={} KB ({:.2f} MB)", total / 1024,
                 static_cast<double>(total) / static_cast<double>(MB));

    // Phase 3: budget-recovery test - free half, then re-import 16 MB until failure.
    u64 freed = 0;
    const size_t half = chunks.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        freed += chunks[i].size;
        ProbeFree(dev, chunks[i]);
    }
    chunks.erase(chunks.begin(), chunks.begin() + half);
    u64 reimported = 0;
    while (true) {
        ProbeChunk c{};
        if (!ProbeImportChunk(pdev, dev, 16 * MB, align, c, err)) {
            break;
        }
        chunks.push_back(c);
        reimported += c.size;
    }
    LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: RECOVERY freed={} MB reimported={} MB", freed / MB,
                 reimported / MB);

    // Budget snapshot at peak (before freeing) - shows which heap the private import consumed.
    if (has_budget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        vk::PhysicalDeviceMemoryProperties2 props2{.pNext = &budget};
        pdev.getMemoryProperties2(&props2);
        const auto& mp = props2.memoryProperties;
        for (u32 i = 0; i < mp.memoryHeapCount; ++i) {
            LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: after heap[{}] budget={} MB usage={} MB", i,
                         budget.heapBudget[i] / MB, budget.heapUsage[i] / MB);
        }
    }

    // Phase 4: free everything probed.
    for (auto& c : chunks) {
        ProbeFree(dev, c);
    }
    chunks.clear();
    LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: done, all freed");
    if (mode == 2) {
        LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: exiting (probe mode 2)");
        TerminateProcess(GetCurrentProcess(), 0);
    }
}

// ---- [GUESTPROBE] measure the pinning ceiling of GUEST memory (file-mapping section views) ----
extern "C" __declspec(dllimport) size_t __stdcall VirtualQuery(const void*, void*, size_t);
// x64 MEMORY_BASIC_INFORMATION layout (48 bytes), declared by hand (no windows.h).
struct ProbeMbi {
    void* BaseAddress;
    void* AllocationBase;
    unsigned long AllocationProtect;
    unsigned long pad0;
    size_t RegionSize;
    unsigned long State;
    unsigned long Protect;
    unsigned long Type;
    unsigned long pad1;
};

// Import an EXISTING host range [host, host+size) (guest memory). No VirtualAlloc/VirtualFree -
// this is the live game's memory. Cleanup destroys only the VkBuffer/VkDeviceMemory.
bool GuestImportChunk(vk::PhysicalDevice pdev, vk::Device dev, void* host, u64 size, u64 align,
                      ProbeChunk& out, vk::Result& err, const char*& stage) {
    const u64 asize = (size + align - 1) & ~(align - 1);
    const auto hp = dev.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, host);
    if (hp.result != vk::Result::eSuccess || hp.value.memoryTypeBits == 0) {
        err = hp.result;
        stage = "getprops";
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
    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_ci{
        vk::BufferCreateInfo{
            .size = asize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        vk::ExternalMemoryBufferCreateInfo{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        },
    };
    auto buffer_res = dev.createBuffer(buffer_ci.get());
    if (buffer_res.result != vk::Result::eSuccess) {
        err = buffer_res.result;
        stage = "buffer";
        return false;
    }
    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT,
                             vk::MemoryAllocateFlagsInfo>
        alloc_ci{
            vk::MemoryAllocateInfo{
                .allocationSize = asize,
                .memoryTypeIndex = type_index,
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = host,
            },
            vk::MemoryAllocateFlagsInfo{
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
            },
        };
    auto memory_res = dev.allocateMemory(alloc_ci.get());
    if (memory_res.result != vk::Result::eSuccess) {
        err = memory_res.result;
        stage = "alloc";
        dev.destroyBuffer(buffer_res.value);
        return false;
    }
    if (dev.bindBufferMemory(buffer_res.value, memory_res.value, 0) != vk::Result::eSuccess) {
        err = vk::Result::eErrorMemoryMapFailed;
        stage = "bind";
        dev.destroyBuffer(buffer_res.value);
        dev.freeMemory(memory_res.value);
        return false;
    }
    out = ProbeChunk{host, asize, buffer_res.value, memory_res.value};
    return true;
}

// Free a guest import: NEVER VirtualFree (live game memory) - only the Vulkan objects.
void GuestFree(vk::Device dev, ProbeChunk& c) {
    dev.destroyBuffer(c.buffer);
    dev.freeMemory(c.memory);
    c = ProbeChunk{};
}

void LogHeapBudget(vk::PhysicalDevice pdev, bool has_budget, const char* tag) {
    constexpr u64 MB = 1ull << 20;
    if (!has_budget) {
        return;
    }
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props2{.pNext = &budget};
    pdev.getMemoryProperties2(&props2);
    const auto& mp = props2.memoryProperties;
    for (u32 i = 0; i < mp.memoryHeapCount; ++i) {
        const bool dev_local =
            static_cast<bool>(mp.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
        LOG_CRITICAL(Render_Vulkan,
                     "GUESTPROBE: {} heap[{}] size={} MB budget={} MB usage={} MB device_local={}",
                     tag, i, mp.memoryHeaps[i].size / MB, budget.heapBudget[i] / MB,
                     budget.heapUsage[i] / MB, dev_local ? 1 : 0);
    }
}

void RunGuestProbe(vk::PhysicalDevice pdev, vk::Device dev, u64 align, bool has_ext,
                   bool has_budget, int mode, u32 delay_s) {
    constexpr u64 MB = 1ull << 20;
    std::this_thread::sleep_for(std::chrono::seconds(delay_s));
    if (!has_ext || align == 0) {
        LOG_CRITICAL(Render_Vulkan, "GUESTPROBE: external_memory_host unavailable (align={}), skip",
                     align);
        if (mode == 2) {
            LOG_CRITICAL(Render_Vulkan, "GUESTPROBE: exiting (probe mode guest2)");
            TerminateProcess(GetCurrentProcess(), 0);
        }
        return;
    }
    LOG_CRITICAL(Render_Vulkan, "GUESTPROBE: start, mapping importability (align={})", align);
    LogHeapBudget(pdev, has_budget, "before");

    // [v3 MAP] Collect committed regions with Type + Protect (not just address spans). This turns
    // the probe into an importability MAP: which regions/types import and which refuse.
    constexpr u64 kGuestLo = 0x1000000000ull;
    constexpr u64 kGuestHi = 0x1400000000ull;
    constexpr unsigned long MEM_PRIVATE = 0x20000;
    constexpr unsigned long MEM_MAPPED = 0x40000;
    constexpr unsigned long MEM_IMAGE = 0x1000000;
    struct Region {
        u64 base;
        u64 end;
        unsigned long type;
        unsigned long protect;
    };
    std::vector<Region> regions;
    u64 committed_bytes = 0, priv_mb = 0, mapped_mb = 0, image_mb = 0;
    for (u64 addr = kGuestLo; addr < kGuestHi;) {
        ProbeMbi mbi{};
        const size_t got = VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi));
        if (got == 0 || mbi.RegionSize == 0) {
            break;
        }
        const u64 region_end = std::min<u64>(addr + mbi.RegionSize, kGuestHi);
        if (mbi.State == PROBE_MEM_COMMIT) {
            regions.push_back({addr, region_end, mbi.Type, mbi.Protect});
            const u64 mb = (region_end - addr) / MB;
            committed_bytes += region_end - addr;
            if (mbi.Type == MEM_PRIVATE) {
                priv_mb += mb;
            } else if (mbi.Type == MEM_MAPPED) {
                mapped_mb += mb;
            } else if (mbi.Type == MEM_IMAGE) {
                image_mb += mb;
            }
        }
        addr = region_end;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "GUESTPROBE: committed private={} MB mapped={} MB image={} MB regions={}", priv_mb,
                 mapped_mb, image_mb, regions.size());

    std::vector<ProbeChunk> chunks;
    u64 imported = 0, failed_bytes = 0;
    u64 priv_ok = 0, priv_fail = 0, mapped_ok = 0, mapped_fail = 0, image_ok = 0, image_fail = 0;
    u32 fail_lines = 0;
    bool first_fail_logged = false;

    const auto type_name = [](unsigned long t) -> const char* {
        return t == MEM_PRIVATE  ? "private"
               : t == MEM_MAPPED ? "mapped"
               : t == MEM_IMAGE  ? "image"
                                 : "other";
    };

    // [v3 SKIP-ON-FAIL] Walk each region; import in 16 MB pieces. On the FIRST failure inside a
    // region, log it (capped) + FIRSTFAIL detail once, then SKIP the rest of THAT region and move
    // to the next - so a bad region cannot mask whether later ones still import (quota vs nature).
    for (const Region& r : regions) {
        u64 region_imported = 0;
        bool region_failed = false;
        for (u64 cur = r.base; cur < r.end;) {
            const u64 sz = std::min<u64>(16 * MB, r.end - cur);
            ProbeChunk c{};
            vk::Result err = vk::Result::eSuccess;
            const char* stage = "?";
            if (GuestImportChunk(pdev, dev, reinterpret_cast<void*>(cur), sz, align, c, err,
                                 stage)) {
                chunks.push_back(c);
                imported += c.size;
                region_imported += c.size;
                cur += sz;
                continue;
            }
            // Failure: record and skip the rest of this region.
            region_failed = true;
            failed_bytes += r.end - cur; // count the untried remainder of the region as failed
            if (fail_lines < 64) {
                ++fail_lines;
                LOG_CRITICAL(Render_Vulkan,
                             "GUESTPROBE: FAIL addr={:#x} size={:#x} err={} stage={} type={} "
                             "protect={:#x}",
                             cur, sz, vk::to_string(err), stage, type_name(r.type), r.protect);
            }
            if (!first_fail_logged) {
                first_fail_logged = true;
                LOG_CRITICAL(Render_Vulkan,
                             "GUESTPROBE: FIRSTFAIL region base={:#x} size={} MB state=0x{:x} "
                             "protect={:#x} type={:#x} ({})",
                             r.base, (r.end - r.base) / MB, PROBE_MEM_COMMIT, r.protect, r.type,
                             type_name(r.type));
            }
            break; // skip rest of this region
        }
        const u64 region_total = r.end - r.base;
        const u64 region_fail = region_total - region_imported;
        if (r.type == MEM_PRIVATE) {
            priv_ok += region_imported;
            priv_fail += region_fail;
        } else if (r.type == MEM_MAPPED) {
            mapped_ok += region_imported;
            mapped_fail += region_fail;
        } else if (r.type == MEM_IMAGE) {
            image_ok += region_imported;
            image_fail += region_fail;
        }
        (void)region_failed;
    }

    LOG_CRITICAL(Render_Vulkan,
                 "GUESTPROBE: MAP imported={} MB failed={} MB | private ok={} MB fail={} MB | "
                 "mapped ok={} MB fail={} MB | image ok={} MB fail={} MB",
                 imported / MB, failed_bytes / MB, priv_ok / MB, priv_fail / MB, mapped_ok / MB,
                 mapped_fail / MB, image_ok / MB, image_fail / MB);

    LogHeapBudget(pdev, has_budget, "after");
    u64 total = imported;
    const size_t freed_chunks = chunks.size();
    for (auto& c : chunks) {
        GuestFree(dev, c);
    }
    chunks.clear();
    LOG_CRITICAL(Render_Vulkan, "GUESTPROBE: done, all freed (peak {} MB, {} chunks)", total / MB,
                 freed_chunks);
    if (mode == 2) {
        LOG_CRITICAL(Render_Vulkan, "GUESTPROBE: exiting (probe mode guest2)");
        TerminateProcess(GetCurrentProcess(), 0);
    }
}
#endif // _WIN32

std::vector<vk::PhysicalDevice> EnumeratePhysicalDevices(vk::UniqueInstance& instance) {
    auto [devices_result, devices] = instance->enumeratePhysicalDevices();
    ASSERT_MSG(devices_result == vk::Result::eSuccess, "Failed to enumerate physical devices: {}",
               vk::to_string(devices_result));
    return std::move(devices);
}

std::vector<std::string> GetSupportedExtensions(vk::PhysicalDevice physical) {
    const auto [extensions_result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (extensions_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Could not query supported extensions: {}",
                  vk::to_string(extensions_result));
        return {};
    }
    std::vector<std::string> supported_extensions;
    supported_extensions.reserve(extensions.size());
    for (const auto& extension : extensions) {
        supported_extensions.emplace_back(extension.extensionName.data());
    }
    return supported_extensions;
}

vk::FormatProperties3 GetFormatProperties(vk::PhysicalDevice physical, vk::Format format) {
    vk::FormatProperties3 properties3{};
    vk::FormatProperties2 properties2 = {
        .pNext = &properties3,
    };
    physical.getFormatProperties2(format, &properties2);
    return properties3;
}

std::unordered_map<vk::Format, vk::FormatProperties3> GetFormatProperties(
    vk::PhysicalDevice physical) {
    std::unordered_map<vk::Format, vk::FormatProperties3> format_properties;
    for (const auto& format_info : LiverpoolToVK::SurfaceFormats()) {
        const auto format = format_info.vk_format;
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    for (const auto& format_info : LiverpoolToVK::DepthFormats()) {
        const auto format = format_info.vk_format;
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    // Other miscellaneous formats, e.g. for color buffers, swizzles, or compatibility
    static constexpr std::array misc_formats = {
        vk::Format::eA2R10G10B10UnormPack32,
        vk::Format::eB8G8R8A8Unorm,
        vk::Format::eB8G8R8A8Srgb,
        vk::Format::eD24UnormS8Uint,
    };
    for (const auto& format : misc_formats) {
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    return format_properties;
}

std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

} // Anonymous namespace

Instance::Instance(bool enable_validation, bool enable_crash_diagnostic)
    : instance{CreateInstance(Frontend::WindowSystemType::Headless, enable_validation,
                              enable_crash_diagnostic)},
      physical_devices{EnumeratePhysicalDevices(instance)} {}

Instance::Instance(Frontend::WindowSDL& window, s32 physical_device_index,
                   bool enable_validation /*= false*/, bool enable_crash_diagnostic /*= false*/)
    : instance{CreateInstance(window.GetWindowInfo().type, enable_validation,
                              enable_crash_diagnostic)},
      physical_devices{EnumeratePhysicalDevices(instance)} {
    if (enable_validation) {
        debug_callback = CreateDebugCallback(*instance);
    }
    if (AddrBindEnabled()) {
        addr_bind_messenger = CreateAddrBindMessenger(*instance);
    }
    const std::size_t num_physical_devices = static_cast<u16>(physical_devices.size());
    ASSERT_MSG(num_physical_devices > 0, "No physical devices found");
    LOG_INFO(Render_Vulkan, "Found {} physical devices", num_physical_devices);

    if (physical_device_index < 0) {
        std::vector<
            std::tuple<size_t, vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMemoryProperties>>
            properties2{};
        for (auto const& physical : physical_devices) {
            properties2.emplace_back(properties2.size(), physical.getProperties2(),
                                     physical.getMemoryProperties());
        }
        std::sort(properties2.begin(), properties2.end(), [](const auto& left, const auto& right) {
            const vk::PhysicalDeviceProperties& left_prop = std::get<1>(left).properties;
            const vk::PhysicalDeviceProperties& right_prop = std::get<1>(right).properties;
            const bool left_supports_api = left_prop.apiVersion >= TargetVulkanApiVersion;
            const bool right_supports_api = right_prop.apiVersion >= TargetVulkanApiVersion;
            if (left_supports_api != right_supports_api) {
                return left_supports_api;
            }

            const bool left_is_discrete =
                left_prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
            const bool right_is_discrete =
                right_prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
            if (left_is_discrete != right_is_discrete) {
                return left_is_discrete;
            }

            constexpr auto get_mem = [](const vk::PhysicalDeviceMemoryProperties& mem) -> size_t {
                size_t max = 0;
                for (u32 i = 0; i < mem.memoryHeapCount; i++) {
                    const vk::MemoryHeap& heap = mem.memoryHeaps[i];
                    if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal && heap.size > max) {
                        max = heap.size;
                    }
                }
                return max;
            };
            size_t left_mem_size = get_mem(std::get<2>(left));
            size_t right_mem_size = get_mem(std::get<2>(right));
            return left_mem_size > right_mem_size;
        });
        physical_device = physical_devices[std::get<0>(properties2[0])];
    } else {
        ASSERT_MSG(physical_device_index < num_physical_devices,
                   "Invalid physical device index {} provided when only {} devices exist",
                   physical_device_index, num_physical_devices);

        physical_device = physical_devices[physical_device_index];
    }

    available_extensions = GetSupportedExtensions(physical_device);
    format_properties = GetFormatProperties(physical_device);
    properties = physical_device.getProperties();
    memory_properties = physical_device.getMemoryProperties();
    CollectDeviceParameters();
    ASSERT_MSG(properties.apiVersion >= TargetVulkanApiVersion,
               "Vulkan {}.{} is required, but only {}.{} is supported by device!",
               VK_VERSION_MAJOR(TargetVulkanApiVersion), VK_VERSION_MINOR(TargetVulkanApiVersion),
               VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion));

    CreateDevice();
    CollectPhysicalMemoryInfo();
    CollectImageFormatInfo();
    CollectToolingInfo();
    Postmortem::MarkStartup("vk_instance");
}

Instance::~Instance() {
    ImGui::Core::Shutdown(GetDevice());
    vmaDestroyAllocator(allocator);
}

std::string Instance::GetDriverVersionName() {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = properties.driverVersion;
    if (driver_id == vk::DriverId::eNvidiaProprietary) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (driver_id == vk::DriverId::eIntelProprietaryWindows) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }
    return GetReadableVersion(version);
}

bool Instance::CreateDevice() {
    const vk::StructureChain feature_chain =
        physical_device
            .getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                          vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                          vk::PhysicalDeviceRobustness2FeaturesEXT,
                          vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
                          vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT,
                          vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT,
                          vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
                          vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT,
                          vk::PhysicalDeviceAddressBindingReportFeaturesEXT>();
    features = feature_chain.get().features;

    const vk::StructureChain properties_chain = physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties,
        vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceVulkan13Properties,
        vk::PhysicalDevicePushDescriptorPropertiesKHR,
        vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
    vk11_props = properties_chain.get<vk::PhysicalDeviceVulkan11Properties>();
    vk12_props = properties_chain.get<vk::PhysicalDeviceVulkan12Properties>();
    vk13_props = properties_chain.get<vk::PhysicalDeviceVulkan13Properties>();
    push_descriptor_props = properties_chain.get<vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    min_imported_host_pointer_alignment =
        properties_chain.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
            .minImportedHostPointerAlignment;
    LOG_INFO(Render_Vulkan, "Physical device subgroup size {}", vk11_props.subgroupSize);

    if (available_extensions.empty()) {
        LOG_CRITICAL(Render_Vulkan, "No extensions supported by device.");
        return false;
    }

    boost::container::static_vector<const char*, 32> enabled_extensions;
    const auto add_extension = [&](std::string_view extension) -> bool {
        const auto result =
            std::find_if(available_extensions.begin(), available_extensions.end(),
                         [&](const std::string& name) { return name == extension; });

        if (result != available_extensions.end()) {
            LOG_INFO(Render_Vulkan, "Enabling extension: {}", extension);
            enabled_extensions.push_back(extension.data());
            return true;
        }

        LOG_WARNING(Render_Vulkan, "Extension {} unavailable.", extension);
        return false;
    };

    // Required
    ASSERT_MSG(add_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}",
               VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);

    const auto robustness2_features = feature_chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
    ASSERT_MSG(robustness2_features.robustBufferAccess2,
               "Required Vulkan feature unavailable: robustBufferAccess2");
    ASSERT_MSG(robustness2_features.robustImageAccess2,
               "Required Vulkan feature unavailable: robustImageAccess2");
    ASSERT_MSG(robustness2_features.nullDescriptor,
               "Required Vulkan feature unavailable: nullDescriptor");

    // Optional
    maintenance_8 = add_extension(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    attachment_feedback_loop = add_extension(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
    if (attachment_feedback_loop) {
        attachment_feedback_loop =
            add_extension(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
        if (!attachment_feedback_loop) {
            // We want both extensions so remove the first if the second isn't available
            enabled_extensions.pop_back();
        }
    }
    depth_range_unrestricted = add_extension(VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME);
    dynamic_state_3 = add_extension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    if (dynamic_state_3) {
        dynamic_state_3_features =
            feature_chain.get<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- extendedDynamicState3ColorWriteMask: {}",
                 dynamic_state_3_features.extendedDynamicState3ColorWriteMask);
        // [PIPESTAT] Reported, NOT enabled. If the pipeline-switch census blames blend state,
        // these are the features that could take it out of the key and collapse the permutation
        // count. Logged at CRITICAL so it lands in the run log next to PIPESTAT.
        LOG_CRITICAL(Render_Vulkan,
                     "DYNSTATE3: colorWriteMask={} colorBlendEquation={} colorBlendEnable={} "
                     "rasterizationSamples={} logicOpEnable={} colorWriteEnable(sep_ext)={}",
                     bool(dynamic_state_3_features.extendedDynamicState3ColorWriteMask),
                     bool(dynamic_state_3_features.extendedDynamicState3ColorBlendEquation),
                     bool(dynamic_state_3_features.extendedDynamicState3ColorBlendEnable),
                     bool(dynamic_state_3_features.extendedDynamicState3RasterizationSamples),
                     bool(dynamic_state_3_features.extendedDynamicState3LogicOpEnable),
                     bool(dynamic_state_3_features.extendedDynamicState3ColorWriteMask));
    }
    custom_border_color = add_extension(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    depth_clip_control = add_extension(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);
    depth_clip_enable = add_extension(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
    vertex_input_dynamic_state = add_extension(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    list_restart = add_extension(VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME);
    if (list_restart) {
        list_restart_features =
            feature_chain.get<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- primitiveTopologyListRestart: {}",
                 list_restart_features.primitiveTopologyListRestart);
        LOG_INFO(Render_Vulkan, "- primitiveTopologyPatchListRestart: {}",
                 list_restart_features.primitiveTopologyPatchListRestart);
    }
    amd_shader_explicit_vertex_parameter =
        add_extension(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);
    if (!amd_shader_explicit_vertex_parameter) {
        fragment_shader_barycentric =
            add_extension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    }
    provoking_vertex = add_extension(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
    shader_stencil_export = add_extension(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    image_load_store_lod = add_extension(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);
    amd_gcn_shader = add_extension(VK_AMD_GCN_SHADER_EXTENSION_NAME);
    amd_shader_trinary_minmax = add_extension(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);
    nv_framebuffer_mixed_samples = add_extension(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);
    amd_mixed_attachment_samples = add_extension(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME);
    shader_atomic_float = add_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    shader_atomic_float2 = add_extension(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    if (shader_atomic_float2) {
        shader_atomic_float2_features =
            feature_chain.get<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- shaderBufferFloat32AtomicMinMax: {}",
                 shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax);
        LOG_INFO(Render_Vulkan, "- shaderImageFloat32AtomicMinMax: {}",
                 shader_atomic_float2_features.shaderImageFloat32AtomicMinMax);
    }
    workgroup_memory_explicit_layout =
        add_extension(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    if (workgroup_memory_explicit_layout) {
        workgroup_memory_explicit_layout_features =
            feature_chain.get<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
        LOG_INFO(Render_Vulkan, "- workgroupMemoryExplicitLayout: {}",
                 workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout);
        LOG_INFO(Render_Vulkan, "- workgroupMemoryExplicitLayoutScalarBlockLayout: {}",
                 workgroup_memory_explicit_layout_features
                     .workgroupMemoryExplicitLayoutScalarBlockLayout);
        LOG_INFO(
            Render_Vulkan, "- workgroupMemoryExplicitLayout16BitAccess: {}",
            workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess);
    }
    image_2d_view_of_3d = add_extension(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME);
    if (image_2d_view_of_3d) {
        image_2d_view_of_3d_features =
            feature_chain.get<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- image2DViewOf3D: {}",
                 image_2d_view_of_3d_features.image2DViewOf3D);
        LOG_INFO(Render_Vulkan, "- sampler2DViewOf3D: {}",
                 image_2d_view_of_3d_features.sampler2DViewOf3D);
    }
    image_view_min_lod = add_extension(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
    supports_memory_budget = add_extension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    external_memory_host = add_extension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    nv_checkpoints = add_extension(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
    device_fault = add_extension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
    // [R8 ADDRBIND] device-side extension for bind/unbind reporting; only when requested.
    // The extension alone is not enough: the reportAddressBinding FEATURE must be queried from
    // the physical device and enabled in the VkDeviceCreateInfo pNext chain, or the driver
    // silently emits no events (observed in run R8: ext on, feature off, zero events).
    const bool ab_ext =
        AddrBindEnabled() && add_extension(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
    const bool ab_feature =
        ab_ext && feature_chain.get<vk::PhysicalDeviceAddressBindingReportFeaturesEXT>()
                          .reportAddressBinding == VK_TRUE;
    address_binding_report = ab_ext && ab_feature;
    if (AddrBindEnabled()) {
        LOG_CRITICAL(Render_Vulkan, "ADDRBIND: requested, ext_supported={} feature_supported={}",
                     ab_ext, ab_feature);
    }
    const bool calibrated_timestamps =
        TRACY_GPU_ENABLED ? add_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) : false;

    const auto family_properties = physical_device.getQueueFamilyProperties();
    if (family_properties.empty()) {
        LOG_CRITICAL(Render_Vulkan, "Physical device reported no queues.");
        return false;
    }

    bool graphics_queue_found = false;
    for (std::size_t i = 0; i < family_properties.size(); i++) {
        const u32 index = static_cast<u32>(i);
        if (family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            queue_family_index = index;
            graphics_queue_found = true;
        }
    }

    if (!graphics_queue_found) {
        LOG_CRITICAL(Render_Vulkan, "Unable to find graphics and/or present queues.");
        return false;
    }

    // A second queue in the family isolates presentation: a wedged present
    // chain (dozing compositor holding images) then cannot park the main
    // graphics queue and with it the whole command processor.
    // Queue 0 - graphics/CP, queue 1 - presentation (isolates a wedged present
    // chain), queue 2 - readback downloads (isolates guest fault servicing from
    // a busy parser; see BufferCache::ReadMemory).
    // [TDR SPINRESCUE] Four priorities so a 4th (rescue) queue never reads past the array end
    // (OOB caught by review). The extra element is harmless when only 3 queues are requested.
    static constexpr std::array queue_priorities = {1.0f, 1.0f, 1.0f, 1.0f};
    // The dedicated rescue queue (index 3) is only requested when the snoop+ contour is armed -
    // otherwise the queue count is unchanged and behaviour is bit-identical.
    static const bool spin_wants_rescue_queue = [] {
        const char* e = std::getenv("SHADPS4_SPINRESCUE");
        return e && (std::strcmp(e, "snoop") == 0 || std::strcmp(e, "import") == 0 ||
                     std::strcmp(e, "1") == 0);
    }();
    const u32 family_queue_count = family_properties[queue_family_index].queueCount;
    const u32 wanted_queues = std::min<u32>(spin_wants_rescue_queue ? 4 : 3, family_queue_count);
    separate_present_queue = wanted_queues >= 2;
    const vk::DeviceQueueCreateInfo queue_info = {
        .queueFamilyIndex = queue_family_index,
        .queueCount = wanted_queues,
        .pQueuePriorities = queue_priorities.data(),
    };

    const auto vk11_features = feature_chain.get<vk::PhysicalDeviceVulkan11Features>();
    vk12_features = feature_chain.get<vk::PhysicalDeviceVulkan12Features>();
    vk13_features = feature_chain.get<vk::PhysicalDeviceVulkan13Features>();
    vk::StructureChain device_chain = {
        vk::DeviceCreateInfo{
            .queueCreateInfoCount = 1u,
            .pQueueCreateInfos = &queue_info,
            .enabledExtensionCount = static_cast<u32>(enabled_extensions.size()),
            .ppEnabledExtensionNames = enabled_extensions.data(),
        },
        vk::PhysicalDeviceFeatures2{
            .features{
                .robustBufferAccess = features.robustBufferAccess,
                .imageCubeArray = features.imageCubeArray,
                .independentBlend = features.independentBlend,
                .geometryShader = features.geometryShader,
                .tessellationShader = features.tessellationShader,
                .sampleRateShading = features.sampleRateShading,
                .dualSrcBlend = features.dualSrcBlend,
                .logicOp = features.logicOp,
                .multiDrawIndirect = features.multiDrawIndirect,
                .depthClamp = features.depthClamp,
                .depthBiasClamp = features.depthBiasClamp,
                .fillModeNonSolid = features.fillModeNonSolid,
                .depthBounds = features.depthBounds,
                .wideLines = features.wideLines,
                .multiViewport = features.multiViewport,
                .samplerAnisotropy = features.samplerAnisotropy,
                .vertexPipelineStoresAndAtomics = features.vertexPipelineStoresAndAtomics,
                .fragmentStoresAndAtomics = features.fragmentStoresAndAtomics,
                .shaderImageGatherExtended = features.shaderImageGatherExtended,
                .shaderStorageImageExtendedFormats = features.shaderStorageImageExtendedFormats,
                .shaderStorageImageMultisample = features.shaderStorageImageMultisample,
                .shaderClipDistance = features.shaderClipDistance,
                .shaderFloat64 = features.shaderFloat64,
                .shaderInt64 = features.shaderInt64,
                .shaderInt16 = features.shaderInt16,
            },
        },
        vk::PhysicalDeviceVulkan11Features{
            .storageBuffer16BitAccess = vk11_features.storageBuffer16BitAccess,
            .uniformAndStorageBuffer16BitAccess = vk11_features.uniformAndStorageBuffer16BitAccess,
            .shaderDrawParameters = vk11_features.shaderDrawParameters,
        },
        vk::PhysicalDeviceVulkan12Features{
            .samplerMirrorClampToEdge = vk12_features.samplerMirrorClampToEdge,
            .drawIndirectCount = vk12_features.drawIndirectCount,
            .storageBuffer8BitAccess = vk12_features.storageBuffer8BitAccess,
            .uniformAndStorageBuffer8BitAccess = vk12_features.uniformAndStorageBuffer8BitAccess,
            .shaderBufferInt64Atomics = vk12_features.shaderBufferInt64Atomics,
            .shaderSharedInt64Atomics = vk12_features.shaderSharedInt64Atomics,
            .shaderFloat16 = vk12_features.shaderFloat16,
            .shaderInt8 = vk12_features.shaderInt8,
            .scalarBlockLayout = vk12_features.scalarBlockLayout,
            .uniformBufferStandardLayout = vk12_features.uniformBufferStandardLayout,
            .separateDepthStencilLayouts = vk12_features.separateDepthStencilLayouts,
            .hostQueryReset = vk12_features.hostQueryReset,
            .timelineSemaphore = vk12_features.timelineSemaphore,
            .bufferDeviceAddress = vk12_features.bufferDeviceAddress,
            .shaderOutputLayer = vk12_features.shaderOutputLayer,
        },
        vk::PhysicalDeviceVulkan13Features{
            .robustImageAccess = vk13_features.robustImageAccess,
            .shaderDemoteToHelperInvocation = vk13_features.shaderDemoteToHelperInvocation,
            .subgroupSizeControl = vk13_features.subgroupSizeControl,
            .synchronization2 = vk13_features.synchronization2,
            .dynamicRendering = vk13_features.dynamicRendering,
            .maintenance4 = vk13_features.maintenance4,
        },
        // Extensions
        vk::PhysicalDeviceCustomBorderColorFeaturesEXT{
            .customBorderColors = true,
            .customBorderColorWithoutFormat = true,
        },
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT{
            .extendedDynamicState3ColorWriteMask =
                dynamic_state_3_features.extendedDynamicState3ColorWriteMask,
        },
        vk::PhysicalDeviceDepthClipControlFeaturesEXT{
            .depthClipControl = true,
        },
        vk::PhysicalDeviceDepthClipEnableFeaturesEXT{
            .depthClipEnable = true,
        },
        vk::PhysicalDeviceRobustness2FeaturesEXT{
            .robustBufferAccess2 = true,
            .robustImageAccess2 = true,
            .nullDescriptor = true,
        },
        vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT{
            .vertexInputDynamicState = true,
        },
        vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT{
            .primitiveTopologyListRestart = list_restart_features.primitiveTopologyListRestart,
            .primitiveTopologyPatchListRestart =
                list_restart_features.primitiveTopologyPatchListRestart,
        },
        vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR{
            .fragmentShaderBarycentric = true,
        },
        vk::PhysicalDeviceProvokingVertexFeaturesEXT{
            .provokingVertexLast = true,
        },
        vk::PhysicalDeviceVertexAttributeDivisorFeatures{
            .vertexAttributeInstanceRateDivisor = true,
        },
        vk::PhysicalDeviceMaintenance8FeaturesKHR{
            .maintenance8 = true,
        },
        vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT{
            .attachmentFeedbackLoopLayout = true,
        },
        vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT{
            .attachmentFeedbackLoopDynamicState = true,
        },
        vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT{
            .shaderBufferFloat32AtomicMinMax =
                shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax,
            .shaderImageFloat32AtomicMinMax =
                shader_atomic_float2_features.shaderImageFloat32AtomicMinMax,
        },
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR{
            .workgroupMemoryExplicitLayout =
                workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout,
            .workgroupMemoryExplicitLayoutScalarBlockLayout =
                workgroup_memory_explicit_layout_features
                    .workgroupMemoryExplicitLayoutScalarBlockLayout,
            .workgroupMemoryExplicitLayout16BitAccess =
                workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess,
        },
        vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT{
            .image2DViewOf3D = image_2d_view_of_3d_features.image2DViewOf3D,
            .sampler2DViewOf3D = image_2d_view_of_3d_features.sampler2DViewOf3D,
        },
        vk::PhysicalDeviceImageViewMinLodFeaturesEXT{
            .minLod = true,
        },
        vk::PhysicalDeviceFaultFeaturesEXT{
            .deviceFault = true,
        },
        vk::PhysicalDeviceAddressBindingReportFeaturesEXT{
            .reportAddressBinding = true,
        },
    };

    if (!device_fault) {
        device_chain.unlink<vk::PhysicalDeviceFaultFeaturesEXT>();
    }
    if (!address_binding_report) {
        device_chain.unlink<vk::PhysicalDeviceAddressBindingReportFeaturesEXT>();
    }
    if (!custom_border_color) {
        device_chain.unlink<vk::PhysicalDeviceCustomBorderColorFeaturesEXT>();
    }
    if (!dynamic_state_3) {
        device_chain.unlink<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
    }
    if (!depth_clip_control) {
        device_chain.unlink<vk::PhysicalDeviceDepthClipControlFeaturesEXT>();
    }
    if (!depth_clip_enable) {
        device_chain.unlink<vk::PhysicalDeviceDepthClipEnableFeaturesEXT>();
    }
    if (!vertex_input_dynamic_state) {
        device_chain.unlink<vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>();
    }
    if (!list_restart) {
        device_chain.unlink<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
    }
    if (!fragment_shader_barycentric) {
        device_chain.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
    }
    if (!provoking_vertex) {
        device_chain.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
    }
    if (!maintenance_8) {
        device_chain.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
    }
    if (!attachment_feedback_loop) {
        device_chain.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
        device_chain.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
    }
    if (!shader_atomic_float2) {
        device_chain.unlink<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
    }
    if (!workgroup_memory_explicit_layout) {
        device_chain.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
    }
    if (!image_2d_view_of_3d) {
        device_chain.unlink<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
    }
    if (!image_view_min_lod) {
        device_chain.unlink<vk::PhysicalDeviceImageViewMinLodFeaturesEXT>();
    }

    auto [device_result, dev] = physical_device.createDeviceUnique(device_chain.get());
    if (device_result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Failed to create device: {}", vk::to_string(device_result));
        return false;
    }
    device = std::move(dev);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

    graphics_queue = device->getQueue(queue_family_index, 0);
    present_queue = device->getQueue(queue_family_index, separate_present_queue ? 1 : 0);
    if (wanted_queues >= 3) {
        readback_queue = device->getQueue(queue_family_index, 2);
    }
    // [TDR SPINRESCUE] Dedicated rescue queue at index 3. Never degrade onto queue 2: at a stall
    // everything behind the TSFF batch's timeline-wait is serialized and would never run.
    if (spin_wants_rescue_queue && wanted_queues >= 4) {
        rescue_queue = device->getQueue(queue_family_index, 3);
    } else if (spin_wants_rescue_queue) {
        LOG_CRITICAL(Render_Vulkan,
                     "SPINRESCUE: family has only {} queues (<4), rescue contour disabled",
                     family_queue_count);
    }
    LOG_CRITICAL(Render_Vulkan, "QUEUES: family {} count {} wanted {} readback {} rescue {}",
                 queue_family_index, family_queue_count, wanted_queues,
                 static_cast<bool>(readback_queue), static_cast<bool>(rescue_queue));
    LOG_INFO(Render_Vulkan, "Present queue: {}",
             separate_present_queue ? "separate (index 1)" : "shared with graphics");

    if (calibrated_timestamps) {
        const auto [time_domains_result, time_domains] =
            physical_device.getCalibrateableTimeDomainsEXT();
        if (time_domains_result == vk::Result::eSuccess) {
#if _WIN64
            const bool has_host_time_domain =
                std::find(time_domains.cbegin(), time_domains.cend(),
                          vk::TimeDomainEXT::eQueryPerformanceCounter) != time_domains.cend();
#elif __linux__
            const bool has_host_time_domain =
                std::find(time_domains.cbegin(), time_domains.cend(),
                          vk::TimeDomainEXT::eClockMonotonicRaw) != time_domains.cend();
#else
            // Tracy limitation means only Windows and Linux can use host time domain.
            // https://github.com/shadps4-emu/tracy/blob/c6d779d78508514102fbe1b8eb28bda10d95bb2a/public/tracy/TracyVulkan.hpp#L384-L389
            const bool has_host_time_domain = false;
#endif
            if (has_host_time_domain) {
                static constexpr std::string_view context_name{"vk_rasterizer"};
                profiler_context = TracyVkContextHostCalibrated(
                    *instance, physical_device, *device,
                    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
                TracyVkContextName(profiler_context, context_name.data(), context_name.size());
            }
        } else {
            LOG_WARNING(Render_Vulkan, "Could not query calibrated time domains for profiling: {}",
                        vk::to_string(time_domains_result));
        }
    }

    // [IMPORTPROBE] SHADPS4_IMPORT_PROBE: measure the raw imported-host-memory ceiling right after
    // device creation, BEFORE any LABELUMA seeds are imported (so the measurement starts from
    // zero). Env-gated: absent -> no effect on the live path. 1 = probe and continue, 2 = probe and
    // exit.
#ifdef _WIN32
    if (const char* pe = std::getenv("SHADPS4_IMPORT_PROBE")) {
        // Private-probe delay + phase-1 cap knobs (guest branch keeps its own default-30 delay).
        u32 priv_delay_s = 0;
        if (const char* ds = std::getenv("SHADPS4_IMPORT_PROBE_DELAY_S")) {
            priv_delay_s = static_cast<u32>(std::strtoul(ds, nullptr, 10));
        }
        u64 cap_mb = 2048;
        if (const char* cm = std::getenv("SHADPS4_IMPORT_PROBE_CAP_MB")) {
            const u64 v = std::strtoull(cm, nullptr, 10);
            if (v != 0) {
                cap_mb = v;
            }
        }
        if (pe[0] == '1' || pe[0] == '2') {
            const int probe_mode = pe[0] == '2' ? 2 : 1;
            const vk::PhysicalDevice pdev = physical_device;
            const vk::Device dev = *device;
            const u64 align = min_imported_host_pointer_alignment;
            const bool has_ext = external_memory_host;
            const bool has_budget = supports_memory_budget;
            if (priv_delay_s == 0) {
                // Original behaviour: inline, right after device creation, before any seeds.
                LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: start (mode {}), align={} cap={} MB",
                             probe_mode, align, cap_mb);
                RunImportProbe(pdev, dev, align, has_ext, has_budget, probe_mode, cap_mb);
            } else {
                // Deferred private probe (control experiment): run under the live game like the
                // guest probe, so we can tell a section-view/BAR limit from a runtime-pressure one.
                const u32 d = priv_delay_s;
                LOG_CRITICAL(Render_Vulkan,
                             "IMPORTPROBE: scheduled (mode {}), delay {}s, align={} cap={} MB",
                             probe_mode, d, align, cap_mb);
                import_probe_thread = std::jthread([pdev, dev, align, has_ext, has_budget,
                                                    probe_mode, d, cap_mb](std::stop_token) {
                    std::this_thread::sleep_for(std::chrono::seconds(d));
                    LOG_CRITICAL(Render_Vulkan, "IMPORTPROBE: start (deferred, mode {})",
                                 probe_mode);
                    RunImportProbe(pdev, dev, align, has_ext, has_budget, probe_mode, cap_mb);
                });
            }
        } else if (std::strcmp(pe, "guest") == 0 || std::strcmp(pe, "guest2") == 0) {
            // [GUESTPROBE] Guest memory (file-mapping section views) may have its own pinning
            // quota. Run on a background thread after a delay so the game has committed its
            // guest RAM. Thread stored on Instance so it outlives CreateDevice.
            const int probe_mode = std::strcmp(pe, "guest2") == 0 ? 2 : 1;
            u32 delay_s = 30;
            if (const char* ds = std::getenv("SHADPS4_IMPORT_PROBE_DELAY_S")) {
                const u32 v = static_cast<u32>(std::strtoul(ds, nullptr, 10));
                delay_s = v; // 0 is allowed (immediate)
            }
            const vk::PhysicalDevice pdev = physical_device;
            const vk::Device dev = *device;
            const u64 align = min_imported_host_pointer_alignment;
            const bool has_ext = external_memory_host;
            const bool has_budget = supports_memory_budget;
            LOG_CRITICAL(Render_Vulkan, "GUESTPROBE: scheduled (mode {}), delay {}s, align={}",
                         probe_mode, delay_s, align);
            import_probe_thread = std::jthread(
                [pdev, dev, align, has_ext, has_budget, probe_mode, delay_s](std::stop_token) {
                    RunGuestProbe(pdev, dev, align, has_ext, has_budget, probe_mode, delay_s);
                });
        }
    }
#endif

    CreateAllocator();
    return true;
}

void Instance::CreateAllocator() {
    const VmaVulkanFunctions functions = {
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };

    const VmaAllocatorCreateInfo allocator_info = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physical_device,
        .device = *device,
        .pVulkanFunctions = &functions,
        .instance = *instance,
        .vulkanApiVersion = TargetVulkanApiVersion,
    };

    const VkResult result = vmaCreateAllocator(&allocator_info, &allocator);
    if (result != VK_SUCCESS) {
        UNREACHABLE_MSG("Failed to initialize VMA with error {}",
                        vk::to_string(vk::Result{result}));
    }
}

void Instance::CollectDeviceParameters() {
    const vk::StructureChain property_chain =
        physical_device
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>();
    const vk::PhysicalDeviceDriverProperties driver =
        property_chain.get<vk::PhysicalDeviceDriverProperties>();

    driver_id = driver.driverID;
    vendor_name = driver.driverName.data();

    const std::string model_name{GetModelName()};
    const std::string driver_version = GetDriverVersionName();
    const std::string driver_name = fmt::format("{} {}", vendor_name, driver_version);
    const std::string api_version = GetReadableVersion(properties.apiVersion);
    const std::string extensions = fmt::format("{}", fmt::join(available_extensions, ", "));

    LOG_INFO(Render_Vulkan, "GPU_Vendor: {}", vendor_name);
    LOG_INFO(Render_Vulkan, "GPU_Model: {}", model_name);
    LOG_INFO(Render_Vulkan, "GPU_Integrated: {}", IsIntegrated() ? "Yes" : "No");
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Driver: {}", driver_name);
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Version: {}", api_version);
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Extensions: {}", extensions);
}

void Instance::CollectPhysicalMemoryInfo() {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = supports_memory_budget ? &budget : nullptr,
    };
    physical_device.getMemoryProperties2(&props);
    const auto& memory_props = props.memoryProperties;
    const size_t num_props = memory_props.memoryHeapCount;
    total_memory_budget = 0;
    u64 device_initial_usage = 0;
    u64 local_memory = 0;
    for (size_t i = 0; i < num_props; ++i) {
        const bool is_device_local =
            (memory_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
            vk::MemoryHeapFlags{};
        if (!IsIntegrated() && !is_device_local) {
            // Ignore non-device local memory on discrete GPUs.
            continue;
        }
        valid_heaps.push_back(i);
        if (is_device_local) {
            local_memory += memory_props.memoryHeaps[i].size;
        }
        if (supports_memory_budget) {
            device_initial_usage += budget.heapUsage[i];
            total_memory_budget += budget.heapBudget[i];
            continue;
        }
        // If memory budget is not supported, use the size of the heap as the budget.
        total_memory_budget += memory_props.memoryHeaps[i].size;
    }
    if (!IsIntegrated()) {
        // We reserve some memory for the system.
        const u64 system_memory = std::min<u64>(total_memory_budget / 8, 1_GB);
        total_memory_budget -= system_memory;
        return;
    }
    // Leave at least 8 GB for the system on integrated GPUs.
    const s64 available_memory = static_cast<s64>(total_memory_budget - device_initial_usage);
    total_memory_budget =
        static_cast<u64>(std::max<s64>(available_memory - 8_GB, static_cast<s64>(local_memory)));
}

void Instance::CollectImageFormatInfo() {
    // Check for block texel view support using basic image format info.
    const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info{
        .format = vk::Format::eBc1RgbaUnormBlock,
        .type = vk::ImageType::e2D,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled,
        .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
    };
    const auto block_texel_view_props =
        physical_device.getImageFormatProperties2(block_texel_view_info);
    supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
    LOG_INFO(Render_Vulkan, "Block Texel View support: {}",
             supports_block_texel_view ? "Yes" : "No");

    // Check and log format support details.
    for (const auto& format : LiverpoolToVK::SurfaceFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan,
                        "Surface format data_format={}, number_format={} is not fully supported "
                        "(vk_format={}, missing features={})",
                        static_cast<u32>(format.data_format),
                        static_cast<u32>(format.number_format), vk::to_string(format.vk_format),
                        vk::to_string(format.flags & ~GetFormatFeatureFlags(format.vk_format)));
        }
    }
    for (const auto& format : LiverpoolToVK::DepthFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan,
                        "Depth format z_format={}, stencil_format={} is not fully supported "
                        "(vk_format={}, missing features={})",
                        static_cast<u32>(format.z_format), static_cast<u32>(format.stencil_format),
                        vk::to_string(format.vk_format),
                        vk::to_string(format.flags & ~GetFormatFeatureFlags(format.vk_format)));
        }
    }
}

void Instance::CollectToolingInfo() const {
    if (driver_id == vk::DriverId::eAmdProprietary ||
        driver_id == vk::DriverId::eIntelProprietaryWindows) {
        // AMD: Causes issues with Reshade.
        // Intel: Causes crash on start.
        return;
    }
    const auto [tools_result, tools] = physical_device.getToolProperties();
    if (tools_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Could not get Vulkan tool properties: {}",
                  vk::to_string(tools_result));
        return;
    }
    for (const vk::PhysicalDeviceToolProperties& tool : tools) {
        const std::string_view name = tool.name;
        LOG_INFO(Render_Vulkan, "Attached debugging tool: {}", name);
    }
}

u64 Instance::GetDeviceMemoryUsage() const {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT memory_budget_props{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = &memory_budget_props,
    };
    physical_device.getMemoryProperties2(&props);

    u64 total_usage = 0;
    for (const size_t heap : valid_heaps) {
        total_usage += memory_budget_props.heapUsage[heap];
    }
    return total_usage;
}

vk::FormatFeatureFlags2 Instance::GetFormatFeatureFlags(vk::Format format) const {
    const auto it = format_properties.find(format);
    if (it == format_properties.end()) {
        UNIMPLEMENTED_MSG("Properties of format {} have not been queried.", vk::to_string(format));
    }

    return it->second.optimalTilingFeatures | it->second.bufferFeatures;
}

bool Instance::IsFormatSupported(const vk::Format format,
                                 const vk::FormatFeatureFlags2 flags) const {
    if (format == vk::Format::eUndefined) [[unlikely]] {
        return true;
    }
    return (GetFormatFeatureFlags(format) & flags) == flags;
}

vk::Format Instance::GetSupportedFormat(const vk::Format format,
                                        const vk::FormatFeatureFlags2 flags) const {
    if (!IsFormatSupported(format, flags)) [[unlikely]] {
        switch (format) {
        case vk::Format::eD16UnormS8Uint:
            if (IsFormatSupported(vk::Format::eD24UnormS8Uint, flags)) {
                return vk::Format::eD24UnormS8Uint;
            }
            if (IsFormatSupported(vk::Format::eD32SfloatS8Uint, flags)) {
                return vk::Format::eD32SfloatS8Uint;
            }
            break;
        case vk::Format::eR8Srgb:
            if (IsFormatSupported(vk::Format::eR8Unorm, flags)) {
                return vk::Format::eR8Unorm;
            }
            break;
        default:
            break;
        }
    }
    return format;
}

} // namespace Vulkan
