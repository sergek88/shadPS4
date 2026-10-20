// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/postmortem.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/backing_import.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

std::string_view BufferTypeName(MemoryUsage type) {
    switch (type) {
    case MemoryUsage::Upload:
        return "Upload";
    case MemoryUsage::Download:
        return "Download";
    case MemoryUsage::Stream:
        return "Stream";
    case MemoryUsage::DeviceLocal:
        return "DeviceLocal";
    default:
        return "Invalid";
    }
}

[[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
    return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                             : VkMemoryPropertyFlagBits{};
}

[[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::Upload:
    case MemoryUsage::Stream:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT |
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case MemoryUsage::Download:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case MemoryUsage::DeviceLocal:
        return {};
    }
    return {};
}

[[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
    case MemoryUsage::Stream:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryUsage::Upload:
    case MemoryUsage::Download:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    if (is_backing) {
        // [BACKIMP] The VkDeviceMemory under this buffer is a shared backing chunk owned by
        // BackingImport and referenced by every other backing buffer - freeing it here would
        // pull guest RAM out from under all of them. Only the VkBuffer is ours.
        if (buffer) {
            device.destroyBuffer(buffer);
            Postmortem::g_res_backing_bufs.fetch_sub(1, std::memory_order_relaxed);
        }
        return;
    }
    if (imported_memory) {
        // Imported path bypasses VMA entirely; free order is buffer first.
        device.destroyBuffer(buffer);
        device.freeMemory(imported_memory);
        return;
    }
    if (buffer) {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
}

void UniqueBuffer::Create(const vk::BufferCreateInfo& buffer_ci, MemoryUsage usage,
                          VmaAllocationInfo* out_alloc_info) {
    const bool with_bda = bool(buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    const VmaAllocationCreateFlags bda_flag =
        with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkBufferCreateInfo buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    VkBuffer unsafe_buffer{};
    VkResult result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                      &allocation, out_alloc_info);
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating buffer with error {}",
               vk::to_string(vk::Result{result}));
    buffer = vk::Buffer{unsafe_buffer};

    if (with_bda) {
        vk::BufferDeviceAddressInfo bda_info{
            .buffer = buffer,
        };
        auto bda_result = device.getBufferAddress(bda_info);
        ASSERT_MSG(bda_result != 0, "Failed to get buffer device address");
        bda_addr = bda_result;
    }
}

std::optional<Buffer> Buffer::CreateBackingBacked(const Vulkan::Instance& instance,
                                                  Vulkan::Scheduler& scheduler, VAddr base,
                                                  u64 size, const char** reject_reason,
                                                  const char* via, bool force_mode) {
    const auto refuse = [reject_reason](const char* why) -> std::optional<Buffer> {
        if (reject_reason != nullptr) {
            *reject_reason = why;
        }
        return std::nullopt;
    };
    auto& backimp = Vulkan::BackingImport::Get();
    if (!backimp.IsAvailable()) {
        return refuse("unavailable");
    }
    if (size == 0) {
        return refuse("empty");
    }
    // [DUALHOME] force_mode: a shadow piece for a range the mode would not import as an owner
    if (!force_mode && !backimp.ModeAllows(base, size)) {
        return refuse("mode");
    }
    // [BACKIMP] Small ephemeral ranges pay the same three driver calls as big ones
    // (vkCreateBuffer + bind + getBufferAddress) but save far less work, and in MODE=all they
    // are the bulk of the traffic. SHADPS4_BACKIMP_MIN_KB (0 = no floor, the default) sends
    // them down the ordinary path; the threshold is left to be chosen from BUFPROF data.
    static const u64 min_bytes = [] {
        const char* e = std::getenv("SHADPS4_BACKIMP_MIN_KB");
        return e != nullptr ? std::strtoull(e, nullptr, 10) * 1024ull : 0ull;
    }();
    if (min_bytes != 0 && size < min_bytes) {
        return refuse("too-small");
    }
    // Guest VA -> offset in the backing section. Refuses anything that is not one contiguous
    // window into the section (private memory, foreign file views, ranges crossing views).
    const char* why = "no-backing";
    const auto& address_space = Core::Memory::Instance()->GetAddressSpace();
    std::optional<u64> offset;
    {
        Postmortem::UsTimer t{Postmortem::g_buf_backing_off_us};
        offset = address_space.GetBackingOffset(base, size, &why);
    }
    if (!offset) {
        return refuse(why);
    }
    Postmortem::UsTimer vk_timer{Postmortem::g_buf_vk_create_us};
    const auto mapping = backimp.GetBuffer(*offset, size);
    if (!mapping) {
        return refuse("cross-chunk");
    }
    if (mapping->delta != 0) {
        // A shifted bind would make Offset() and BufferDeviceAddress() disagree in every
        // consumer. Page aligned guest ranges never produce one, so treat it as a hard refusal
        // rather than patching offsets in a dozen places.
        backimp.DestroyBuffer(mapping->buffer);
        return refuse("delta");
    }
    const auto device = instance.GetDevice();
    const vk::DeviceAddress bda =
        device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = mapping->buffer});
    if (bda == 0) {
        backimp.DestroyBuffer(mapping->buffer);
        return refuse("no-bda");
    }

    Buffer result{Buffer::HostImportTag{}, instance, scheduler, base, size};
    result.is_import_backed = true;
    result.is_backing_backed = true;
    result.backing_offset = *offset;
    result.backing_delta = mapping->delta;
    result.buffer.device = device;
    result.buffer.buffer = mapping->buffer;
    result.buffer.bda_addr = bda;
    // Memory is NOT ours: imported_memory stays null and is_backing tells the destructor to
    // take the VkBuffer only.
    result.buffer.is_backing = true;
    Postmortem::g_res_backing_bufs.fetch_add(1, std::memory_order_relaxed);
    Postmortem::g_backing_hist[size < 4096          ? 0
                               : size < 64 * 1024   ? 1
                               : size < 1024 * 1024 ? 2
                                                    : 3]
        .fetch_add(1, std::memory_order_relaxed);
    // Unlike the private import path this buffer DOES get a CPU pointer: it is the same RW view
    // of the section the GPU is bound to, so host_ptr + delta is exactly the first byte of the
    // requested range. Nothing in the cache reads a slot buffer's mapped_data today, but leaving
    // it null is a null-deref waiting to happen the moment something does - and every VMA path
    // that could touch it (StreamBuffer::Map/Commit, vmaFlush/Invalidate/GetAllocationMemory-
    // Properties) is StreamBuffer-only and can never see a backing buffer.
    if (mapping->host_ptr != nullptr) {
        result.mapped_data = std::span<u8>{mapping->host_ptr + mapping->delta, size};
    }
    // The import picked a HOST_VISIBLE|HOST_COHERENT memory type, so no flush/invalidate is ever
    // required (and none is reachable: is_coherent is read by StreamBuffer only).
    result.is_coherent = true;
    Vulkan::SetObjectName(device, result.Handle(), "Backing {:#x}:{:#x}", base, size);
    backimp.NoteBind(base, size, *offset, static_cast<u64>(bda), mapping->delta, via);
    return result;
}

std::optional<Buffer> Buffer::CreateHostImport(const Vulkan::Instance& instance,
                                               Vulkan::Scheduler& scheduler, VAddr base, u64 size,
                                               const char* via) {
    // [BACKIMP] Prefer a window into the already imported backing section: no pinning budget,
    // no vkAllocateMemory, no per-range import that the driver has to page-lock again.
    {
        const char* reason = "unavailable";
        if (auto backed = CreateBackingBacked(instance, scheduler, base, size, &reason, via)) {
            return backed;
        }
        auto& backimp = Vulkan::BackingImport::Get();
        if (backimp.IsAvailable() && std::strcmp(reason, "mode") != 0) {
            backimp.NoteReject(base, size, reason);
        }
    }
    const auto device = instance.GetDevice();
    const auto host_props = device.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, reinterpret_cast<void*>(base));
    if (host_props.result != vk::Result::eSuccess || host_props.value.memoryTypeBits == 0) {
        LOG_CRITICAL(Render_Vulkan, "LABELUMA import: host props rejected {:#x}:{:#x} ({})", base,
                     size, vk::to_string(host_props.result));
        return std::nullopt;
    }
    // Strictly coherent host-visible or nothing: a non-coherent import would
    // need flush/invalidate bookkeeping the callers do not have.
    const auto mem_props = instance.GetPhysicalDevice().getMemoryProperties();
    u32 type_index = std::numeric_limits<u32>::max();
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (!(host_props.value.memoryTypeBits & (1u << i))) {
            continue;
        }
        const auto flags = mem_props.memoryTypes[i].propertyFlags;
        if ((flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            type_index = i;
            break;
        }
    }
    if (type_index == std::numeric_limits<u32>::max()) {
        LOG_CRITICAL(Render_Vulkan, "LABELUMA import: no coherent type for {:#x}:{:#x}", base,
                     size);
        return std::nullopt;
    }
    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_ci{
        vk::BufferCreateInfo{
            .size = size,
            .usage = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        vk::ExternalMemoryBufferCreateInfo{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        },
    };
    auto buffer_res = device.createBuffer(buffer_ci.get());
    if (buffer_res.result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "LABELUMA import: createBuffer failed {:#x}:{:#x} ({})", base,
                     size, vk::to_string(buffer_res.result));
        return std::nullopt;
    }
    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT,
                             vk::MemoryAllocateFlagsInfo>
        alloc_ci{
            vk::MemoryAllocateInfo{
                .allocationSize = size,
                .memoryTypeIndex = type_index,
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = reinterpret_cast<void*>(base),
            },
            vk::MemoryAllocateFlagsInfo{
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
            },
        };
    auto memory_res = device.allocateMemory(alloc_ci.get());
    if (memory_res.result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "LABELUMA import: allocate/import failed {:#x}:{:#x} ({})",
                     base, size, vk::to_string(memory_res.result));
        device.destroyBuffer(buffer_res.value);
        return std::nullopt;
    }
    if (device.bindBufferMemory(buffer_res.value, memory_res.value, 0) != vk::Result::eSuccess) {
        device.destroyBuffer(buffer_res.value);
        device.freeMemory(memory_res.value);
        return std::nullopt;
    }
    const vk::DeviceAddress bda =
        device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = buffer_res.value});

    Buffer result{Buffer::HostImportTag{}, instance, scheduler, base, size};
    result.is_import_backed = true;
    result.buffer.device = device;
    result.buffer.buffer = buffer_res.value;
    result.buffer.bda_addr = bda;
    result.buffer.imported_memory = memory_res.value;
    // mapped_data stays EMPTY on purpose: every cache branch must keep
    // treating this as a device-local buffer.
    return result;
}

Buffer::Buffer(HostImportTag, const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
               VAddr cpu_addr_, u64 size_bytes_)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{MemoryUsage::DeviceLocal}, buffer{instance->GetDevice(), instance->GetAllocator()} {}

Buffer::Buffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, MemoryUsage usage_,
               VAddr cpu_addr_, vk::BufferUsageFlags flags, u64 size_bytes_)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{usage_}, buffer{instance->GetDevice(), instance->GetAllocator()} {
    // Create buffer object.
    const vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = flags,
    };
    VmaAllocationInfo alloc_info{};
    buffer.Create(buffer_ci, usage, &alloc_info);

    const auto device = instance->GetDevice();
    Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);

    // Map it if it is host visible.
    VkMemoryPropertyFlags property_flags{};
    vmaGetAllocationMemoryProperties(instance->GetAllocator(), buffer.allocation, &property_flags);
    if (alloc_info.pMappedData) {
        mapped_data = std::span<u8>{std::bit_cast<u8*>(alloc_info.pMappedData), size_bytes};
    }
    is_coherent = property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

void Buffer::Fill(u64 offset, u32 num_bytes, u32 value) {
    scheduler->EndRendering();
    ASSERT_MSG(offset % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer size must be a multiple of 4 bytes");
    const auto cmdbuf = scheduler->CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.fillBuffer(buffer, offset, num_bytes, value);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

constexpr u64 WATCHES_INITIAL_RESERVE = 0x4000;
constexpr u64 WATCHES_RESERVE_CHUNK = 0x1000;

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           MemoryUsage usage, u64 size_bytes, vk::BufferUsageFlags extra_flags)
    : Buffer{instance, scheduler, usage, 0, AllFlags | extra_flags, size_bytes} {
    ReserveWatches(current_watches, WATCHES_INITIAL_RESERVE);
    ReserveWatches(previous_watches, WATCHES_INITIAL_RESERVE);
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, Handle(), "StreamBuffer({}):{:#x}", BufferTypeName(usage),
                          size_bytes);
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment, bool allow_wait) {
    if (!is_coherent && usage == MemoryUsage::Stream) {
        size = Common::AlignUp(size, instance->NonCoherentAtomSize());
    }

    if (size > this->size_bytes) {
        return {nullptr, 0};
    }

    mapped_size = size;

    if (alignment > 0) {
        offset = Common::AlignUp(offset, alignment);
    }

    if (offset + size > this->size_bytes) {
        // The buffer would overflow, save the amount of used watches and reset the state.
        invalidation_mark = current_watch_cursor;
        current_watch_cursor = 0;
        offset = 0;
        ++wrap_generation;

        // Swap watches and reset waiting cursors.
        std::swap(previous_watches, current_watches);
        wait_cursor = 0;
        wait_bound = 0;
    }

    const u64 mapped_upper_bound = offset + size;
    if (!WaitPendingOperations(mapped_upper_bound, allow_wait)) {
        return {nullptr, 0};
    }

    return {mapped_data.data() + offset, offset};
}

void StreamBuffer::HostInvalidate(u64 region_offset, u64 region_size) {
    if (is_coherent) {
        return;
    }
    vmaInvalidateAllocation(instance->GetAllocator(), buffer.allocation, region_offset,
                            region_size);
}

void StreamBuffer::Commit() {
    if (!is_coherent) {
        if (usage == MemoryUsage::Download) {
            vmaInvalidateAllocation(instance->GetAllocator(), buffer.allocation, offset,
                                    mapped_size);
        } else {
            vmaFlushAllocation(instance->GetAllocator(), buffer.allocation, offset, mapped_size);
        }
    }

    offset += mapped_size;
    if (current_watch_cursor != 0 &&
        current_watches[current_watch_cursor].tick == scheduler->CurrentTick()) {
        current_watches[current_watch_cursor].upper_bound = offset;
        return;
    }

    if (current_watch_cursor + 1 >= current_watches.size()) {
        // Ensure that there are enough watches.
        ReserveWatches(current_watches, WATCHES_RESERVE_CHUNK);
    }

    auto& watch = current_watches[current_watch_cursor++];
    watch.upper_bound = offset;
    watch.tick = scheduler->CurrentTick();
}

void StreamBuffer::ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size) {
    watches.resize(watches.size() + grow_size);
}

bool StreamBuffer::WaitPendingOperations(u64 requested_upper_bound, bool allow_wait) {
    if (!invalidation_mark) {
        return true;
    }
    while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
        auto& watch = previous_watches[wait_cursor];
        if (!scheduler->IsFree(watch.tick) && !allow_wait) {
            return false;
        }
        scheduler->Wait(watch.tick);
        wait_bound = watch.upper_bound;
        ++wait_cursor;
    }
    return true;
}

} // namespace VideoCore
