// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other)
        : device{other.device}, allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
          bda_addr{std::exchange(other.bda_addr, 0)},
          imported_memory{std::exchange(other.imported_memory, VK_NULL_HANDLE)},
          is_backing{std::exchange(other.is_backing, false)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) {
        device = other.device;
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        bda_addr = std::exchange(other.bda_addr, 0);
        imported_memory = std::exchange(other.imported_memory, VK_NULL_HANDLE);
        is_backing = std::exchange(other.is_backing, false);
        return *this;
    }

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
    // Non-null for import-backed buffers (VK_EXT_external_memory_host): the
    // memory IS guest RAM, allocated outside VMA and freed after the buffer.
    vk::DeviceMemory imported_memory{};
    // [BACKIMP] Bound over a chunk of the globally imported backing section. The memory is
    // shared by every such buffer and owned by BackingImport, so destruction must take the
    // VkBuffer and NOTHING else - no vkFreeMemory, no VMA. Checked first in the destructor.
    bool is_backing{};
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    /// Import-backed buffer over guest RAM (VK_EXT_external_memory_host):
    /// requires a strictly HOST_VISIBLE|HOST_COHERENT memory type; returns
    /// nullopt on any failure so callers fall back to the ordinary VMA path.
    /// `via` names the calling path (label / rescue / regular) and only reaches the BACKIMP log.
    static std::optional<Buffer> CreateHostImport(const Vulkan::Instance& instance,
                                                  Vulkan::Scheduler& scheduler, VAddr base,
                                                  u64 size, const char* via = "import");

    /// [BACKIMP] Buffer bound over the already imported backing section - no vkAllocateMemory
    /// at all, just vkCreateBuffer + vkBindBufferMemory at an offset. Returns nullopt (with
    /// *reject_reason set) whenever the range is not a plain window into the backing, so the
    /// caller falls back to CreateHostImport / the ordinary VMA buffer.
    static std::optional<Buffer> CreateBackingBacked(const Vulkan::Instance& instance,
                                                     Vulkan::Scheduler& scheduler, VAddr base,
                                                     u64 size, const char** reject_reason,
                                                     const char* via, bool force_mode = false);

    struct HostImportTag {};
    /// No-allocation constructor used only by CreateHostImport.
    Buffer(HostImportTag, const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
           VAddr cpu_addr_, u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return buffer.bda_addr;
    }

    /// [CSBATCH] True when BufferDeviceAddress() may be asked (the stream_buffer and the batch
    /// ring carry eShaderDeviceAddress; SNAPRO shadows, the download/staging rings do not).
    bool HasDeviceAddress() const noexcept {
        return buffer.bda_addr != 0;
    }

    // [PRECISE-STAGES] SHADPS4_PRECISE_STAGES=1 (read once): barriers target the using stage and
    // a read of memory that an explicit post-barrier already made visible to every stage
    // (MemoryRead @ ALL_COMMANDS - the state after an upload or a JoinOverlap copy) needs no
    // further barrier. liveloop108: every worker dispatch bound freshly created buffers and paid
    // a full-drain barrier (~25 us) just to read them.
    static bool PreciseStages() {
        static const bool on = [] {
            const char* e = std::getenv("SHADPS4_PRECISE_STAGES");
            return e && e[0] == '1';
        }();
        return on;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0) {
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }
        if (PreciseStages()) {
            constexpr vk::AccessFlags2 kReads =
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eIndexRead |
                vk::AccessFlagBits2::eVertexAttributeRead |
                vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eUniformRead |
                vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eMemoryRead;
            const bool read_only = (dst_acess_mask & ~kReads) == vk::AccessFlags2{};
            const bool visible_to_all = (access_mask & vk::AccessFlagBits2::eMemoryRead) &&
                                        stage == vk::PipelineStageFlagBits2::eAllCommands;
            if (read_only && visible_to_all) {
                return {}; // already visible to every reader; keep the state
            }
        }

        DEBUG_ASSERT(offset < size_bytes);

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    // Scheduler tick after which this buffer's backing holds meaningful data:
    // a JoinOverlap copy into a freshly created buffer executes at this tick,
    // and a readback that skips waiting for it would copy garbage.
    u64 content_tick{};
    // Tick of the last CPU->GPU upload recorded into this buffer: a readback
    // that does not wait for it can return PRE-upload bytes and resurrect
    // stale data over fresher CPU writes.
    u64 last_upload_tick{};
    // The buffer's memory IS guest RAM: never upload into it, never deliver
    // readbacks from it, never evict it for VRAM pressure.
    bool is_import_backed{};
    // [BACKIMP] Subset of is_import_backed: guest RAM reached through the shared backing
    // import rather than a private per-buffer host import. Diagnostics only - every cache
    // rule that matters is already keyed off is_import_backed.
    bool is_backing_backed{};
    // [WRITESTAT] Guest-write history. 0 = the guest has never written this buffer.
    u64 write_first_frame{};
    bool counted_written{};
    // [TOUCHSTAT] Frame number of the last ObtainBuffer touch. Lets the per-frame "unique bytes"
    // census be O(1) with no side table: the first touch in a frame is the one that counts.
    u64 touch_stamp{};
    bool counted_late{};
    // [BYWRITE] Already standing in the deferred-correction queue; keeps the queue O(1) to
    // deduplicate without scanning it on every write.
    bool bywrite_queued{};
    u64 backing_offset{};
    // Distance between the bind offset and the requested one. CreateBackingBacked refuses a
    // non-zero delta outright (every guest range it sees is page aligned, and a shifted bind
    // would desync BufferDeviceAddress() from Offset()), so this is always 0 today - it is
    // kept because the bind alignment is a driver property, not a constant.
    u64 backing_delta{};
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 lru_id = 0;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
};

class StreamBuffer : public Buffer {
public:
    /// [CSBATCH] extra_flags: usage bits on top of AllFlags (eShaderDeviceAddress for the
    /// buffer cache's stream_buffer and the batch ring; UniqueBuffer::Create then takes a
    /// dedicated allocation and fetches the address).
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_,
                          vk::BufferUsageFlags extra_flags = {});

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// [CSBATCH] Makes GPU writes into a mapped region visible to the CPU (no-op on coherent
    /// memory). Only for regions the GPU wrote and whose tick has retired.
    void HostInvalidate(u64 region_offset, u64 region_size);

    /// Incremented every time the ring wraps back to offset 0 and starts overwriting. Any
    /// cached offset taken before the bump points at memory that has since been reused.
    [[nodiscard]] u64 WrapGeneration() const noexcept {
        return wrap_generation;
    }

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    u64 wrap_generation{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

} // namespace VideoCore
