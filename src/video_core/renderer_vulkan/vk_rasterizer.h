// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <span>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    // Flush the command buffer every N draws so a level-load frame never
    // exceeds the host GPU watchdog budget (see vk_rasterizer.cpp).
    void ChunkHeavyFrame();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size);
    void ProcessDownloadImages();
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void CpSync();
    u64 Flush();
    void Finish();
    void OnSubmit();
    /// [EOPRETIRE] Queues the pending fault-buffer processing (BDA-journal replay op) on the
    /// current tick now; the EOP/ReleaseMem sites call it before deferring the fence.
    void ProcessFaultBufferIfPending();

    /// Runs the callback on the priority-ops thread once the GPU retires every
    /// command recorded so far. Used to hold guest-visible completion interrupts
    /// (EOP/RELEASE_MEM) until the work they announce has actually executed.
    void DeferSignalToRetire(std::function<void()>&& func);

    /// [CSBATCH] Called by ExecuteShaderHLE for the skinning worker: resolves the dispatch's 7
    /// ranges through the buffer cache NOW (guest order) and appends one job to the pending
    /// batch. false = decline -> the dispatch takes the normal path (the batch is flushed first).
    bool SkinBatchAppend(const Shader::Info& cs, const AmdGpu::ComputeProgram& p);
    /// [CSBATCH] Records the pending batch as ONE host dispatch. Called before any other GPU
    /// command and from the scheduler's pre-submit hook; one `empty` test when idle.
    void FlushSkinBatch(SkinBatch::Reason why);

    /// [CSBATCH-DEFERLABEL] Queue a guest completion label behind the pending batch instead of
    /// flushing the batch for it (run 359: flush{label}=60695 -> subs/batch=1, no batching at
    /// all). Returns false when there is nothing pending, and the caller signals as it does
    /// today. The queued labels are registered as priority operations right after the batch is
    /// recorded, so they retire on the tick that executes it.
    bool SkinBatchDeferLabel(std::function<void()>&& fence);

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

    /// The uses_dma block of BindResources (once-per-flip sweep / DMASYNC_ZONE), extracted so
    /// the [CSBATCH] append keeps today's per-worker-dispatch cadence. Behaviour unchanged.
    void SyncDmaZones();
    // [CSBATCH] internals (vk_rasterizer.cpp)
    void RecordSkinDispatch(std::span<const SkinJobGpu> jobs, std::span<const u32> group_map,
                            std::span<const vk::BufferMemoryBarrier2> barriers, bool prebar,
                            bool postbar, SkinBatch::Reason why);
    bool SkinBatchShadowMaybe(const SkinJobGpu& j, u32 n, VAddr out_base, u64 out_size,
                              const VideoCore::Buffer& out_owner);
    void SkinBatchShadowArm(); // after the guest dispatch of the shadowed job
    void SkinBatchShadowCompare();
    void SkinBatchReport(); // 5 s census

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    bool fault_process_pending{};
    // Frame stamp of the last full mapped-memory sync for DMA shaders (see BindResources).
    u64 dma_sync_stamp{~0ULL};
    bool attachment_feedback_loop{};
    u32 draws_since_flush{0};
    SkinBatch skin_batch; // [CSBATCH]
    /// [CSBATCH-DEFERLABEL] guest completion labels waiting for the batch's tick.
    std::vector<std::function<void()>> skin_batch_labels;
};

} // namespace Vulkan
