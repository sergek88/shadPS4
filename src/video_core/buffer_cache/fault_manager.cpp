// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "common/div_ceil.h"
#include "common/postmortem.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/fault_buffer_process_comp.h"
#include "video_core/host_shaders/dual_writeback_comp.h"

namespace VideoCore {

static constexpr size_t MaxPageFaults = 1024;
static constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(u64);

// [POLLJ] SHADPS4_PAGETRACE_POLLJ=1 + SHADPS4_PAGETRACE: the presence window (one u32 per page
// of the DUAL window, BufferCache::POLLJ_*) is appended to the fault buffer, and the slices of it
// that the PAGETRACE ranges cover are copied out and cleared every batch, then folded into the
// per-page counters (PJ). Slices are fixed for the session (the ranges come from the env).
namespace {
struct PollSlice {
    u64 first_page; // absolute 16 KiB page index (>= DUAL_WINDOW_BASE_PAGE)
    u32 pages;
    u64 dl_offset; // byte offset inside the POLLJ region of the download buffer
};
bool PollTraceActive() {
    static const bool on = BufferCache::PollTraceEnabled() && PageTrace::Enabled();
    return on;
}
const std::vector<PollSlice>& PollSlices() {
    static const std::vector<PollSlice> slices = [] {
        std::vector<PollSlice> out;
        if (!PollTraceActive()) {
            return out;
        }
        constexpr u64 win_first = BufferCache::DUAL_WINDOW_BASE_PAGE;
        constexpr u64 win_end = win_first + BufferCache::POLLJ_WINDOW_PAGES;
        u64 dl = 0;
        for (const auto& [base, size] : PageTrace::Ranges()) {
            const u64 first = std::max<u64>(base >> BufferCache::CACHING_PAGEBITS, win_first);
            const u64 end = std::min<u64>((base + size) >> BufferCache::CACHING_PAGEBITS, win_end);
            if (first >= end) {
                continue;
            }
            out.push_back({first, static_cast<u32>(end - first), dl});
            dl += (end - first) * 4;
        }
        return out;
    }();
    return slices;
}
// Bytes of ONE fault area's presence slices in the download buffer (the region is per area).
u64 PollDownloadBytes() {
    u64 total = 0;
    for (const auto& s : PollSlices()) {
        total += u64{s.pages} * 4;
    }
    return total;
}
// The window itself follows the env alone: shaders compiled with it on store into it whether or
// not PAGETRACE names any range, so it must exist whenever they can be built.
u64 PollFaultBytes() {
    return BufferCache::PollTraceEnabled() ? BufferCache::POLLJ_BYTES : 0;
}
// [POLLMAP] address slot of the poll bitmap, behind the presence window: shaders compiled with
// the env on read it, so it exists whenever they can be built. BufferCache writes the address
// once at construction; nothing here ever copies or clears the tail.
u64 PollMapFaultBytes() {
    return BufferCache::PollMapEnabled() ? BufferCache::POLLMAP_SLOT_BYTES : 0;
}
} // namespace

FaultManager::FaultManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                           BufferCache& buffer_cache_, u32 caching_pagebits, u64 caching_num_pages_)
    : scheduler{scheduler_}, buffer_cache{buffer_cache_},
      caching_pagesize{1ULL << caching_pagebits}, caching_num_pages{caching_num_pages_},
      fault_buffer_size{caching_num_pages_ / 8},
      fault_buffer{instance, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                   fault_buffer_size + BufferCache::BDA_LOGS_BYTES + BufferCache::PAGEJ_BYTES +
                       PollFaultBytes() + PollMapFaultBytes()},
      download_buffer{instance, scheduler, MemoryUsage::Download, 0, AllFlags,
                      MaxPendingFaults * PageFaultAreaSize + BufferCache::BDA_LOGS_BYTES + BufferCache::PAGEJ_LIST_BYTES +
                          MaxPendingFaults * PollDownloadBytes()} {
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, fault_buffer.Handle(), "Fault Buffer");

    // Same reasoning as the page table: the bitmap is device local, so without an explicit
    // clear the host would start replaying page faults for pages nobody ever touched.
    // [POLLJ] the presence window included: a stale slot would read as a poll.
    // [POLLMAP] the address slot included: zero means "no bitmap" to the shader.
    fault_buffer.Fill(0,
                      static_cast<u32>(fault_buffer_size + BufferCache::BDA_LOGS_BYTES +
                                       BufferCache::PAGEJ_BYTES + PollFaultBytes() +
                                       PollMapFaultBytes()),
                      0);
    if (BufferCache::PollTraceEnabled()) {
        // The shader indexes the window from POLLJ_BASE_DWORDS; the host must agree with it.
        ASSERT_MSG(BufferCache::POLLJ_BASE_BYTES ==
                       fault_buffer_size + BufferCache::BDA_LOGS_BYTES + BufferCache::PAGEJ_BYTES,
                   "POLLJ: presence window base drifted from the fault buffer layout");
        if (PollTraceActive()) {
            LOG_CRITICAL(Render_Vulkan,
                         "POLLJ: armed, {} slices, {} traced pages inside the window, fault "
                         "buffer +{} KB, download +{} KB ({} areas)",
                         PollSlices().size(), PollDownloadBytes() / 4,
                         BufferCache::POLLJ_BYTES >> 10,
                         (MaxPendingFaults * PollDownloadBytes()) >> 10, MaxPendingFaults);
        } else {
            LOG_CRITICAL(Render_Vulkan,
                         "POLLJ: window allocated (+{} KB) but SHADPS4_PAGETRACE names no range "
                         "- nothing is copied out or folded",
                         BufferCache::POLLJ_BYTES >> 10);
        }
    }
    if (BufferCache::PollMapEnabled()) {
        // The shader reads the slot at PollMapSlotDwords(); the host must agree with it. Not an
        // assert: BufferCache checks the same thing against the buffer's tail when it arms and
        // keeps the marking off on a mismatch (a zero slot is "no bitmap" to the shader).
        const u64 expected = fault_buffer_size + BufferCache::BDA_LOGS_BYTES +
                             BufferCache::PAGEJ_BYTES + PollFaultBytes();
        if (BufferCache::PollMapSlotOffset() != expected) {
            LOG_CRITICAL(Render_Vulkan,
                         "POLLMAP: address slot {:#x} drifted from the fault buffer layout "
                         "(expected {:#x}) - the marking will stay disabled",
                         BufferCache::PollMapSlotOffset(), expected);
        } else {
            LOG_CRITICAL(Render_Vulkan,
                         "POLLMAP: address slot reserved at fault buffer +{:#x} ({} B)",
                         BufferCache::PollMapSlotOffset(), BufferCache::POLLMAP_SLOT_BYTES);
        }
    }

    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 2,
        .pBindings = bindings.data(),
    };
    fault_process_desc_layout =
        Vulkan::Check(device.createDescriptorSetLayoutUnique(desc_layout_ci));

    std::vector<std::string> defines{{fmt::format("CACHING_PAGEBITS={}", caching_pagebits),
                                      fmt::format("MAX_PAGE_FAULTS={}", MaxPageFaults)}};
    const auto module = Vulkan::Compile(HostShaders::FAULT_BUFFER_PROCESS_COMP,
                                        vk::ShaderStageFlagBits::eCompute, device, defines);
    Vulkan::SetObjectName(device, module, "Fault Buffer Parser");

    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &(*fault_process_desc_layout),
    };
    fault_process_pipeline_layout = Vulkan::Check(device.createPipelineLayoutUnique(layout_info));

    const vk::ComputePipelineCreateInfo pipeline_info = {
        .stage = shader_ci,
        .layout = *fault_process_pipeline_layout,
    };
    fault_process_pipeline = Vulkan::Check(device.createComputePipelineUnique({}, pipeline_info));
    Vulkan::SetObjectName(device, *fault_process_pipeline, "Fault Buffer Parser Pipeline");

    device.destroyShaderModule(module);

    // [DUALHOME v2] writeback kernel: binding 0 = page table (primary + window), binding 1 =
    // the page-journal list inside the fault buffer.
    if (BufferCache::DualHomeEnabled()) {
        const std::array<vk::DescriptorSetLayoutBinding, 2> wb_bindings = {{
            {
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
            {
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
        }};
        const vk::DescriptorSetLayoutCreateInfo wb_layout_ci = {
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = 2,
            .pBindings = wb_bindings.data(),
        };
        dual_wb_desc_layout = Vulkan::Check(device.createDescriptorSetLayoutUnique(wb_layout_ci));
        std::vector<std::string> wb_defines{{
            fmt::format("LIST_CAP={}u", DualWritebackCap),
            fmt::format("NUMPAGES={}u", caching_num_pages_),
            // plain 32-bit literals: the preamble is parsed before the int64 extension line
            fmt::format("WINDOW_BASE_PAGE={}u", BufferCache::DUAL_WINDOW_BASE_PAGE),
            fmt::format("WINDOW_PAGES={}u", BufferCache::DUAL_WINDOW_PAGES),
        }};
        const auto wb_module = Vulkan::Compile(HostShaders::DUAL_WRITEBACK_COMP,
                                               vk::ShaderStageFlagBits::eCompute, device, wb_defines);
        Vulkan::SetObjectName(device, wb_module, "Dual Home Writeback");
        const vk::PipelineShaderStageCreateInfo wb_stage = {
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = wb_module,
            .pName = "main",
        };
        const vk::PipelineLayoutCreateInfo wb_layout_info = {
            .setLayoutCount = 1U,
            .pSetLayouts = &(*dual_wb_desc_layout),
        };
        dual_wb_pipeline_layout = Vulkan::Check(device.createPipelineLayoutUnique(wb_layout_info));
        const vk::ComputePipelineCreateInfo wb_pipeline_info = {
            .stage = wb_stage,
            .layout = *dual_wb_pipeline_layout,
        };
        dual_wb_pipeline = Vulkan::Check(device.createComputePipelineUnique({}, wb_pipeline_info));
        Vulkan::SetObjectName(device, *dual_wb_pipeline, "Dual Home Writeback Pipeline");
        device.destroyShaderModule(wb_module);
        LOG_CRITICAL(Render_Vulkan, "DUALHOME: writeback kernel ready (cap {} pages/batch)",
                     DualWritebackCap);
    }
}

// [PAGEJOURNAL] SHADPS4_PAGEJOURNAL=1 (needs BDATRACK=1 for the buffer plumbing)
static bool PageJournalEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_PAGEJOURNAL");
        return e && (e[0] == '1' || e[0] == '2');
    }();
    return on;
}

void FaultManager::ProcessFaultBuffer() {
    if (u64 wait_tick = fault_areas[current_area]) {
        // [Stage1b PARSERPROF] This wait runs on the parser thread and stalls it
        // until the previous fault-parse tick retires - a suspected slice of the
        // fatal EOP-silence window at level load.
        const s64 fb_t0 = Parserprof::Enabled()
                              ? std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count()
                              : 0;
        scheduler.Wait(wait_tick);
        scheduler.PopPendingOperations();
        if (fb_t0 != 0) {
            const s64 fb_now = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
            Parserprof::g_fb_wait_us.fetch_add(static_cast<u64>(fb_now - fb_t0),
                                               std::memory_order_relaxed);
        }
    }

    const u32 offset = current_area * PageFaultAreaSize;
    u8* mapped = download_buffer.mapped_data.data() + offset;
    std::memset(mapped, 0, PageFaultAreaSize);

    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_buffer_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_buffer_size,
    };
    const vk::DescriptorBufferInfo fault_buffer_info = {
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .range = fault_buffer_size,
    };
    const vk::DescriptorBufferInfo download_info = {
        .buffer = download_buffer.Handle(),
        .offset = offset,
        .range = PageFaultAreaSize,
    };
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &fault_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &download_info,
        },
    }};
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    // Pull the raw-address write log across and clear its counter for the next frame.
    // Only when the log is actually being collected: otherwise this copy-then-clear pair
    // runs every frame for nothing, and the clear races the copy (write-after-read on the
    // fault buffer, flagged by synchronization validation).
    static const bool write_log_enabled = [] {
        const char* e = std::getenv("SHADPS4_BDATRACK");
        return e && e[0] == '1';
    }();
    const u64 log_download_offset = MaxPendingFaults * PageFaultAreaSize;
    u8* log_mapped = download_buffer.mapped_data.data() + log_download_offset;
    if (write_log_enabled) {
        std::memset(log_mapped, 0, BufferCache::BDA_LOGS_BYTES + BufferCache::PAGEJ_LIST_BYTES);
        cmdbuf.copyBuffer(fault_buffer.Handle(), download_buffer.Handle(),
                          vk::BufferCopy{
                              .srcOffset = fault_buffer_size,
                              .dstOffset = log_download_offset,
                              .size = BufferCache::BDA_LOGS_BYTES,
                          });
        const vk::BufferMemoryBarrier2 log_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eClear,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = fault_buffer.Handle(),
            .offset = fault_buffer_size,
            .size = BufferCache::BDA_LOGS_BYTES,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &log_barrier,
        });
        cmdbuf.fillBuffer(fault_buffer.Handle(), fault_buffer_size, sizeof(u32), 0);
        // [DROPLOG] and the counter of the dropped-write log behind it
        cmdbuf.fillBuffer(fault_buffer.Handle(), fault_buffer_size + BufferCache::BDA_WRITE_LOG_BYTES,
                          sizeof(u32), 0);
        // [DUALHOME v2] before the list is copied out and cleared: copy every listed page from
        // its VRAM owner into its guest-memory home, in command order after this batch's work.
        if (PageJournalEnabled() && BufferCache::DualHomeEnabled() && dual_wb_pipeline) {
            const vk::MemoryBarrier2 wb_pre = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &wb_pre,
            });
            Vulkan::InvalidateBoundPipeline();
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *dual_wb_pipeline);
            Buffer* pt = buffer_cache.GetBdaPageTableBuffer();
            const vk::DescriptorBufferInfo wb_pt_info = {
                .buffer = pt->Handle(),
                .offset = 0,
                .range = pt->SizeBytes(),
            };
            const vk::DescriptorBufferInfo wb_list_info = {
                .buffer = fault_buffer.Handle(),
                .offset = fault_buffer_size + BufferCache::BDA_LOGS_BYTES + BufferCache::PAGEJ_BITMAP_BYTES,
                .range = BufferCache::PAGEJ_LIST_BYTES,
            };
            const std::array<vk::WriteDescriptorSet, 2> wb_writes = {{
                {
                    .dstSet = VK_NULL_HANDLE,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &wb_pt_info,
                },
                {
                    .dstSet = VK_NULL_HANDLE,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &wb_list_info,
                },
            }};
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *dual_wb_pipeline_layout,
                                        0, wb_writes);
            cmdbuf.dispatch(DualWritebackCap, 1, 1);
            const vk::MemoryBarrier2 wb_post = {
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &wb_post,
            });
        }
        // [PAGEJOURNAL] pull the page list across, then clear the bitmap and the list count
        if (PageJournalEnabled()) {
            const u64 pj_list_src = fault_buffer_size + BufferCache::BDA_LOGS_BYTES +
                                    BufferCache::PAGEJ_BITMAP_BYTES;
            cmdbuf.copyBuffer(fault_buffer.Handle(), download_buffer.Handle(),
                              vk::BufferCopy{
                                  .srcOffset = pj_list_src,
                                  .dstOffset = log_download_offset + BufferCache::BDA_LOGS_BYTES,
                                  .size = BufferCache::PAGEJ_LIST_BYTES,
                              });
            const vk::BufferMemoryBarrier2 pj_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eClear,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .buffer = fault_buffer.Handle(),
                .offset = fault_buffer_size + BufferCache::BDA_LOGS_BYTES,
                .size = BufferCache::PAGEJ_BYTES,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &pj_barrier,
            });
            cmdbuf.fillBuffer(fault_buffer.Handle(), fault_buffer_size + BufferCache::BDA_LOGS_BYTES,
                              BufferCache::PAGEJ_BITMAP_BYTES + sizeof(u32), 0);
        }
    }
    // [POLLJ] pull the traced slices of the presence window across, then clear them for the
    // next batch - the same copy -> clear shape as the page journal above. The download region
    // is per fault area like the fault lists (not single-buffered like the logs): the wait at
    // the top of this function retired this area's previous batch, so one batch's GPU copy can
    // never land on bytes another batch's fold is still reading. No host memset either - the
    // slice copies tile the area exactly, so the GPU is the only writer. What remains is the
    // shape the logs already live with: a priority fold lagging MaxPendingFaults batches behind.
    const u64 pollj_download_offset = log_download_offset + BufferCache::BDA_LOGS_BYTES +
                                      BufferCache::PAGEJ_LIST_BYTES +
                                      current_area * PollDownloadBytes();
    u8* pollj_mapped = download_buffer.mapped_data.data() + pollj_download_offset;
    const bool pollj_active = PollTraceActive() && !PollSlices().empty();
    if (pollj_active) {
        std::vector<vk::BufferMemoryBarrier2> pollj_barriers;
        pollj_barriers.reserve(PollSlices().size());
        const auto slice_src = [](const PollSlice& s) {
            return BufferCache::POLLJ_BASE_BYTES +
                   (s.first_page - BufferCache::DUAL_WINDOW_BASE_PAGE) * 4;
        };
        // The slots are written by this batch's waves: make those stores visible to the copy
        // (the journal copies above rely on the parse barrier that follows; this one is a
        // diagnostic-only path, so it pays for its own).
        for (const auto& s : PollSlices()) {
            pollj_barriers.push_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .buffer = fault_buffer.Handle(),
                .offset = slice_src(s),
                .size = u64{s.pages} * 4,
            });
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(pollj_barriers.size()),
            .pBufferMemoryBarriers = pollj_barriers.data(),
        });
        pollj_barriers.clear();
        for (const auto& s : PollSlices()) {
            cmdbuf.copyBuffer(fault_buffer.Handle(), download_buffer.Handle(),
                              vk::BufferCopy{
                                  .srcOffset = slice_src(s),
                                  .dstOffset = pollj_download_offset + s.dl_offset,
                                  .size = u64{s.pages} * 4,
                              });
            pollj_barriers.push_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eClear,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .buffer = fault_buffer.Handle(),
                .offset = slice_src(s),
                .size = u64{s.pages} * 4,
            });
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(pollj_barriers.size()),
            .pBufferMemoryBarriers = pollj_barriers.data(),
        });
        for (const auto& s : PollSlices()) {
            cmdbuf.fillBuffer(fault_buffer.Handle(), slice_src(s), u64{s.pages} * 4, 0);
        }
        // ... and the clear must land before the next batch's waves stamp the same slots: a
        // write-after-write needs a memory dependency, and the parse barriers below only span
        // the fault bitmap.
        pollj_barriers.clear();
        for (const auto& s : PollSlices()) {
            pollj_barriers.push_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eClear,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask =
                    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
                .buffer = fault_buffer.Handle(),
                .offset = slice_src(s),
                .size = u64{s.pages} * 4,
            });
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(pollj_barriers.size()),
            .pBufferMemoryBarriers = pollj_barriers.data(),
        });
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    Vulkan::InvalidateBoundPipeline();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *fault_process_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *fault_process_pipeline_layout, 0,
                                writes);
    // 1 bit per page, 32 pages per workgroup
    const u32 num_threads = caching_num_pages / 32;
    const u32 num_workgroups = Common::DivCeil(num_threads, 64u);
    cmdbuf.dispatch(num_workgroups, 1, 1);

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });

    // Priority defer, not the submit-pumped queue: plain deferred ops only run when the
    // NEXT submit arrives, so during a submit lull (level-load frames) replayed raw
    // stores sat invisible for seconds while the guest already saw this frame's EOP
    // label - it then read a zero frame tick and tripped its own assert
    // (fg-draw-mgr.cpp:234). The priority thread lands the bytes right after the GPU
    // tick retires, which is the ordering real hardware gives these stores.
    // [POLLJ] fold of the presence slices, on the same priority thread as the logs: a non-zero
    // slot means a poll loop of the shader whose hash (low 32 bits | 1) sits in it read the
    // page this batch. PageTrace::Note is relaxed-atomic, so the thread does not matter to it;
    // the poller table and the 5 s summary are this thread's own.
    const auto pollj_fold = [pollj_mapped, pollj_active] {
        if (!pollj_active) {
            return;
        }
        struct Poller {
            u32 hash;    // as stored: low 32 bits of the shader hash, bit 0 forced to 1
            u32 batches; // batches this window with the page present
        };
        static std::unordered_map<VAddr, Poller> pollers;
        static u64 pj_batches = 0;
        static u64 pj_marks = 0;
        static auto pj_next = std::chrono::steady_clock::now();
        for (const auto& s : PollSlices()) {
            const u32* slots = std::bit_cast<const u32*>(pollj_mapped + s.dl_offset);
            for (u32 i = 0; i < s.pages; ++i) {
                const u32 v = slots[i];
                if (v == 0) {
                    continue;
                }
                const VAddr page = (s.first_page + i) << BufferCache::CACHING_PAGEBITS;
                PageTrace::Note(page, PageTrace::PJ);
                auto& p = pollers[page];
                p.hash = v;
                ++p.batches;
                ++pj_marks;
            }
        }
        ++pj_batches;
        const auto now = std::chrono::steady_clock::now();
        if (now >= pj_next) {
            pj_next = now + std::chrono::seconds(5);
            std::vector<std::pair<VAddr, Poller>> top(pollers.begin(), pollers.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
                return a.second.batches != b.second.batches ? a.second.batches > b.second.batches
                                                            : a.first < b.first;
            });
            std::string line;
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                fmt::format_to(std::back_inserter(line), "{:#x} by {:#x} x{} | ", top[i].first,
                               top[i].second.hash, top[i].second.batches);
            }
            LOG_CRITICAL(Render_Vulkan,
                         "POLLJ: {} pages polled in {} batches ({} marks), poller = hash32|1 | {}",
                         pollers.size(), pj_batches, pj_marks, line.empty() ? "none" : line);
            pollers.clear();
            pj_batches = 0;
            pj_marks = 0;
        }
    };
    if (!write_log_enabled && pollj_active) {
        // [POLLJ] without BDATRACK there is no log op to ride on: the fold gets its own
        scheduler.DeferPriorityOperation(pollj_fold);
    }
    if (write_log_enabled)
    scheduler.DeferPriorityOperation([this, log_mapped, pollj_fold] {
        const u32* log = std::bit_cast<const u32*>(log_mapped);
        const u32 requested = log[0];
        const u32 applied = std::min(requested, BufferCache::BDA_WRITE_LOG_ENTRIES);
        if (requested > BufferCache::BDA_WRITE_LOG_ENTRIES) {
            LOG_CRITICAL(Render_Vulkan, "BDALOG-OVERFLOW: raw-address write log overflowed: {} of {} entries kept",
                        BufferCache::BDA_WRITE_LOG_ENTRIES, requested);
        }
        {
            // How much the GPU actually writes through raw addresses, once every 5 s.
            static u64 total = 0;
            static u64 batches = 0;
            static auto next = std::chrono::steady_clock::now();
            total += applied;
            ++batches;
            const auto now = std::chrono::steady_clock::now();
            if (now >= next) {
                next = now + std::chrono::seconds(5);
                LOG_CRITICAL(Render_Vulkan, "BDALOG: {} writes replayed over {} batches", total,
                             batches);
                total = 0;
                batches = 0;
            }
        }
        auto* memory = Core::Memory::Instance();
        // [LIVEREPLAY] SHADPS4_LIVEREPLAY_SKIP=1: a write that the GPU already put into an
        // imported backing page (= guest memory) is not replayed; the replay would land up to a
        // submit later and could overwrite what the CPU wrote to that slot in the meantime.
        // =1: the GPU decides (page-table bit 63, write_const_live skips the log itself) and
        //     the host replays everything that still reaches the log - a write made while the
        //     entry was not yet published (fault path) must still land through the replay.
        // =2: host-side decision by CPU page_table ownership (lossy in that publish window;
        //     liveloop12 crashed on it), kept for comparison only.
        static const bool skip_landed = [] {
            const char* e = std::getenv("SHADPS4_LIVEREPLAY_SKIP");
            return e && e[0] == '2';
        }();
        static const bool skip_stats = [] {
            const char* e = std::getenv("SHADPS4_LIVEREPLAY_SKIP");
            return e && (e[0] == '1' || e[0] == '2');
        }();
        static u64 skipped_total = 0;
        static u64 replayed_total = 0;
        static auto skip_next = std::chrono::steady_clock::now();
        u32 skipped = 0;
        for (u32 i = 0; i < applied; ++i) {
            const u32* entry = log + 1 + i * BufferCache::BDA_WRITE_LOG_WORDS_PER_ENTRY;
            const VAddr addr = (u64{entry[1]} << 32) | entry[0];
            const u32 value = entry[2];
            if (skip_landed && buffer_cache.LiveWriteLandsInGuest(addr)) {
                ++skipped;
                continue;
            }
            memory->TryWriteBacking(std::bit_cast<u8*>(addr), &value, sizeof(value));
        }
        // [DROPLOG] first-touch writes the GPU had to drop: land them now. Nothing here
        // depends on caching state - guest memory is the home the CPU reads.
        {
            const u32* dlog =
                std::bit_cast<const u32*>(log_mapped + BufferCache::BDA_WRITE_LOG_BYTES);
            const u32 drequested = dlog[0];
            const u32 dapplied = std::min(drequested, BufferCache::BDA_DROP_LOG_ENTRIES);
            for (u32 i = 0; i < dapplied; ++i) {
                const u32* entry = dlog + 1 + i * BufferCache::BDA_WRITE_LOG_WORDS_PER_ENTRY;
                const VAddr addr = (u64{entry[1]} << 32) | entry[0];
                const u32 value = entry[2];
                memory->TryWriteBacking(std::bit_cast<u8*>(addr), &value, sizeof(value));
                PageTrace::Note(addr, PageTrace::DL);
            }
            static u64 drop_total = 0;
            static u64 drop_lost = 0;
            static u64 drop_batches = 0;
            static VAddr drop_last = 0;
            static auto drop_next = std::chrono::steady_clock::now();
            drop_total += dapplied;
            drop_lost += drequested - dapplied;
            ++drop_batches;
            if (dapplied != 0) {
                const u32* last = dlog + 1 + (dapplied - 1) * BufferCache::BDA_WRITE_LOG_WORDS_PER_ENTRY;
                drop_last = (u64{last[1]} << 32) | last[0];
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= drop_next) {
                drop_next = now + std::chrono::seconds(5);
                if (drop_total != 0 || drop_lost != 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "DROPLOG: {} first-touch writes landed, {} lost (overflow), {} "
                                 "batches, last {:#x}",
                                 drop_total, drop_lost, drop_batches, drop_last);
                }
                drop_total = 0;
                drop_lost = 0;
                drop_batches = 0;
            }
        }
        // [PAGEJOURNAL] pages the GPU wrote this batch -> readback protection (VRAM pages only)
        if (PageJournalEnabled()) {
            const u32* plist = std::bit_cast<const u32*>(log_mapped + BufferCache::BDA_LOGS_BYTES);
            const u32 prequested = plist[0];
            const u32 papplied = std::min(prequested, BufferCache::PAGEJ_LIST_ENTRIES);
            for (u32 i = 0; i < papplied; ++i) {
                const u64 page = plist[1 + i];
                if (page < BufferCache::CACHING_NUMPAGES) {
                    buffer_cache.NoteGpuWrittenPage(page << BufferCache::CACHING_PAGEBITS);
                    PageTrace::Note(page << BufferCache::CACHING_PAGEBITS, PageTrace::GW);
                }
            }
            static u64 pj_total = 0;
            static u64 pj_lost = 0;
            static u64 pj_batches = 0;
            static auto pj_next = std::chrono::steady_clock::now();
            pj_total += papplied;
            pj_lost += prequested - papplied;
            ++pj_batches;
            const auto now = std::chrono::steady_clock::now();
            if (now >= pj_next) {
                pj_next = now + std::chrono::seconds(5);
                LOG_CRITICAL(Render_Vulkan, "PAGEJ: {} written pages marked, {} lost, {} batches",
                             pj_total, pj_lost, pj_batches);
                pj_total = 0;
                pj_lost = 0;
                pj_batches = 0;
            }
        }
        if (skip_stats) {
            skipped_total += skipped;
            replayed_total += applied - skipped;
            const auto now = std::chrono::steady_clock::now();
            if (now >= skip_next) {
                skip_next = now + std::chrono::seconds(5);
                LOG_CRITICAL(Render_Vulkan, "LIVEREPLAY: {} landed live (skipped), {} replayed",
                             skipped_total, replayed_total);
                skipped_total = 0;
                replayed_total = 0;
            }
        }
        pollj_fold();
    });

    scheduler.DeferOperation([this, mapped, area = current_area] {
        fault_ranges.Clear();
        const u64* fault_buf = std::bit_cast<const u64*>(mapped);
        const u32 fault_count = fault_buf[0];
        for (u32 i = 1; i <= fault_count; ++i) {
            fault_ranges.Add(fault_buf[i], caching_pagesize);
            LOG_INFO(Render_Vulkan, "Accessed non-GPU cached memory at {:#x}", fault_buf[i]);
        }
        // [GPUFAULT] How often the GPU touches a page nobody cached yet (each such write is
        // dropped or, with DROPLOG, landed late). Once per 5 s, with the last addresses.
        {
            static u64 gf_pages = 0;
            static u64 gf_batches = 0;
            static u64 gf_batches_with = 0;
            static std::array<u64, 3> gf_last{};
            static auto gf_next = std::chrono::steady_clock::now();
            // Only guest-range pages count as real first touches; stray pointers (0x0,
            // 0x10000, 0x8000000000...) are noise and are counted separately.
            static u64 gf_stray = 0;
            u32 real = 0;
            for (u32 i = 1; i <= fault_count; ++i) {
                const u64 a = fault_buf[i];
                if (a >= 0x1000000000ULL && a < 0x1200000000ULL) {
                    ++real;
                    buffer_cache.NoteGpuFaultPage(a);
                    PageTrace::Note(a, PageTrace::FT);
                    gf_last[2] = gf_last[1];
                    gf_last[1] = gf_last[0];
                    gf_last[0] = a;
                } else {
                    ++gf_stray;
                }
            }
            gf_pages += real;
            ++gf_batches;
            if (real != 0) {
                ++gf_batches_with;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= gf_next) {
                gf_next = now + std::chrono::seconds(5);
                LOG_CRITICAL(Render_Vulkan,
                             "GPUFAULT: {} real uncached pages in {} of {} batches (+{} stray) | "
                             "last {:#x} {:#x} {:#x}",
                             gf_pages, gf_batches_with, gf_batches, gf_stray, gf_last[0],
                             gf_last[1], gf_last[2]);
                gf_pages = 0;
                gf_batches = 0;
                gf_batches_with = 0;
                gf_stray = 0;
            }
        }
        fault_ranges.ForEach([&](VAddr start, VAddr end) {
            ASSERT_MSG((end - start) <= std::numeric_limits<u32>::max(),
                       "Buffer size is too large");
            // fault_ranges is a joining set: adjacent pages coalesce into one span, and a
            // whole-span validity check let a single unmapped page sink every valid
            // neighbour with it. A rejected page never heals - it faults again each frame
            // and reads zeros forever - so validate per page and cache each valid run.
            VAddr run_begin = 0;
            const auto flush_run = [&](VAddr run_end) {
                if (run_begin != 0) {
                    // Raw-address writes are carried by the write log below, which puts the
                    // bytes straight into guest memory; flagging these pages GPU-modified
                    // instead was tried and produced a write-only page (GPU-modified plus
                    // CPU-modified), which is not a legal protection combination.
                    buffer_cache.FindBuffer(run_begin, static_cast<u32>(run_end - run_begin));
                    run_begin = 0;
                }
            };
            for (VAddr page = start; page < end; page += caching_pagesize) {
                // [TDR ETAP5] A page held by a live rescue import must NOT get a VRAM buffer
                // stretched over it: that would overwrite the protocol page and re-spin invisibly.
                if (buffer_cache.RescueBdaFor(page) != 0) {
                    flush_run(page);
                    continue;
                }
                // [HOMEPOLICY] keep each run homogeneous in its desired home, so the owner the
                // cache creates is never a union of import-wanting and VRAM pages.
                if (run_begin != 0 && HomePolicy::Enabled() &&
                    HomePolicy::WantsImport(page, caching_pagesize) !=
                        HomePolicy::WantsImport(run_begin, caching_pagesize)) {
                    flush_run(page);
                }
                if (buffer_cache.IsFaultRegionValid(page, caching_pagesize)) {
                    if (run_begin == 0) {
                        run_begin = page;
                    }
                    continue;
                }
                flush_run(page);
                static std::atomic<u64> rejected{0};
                const u64 n = rejected.fetch_add(1, std::memory_order_relaxed) + 1;
                // [SKINVRAM] A page of the synthetic window arriving here means the shader
                // dereferenced an entry that never landed - the one failure mode that otherwise
                // looks exactly like success (redirected= climbing, every skip at zero). The
                // stand keeps only Critical lines, so this one must be Critical.
                if (buffer_cache.SkinVramNoteFaultPage(page)) {
                    static std::atomic<u64> skin_rejected{0};
                    const u64 sn = skin_rejected.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (sn <= 16 || (sn % 512) == 0) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "SKINVRAM: GPU touched synthetic page {:#x} with no page-table "
                                     "entry ({} such pages) - the publish did not land",
                                     page, sn);
                    }
                } else if (n <= 16 || (n % 512) == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "GPU touched unmapped page {:#x}, leaving it uncached ({} total)",
                                page, n);
                }
            }
            flush_run(end);
        });
        fault_areas[area] = 0;
    });

    fault_areas[current_area++] = scheduler.CurrentTick();
    current_area %= MaxPendingFaults;
}

} // namespace VideoCore
