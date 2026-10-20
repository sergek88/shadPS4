// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring> // [CSBATCH]
#include <deque>
#include <span> // [CSBATCH]
#include <string>
#include <thread> // [CSBATCH]
#include <tuple>  // [CSBATCH]
#include <unordered_map>
#include <unordered_set> // [CSSPLIT] once-per-shader census notes
#include <vector>
#include <boost/container/static_vector.hpp> // [CSBATCH]
#include "common/alignment.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/postmortem.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

// [debug] общий выключатель шумных отладочных логов: SHADPS4_DBGLOG=1
static bool DbgLogEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DBGLOG");
        return e && e[0] == '1';
    }();
    return on;
}

// [debug] счётчики работы GPU за период — читает liverpool.cpp
std::atomic<u32> g_dbg_draws{0};
// Байты, залитые в GPU-буферы за кадр (SynchronizeBuffer/CopyBuffer) —
// печатается и сбрасывается в FLIPSTAT.
std::atomic<u64> g_dbg_upload_bytes{0};
std::atomic<u32> g_dbg_dispatches{0};
std::atomic<u32> g_dbg_frame_no{0};

// Кольцо последних диспатчей: при зависании present-фенса презентер
// вызывает DumpRecentDispatches() — хвост кольца показывает, какой
// шейдер GPU получил последним (кандидат в вечный спин).
struct RecentDispatch {
    u64 hash;
    u32 gx, gy, gz;
    u32 frame;
};
static std::array<RecentDispatch, 64> g_recent_disp{};
static std::atomic<u32> g_recent_disp_idx{0};

// [DESCSTAT] How many descriptor writes repeat what that slot already held. Compares only the
// three fields that identify a buffer binding, so it is a few loads per write - cheap enough to
// leave on, and it answers whether "write everything every draw" is worth fixing.
struct DescSig {
    VkBuffer buffer;
    u64 offset;
    u64 range;
};
static std::vector<DescSig> g_last_desc;

static void CountRedundantDescriptors(const auto& set_writes) {
    if (g_last_desc.size() < set_writes.size()) {
        g_last_desc.resize(set_writes.size(), DescSig{VK_NULL_HANDLE, ~0ull, ~0ull});
    }
    u64 redundant = 0;
    for (size_t i = 0; i < set_writes.size(); ++i) {
        const auto* bi = set_writes[i].pBufferInfo;
        if (bi == nullptr) {
            continue; // image/sampler writes are not compared here
        }
        const DescSig sig{static_cast<VkBuffer>(bi->buffer), bi->offset, bi->range};
        if (g_last_desc[i].buffer == sig.buffer && g_last_desc[i].offset == sig.offset &&
            g_last_desc[i].range == sig.range) {
            ++redundant;
        } else {
            g_last_desc[i] = sig;
        }
    }
    Parserprof::g_desc_redundant.fetch_add(redundant, std::memory_order_relaxed);
}

static void RecordDispatch(u64 hash, u32 gx, u32 gy, u32 gz) {
    const u32 slot = g_recent_disp_idx.fetch_add(1, std::memory_order_relaxed) %
                     static_cast<u32>(g_recent_disp.size());
    g_recent_disp[slot] = {hash, gx, gy, gz, g_dbg_frame_no.load(std::memory_order_relaxed)};
}

// [GPUSTALL] Per-SUBMIT contents, keyed by scheduler tick. When the master semaphore stops
// advancing, the ticks between the GPU's value and the awaited one are exactly the submits still
// executing - this says WHAT is in them instead of leaving it to guesswork.
// Written only by the parser thread (the sole recorder), read by the watchdog.
struct SubmitRec {
    std::atomic<u64> tick{0};
    std::atomic<u32> draws{0};
    std::atomic<u32> dispatches{0};
    std::atomic<u64> hashes[12]{}; // [GPUSTALL] first 12 distinct-ish shaders of the submit
    std::atomic<u32> gx{0}, gy{0}, gz{0};
    std::atomic<s64> first_ms{0};
};
static constexpr u32 kSubmitRing = 16;
static std::array<SubmitRec, kSubmitRing> g_submit_ring{};

static SubmitRec& SubmitSlot(u64 tick) {
    SubmitRec& r = g_submit_ring[tick % kSubmitRing];
    if (r.tick.load(std::memory_order_relaxed) != tick) {
        r.tick.store(tick, std::memory_order_relaxed);
        r.draws.store(0, std::memory_order_relaxed);
        r.dispatches.store(0, std::memory_order_relaxed);
        for (auto& h : r.hashes) {
            h.store(0, std::memory_order_relaxed);
        }
        r.gx.store(0, std::memory_order_relaxed);
        r.gy.store(0, std::memory_order_relaxed);
        r.gz.store(0, std::memory_order_relaxed);
        r.first_ms.store(Postmortem::NowMs(), std::memory_order_relaxed);
    }
    return r;
}

namespace AddrBind {
extern std::atomic<u64> g_draw_tick; // vk_platform.cpp, inside namespace Vulkan
void DumpHandle(u64 handle);         // vk_platform.cpp: VA history of one object handle
} // namespace AddrBind

// [BINDLOG] color target 0 of the draw being recorded (VkImage handle), set in BeginRendering.
static u64 g_rt_image[8]{};

// [BINDLOG] SHADPS4_BINDLOG=<hex pgm hash>[,...]: ring of the image bindings those shaders
// actually received (VkImage/view handles + the cache slot + the guest range), stamped with the
// scheduler tick. Dumped at DEVFAULT: matches the faulting VkImage to a draw and its slot.
namespace BindLog {
struct Entry {
    u64 tick;
    u64 pgm;
    u64 tsharp;
    u64 guest;
    u64 guest_size;
    u64 img;
    u64 view;
    u64 rt0; // color target 0 VkImage of the same draw
    u32 slot;
    u32 w;
    u32 h;
    u32 idx;
};
static constexpr size_t kSize = 128;
static Entry ring[kSize];
static size_t head = 0;
static u64 total = 0;

static bool Wants(u64 hash) {
    static const std::vector<u64> wanted = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_BINDLOG");
        const char* cursor = e ? e : "";
        while (*cursor != '\0') {
            char* end = nullptr;
            const u64 v = std::strtoull(cursor, &end, 16);
            if (end == cursor) {
                break;
            }
            if (v != 0) {
                out.push_back(v);
            }
            cursor = (*end == ',') ? end + 1 : end;
        }
        return out;
    }();
    return !wanted.empty() && std::find(wanted.begin(), wanted.end(), hash) != wanted.end();
}

static void Note(const Entry& e) {
    ring[head] = e;
    head = (head + 1) % kSize;
    ++total;
}

void Dump() {
    const size_t n = std::min<u64>(total, kSize);
    LOG_CRITICAL(Render_Vulkan, "BINDLOG: total={} showing last {}", total, n);
    for (size_t k = 0; k < n; k++) {
        const auto& e = ring[(head + kSize - n + k) % kSize];
        LOG_CRITICAL(Render_Vulkan,
                     "BINDLOG[{}]: tick={} pgm={:#x} idx={} T#={:#x} slot={} img={:#x} view={:#x} "
                     "rt0={:#x} guest={:#x}+{:#x} {}x{}",
                     k, e.tick, e.pgm, e.idx, e.tsharp, e.slot, e.img, e.view, e.rt0, e.guest,
                     e.guest_size, e.w, e.h);
    }
    // VA history of every distinct image/target handle in the last entries: is the bound
    // image (or the target) the object whose memory the fault address falls into?
    u64 seen[12]{};
    size_t nseen = 0;
    for (size_t k = n; k > 0 && nseen < 12; k--) {
        const auto& e = ring[(head + kSize - k) % kSize];
        for (const u64 hnd : {e.img, e.rt0}) {
            if (hnd == 0) {
                continue;
            }
            bool dup = false;
            for (size_t j = 0; j < nseen; j++) {
                dup |= seen[j] == hnd;
            }
            if (!dup && nseen < 12) {
                seen[nseen++] = hnd;
            }
        }
    }
    for (size_t j = 0; j < nseen; j++) {
        AddrBind::DumpHandle(seen[j]);
    }
}
} // namespace BindLog

// [SEQLOG] SHADPS4_SEQLOG=<hex worker hash>:<delay_s>:<count>: once the process is delay_s
// seconds old, log the next `count` dispatches of that worker and, in order, every draw and
// every other dispatch recorded between them, with the guest ranges each one touches. Answers
// "which draw consumes which worker output, and how many worker dispatches sit back-to-back"
// without a full GPUPROF. Parser thread only.
namespace SeqLog {
struct Cfg {
    u64 hash = 0;
    u32 delay_s = 0;
    u32 count = 0;
};
static const Cfg& Get() {
    static const Cfg cfg = [] {
        Cfg c;
        const char* e = std::getenv("SHADPS4_SEQLOG");
        if (!e || !*e) {
            return c;
        }
        char* end = nullptr;
        c.hash = std::strtoull(e, &end, 16);
        if (end && *end == ':') {
            c.delay_s = static_cast<u32>(std::strtoul(end + 1, &end, 10));
        }
        if (end && *end == ':') {
            c.count = static_cast<u32>(std::strtoul(end + 1, &end, 10));
        }
        if (c.hash != 0 && c.count != 0) {
            LOG_CRITICAL(Render_Vulkan, "SEQLOG: worker {:#x}, window after {} s, {} dispatches",
                         c.hash, c.delay_s, c.count);
        }
        return c;
    }();
    return cfg;
}
static u32 g_seen = 0;
static bool g_active = false;
static const auto g_t0 = std::chrono::steady_clock::now();
static bool Active() {
    return g_active;
}
static bool Wants(u64 hash) {
    const Cfg& c = Get();
    return c.hash != 0 && c.count != 0 && hash == c.hash;
}
// Called for every dispatch of the watched worker; true when this one falls in the window.
static bool OnWorker() {
    const Cfg& c = Get();
    if (g_seen >= c.count) {
        if (g_active) {
            g_active = false;
            LOG_CRITICAL(Render_Vulkan, "SEQLOG: window closed after {} worker dispatches", g_seen);
        }
        return false;
    }
    if (!g_active) {
        const auto el = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - g_t0)
                            .count();
        if (el < static_cast<long long>(c.delay_s)) {
            return false;
        }
        g_active = true;
        LOG_CRITICAL(Render_Vulkan, "SEQLOG: window open at {} s", el);
    }
    ++g_seen;
    return true;
}
} // namespace SeqLog

static void RecordDrawTick(u64 tick) {
    // [ATTRIB] the address-bind ring stamps driver bind/unbind events with this tick.
    AddrBind::g_draw_tick.store(tick, std::memory_order_relaxed);
    SubmitSlot(tick).draws.fetch_add(1, std::memory_order_relaxed);
}

static void RecordDispatchTick(u64 tick, u64 hash, u32 gx, u32 gy, u32 gz) {
    SubmitRec& r = SubmitSlot(tick);
    const u32 n = r.dispatches.fetch_add(1, std::memory_order_relaxed);
    if (n < 12) {
        r.hashes[n].store(hash, std::memory_order_relaxed);
    }
    if (n == 0) {
        r.gx.store(gx, std::memory_order_relaxed);
        r.gy.store(gy, std::memory_order_relaxed);
        r.gz.store(gz, std::memory_order_relaxed);
    }
}

// Called by the stall watchdog. Prints the still-executing submits and flags any compute that
// has been in flight beyond `slow_ms` - the signature of a shader that never exits its loop.
void DumpStalledSubmits(u64 gpu_tick, u64 waiting_tick, u64 slow_ms) {
    const s64 now = Postmortem::NowMs();
    std::string line =
        fmt::format("GPUSTALL-SUBMIT: waiting_tick={} current={} |", waiting_tick, gpu_tick);
    for (u64 t = gpu_tick + 1; t <= waiting_tick && t <= gpu_tick + kSubmitRing; ++t) {
        const SubmitRec& r = g_submit_ring[t % kSubmitRing];
        if (r.tick.load(std::memory_order_relaxed) != t) {
            line += fmt::format(" tick{}: <not recorded> |", t);
            continue;
        }
        const u32 nd = r.dispatches.load(std::memory_order_relaxed);
        line += fmt::format(" tick{}: draws={} disp={}", t, r.draws.load(std::memory_order_relaxed),
                            nd);
        for (u32 i = 0; i < std::min<u32>(nd, 12); ++i) {
            line += fmt::format(" pgm={:#x}", r.hashes[i].load(std::memory_order_relaxed));
        }
        if (nd != 0) {
            line += fmt::format(" groups=({},{},{})", r.gx.load(std::memory_order_relaxed),
                                r.gy.load(std::memory_order_relaxed),
                                r.gz.load(std::memory_order_relaxed));
        }
        line += " |";
    }
    LOG_CRITICAL(Render_Vulkan, "{}", line);

    for (u64 t = gpu_tick + 1; t <= waiting_tick && t <= gpu_tick + kSubmitRing; ++t) {
        const SubmitRec& r = g_submit_ring[t % kSubmitRing];
        if (r.tick.load(std::memory_order_relaxed) != t) {
            continue;
        }
        const u32 nd = r.dispatches.load(std::memory_order_relaxed);
        const s64 t0 = r.first_ms.load(std::memory_order_relaxed);
        const s64 age = t0 ? now - t0 : -1;
        if (nd == 0 || age < static_cast<s64>(slow_ms)) {
            continue;
        }
        for (u32 i = 0; i < std::min<u32>(nd, 12); ++i) {
            LOG_CRITICAL(
                Render_Vulkan, "GPUSTALL-SHADER: pgm={:#x} groups=({},{},{}) inflight={}ms tick={}",
                r.hashes[i].load(std::memory_order_relaxed), r.gx.load(std::memory_order_relaxed),
                r.gy.load(std::memory_order_relaxed), r.gz.load(std::memory_order_relaxed), age, t);
        }
    }
}

void DumpRecentDispatches() {
    const u32 total = g_recent_disp_idx.load(std::memory_order_relaxed);
    const u32 n = std::min<u32>(total, static_cast<u32>(g_recent_disp.size()));
    LOG_WARNING(Render_Vulkan, "STUCKDISP: last {} dispatches (of {} total), newest last:", n,
                total);
    for (u32 i = 0; i < n; ++i) {
        const auto& d = g_recent_disp[(total - n + i) % static_cast<u32>(g_recent_disp.size())];
        LOG_WARNING(Render_Vulkan, "STUCKDISP: frame={} cs={:#x} groups={}x{}x{}", d.frame, d.hash,
                    d.gx, d.gy, d.gz);
    }
}

static u32 DumpFrameTarget() {
    static const char* e = std::getenv("SHADPS4_DUMPFRAME");
    static const u32 v = e ? (u32)std::strtoul(e, nullptr, 10) : 0;
    return v;
}

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    // [CSBATCH] SHADPS4_CSBATCH=8f067540: compile the host skinning kernel, create its table
    // ring and install the pre-submit hook. Env unset -> nothing here runs (Wanted() is one
    // static-list test) and every hook below is one empty() test.
    if (SkinBatch::Wanted(SkinBatch::kHash)) {
        const bool stream_bda =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream).HasDeviceAddress();
        skin_batch.Create(instance, scheduler, stream_bda);
        if (skin_batch.Ready()) {
            scheduler.SetPreSubmitHook([this] { FlushSkinBatch(SkinBatch::Reason::Hook); });
        }
    }
    // GPU fault addresses come from guest shaders and may point at unmapped
    // memory (e.g. a null job pointer read through an unbounded V#). Such
    // pages must stay uncached and read as zero instead of being cached.
    buffer_cache.SetFaultRegionValidator(
        [this](VAddr addr, u64 size) { return addr != 0 && IsMapped(addr, size); });
    if (liverpool) {
        // Lets the no-flip frame-pacing hold (SHADPS4_PACERETIRE) ask whether the
        // GPU has actually executed everything submitted so far.
        liverpool->SetGpuRetireQuery([this] { return scheduler.AllSubmittedRetired(); });
    }
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::CpSync() {
    FlushSkinBatch(SkinBatch::Reason::Sync); // [CSBATCH]
    {
        Parserprof::PhaseProbe p{Parserprof::PhSched};
        scheduler.EndRendering();
    }
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

// [GPUPROF] SHADPS4_GPUPROF=1: GPU time per draw/dispatch via timestamp queries, summed by shader
// hash and reported every 300 frames. Answers "where do the 50+ ms of GPU time per frame go" -
// DRAWPROF/PARSERPROF only see the CPU side. Queries live in a ring; a pair is read back and
// host-reset once the tick it was recorded in has retired (VK 1.2 hostQueryReset is enabled).
namespace {
struct GpuProfEvent {
    u64 tick;
    u32 query;
    u64 h1;
    u64 h2;
    u8 kind; // 0 draw, 1 draw-indirect, 2 dispatch, 3 dispatch-indirect
};
struct GpuProfAcc {
    u64 ns{};
    u64 n{};
    u8 kind{};
    u64 h2{};
};
struct GpuProf {
    bool checked{false};
    bool enabled{false};
    vk::Device dev{};
    vk::QueryPool pool{};
    double period_ns{1.0};
    static constexpr u32 kCap = 32768; // queries (pairs use 2)
    u32 next{0};
    std::deque<GpuProfEvent> pending;
    std::unordered_map<u64, GpuProfAcc> acc;
    u64 total_ns{0};
    u64 events{0};
    u64 dropped{0};
    u64 last_report_frame{0};
    u64 last_frame_seen{0};
    u64 frames_in_window{0};
    u64 prev_end{0}; // end timestamp of the previous collected event (submission order)
    u64 gap_ns{0};   // GPU idle between consecutive events
    u64 gap_n{0};
    u64 rp_last{0}; // Parserprof::g_rp_ends at the previous report
    // [GPUPROF-GAPS] where the idle time sits: inside one submit (barriers/drains between
    // events) or between submits (the GPU ran dry waiting for the CPU), split by the kind of
    // the event that preceded the gap, plus a size histogram (<10us, 10-50, 50-200, >200us).
    u64 prev_tick{0};
    u8 prev_kind{0};
    u64 gap_in_ns{0}, gap_in_n{0};
    u64 gap_cross_ns{0}, gap_cross_n{0};
    std::array<u64, 8> gap_after_ns{}; // by kind (0..4)
    std::array<u64, 4> gap_hist_n{};
    std::array<u64, 4> gap_hist_ns{};
};
GpuProf g_gpuprof;
std::atomic<u64> g_gpuprof_rp_breaks{
    0}; // render passes ended by compute/other work (see Scheduler)
} // namespace
// [CSBARRIER] set by the command processor on ACQUIRE_MEM (the game's own sync point); the next
// dispatch keeps its buffer barriers even in lax mode. Same thread (GpuCommandProcessor).
bool g_cs_sync_pending = false;
// [CSBARRIER] which of the game's packets count as its sync point (SHADPS4_CSBARRIER_SYNCS,
// bitmask, default 1): 1 = ACQUIRE_MEM, 2 = WAIT_REG_MEM, 4 = EVENT_WRITE (CS_PARTIAL_FLUSH and
// friends). Read once at load.
u32 g_cs_sync_mask = [] {
    const char* e = std::getenv("SHADPS4_CSBARRIER_SYNCS");
    return e && *e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 1u;
}();
namespace {

bool GpuProfInit(const Instance& instance) {
    GpuProf& g = g_gpuprof;
    if (g.checked) {
        return g.enabled;
    }
    g.checked = true;
    const char* e = std::getenv("SHADPS4_GPUPROF");
    if (e == nullptr || e[0] != '1') {
        return false;
    }
    g.dev = instance.GetDevice();
    const auto props = instance.GetPhysicalDevice().getProperties();
    g.period_ns = props.limits.timestampPeriod;
    const vk::QueryPoolCreateInfo ci{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = GpuProf::kCap,
    };
    auto [r, pool] = g.dev.createQueryPool(ci);
    if (r != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "GPUPROF: query pool creation failed ({})", vk::to_string(r));
        return false;
    }
    g.pool = pool;
    g.dev.resetQueryPool(g.pool, 0, GpuProf::kCap);
    g.enabled = true;
    LOG_CRITICAL(Render_Vulkan, "GPUPROF: armed, {} queries, timestampPeriod={} ns", GpuProf::kCap,
                 g.period_ns);
    return true;
}

// [GPUPROF-ONLY] SHADPS4_GPUPROF_ONLY=<hex hash>[,...]: timestamp ONLY these shaders. The full
// instrument writes a pair around every draw and dispatch (~1450 events per frame) and twice
// took the host down with it (runs 146, 195); one shader is ~217 pairs per frame.
static bool GpuProfWants(u64 h1, u64 h2) {
    static const std::vector<u64> only = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_GPUPROF_ONLY");
        const char* cursor = e ? e : "";
        while (*cursor != '\0') {
            char* end = nullptr;
            const u64 v = std::strtoull(cursor, &end, 16);
            if (end == cursor) {
                break;
            }
            if (v != 0) {
                out.push_back(v);
            }
            cursor = (*end == ',') ? end + 1 : end;
        }
        return out;
    }();
    if (only.empty()) {
        return true;
    }
    return std::find(only.begin(), only.end(), h1) != only.end() ||
           std::find(only.begin(), only.end(), h2) != only.end();
}

// Returns the query index of the pair or UINT32_MAX when the ring is full.
u32 GpuProfBegin(vk::CommandBuffer cmdbuf, u64 tick, u64 h1, u64 h2, u8 kind) {
    GpuProf& g = g_gpuprof;
    if (!g.enabled) {
        return UINT32_MAX;
    }
    if (!GpuProfWants(h1, h2)) {
        return UINT32_MAX;
    }
    if (g.pending.size() * 2 + 2 > GpuProf::kCap) {
        ++g.dropped;
        return UINT32_MAX;
    }
    const u32 q = g.next;
    g.next = (g.next + 2) % GpuProf::kCap;
    cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, q);
    g.pending.push_back(GpuProfEvent{tick, q, h1, h2, kind});
    return q;
}

void GpuProfEnd(vk::CommandBuffer cmdbuf, u32 q) {
    if (q == UINT32_MAX) {
        return;
    }
    cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gpuprof.pool, q + 1);
}

const char* GpuProfKindName(u8 k) {
    switch (k) {
    case 0:
        return "draw";
    case 1:
        return "drawind";
    case 2:
        return "disp";
    case 4:
        return "prep-bind"; // [GPUPROF-PREP] Rasterizer::BindResources: uploads, page-table writes
    case 5:
        return "prep-rp"; // EndRendering before a dispatch
    case 6:
        return "prep-bar"; // Pipeline::BindResources: buffer barriers
    default:
        return "dispind";
    }
}

// [GPUPROF-PREP] Everything the emulator records between two dispatches (uploads of CPU-dirty
// pages, buffer barriers, render-pass ends) shows up in GPUPROF-GAPS as idle "after disp".
// A prep pair brackets that recorded work so its GPU time gets its own row (kind 4, h1 = the
// dispatch's shader). The event is pushed at the END with the tick current then: a flush inside
// the bracket moves the end timestamp to a later tick, and the pair must only be read once
// that tick retired.
u32 GpuProfPrepBegin(vk::CommandBuffer cmdbuf) {
    GpuProf& g = g_gpuprof;
    if (!g.enabled) {
        return UINT32_MAX;
    }
    if (g.pending.size() * 2 + 2 > GpuProf::kCap) {
        ++g.dropped;
        return UINT32_MAX;
    }
    const u32 q = g.next;
    g.next = (g.next + 2) % GpuProf::kCap;
    cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, q);
    return q;
}

void GpuProfPrepEnd(vk::CommandBuffer cmdbuf, u32 q, u64 tick, u64 h1, u8 kind = 4) {
    if (q == UINT32_MAX) {
        return;
    }
    cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gpuprof.pool, q + 1);
    g_gpuprof.pending.push_back(GpuProfEvent{tick, q, h1, 0, kind});
}

void GpuProfCollect(Scheduler& scheduler) {
    GpuProf& g = g_gpuprof;
    if (!g.enabled) {
        return;
    }
    while (!g.pending.empty()) {
        const GpuProfEvent& ev = g.pending.front();
        if (!scheduler.IsFree(ev.tick)) {
            break;
        }
        std::array<u64, 2> ts{};
        const auto r = g.dev.getQueryPoolResults(g.pool, ev.query, 2, sizeof(ts), ts.data(),
                                                 sizeof(u64), vk::QueryResultFlagBits::e64);
        g.dev.resetQueryPool(g.pool, ev.query, 2);
        if (r == vk::Result::eSuccess && ts[1] >= ts[0]) {
            if (g.prev_end != 0 && ts[0] > g.prev_end) {
                const u64 gap =
                    static_cast<u64>(static_cast<double>(ts[0] - g.prev_end) * g.period_ns);
                g.gap_ns += gap;
                ++g.gap_n;
                if (ev.tick == g.prev_tick) {
                    g.gap_in_ns += gap;
                    ++g.gap_in_n;
                } else {
                    g.gap_cross_ns += gap;
                    ++g.gap_cross_n;
                }
                g.gap_after_ns[g.prev_kind & 7] += gap;
                const u32 bucket = gap < 10'000 ? 0 : gap < 50'000 ? 1 : gap < 200'000 ? 2 : 3;
                g.gap_hist_n[bucket] += 1;
                g.gap_hist_ns[bucket] += gap;
            }
            g.prev_end = ts[1];
            g.prev_tick = ev.tick;
            g.prev_kind = ev.kind;
            const u64 ns = static_cast<u64>(static_cast<double>(ts[1] - ts[0]) * g.period_ns);
            const u64 key = ev.h1 ^ (ev.h2 << 1) ^ (static_cast<u64>(ev.kind) << 60);
            GpuProfAcc& a = g.acc[key];
            a.ns += ns;
            a.n += 1;
            a.kind = ev.kind;
            a.h2 = ev.h2;
            if (a.n == 1) {
                a.h2 = ev.h2;
            }
            g.total_ns += ns;
            ++g.events;
        }
        g.pending.pop_front();
    }
    const u64 frame = g_dbg_frame_no.load(std::memory_order_relaxed);
    if (frame != g.last_frame_seen) {
        g.last_frame_seen = frame;
        ++g.frames_in_window;
    }
    if (g.frames_in_window >= 300) {
        const double fr = static_cast<double>(g.frames_in_window);
        std::vector<std::pair<u64, GpuProfAcc>> rows(g.acc.begin(), g.acc.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
        std::array<u64, 8> by_kind{};
        for (const auto& [k, a] : rows) {
            by_kind[a.kind & 7] += a.ns;
        }
        const u64 rp_now = Parserprof::g_rp_ends.load(std::memory_order_relaxed);
        const u64 rp_breaks = rp_now >= g.rp_last ? rp_now - g.rp_last : rp_now;
        g.rp_last = rp_now;
        LOG_CRITICAL(Render_Vulkan,
                     "GPUPROF: frames={} gpu_ms/frame={:.2f} events/frame={:.0f} dropped={} | "
                     "draw={:.2f} drawind={:.2f} disp={:.2f} dispind={:.2f} prep-bind={:.2f} "
                     "prep-rp={:.2f} prep-bar={:.2f} ms/frame | "
                     "gaps={:.2f} ms/frame ({:.0f}/frame) rp_breaks/frame={:.1f}",
                     g.frames_in_window, g.total_ns / 1e6 / fr, g.events / fr, g.dropped,
                     by_kind[0] / 1e6 / fr, by_kind[1] / 1e6 / fr, by_kind[2] / 1e6 / fr,
                     by_kind[3] / 1e6 / fr, by_kind[4] / 1e6 / fr, by_kind[5] / 1e6 / fr,
                     by_kind[6] / 1e6 / fr, g.gap_ns / 1e6 / fr, g.gap_n / fr, rp_breaks / fr);
        LOG_CRITICAL(Render_Vulkan,
                     "GPUPROF-GAPS: in-submit={:.2f} ms/frame ({:.0f}/frame) cross-submit={:.2f} "
                     "ms/frame ({:.1f}/frame) | after draw={:.2f} drawind={:.2f} disp={:.2f} "
                     "dispind={:.2f} prep={:.2f} ms/frame | hist <10us {:.0f}/{:.2f}ms  10-50us "
                     "{:.0f}/{:.2f}ms  "
                     "50-200us {:.0f}/{:.2f}ms  >200us {:.1f}/{:.2f}ms (per frame)",
                     g.gap_in_ns / 1e6 / fr, g.gap_in_n / fr, g.gap_cross_ns / 1e6 / fr,
                     g.gap_cross_n / fr, g.gap_after_ns[0] / 1e6 / fr, g.gap_after_ns[1] / 1e6 / fr,
                     g.gap_after_ns[2] / 1e6 / fr, g.gap_after_ns[3] / 1e6 / fr,
                     g.gap_after_ns[4] / 1e6 / fr, g.gap_hist_n[0] / fr,
                     g.gap_hist_ns[0] / 1e6 / fr, g.gap_hist_n[1] / fr, g.gap_hist_ns[1] / 1e6 / fr,
                     g.gap_hist_n[2] / fr, g.gap_hist_ns[2] / 1e6 / fr, g.gap_hist_n[3] / fr,
                     g.gap_hist_ns[3] / 1e6 / fr);
        g.gap_in_ns = g.gap_in_n = g.gap_cross_ns = g.gap_cross_n = 0;
        g.gap_after_ns.fill(0);
        g.gap_hist_n.fill(0);
        g.gap_hist_ns.fill(0);
        u32 shown = 0;
        for (const auto& [k, a] : rows) {
            if (shown++ >= 14) {
                break;
            }
            const u64 h1 = k ^ (a.h2 << 1) ^ (static_cast<u64>(a.kind) << 60);
            LOG_CRITICAL(
                Render_Vulkan,
                "GPUPROF-TOP: {} h1={:#x} h2={:#x} ms/frame={:.2f} n/frame={:.1f} avg_us={:.0f}",
                GpuProfKindName(a.kind), h1, a.h2, a.ns / 1e6 / fr, a.n / fr,
                a.n ? (a.ns / 1e3) / a.n : 0.0);
        }
        g.acc.clear();
        g.total_ns = 0;
        g.events = 0;
        g.dropped = 0;
        g.gap_ns = 0;
        g.gap_n = 0;
        g.frames_in_window = 0;
    }
}
} // namespace

// [FRAMECHAIN] scheduler.PopPendingOperations() runs on every draw/dispatch (~1400/frame in
// run 149, one vkGetSemaphoreCounterValue each): timed here to size measure #5. Off => the
// plain call, nothing else.
static void PopPendingTimed(Scheduler& scheduler) {
    if (!Parserprof::Enabled()) {
        scheduler.PopPendingOperations();
        return;
    }
    const s64 t0 = Parserprof::NowNs();
    scheduler.PopPendingOperations();
    const s64 t1 = Parserprof::NowNs();
    Parserprof::g_fc_pop_ns.fetch_add(t1 > t0 ? static_cast<u64>(t1 - t0) : 0,
                                      std::memory_order_relaxed);
    Parserprof::g_fc_pop_n.fetch_add(1, std::memory_order_relaxed);
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    // [SLOWPKT] vk phase = whole draw minus the nested compile/tex/buf/desc probes.
    Parserprof::PhaseProbe slow_probe{Parserprof::PhVk};
    // [CSBATCH] first: before FilterDraw (EliminateFastClear/Resolve/DepthStencilCopy record
    // commands) and before RecordDrawTick - the flush may chunk-submit and end the tick.
    FlushSkinBatch(SkinBatch::Reason::Draw);
    RecordDrawTick(scheduler.CurrentTick());
    ++g_dbg_draws;
    RENDERER_TRACE;

    PopPendingTimed(scheduler);

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    // [debug] эксперимент: пропуск дро с заданным хэшем пиксельника
    // (SHADPS4_SKIPFS=b1032a3f) — проверка, что именно оно убивает device.
    {
        static const u64 skip_fs = [] {
            const char* e = std::getenv("SHADPS4_SKIPFS");
            return e ? std::strtoull(e, nullptr, 16) : 0ull;
        }();
        if (skip_fs != 0 &&
            pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash == skip_fs) {
            static u32 s_skipped = 0;
            if ((++s_skipped % 500) == 1) {
                LOG_WARNING(Render_Vulkan, "SKIPFS: skipped {} draws of fs={:#x}", s_skipped,
                            skip_fs);
            }
            return;
        }
    }

    // [SEQLOG] one line per draw inside the window: VS/FS hashes, index count and the guest
    // range of every buffer the vertex stage binds ("w" = written by the shader).
    if (SeqLog::Active()) {
        const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
        const auto& fs_info = pipeline->GetStage(Shader::LogicalStage::Fragment);
        std::string bufs;
        for (const auto& desc : vs_info.buffers) {
            const auto sharp = desc.GetSharp(vs_info);
            bufs += fmt::format(" {:#x}+{:#x}{}", u64(sharp.base_address), u64(sharp.GetSize()),
                                desc.is_written ? "w" : "");
        }
        LOG_CRITICAL(Render_Vulkan, "SEQ W tick={} vs={:#x} fs={:#x} idx={} n={} bufs:{}",
                     scheduler.CurrentTick(), vs_info.pgm_hash, fs_info.pgm_hash,
                     is_indexed ? 1 : 0, regs.num_indices, bufs);
    }

    PrepareRenderState(pipeline);
    const auto tt0 = std::chrono::steady_clock::now();
    // [debug] журнал биндингов дро-виновника HUD-фриза (fs=0xb1032a3f):
    // все текстуры этого дро проходят через FindImage под тегом.
    const bool is_blur_draw =
        pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash == 0xb1032a3f;
    const bool watch_blur_bind = is_blur_draw && DbgLogEnabled();
    // Log the textures this draw samples for the first handful of draws: the shader is a
    // plain downsample, so a wrong result can only come from its input.
    static u32 s_blur_binds = 0;
    const bool log_blur_inputs = is_blur_draw && s_blur_binds < 14;
    if (log_blur_inputs) {
        ++s_blur_binds;
        texture_cache.SetBindLogTag(0xb1032a3f);
    } else if (watch_blur_bind) {
        texture_cache.SetBindLogTag(0xb1032a3f);
    }
    const bool bind_ok = BindResources(pipeline);
    if (log_blur_inputs || watch_blur_bind) {
        texture_cache.SetBindLogTag(0);
    }
    if (!bind_ok) {
        return;
    }
    const auto tt1 = std::chrono::steady_clock::now();
    const auto state = [&] {
        Parserprof::PhaseProbe p{Parserprof::PhRp};
        return BeginRendering(pipeline);
    }();
    {
        static const bool s_drawprof = std::getenv("SHADPS4_DRAWPROF") != nullptr;
        if (s_drawprof) {
            const auto tt2 = std::chrono::steady_clock::now();
            const auto бинд =
                std::chrono::duration_cast<std::chrono::microseconds>(tt1 - tt0).count();
            const auto рендер =
                std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1).count();
            if (бинд + рендер > 3000) {
                LOG_WARNING(Render_Vulkan, "DRAWPROF: bind={}us begin_rendering={}us", бинд,
                            рендер);
            }
        }
    }

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }

    // [debug] журнал биндингов дро-виновника HUD-фриза (fs=0xb1032a3f):
    // FindImage печатает адреса всех текстур, которые получит это дро.
    const bool watch_blur_draw =
        DbgLogEnabled() &&
        pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash == 0xb1032a3f;
    if (watch_blur_draw) {
        texture_cache.SetBindLogTag(0xb1032a3f);
    }
    {
        Parserprof::PhaseProbe p{Parserprof::PhBarr};
        Parserprof::t_pkt_barriers += static_cast<u32>(buffer_barriers.size());
        Parserprof::t_pkt_writes += static_cast<u32>(set_writes.size());
        Parserprof::g_bar_calls.fetch_add(1, std::memory_order_relaxed);
        Parserprof::g_bar_buffers.fetch_add(buffer_barriers.size(), std::memory_order_relaxed);
        Parserprof::g_desc_writes.fetch_add(set_writes.size(), std::memory_order_relaxed);
        CountRedundantDescriptors(set_writes);
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
    }
    if (watch_blur_draw) {
        texture_cache.SetBindLogTag(0);
    }
    {
        Parserprof::PhaseProbe p{Parserprof::PhBind};
        UpdateDynamicState(pipeline, is_indexed);
    }
    {
        Parserprof::PhaseProbe p{Parserprof::PhRp};
        scheduler.BeginRendering(state);
    }

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    // [debug] вскрытие кадра: SHADPS4_DUMPFRAME=N
    {
        const u32 цель = DumpFrameTarget();
        const u32 кадр = g_dbg_frame_no.load();
        static bool s_once = false;
        if (!s_once) {
            s_once = true;
            LOG_WARNING(Render_Vulkan, "DUMPDRAW armed: target={} frame_now={}", цель, кадр);
        }
        if ((цель && кадр >= цель && кадр < цель + 2) || (кадр % 500) == 250) {
            const auto& ps_info = pipeline->GetStage(Shader::LogicalStage::Fragment);
            const auto& cb0 = regs.color_buffers[0];
            const auto& bc0 = regs.blend_control[0];
            LOG_WARNING(Render_Vulkan,
                        "DUMPDRAW: vs={:#x} ps={:#x} indices={} indexed={} rt0={:#x} "
                        "fmt={} blend_en={} src={} dst={} db={:#x} depth_en={}",
                        vs_info.pgm_hash, ps_info.pgm_hash, regs.num_indices, is_indexed,
                        cb0.Address(), (u32)cb0.GetDataFmt(), (u32)bc0.enable,
                        (u32)bc0.color_src_factor, (u32)bc0.color_dst_factor,
                        regs.depth_buffer.DepthAddress(), (u32)regs.depth_control.depth_enable);
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    {
        Parserprof::PhaseProbe p{Parserprof::PhBind};
        ++Parserprof::t_pkt_binds;
        Parserprof::g_bind_calls.fetch_add(1, std::memory_order_relaxed);
        if (Vulkan::ShouldBindPipeline(pipeline->Handle())) {
            Parserprof::g_bind_real.fetch_add(1, std::memory_order_relaxed);
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
        }
    }

    if (instance.HasNvCheckpoints()) {
        // Top bit marks draws; low 32 bits carry the fragment program hash; bits 32..62 the
        // scheduler tick the draw is recorded in ([TICKMARK] 06.09: the 161 device fault named
        // an image unbound at tick 27663 - the fault dump needs the draw's tick next to it).
        const u64 marker =
            (1ULL << 63) | ((scheduler.CurrentTick() & 0x7fffffffULL) << 32) |
            (pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash & 0xffffffffULL);
        cmdbuf.setCheckpointNV(reinterpret_cast<const void*>(marker));
    }
    if (is_blur_draw) {
        // Skipping this one draw fixes both the black world and the GPU fault, which a
        // plain bloom blur could not do - so record exactly what it renders into.
        static u32 s_reported = 0;
        if (s_reported < 12) {
            ++s_reported;
            const auto& cb0 = regs.color_buffers[0];
            const auto& vp = regs.viewports[0];
            const auto& sc = regs.screen_scissor;
            LOG_CRITICAL(Render_Vulkan,
                         "BLURTARGET: rt0={:#x} fmt={} slice={} pitch={} viewport=({},{} {}x{}) "
                         "scissor=({},{} {}x{}) indices={}",
                         cb0.Address(), static_cast<u32>(cb0.GetDataFmt()), cb0.slice.tile_max,
                         cb0.Pitch(), vp.xoffset, vp.yoffset, vp.xscale * 2.0f, vp.yscale * 2.0f,
                         sc.top_left_x, sc.top_left_y, sc.GetWidth(), sc.GetHeight(),
                         regs.num_indices);
        }
    }
    if (watch_blur_bind && DbgLogEnabled()) {
        // [debug] отметка запуска подозреваемого дро в потоке сабмитов
        LOG_WARNING(Render_Vulkan, "BLURDRAW: tick={} indices={}", scheduler.CurrentTick(),
                    regs.num_indices);
    }
    const u32 gpuprof_q =
        GpuProfInit(instance)
            ? GpuProfBegin(cmdbuf, scheduler.CurrentTick(),
                           pipeline->GetStage(Shader::LogicalStage::Vertex).pgm_hash,
                           pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash, 0)
            : UINT32_MAX;
    if (is_indexed) {
        Parserprof::PhaseProbe p{Parserprof::PhDraw};
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        Parserprof::PhaseProbe p{Parserprof::PhDraw};
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    GpuProfEnd(cmdbuf, gpuprof_q);
    DebugState.IncDrawCall();
    ChunkHeavyFrame();

    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    // [SLOWPKT] vk phase (indirect draws).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhVk};
    ++g_dbg_draws;
    RENDERER_TRACE;

    PopPendingTimed(scheduler);
    FlushSkinBatch(SkinBatch::Reason::DrawIndirect); // [CSBATCH]

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    // [debug] журнал биндингов дро-виновника HUD-фриза (fs=0xb1032a3f):
    // FindImage печатает адреса всех текстур, которые получит это дро.
    const bool watch_blur_draw =
        DbgLogEnabled() &&
        pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash == 0xb1032a3f;
    if (watch_blur_draw) {
        texture_cache.SetBindLogTag(0xb1032a3f);
    }
    {
        Parserprof::PhaseProbe p{Parserprof::PhBarr};
        Parserprof::t_pkt_barriers += static_cast<u32>(buffer_barriers.size());
        Parserprof::t_pkt_writes += static_cast<u32>(set_writes.size());
        Parserprof::g_bar_calls.fetch_add(1, std::memory_order_relaxed);
        Parserprof::g_bar_buffers.fetch_add(buffer_barriers.size(), std::memory_order_relaxed);
        Parserprof::g_desc_writes.fetch_add(set_writes.size(), std::memory_order_relaxed);
        CountRedundantDescriptors(set_writes);
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
    }
    if (watch_blur_draw) {
        texture_cache.SetBindLogTag(0);
    }
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    {
        Parserprof::PhaseProbe p{Parserprof::PhBind};
        ++Parserprof::t_pkt_binds;
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
    }

    if (instance.HasNvCheckpoints()) {
        // Bit 62 marks indirect draws, bit 61 the count-from-buffer variant.
        const u64 marker = (1ULL << 62) | (count_address != 0 ? (1ULL << 61) : 0) |
                           pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash;
        cmdbuf.setCheckpointNV(reinterpret_cast<const void*>(marker));
    }
    const u32 gpuprof_q =
        GpuProfInit(instance)
            ? GpuProfBegin(cmdbuf, scheduler.CurrentTick(),
                           pipeline->GetStage(Shader::LogicalStage::Vertex).pgm_hash,
                           pipeline->GetStage(Shader::LogicalStage::Fragment).pgm_hash, 1)
            : UINT32_MAX;
    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
        ChunkHeavyFrame();
        ChunkHeavyFrame();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
        ChunkHeavyFrame();
        ChunkHeavyFrame();
    }
    GpuProfEnd(cmdbuf, gpuprof_q);

    ResetBindings();
}

// Split monster frames into GPU-sized bites. A level-load frame can carry
// tens of thousands of draws; submitted as one command buffer it exceeds the
// Windows GPU watchdog budget and the driver resets mid-load (observed as
// "Device lost during waiting for a frame"). Console GPUs have no watchdog,
// so the work has to be chunked to emulate that freedom.
// [CHUNK] SHADPS4_CHUNK_DRAWS=<n>: submit every n draws+dispatches (default 2048, draws only).
// A guest submit of ~1000 draws + ~700 dispatches is parsed for ~8 ms before the GPU sees any
// of it (GPUPROF-GAPS: 11.8 ms/frame of cross-submit idle at 28 FPS). Smaller chunks let the
// GPU start on the first part while the CPU still parses the rest.
static u32 ChunkDrawsPerFlush() {
    static const u32 v = [] {
        const char* e = std::getenv("SHADPS4_CHUNK_DRAWS");
        const u32 n = e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
        return n != 0 ? n : 2048u;
    }();
    return v;
}
static bool ChunkCountsDispatches() {
    static const bool v = std::getenv("SHADPS4_CHUNK_DRAWS") != nullptr;
    return v;
}

void Rasterizer::ChunkHeavyFrame() {
    if (++draws_since_flush < ChunkDrawsPerFlush()) {
        return;
    }
    draws_since_flush = 0;
    FlushSkinBatch(SkinBatch::Reason::Chunk); // [CSBATCH] the batch stays in this tick
    {
        // [FRAMECHAIN] The chunk flush (a vkQueueSubmit, ~19/frame at CHUNK_DRAWS=64) is
        // charged to `sched` too, not to the enclosing vk bucket. Same calls, same order.
        Parserprof::PhaseProbe p{Parserprof::PhSched};
        scheduler.EndRendering();
        scheduler.Flush();
    }
}

// [SKIPCS] SHADPS4_SKIPCS=<hex hash>[,<hex hash>...]: skip compute dispatches of the named
// shaders. Diagnostic only - a stand experiment to price the GPU-side polling shaders (the ones
// GPUSTALL-SHADER names) by removing them; the game is expected to misbehave.
static bool SkipComputeShader(u64 hash) {
    static const std::vector<u64> wanted = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_SKIPCS");
        if (e == nullptr) {
            return out;
        }
        const char* cursor = e;
        while (*cursor != '\0') {
            char* end = nullptr;
            const u64 v = std::strtoull(cursor, &end, 16);
            if (end == cursor) {
                break;
            }
            if (v != 0) {
                out.push_back(v);
            }
            cursor = (*end == ',') ? end + 1 : end;
        }
        return out;
    }();
    if (wanted.empty()) {
        return false;
    }
    // [SKIPCS-AFTER] SHADPS4_SKIPCS_AFTER_S=<sec>: start skipping only this long after launch,
    // so the level loads normally and the deck GPUBusy (PresentMon v2) prices the dispatch.
    {
        static const u64 after_s = [] {
            const char* e = std::getenv("SHADPS4_SKIPCS_AFTER_S");
            return e ? static_cast<u64>(std::strtoull(e, nullptr, 10)) : u64{0};
        }();
        static const auto t0 = std::chrono::steady_clock::now();
        if (after_s != 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(after_s)) {
            return false;
        }
    }
    for (const u64 h : wanted) {
        if (h == hash) {
            static std::atomic<u32> s_skipped{0};
            const u32 n = ++s_skipped;
            if ((n % 500) == 1) {
                LOG_CRITICAL(Render_Vulkan, "SKIPCS: skipped {} dispatches (last cs={:#x})", n,
                             hash);
            }
            return true;
        }
    }
    return false;
}

// [CSSPLIT] 5-second census: split dispatches, guest -> host group counts, and how many
// dispatches of a listed shader went out unsplit (indirect args or a 2-D/3-D grid). The
// once-per-shader line explains the first unsplit dispatch of each listed hash. Never called
// unless the shader is listed in SHADPS4_CSSPLIT.
static void CsSplitAccount(u64 hash, u32 dim_x, u32 k, bool unsplit) {
    static u64 n_split = 0, groups_guest = 0, groups_host = 0, n_unsplit = 0;
    static std::unordered_set<u64> explained;
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    if (unsplit) {
        ++n_unsplit;
        if (explained.insert(hash).second) {
            LOG_CRITICAL(Render_Vulkan,
                         "[CSSPLIT] shader {:#x}: dispatch left unsplit (indirect args or "
                         "dim_y/dim_z != 1) - K=1 permutation",
                         hash);
        }
    } else {
        ++n_split;
        groups_guest += dim_x;
        groups_host += u64(dim_x) * k;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds(5);
    LOG_CRITICAL(Render_Vulkan,
                 "[CSSPLIT] 5s: {} split dispatches, groups {} -> {} (last {:#x}: {} -> {} "
                 "groups), {} listed dispatches unsplit",
                 n_split, groups_guest, groups_host, hash, dim_x, dim_x * k, n_unsplit);
    n_split = groups_guest = groups_host = n_unsplit = 0;
}

void Rasterizer::DispatchDirect() {
    // [SLOWPKT] vk phase (compute dispatch).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhVk};
    ++g_dbg_dispatches;
    RENDERER_TRACE;

    PopPendingTimed(scheduler);

    const auto& cs_program = liverpool->GetCsRegs();
    // [CSSPLIT] only a 1-D grid is split; dim_y/dim_z != 1 selects the unsplit permutation
    const bool allow_cssplit = cs_program.dim_y == 1 && cs_program.dim_z == 1;
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(allow_cssplit);
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (SkipComputeShader(cs.pgm_hash)) {
        return;
    }
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }
    // [CSBATCH] a foreign shader (or a declined/shadowed worker dispatch): the batch goes first.
    // Counted as flush{disp} even when BindResources then eats the dispatch (meta clear / image
    // copy / image clear early-outs): those record texture-cache work of their own.
    FlushSkinBatch(SkinBatch::Reason::Dispatch);

    // [SEQLOG] the worker's own line (V# bases peeked from its user-data block like DESCDUMP),
    // or a short line for any other dispatch inside the window.
    if (SeqLog::Wants(cs.pgm_hash)) {
        if (SeqLog::OnWorker() && cs.user_data.size() >= 4) {
            const u64 blk = u64(cs.user_data[0]) | (u64(cs.user_data[1]) << 32);
            const u64 cnt_addr = u64(cs.user_data[2]) | (u64(cs.user_data[3]) << 32);
            u32 d[32]{};
            u32 home = 0;
            for (u32 i = 16; i < 28; i++) {
                buffer_cache.PeekGuestDword(blk + i * 4, &d[i], &home);
            }
            u32 cnt_v = 0;
            buffer_cache.PeekGuestDword(cnt_addr, &cnt_v, &home);
            const auto vbase = [&](u32 i) { return u64(d[i]) | (u64(d[i + 1] & 0xffff) << 32); };
            const auto vsize = [&](u32 i) {
                const u64 st = (d[i + 1] >> 16) & 0x3fff;
                return st ? st * d[i + 2] : u64(d[i + 2]);
            };
            LOG_CRITICAL(Render_Vulkan,
                         "SEQ D k={} tick={} grid={}x{}x{} cnt={} blk={:#x} v16={:#x}+{:#x} "
                         "v20={:#x}+{:#x} v24={:#x}+{:#x}",
                         SeqLog::g_seen, scheduler.CurrentTick(), cs_program.dim_x,
                         cs_program.dim_y, cs_program.dim_z, cnt_v, blk, vbase(16), vsize(16),
                         vbase(20), vsize(20), vbase(24), vsize(24));
        }
    } else if (SeqLog::Active()) {
        LOG_CRITICAL(Render_Vulkan, "SEQ C tick={} pgm={:#x} grid={}x{}x{}",
                     scheduler.CurrentTick(), cs.pgm_hash, cs_program.dim_x, cs_program.dim_y,
                     cs_program.dim_z);
    }

    // [GPUPROF-PREP] three brackets around everything recorded before the dispatch itself:
    // bind (uploads, page-table writes), rp (EndRendering), bar (Pipeline::BindResources barriers)
    const bool prof = GpuProfInit(instance);
    u32 gpuprof_prep_q = prof ? GpuProfPrepBegin(scheduler.CommandBuffer()) : UINT32_MAX;
    if (!BindResources(pipeline)) {
        GpuProfPrepEnd(scheduler.CommandBuffer(), gpuprof_prep_q, scheduler.CurrentTick(),
                       cs.pgm_hash, 4);
        if (skin_batch.shadow.armed && !skin_batch.shadow.registered) {
            // [CSBATCH-SHADOW] the guest never computes the shadowed job (meta-clear early-out)
            skin_batch.shadow.armed = false;
            ++skin_batch.st.shadow_skip;
        }
        return;
    }
    GpuProfPrepEnd(scheduler.CommandBuffer(), gpuprof_prep_q, scheduler.CurrentTick(), cs.pgm_hash,
                   4);

    gpuprof_prep_q = prof ? GpuProfPrepBegin(scheduler.CommandBuffer()) : UINT32_MAX;
    {
        Parserprof::PhaseProbe p{Parserprof::PhSched};
        scheduler.EndRendering();
    }
    GpuProfPrepEnd(scheduler.CommandBuffer(), gpuprof_prep_q, scheduler.CurrentTick(), cs.pgm_hash,
                   5);
    gpuprof_prep_q = prof ? GpuProfPrepBegin(scheduler.CommandBuffer()) : UINT32_MAX;
    // [CSBARRIER] SHADPS4_CSBARRIER=lax: a buffer barrier between two compute dispatches is a
    // compute-pipe drain (~25 us each on RTX 5080; the ND job workers take 107-196 of them per
    // frame = 2.8-4.6 ms). GNM has no implicit hazard tracking between dispatches - the game
    // orders its jobs itself - so compute->compute buffer barriers are dropped; anything that
    // involves a transfer, the host, indirect args or a graphics stage stays.
    // lax = keep the barriers of a dispatch preceded by the game's ACQUIRE_MEM; drop = drop them
    // regardless (the game sends ACQUIRE_MEM before every worker dispatch - liveloop105 kept all)
    static const int cs_barrier_mode = [] {
        const char* e = std::getenv("SHADPS4_CSBARRIER");
        const int v = !e                              ? 0
                      : std::string_view{e} == "lax"  ? 1
                      : std::string_view{e} == "drop" ? 2
                                                      : 0;
        if (v) {
            LOG_CRITICAL(Render_Vulkan,
                         "CSBARRIER: mode {} - compute->compute buffer barriers dropped{}",
                         v == 1 ? "lax" : "drop", v == 1 ? " unless ACQUIRE_MEM asked" : " always");
        }
        return v;
    }();
    const bool cs_barrier_lax = cs_barrier_mode != 0;
    if (cs_barrier_lax && !buffer_barriers.empty()) {
        static u64 dropped_total = 0, kept_total = 0, sync_total = 0;
        static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        if (g_cs_sync_pending && cs_barrier_mode == 1) {
            // the game asked for a sync (ACQUIRE_MEM) since the previous dispatch: honour it
            g_cs_sync_pending = false;
            ++sync_total;
            kept_total += buffer_barriers.size();
        } else {
            if (g_cs_sync_pending) {
                ++sync_total; // counted even when ignored (drop mode)
                g_cs_sync_pending = false;
            }
            static u32 shown = 0;
            static auto sample_next = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() >= sample_next && cs.pgm_hash == 0x8f067540ull) {
                sample_next = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                shown = 0;
            }
            if (shown < 6 && cs.pgm_hash == 0x8f067540ull) {
                for (const auto& b : buffer_barriers) {
                    if (shown++ >= 6)
                        break;
                    LOG_CRITICAL(Render_Vulkan,
                                 "CSBARRIER-SAMPLE: pgm={:#x} src_stage={:#x} src_access={:#x} "
                                 "dst_stage={:#x} "
                                 "dst_access={:#x} size={:#x}",
                                 cs.pgm_hash, static_cast<u64>(b.srcStageMask),
                                 static_cast<u64>(b.srcAccessMask),
                                 static_cast<u64>(b.dstStageMask),
                                 static_cast<u64>(b.dstAccessMask), b.size);
                }
            }
            // v2 (08.09): the buffer cache emits its hazard barriers as ALL_COMMANDS ->
            // ALL_COMMANDS with SHADER_READ -> SHADER_WRITE (liveloop269 samples: src_stage
            // 0x10000, src_access 0x20, dst_access 0x40), so the old "COMPUTE -> COMPUTE only"
            // test matched nothing (dropped=0 kept=847882). Drop every shader<->shader buffer
            // barrier of this dispatch; keep anything that involves a transfer, the host or
            // indirect arguments (those guard emulator-internal uploads, not game ordering).
            const auto is_droppable = [](const vk::BufferMemoryBarrier2& b) {
                using S = vk::PipelineStageFlagBits2;
                using A = vk::AccessFlagBits2;
                const vk::PipelineStageFlags2 keep_stages = S::eTransfer | S::eCopy | S::eClear |
                                                            S::eBlit | S::eResolve | S::eHost |
                                                            S::eDrawIndirect;
                const vk::AccessFlags2 keep_access = A::eTransferWrite | A::eTransferRead |
                                                     A::eHostWrite | A::eHostRead |
                                                     A::eIndirectCommandRead;
                return !(b.srcStageMask & keep_stages) && !(b.dstStageMask & keep_stages) &&
                       !(b.srcAccessMask & keep_access) && !(b.dstAccessMask & keep_access);
            };
            static u32 drop_shown = 0;
            static auto drop_next = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() >= drop_next) {
                drop_next = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                drop_shown = 0;
            }
            for (const auto& b : buffer_barriers) {
                if (drop_shown >= 4 || !is_droppable(b)) {
                    continue;
                }
                ++drop_shown;
                LOG_CRITICAL(Render_Vulkan,
                             "CSBARRIER-DROP: pgm={:#x} src_stage={:#x} src_access={:#x} "
                             "dst_stage={:#x} dst_access={:#x} size={:#x}",
                             cs.pgm_hash, static_cast<u64>(b.srcStageMask),
                             static_cast<u64>(b.srcAccessMask), static_cast<u64>(b.dstStageMask),
                             static_cast<u64>(b.dstAccessMask), b.size);
            }
            const auto before = buffer_barriers.size();
            buffer_barriers.erase(
                std::remove_if(buffer_barriers.begin(), buffer_barriers.end(), is_droppable),
                buffer_barriers.end());
            dropped_total += before - buffer_barriers.size();
            kept_total += buffer_barriers.size();
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= next) {
            next = now + std::chrono::seconds(10);
            LOG_CRITICAL(Render_Vulkan,
                         "CSBARRIER: dropped={} kept={} acquire_syncs={} (cumulative)",
                         dropped_total, kept_total, sync_total);
        }
    }
    // [SKINVRAM] SHADPS4_SKINVRAM: serve the ND skinning worker's raw-pointer inputs from a VRAM
    // ring; rewrites ud_0..3 in push_data before they are pushed. Sits after BindResources (all
    // cache uploads recorded, render pass ended) and before Pipeline::BindResources, so the slot
    // write, the push constants and the dispatch share one command buffer. Deliberately AFTER
    // the CSBARRIER filter: with a non-ReBAR ring the redirect hands back one ranged
    // transfer->compute barrier, which is folded into the single pipelineBarrier2 that
    // Pipeline::BindResources issues anyway and must not be dropped by the filter. No-op when
    // the env is unset.
    if (VideoCore::BufferCache::SkinVramWanted(cs.pgm_hash)) {
        buffer_cache.SkinVramRedirect(cs.pgm_hash, cs.ud_mask.mask, push_data.ud_regs.data(),
                                      buffer_barriers);
    }
    {
        Parserprof::PhaseProbe p{Parserprof::PhBarr};
        Parserprof::t_pkt_barriers += static_cast<u32>(buffer_barriers.size());
        Parserprof::t_pkt_writes += static_cast<u32>(set_writes.size());
        Parserprof::g_bar_calls.fetch_add(1, std::memory_order_relaxed);
        Parserprof::g_bar_buffers.fetch_add(buffer_barriers.size(), std::memory_order_relaxed);
        Parserprof::g_desc_writes.fetch_add(set_writes.size(), std::memory_order_relaxed);
        CountRedundantDescriptors(set_writes);
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
    }

    // [debug] вскрытие кадра: параметры диспатчей чаще, чем DUMPDRAW —
    // тяжёлая фаза живёт в узком окне флипов 700-800
    {
        const u32 кадр = g_dbg_frame_no.load();
        if ((кадр % 100) == 50) {
            LOG_WARNING(Render_Vulkan, "DUMPDISP: cs={:#x} groups={}x{}x{} tg={}x{}x{}",
                        cs.pgm_hash, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z,
                        cs_program.num_thread_x.full, cs_program.num_thread_y.full,
                        cs_program.num_thread_z.full);
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    {
        Parserprof::PhaseProbe p{Parserprof::PhBind};
        ++Parserprof::t_pkt_binds;
        Parserprof::g_bind_calls.fetch_add(1, std::memory_order_relaxed);
        if (Vulkan::ShouldBindPipeline(pipeline->Handle())) {
            Parserprof::g_bind_real.fetch_add(1, std::memory_order_relaxed);
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
        }
    }
    RecordDispatch(cs.pgm_hash, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    RecordDispatchTick(scheduler.CurrentTick(), cs.pgm_hash, cs_program.dim_x, cs_program.dim_y,
                       cs_program.dim_z);
    if (instance.HasNvCheckpoints()) {
        cmdbuf.setCheckpointNV(reinterpret_cast<const void*>(cs.pgm_hash));
    }
    GpuProfPrepEnd(cmdbuf, gpuprof_prep_q, scheduler.CurrentTick(), cs.pgm_hash, 6);
    const u32 gpuprof_q = GpuProfInit(instance)
                              ? GpuProfBegin(cmdbuf, scheduler.CurrentTick(), cs.pgm_hash,
                                             (u64(cs_program.dim_x) << 40) |
                                                 (u64(cs_program.dim_y) << 20) | cs_program.dim_z,
                                             2)
                              : UINT32_MAX;
    // [CSSPLIT] 1 when off or declined; K of the permutation GetComputePipeline() just selected,
    // so the LocalSize baked into the bound module and this grid were compiled together. The
    // profiler brackets and RecordDispatch above keep the guest dims.
    const u32 split_k = pipeline_cache.ComputeSplitK();
    {
        Parserprof::PhaseProbe p{Parserprof::PhDraw};
        cmdbuf.dispatch(cs_program.dim_x * split_k, cs_program.dim_y, cs_program.dim_z);
    }
    if (split_k > 1) {
        CsSplitAccount(cs.pgm_hash, cs_program.dim_x, split_k, false);
    } else if (pipeline_cache.ComputeSplitListed()) {
        CsSplitAccount(cs.pgm_hash, cs_program.dim_x, 1, true); // listed, but 2-D/3-D grid
    }
    // [SKINVRAM] the slot's lifetime is the tick of the cmdbuf that holds the dispatch
    buffer_cache.SkinVramStamp(scheduler.CurrentTick());
    GpuProfEnd(cmdbuf, gpuprof_q);
    if (skin_batch.shadow.armed && !skin_batch.shadow.registered) {
        // [CSBATCH-SHADOW] this is the guest permutation computing the shadowed job into the
        // real output: both results are in this cmdbuf now, compare once the tick retires.
        SkinBatchShadowArm();
    }
    DebugState.IncDispatch();
    if (ChunkCountsDispatches()) {
        ChunkHeavyFrame();
    }

    ResetBindings();
}

// ---------------------------------------------------------------------------------------------
// [CSBATCH] SHADPS4_CSBATCH=8f067540 (vk_shader_hle.h): the ND skinning worker is computed by
// a host kernel, one dispatch per batch of guest dispatches. Append (from ExecuteShaderHLE)
// resolves the dispatch's 7 ranges through the buffer cache NOW - snapshots and uploads are
// recorded in guest order, exactly what BindBuffers would do - and stores raw device addresses
// in an 88-byte job record. Flush records the barriers, the table, and ONE vkCmdDispatch; it is
// called before any other GPU command (Draw/DrawIndirect/DispatchDirect/DispatchIndirect/
// FillBuffer/CopyBuffer/CpSync/Flush/Finish/OnSubmit/ProcessFaultBufferIfPending/
// ProcessDownloadImages/DeferSignalToRetire/ChunkHeavyFrame here, the label/WriteData packets
// and ProcessCommands in liverpool.cpp) and from the scheduler's pre-submit hook, so the batch
// always lands in the tick of its uploads. Addresses follow the cache OWNER of each range (a
// stream snapshot, the VRAM copy or a backing window) - what the SKINDIRECT bindings read
// today; the page table and the fault path are not involved.
// ---------------------------------------------------------------------------------------------
static vk::PipelineStageFlagBits2 SkinBatchDstStage() {
    // BindBuffers' rule (PRECISE-STAGES): GetBarrier then compares the same tracked states.
    return VideoCore::Buffer::PreciseStages() ? vk::PipelineStageFlagBits2::eComputeShader
                                              : vk::PipelineStageFlagBits2::eAllCommands;
}

bool Rasterizer::SkinBatchAppend(const Shader::Info& cs, const AmdGpu::ComputeProgram& p) {
    auto& sb = skin_batch;
    if (!sb.Ready()) {
        return false;
    }
    if (sb.gpu_thread == std::thread::id{}) {
        sb.gpu_thread = std::this_thread::get_id();
    }
    using D = SkinBatch::Decline;
    const auto ud = [&](u32 i) -> u32 { return cs.user_data.size() > i ? cs.user_data[i] : 0u; };
    VAddr cl_base = 0; // the output range once decoded: a declined job takes the guest path and
    u64 cl_size = 0;   // still rewrites it - the shadow compare must know (clobbered)
    const auto decline = [&](D why) {
        ++sb.st.decl[size_t(why)];
        if (sb.shadow.armed && cl_size != 0 && cl_base < sb.shadow.out_base + sb.shadow.out_size &&
            sb.shadow.out_base < cl_base + cl_size) {
            sb.shadow.clobbered = true;
        }
        if (!sb.st.decl_logged[size_t(why)]) {
            sb.st.decl_logged[size_t(why)] = true;
            // The descriptor list comes with it: run 341 declined 46762 jobs on "layout" and the
            // line alone could not say which of the three slots was wrong.
            std::string bufs;
            for (u32 i = 0; i < cs.buffers.size() && i < 10; ++i) {
                const auto& b = cs.buffers[i];
                bufs += fmt::format(" [{}]sharp={}/st{}{}{}", i, b.sharp_idx,
                                    b.IsSpecial() ? 0u : u32(b.GetSharp(cs).stride),
                                    b.IsSpecial() ? ",special" : "", b.is_written ? ",w" : "");
            }
            LOG_CRITICAL(Render_Vulkan,
                         "[CSBATCH] decline {}: ud={:#x} {:#x} {:#x} {:#x} grid={}x{}x{} tg={} "
                         "bufs(n={}):{}",
                         SkinBatch::DeclineName(why), ud(0), ud(1), ud(2), ud(3), p.dim_x, p.dim_y,
                         p.dim_z, p.num_thread_x.full, cs.buffers.size(), bufs);
        }
        // the normal path's BindBuffers may CreateBuffer over a pending owner: batch first
        FlushSkinBatch(SkinBatch::Reason::Fallback);
        SkinBatchReport();
        return false; // -> CSSPLIT+SKINDIRECT path, unchanged
    };
    // 1. shape: guest index = wg*1024 + local, x only; three user-data V#s (pos, nrm, tan).
    //    spv/cs_8f067540.dis: binding 0 = position f32, 1 = normal, 2 = tangent, all three in
    //    user data (sharp_idx < 16); a converted block read (SKINDIRECT) sits beyond that.
    if (p.dim_y != 1 || p.dim_z != 1 || p.num_thread_x.full != 1024 || cs.user_data.size() < 4 ||
        cs.buffers.size() < 3) {
        return decline(D::Dims);
    }
    // 09.09: the three source V#s are NOT buffers[0..2]. The shader reads its vertex count
    // through ReadConst first, so the flat buffer (and the BDA page table / fault buffer when
    // they are used) are added to the descriptor list before them; every job of run 341 was
    // declined here with buffers[0] special. Take the first three descriptors that are plain
    // read-only user-data buffers instead - in IR order that is position, normal, tangent
    // (spv/cs_8f067540.dis: bindings 0/1/2), with the SKINDIRECT-converted influence/palette
    // buffers appearing after them.
    // Runs 346/352 printed the real descriptor list: 12 entries, sharp_idx 28/40/44/48/32/36
    // read-only, 16/20/24 written (the interleaved output), one special - so the sources are
    // neither the first three entries nor user-data registers (sharp_idx indexes the flat
    // buffer). They are picked by shape below, AFTER the block is read: position is the
    // stride-12 V#, normal and tangent the two stride-8 ones that are not the influence header
    // (which also has stride 8 - run 352 selected it as "normal", giving nrm == hdr).
    // 2. guest facts. IsMapped keeps a garbage pointer out; the reads themselves go through
    //    ReadProcessMemory (PeekGuestBytes): a page under the tracker's read watchers is
    //    PAGE_NOACCESS (PageState::Perms) and a plain load would trap the GPU thread into the
    //    readback handler - the rule SKINVRAM and SEQLOG already follow for these two objects.
    //    DESCDUMP: the count dword sits 16 B below the block in every sample, so that layout
    //    costs one read; anything else two.
    const VAddr blk = VAddr(cs.user_data[0]) | (VAddr(cs.user_data[1]) << 32);
    const VAddr cnt_addr = VAddr(cs.user_data[2]) | (VAddr(cs.user_data[3]) << 32);
    if (blk == 0 || cnt_addr == 0 || !IsMapped(blk, 128) || !IsMapped(cnt_addr, sizeof(u32))) {
        return decline(D::Unmapped);
    }
    u32 d[32];
    u32 count;
    if (blk >= cnt_addr + sizeof(u32) && blk - cnt_addr <= 64) {
        u32 span[16 + 32];
        const u32 lead = static_cast<u32>(blk - cnt_addr);
        if (!buffer_cache.PeekGuestBytes(cnt_addr, span, lead + sizeof(d))) {
            return decline(D::Unmapped);
        }
        count = span[0];
        std::memcpy(d, reinterpret_cast<const u8*>(span) + lead, sizeof(d));
    } else if (!buffer_cache.PeekGuestBytes(blk, d, sizeof(d)) ||
               !buffer_cache.PeekGuestBytes(cnt_addr, &count, sizeof(count))) {
        return decline(D::Unmapped);
    }
    const u32 n = static_cast<u32>(std::min<u64>(count, u64(p.dim_x) * 1024u));
    if (n == 0) {
        ++sb.st.count0; // every guest thread returns at index < count
        return true;
    }
    // the shader masks base_hi & 0xfff; stride/records as DESCDUMP decodes them
    const auto vbase = [&](u32 w) { return VAddr(d[w]) | (VAddr(d[w + 1] & 0xfffu) << 32); };
    const auto vsize = [&](u32 w) -> u64 {
        const u64 st = (d[w + 1] >> 16) & 0x3fffu;
        return st ? st * d[w + 2] : u64(d[w + 2]);
    };
    if (vbase(0) == 0) {
        ++sb.st.null_out; // SPECMISS/FMT0 empty slot: the guest stores nothing
        return true;
    }
    AmdGpu::Buffer s0{}, s1{}, s2{};
    {
        const VAddr hdr_base = vbase(16), infl_base = vbase(20), pal_base = vbase(24);
        bool have_pos = false;
        u32 got8 = 0;
        for (u32 i = 0; i < cs.buffers.size(); ++i) {
            const auto& b = cs.buffers[i];
            if (b.IsSpecial() || b.is_written) {
                continue;
            }
            const auto sh = b.GetSharp(cs);
            const VAddr base = sh.base_address;
            if (base == hdr_base || base == infl_base || base == pal_base) {
                continue; // block-side buffers (header / influences / palette), not the mesh
            }
            const u32 stride = static_cast<u32>(sh.stride);
            if (stride == 12 && !have_pos) {
                s0 = sh;
                have_pos = true;
            } else if (stride == 8 && got8 < 2) {
                (got8 == 0 ? s1 : s2) = sh;
                ++got8;
            }
        }
        if (!have_pos || got8 < 2) {
            return decline(D::Layout);
        }
    }
    struct R {
        VAddr base;
        u64 size;
        bool written;
    };
    const R out{vbase(0), u64(n) * 28, true};
    const R hdr{vbase(16), u64(n) * 8, false};
    const R infl{vbase(20), vsize(20), false};
    const R pal{vbase(24), vsize(24), false};
    const R pos{s0.base_address, u64(n) * 12, false};
    const R nrm{s1.base_address, u64(n) * 8, false};
    const R tan{s2.base_address, u64(n) * 8, false};
    cl_base = out.base;
    cl_size = out.size;
    // 3. sanity -> normal path (each one counted and logged once)
    if (vbase(4) != out.base + 12 || vbase(8) != out.base + 20) {
        return decline(D::Layout); // all 293 DESCDUMP samples interleaved
    }
    for (const u32 w : {2u, 18u, 22u, 26u}) {
        if (d[w] == 0xffffffffu) {
            return decline(D::Unbounded);
        }
    }
    if (s0.num_records == 0xffffffffu || s1.num_records == 0xffffffffu ||
        s2.num_records == 0xffffffffu) {
        return decline(D::Unbounded);
    }
    if (s0.GetSize() < pos.size || s1.GetSize() < nrm.size || s2.GetSize() < tan.size) {
        return decline(D::ShortSrc); // today: robust OOB reads 0
    }
    if (vsize(0) < out.size || vsize(16) < hdr.size) {
        // the guest clips stores / reads 0 past its V#, the kernel would not (raw pointers).
        // DESCDUMP: records == count for both, so a real job never lands here.
        return decline(D::ShortSrc);
    }
    if (infl.size < 4 || pal.size < 48) {
        return decline(D::Empty);
    }
    if (SkinBatch::PalVec4() && (pal.base & 15) != 0) {
        return decline(D::Align);
    }
    const R all[7] = {out, hdr, infl, pal, pos, nrm, tan};
    for (const R& r : all) {
        if (r.base == 0 || r.size > (u64{64} << 20)) {
            return decline(D::Oversize);
        }
        if ((r.base & 3) != 0) {
            return decline(D::Align); // buffer_reference_align = 4; BindBuffers rounds down
        }
        if (memory->ClampRangeSize(r.base, r.size) != r.size) {
            return decline(D::Unmapped);
        }
    }
    const u32 wg = SkinBatch::Wg();
    const u32 groups = (n + wg - 1) / wg;
    if (groups >= (1u << SkinBatch::JobShift) || groups > sb.max_groups) {
        return decline(D::Dims); // group_in_job must fit the map entry and one vkCmdDispatch
    }
    // 4. batch boundaries BEFORE touching the cache
    if (!sb.jobs.empty()) {
        if (sb.jobs.size() >= SkinBatch::MaxJobs() ||
            sb.group_map.size() + groups > sb.max_groups) {
            FlushSkinBatch(SkinBatch::Reason::Full);
        } else if (scheduler.CurrentTick() != sb.first_tick) {
            ++sb.st.tick_splits; // must stay 0: the pre-submit hook flushes before a tick ends
            FlushSkinBatch(SkinBatch::Reason::Hook);
        }
    }
    const auto iv = [](const R& r) {
        return boost::icl::interval<VAddr>::right_open(r.base, r.base + r.size);
    };
    if (!sb.jobs.empty()) {
        // chained: this job READS (or rewrites) what a pending job WRITES, or WRITES what a
        // pending job READS - one dispatch cannot order either
        bool chained = boost::icl::intersects(sb.pending_in, iv(out));
        for (const R* r : {&out, &hdr, &infl, &pal, &pos, &nrm, &tan}) {
            chained = chained || boost::icl::intersects(sb.pending_out, iv(*r));
        }
        if (chained) {
            ++sb.st.chained;
            FlushSkinBatch(SkinBatch::Reason::Chained);
        }
    }
    const u64 tick_before = scheduler.CurrentTick();
    // 5. resolve. Pass A: make every cache-path owner exist. A CreateBuffer JoinOverlaps and
    //    deletes the owners a pending job already points at, so the batch is flushed before the
    //    first creation; repeat while a creation invalidated an earlier answer of THIS job
    //    (adjacent header/list ranges merging).
    bool creates[7];
    bool any_create = false;
    for (u32 i = 0; i < 7; ++i) {
        creates[i] = buffer_cache.WouldCreate(all[i].base, u32(all[i].size), all[i].written);
        any_create |= creates[i];
    }
    if (any_create && !sb.jobs.empty()) {
        FlushSkinBatch(SkinBatch::Reason::Create);
    }
    for (int pass = 0; any_create; ++pass) {
        if (pass == 3) {
            ++sb.st.unstable;
            return decline(D::Unstable);
        }
        for (u32 i = 0; i < 7; ++i) {
            if (creates[i]) {
                (void)buffer_cache.FindBuffer(all[i].base, u32(all[i].size));
            }
        }
        any_create = false;
        for (u32 i = 0; i < 7; ++i) {
            creates[i] = buffer_cache.WouldCreate(all[i].base, u32(all[i].size), all[i].written);
            any_create |= creates[i];
        }
    }
    if (!sb.jobs.empty()) {
        // reupload: a cache-path input a pending job already reads (the same VRAM owner) with
        // CPU-dirty pages - this job's ObtainBuffer would upload them BEFORE the one dispatch
        // that serves both, and the pending job would read the new bytes where today it reads
        // the old ones. Stream-path ranges snapshot privately, import-backed owners never
        // upload: WouldUpload answers false for both.
        for (const R* r : {&hdr, &infl, &pal, &pos, &nrm, &tan}) {
            if (boost::icl::intersects(sb.pending_in, iv(*r)) &&
                buffer_cache.WouldUpload(r->base, u32(r->size), false)) {
                ++sb.st.reupload;
                FlushSkinBatch(SkinBatch::Reason::Reupload);
                break;
            }
        }
    }
    //    Pass B: obtain -> uploads and snapshots recorded NOW in guest order (BindBuffers'
    //    ObtainBuffer), barriers collected, address = owner BDA + byte offset.
    boost::container::static_vector<VideoCore::Buffer*, 8> owners;
    SkinJobGpu j{};
    const auto resolve = [&](const R& r, u64* bda) -> bool {
        const auto [buf, off] = buffer_cache.ObtainBuffer(r.base, u32(r.size), r.written);
        if (!buf->HasDeviceAddress()) {
            return false; // SNAPRO shadow etc.
        }
        *bda = buf->BufferDeviceAddress() + off; // byte-exact, no StorageMinAlignment "adjust"
        if (auto b = buf->GetBarrier(r.written ? vk::AccessFlagBits2::eShaderWrite
                                               : vk::AccessFlagBits2::eShaderRead,
                                     SkinBatchDstStage())) {
            sb.barriers.push_back(*b);
        }
        owners.push_back(buf);
        return true;
    };
    if (!resolve(hdr, &j.hdr) || !resolve(infl, &j.infl) || !resolve(pal, &j.pal) ||
        !resolve(pos, &j.pos) || !resolve(nrm, &j.nrm) || !resolve(tan, &j.tan) ||
        !resolve(out, &j.out_pos)) {
        return decline(D::NoBda); // uploads already recorded: harmless
    }
    for (const auto* o : owners) {
        if (o->is_deleted) {
            ++sb.st.unstable;
            return decline(D::Unstable);
        }
    }
    VideoCore::Buffer* out_owner = owners.back();
    j.out_nrm = j.out_pos + 12;
    j.out_tan = j.out_pos + 20;
    j.count = n;
    j.infl_dw = u32(infl.size / 4);
    j.pal_dw = u32(pal.size / 4);
    // [R1] home-4 divergence, SkinDirectNoteStoreBind's rule
    if (!out_owner->is_import_backed && buffer_cache.PtRedirectedInside(out.base, out.size)) {
        return decline(D::PtDiverge);
    }
    // [SHADOW] one job per N-th frame is computed by the kernel into a scratch area of the ring
    // AND by the guest permutation into the real output; this dispatch takes the normal path.
    if (SkinBatch::Shadow() != 0 && SkinBatchShadowMaybe(j, n, out.base, out.size, *out_owner)) {
        return false;
    }
    // 6. consumer plumbing, per sub-dispatch (BindBuffers' tail for a written descriptor)
    buffer_cache.CsBatchNoteOutput(out.base, out.size); // BindVertexBuffers barrier, no snapshot
    if (SkinBatch::ImgInv()) {
        texture_cache.InvalidateMemoryFromGPU(out.base, out.size);
    }
    sb.pending_out += iv(out);
    for (const R* r : {&hdr, &infl, &pal, &pos, &nrm, &tan}) {
        sb.pending_in += iv(*r);
    }
    if (sb.shadow.armed && out.base < sb.shadow.out_base + sb.shadow.out_size &&
        sb.shadow.out_base < out.base + out.size) {
        sb.shadow.clobbered = true; // the guest output is rewritten before the compare
    }
    const VideoCore::BufferId out_id = buffer_cache.FindBuffer(out.base, u32(out.size));
    if (std::find(sb.out_owners.begin(), sb.out_owners.end(), out_id) == sb.out_owners.end()) {
        sb.out_owners.push_back(out_id);
    }
    // 7. today's side cadence
    if (SkinBatch::ZoneSync()) {
        SyncDmaZones(); // DMASYNC_ZONE ran on every worker dispatch (BindResources' uses_dma)
    }
    RecordDispatch(cs.pgm_hash, p.dim_x, 1, 1); // STUCKDISP ring keeps per-guest-dispatch entries
    // 8. append
    const u32 jid = u32(sb.jobs.size());
    if (sb.jobs.empty()) {
        sb.first_tick = scheduler.CurrentTick();
    }
    if (scheduler.CurrentTick() != tick_before) {
        // a hidden Wait->Flush inside resolve (stream/staging Map, SynchronizeBuffersInRange):
        // the uploads sit in the retired tick, the dispatch will sit in this one - the same
        // straddle BindBuffers has today. Re-stamp the output so a CPU read of it waits for
        // the tick that actually writes it.
        ++sb.st.straddle;
        (void)buffer_cache.ObtainBuffer(out.base, u32(out.size), true);
    }
    for (u32 gi = 0; gi < groups; ++gi) {
        sb.group_map.push_back((jid << SkinBatch::JobShift) | gi);
    }
    sb.jobs.push_back(j);
    sb.verts += n;
    ++sb.st.subs;
    sb.st.verts += n;
    // [SEQLOG] the worker's line is "SEQ A" here (the D line of DispatchDirect is never reached)
    const bool seq = SeqLog::Wants(cs.pgm_hash) ? SeqLog::OnWorker() : SeqLog::Active();
    if (seq) {
        LOG_CRITICAL(Render_Vulkan, "SEQ A tick={} job={} grid={} n={} out={:#x}",
                     scheduler.CurrentTick(), jid, p.dim_x, n, out.base);
    }
    static u32 logged_jobs = 0;
    if (SkinBatch::Log() && logged_jobs < 24) {
        ++logged_jobs;
        LOG_CRITICAL(Render_Vulkan,
                     "[CSBATCH] job#{} blk={:#x} cnt@{:#x}={} grid={} n={} out={:#x}+{:#x} "
                     "hdr={:#x} infl={:#x}+{:#x} pal={:#x}+{:#x} pos={:#x} nrm={:#x} tan={:#x} "
                     "out_home={} src_ud={}/{}/{}",
                     jid, blk, cnt_addr, count, p.dim_x, n, out.base, out.size, hdr.base, infl.base,
                     infl.size, pal.base, pal.size, pos.base, nrm.base, tan.base,
                     out_owner->is_backing_backed  ? 3
                     : out_owner->is_import_backed ? 2
                                                   : 1,
                     cs.buffers[0].sharp_idx, cs.buffers[1].sharp_idx, cs.buffers[2].sharp_idx);
    }
    if (SkinBatch::FlushEvery() != 0 && sb.jobs.size() >= SkinBatch::FlushEvery()) {
        FlushSkinBatch(SkinBatch::Reason::Every);
    }
    return true;
}

void Rasterizer::RecordSkinDispatch(std::span<const SkinJobGpu> jobs,
                                    std::span<const u32> group_map,
                                    std::span<const vk::BufferMemoryBarrier2> barriers, bool prebar,
                                    bool postbar, SkinBatch::Reason why) {
    auto& sb = skin_batch;
    // (a) table + map into the private ring: [SkinTableHdr][jobs n x 88 B] then, at the next
    //     StorageMinAlignment, [group_map n_groups x 4 B]. allow_wait=false: a wait would be
    //     Scheduler::Wait -> Flush -> the pre-submit hook re-entered (guarded) and the batch
    //     would land a tick later than its uploads.
    const u64 align = std::max<u64>(instance.StorageMinAlignment(), 16);
    const u64 tbytes = sizeof(SkinTableHdr) + jobs.size() * sizeof(SkinJobGpu);
    const u64 moff_rel = Common::AlignUp(tbytes, align);
    const u64 mbytes = group_map.size() * sizeof(u32);
    const u64 total = moff_rel + mbytes;
    auto [tp, toff] = sb.ring->Map(total, align, /*allow_wait=*/false);
    if (tp == nullptr) {
        ++sb.st.ring_wait;
        std::tie(tp, toff) = sb.ring->Map(total, align, true);
    }
    ASSERT_MSG(tp != nullptr, "[CSBATCH] a table of {} bytes does not fit the ring", total);
    const SkinTableHdr h{u32(jobs.size()), u32(group_map.size()), 0, 0};
    std::memcpy(tp, &h, sizeof(h));
    std::memcpy(tp + sizeof(h), jobs.data(), jobs.size() * sizeof(SkinJobGpu));
    std::memcpy(tp + moff_rel, group_map.data(), mbytes);
    sb.ring->Commit();

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const u64 tick = scheduler.CurrentTick();
    const bool prof = GpuProfInit(instance);
    u32 q = prof ? GpuProfPrepBegin(cmdbuf) : UINT32_MAX;
    vk::Buffer table = sb.ring->Handle();
    boost::container::small_vector<vk::BufferMemoryBarrier2, 34> bufbars(barriers.begin(),
                                                                         barriers.end());
    if (sb.vram_table) {
        // [TABLE_VRAM] staging copy into DEVICE_LOCAL at the same offsets: one ranged barrier
        // against the previous batch's reads of the region (WAR), one for the copy's
        // visibility (RAW) folded into the batch's pipelineBarrier2 below.
        const vk::BufferMemoryBarrier2 war = {
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = sb.vram_table->Handle(),
            .offset = toff,
            .size = total,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &war,
        });
        cmdbuf.copyBuffer(sb.ring->Handle(), sb.vram_table->Handle(),
                          vk::BufferCopy{.srcOffset = toff, .dstOffset = toff, .size = total});
        bufbars.push_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .buffer = sb.vram_table->Handle(),
            .offset = toff,
            .size = total,
        });
        table = sb.vram_table->Handle();
    }
    // (b) barriers: the collected per-owner transitions plus (PREBAR) one global memory barrier
    const vk::MemoryBarrier2 pre = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = prebar ? 1u : 0u,
        .pMemoryBarriers = &pre,
        .bufferMemoryBarrierCount = static_cast<u32>(bufbars.size()),
        .pBufferMemoryBarriers = bufbars.data(),
    });
    GpuProfPrepEnd(cmdbuf, q, tick, SkinBatch::kHash, 6); // GPUPROF-TOP "prep-bar": one per batch
    // (c) bind + ONE dispatch
    Vulkan::InvalidateBoundPipeline(); // BINDSKIP tracker (vk_common.h)
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *sb.pipeline);
    const vk::DescriptorBufferInfo tinfo = {
        .buffer = table,
        .offset = toff,
        .range = tbytes,
    };
    const vk::DescriptorBufferInfo minfo = {
        .buffer = table,
        .offset = toff + moff_rel,
        .range = mbytes,
    };
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tinfo,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &minfo,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *sb.layout, 0, writes);
    const u32 groups = static_cast<u32>(group_map.size());
    RecordDispatchTick(tick, SkinBatch::kHash, groups, 1, 1);
    if (instance.HasNvCheckpoints()) {
        cmdbuf.setCheckpointNV(reinterpret_cast<const void*>(SkinBatch::kHash));
    }
    // GPUPROF-TOP "disp" under 0x8f067540, one per batch: dims = groups<<40 | subs<<20 | kind
    // (1 = batch, 2 = shadow). The rows split per distinct dims; the by-kind `disp=` total and
    // the [CSBATCH] census carry the per-frame numbers.
    const u64 dims = (u64(groups) << 40) | (u64(jobs.size()) << 20) |
                     (why == SkinBatch::Reason::Shadow ? 2u : 1u);
    q = prof ? GpuProfBegin(cmdbuf, tick, SkinBatch::kHash, dims, 2) : UINT32_MAX;
    cmdbuf.dispatch(groups, 1, 1);
    GpuProfEnd(cmdbuf, q);
    if (postbar) {
        const vk::MemoryBarrier2 post = {
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &post,
        });
    }
    DebugState.IncDispatch();
}

bool Rasterizer::SkinBatchDeferLabel(std::function<void()>&& fence) {
    if (skin_batch.jobs.empty()) {
        return false; // nothing to wait for: the caller signals at parse time as today
    }
    // A cap keeps a stalled guest from waiting forever on labels this side is still holding:
    // once it is reached the batch is recorded, its queued labels are registered on that tick,
    // and this label takes the ordinary parse-time path.
    if (skin_batch_labels.size() >= SkinBatch::MaxHeldLabels()) {
        FlushSkinBatch(SkinBatch::Reason::Label);
        return false;
    }
    skin_batch_labels.push_back(std::move(fence));
    return true;
}

void Rasterizer::FlushSkinBatch(SkinBatch::Reason why) {
    auto& sb = skin_batch;
    if (why == SkinBatch::Reason::Hook && sb.gpu_thread != std::thread::id{} &&
        std::this_thread::get_id() != sb.gpu_thread) {
        // invariant 9: submits come from the GPU thread. Counted before anything is touched.
        sb.st.hook_offthread.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (sb.jobs.empty()) {
        // the only cost at every hook: this test. Barriers here belong to a job declined after
        // resolve or shadowed: resolve() already advanced the owners' tracked state through
        // GetBarrier, so the guest path's BindBuffers will not ask for these transitions again
        // - record them now, in the same tick (every VkBuffer is alive: a deleted owner is
        // erased only after its tick retires).
        if (!sb.barriers.empty()) {
            scheduler.EndRendering();
            scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = static_cast<u32>(sb.barriers.size()),
                .pBufferMemoryBarriers = sb.barriers.data(),
            });
            sb.st.orphan_bars += sb.barriers.size();
            sb.barriers.clear();
        }
        return;
    }
    if (sb.flushing) {
        return;
    }
    sb.flushing = true;
    ++sb.st.batches;
    ++sb.st.flush[size_t(why)];
    sb.st.groups += sb.group_map.size();
    sb.st.max_subs = std::max<u64>(sb.st.max_subs, sb.jobs.size());
    // Re-arm every output owner to ShaderWrite@dst so the consumer's GetBarrier(VertexAttribute-
    // Read) in BindVertexBuffers sees a compute WRITE as source even if a read bind flipped it
    // meanwhile. An owner deleted since its append (JoinOverlap by a foreign CreateBuffer) has
    // had its bytes copied away already: counted, expected 0.
    for (const auto id : sb.out_owners) {
        if (!buffer_cache.IsBufferLive(id)) {
            ++sb.st.deleted_owner;
            continue;
        }
        auto& o = buffer_cache.GetBuffer(id);
        if (auto b = o.GetBarrier(vk::AccessFlagBits2::eShaderWrite, SkinBatchDstStage())) {
            sb.barriers.push_back(*b);
        }
    }
    RecordSkinDispatch(
        sb.jobs, sb.group_map,
        std::span<const vk::BufferMemoryBarrier2>{sb.barriers.data(), sb.barriers.size()},
        SkinBatch::PreBar(), SkinBatch::PostBar(), why);
    if (scheduler.CurrentTick() != sb.first_tick) {
        ++sb.st.tick_splits;
    }
    sb.st.barriers += sb.barriers.size();
    if (SeqLog::Active()) {
        LOG_CRITICAL(Render_Vulkan, "SEQ B tick={} subs={} groups={} why={}",
                     scheduler.CurrentTick(), sb.jobs.size(), sb.group_map.size(),
                     SkinBatch::ReasonName(why));
    }
    sb.Reset();
    sb.flushing = false;
    SkinBatchReport();
    // the batch = ONE dispatch for CHUNK_DRAWS - except where a submit follows anyway (a chunk
    // submit right before Flush/Finish/a queued command would only add an empty tick)
    switch (why) {
    case SkinBatch::Reason::Hook:
    case SkinBatch::Reason::Chunk:
    case SkinBatch::Reason::Submit:
    case SkinBatch::Reason::Cmd:
    case SkinBatch::Reason::Fault:
    case SkinBatch::Reason::Readback:
        break;
    default:
        if (ChunkCountsDispatches() && SkinBatch::CountsForChunk()) {
            ChunkHeavyFrame();
        }
    }

    // [CSBATCH-DEFERLABEL] the batch is now in the current cmdbuf; the labels the command
    // processor held back are registered on this tick, so the guest sees "done" only after the
    // GPU has executed the work they announce.
    if (!skin_batch_labels.empty()) {
        for (auto& fence : skin_batch_labels) {
            scheduler.DeferPriorityOperation(std::move(fence));
        }
        skin_batch_labels.clear();
    }
}

// [CSBATCH-SHADOW] SHADPS4_CSBATCH_SHADOW=N: in a frame that is a multiple of N, the first job
// with n <= 4096 whose output owner is import/backing-backed (the CPU can read the guest's
// result from guest RAM without a readback) is computed twice: the kernel writes a scratch
// area of the ring (a one-job batch recorded right here), the guest permutation writes the
// real output (SkinBatchAppend returns false). SkinBatchShadowArm runs after that guest
// dispatch and defers the compare to the tick's retirement.
bool Rasterizer::SkinBatchShadowMaybe(const SkinJobGpu& j, u32 n, VAddr out_base, u64 out_size,
                                      const VideoCore::Buffer& out_owner) {
    auto& sb = skin_batch;
    auto& sh = sb.shadow;
    const u32 every = SkinBatch::Shadow();
    if (every == 0 || sh.armed) {
        return false;
    }
    const u64 frame = g_dbg_frame_no.load(std::memory_order_relaxed);
    if (frame % every != 0 || frame == sh.frame) {
        return false;
    }
    if (n > 4096 || !out_owner.is_import_backed) {
        return false;
    }
    const u64 bytes = u64(n) * 28;
    auto [sp, soff] = sb.ring->Map(bytes, 64, /*allow_wait=*/false);
    if (sp == nullptr) {
        ++sb.st.shadow_skip;
        return false;
    }
    std::memset(sp, 0xCD, bytes); // a vertex the kernel never writes shows up as a mismatch
    sb.ring->Commit();
    SkinJobGpu sj = j;
    sj.out_pos = sb.ring->BufferDeviceAddress() + soff;
    sj.out_nrm = sj.out_pos + 12;
    sj.out_tan = sj.out_pos + 20;
    const u32 wg = SkinBatch::Wg();
    const u32 groups = (n + wg - 1) / wg;
    boost::container::small_vector<u32, 128> map;
    for (u32 gi = 0; gi < groups; ++gi) {
        map.push_back(gi); // job 0
    }
    ++sb.st.flush[size_t(SkinBatch::Reason::Shadow)];
    RecordSkinDispatch(std::span<const SkinJobGpu>{&sj, 1},
                       std::span<const u32>{map.data(), map.size()}, {}, /*prebar=*/true,
                       /*postbar=*/true, SkinBatch::Reason::Shadow);
    sh.armed = true;
    sh.registered = false;
    sh.clobbered = false;
    sh.frame = frame;
    sh.tick = scheduler.CurrentTick();
    sh.wrap = sb.ring->WrapGeneration();
    sh.off = soff;
    sh.out_base = out_base;
    sh.out_size = out_size;
    sh.n = n;
    return true;
}

void Rasterizer::SkinBatchShadowArm() {
    auto& sb = skin_batch;
    auto& sh = sb.shadow;
    sh.registered = true;
    sh.tick = scheduler.CurrentTick();
    // the kernel's scratch write and the guest's real output are both recorded now: make them
    // visible to the host read of the compare (NVIDIA sysmem imports want the explicit barrier)
    scheduler.EndRendering();
    const vk::MemoryBarrier2 host = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &host,
    });
    scheduler.DeferOperation([this] { SkinBatchShadowCompare(); });
}

namespace {
// position dwords: exact, or both NaN, or +-0, or within `ulps` representable floats
bool ShadowF32Close(u32 a, u32 b, u32 ulps) {
    if (a == b) {
        return true;
    }
    const bool nan_a = (a & 0x7f800000u) == 0x7f800000u && (a & 0x007fffffu) != 0;
    const bool nan_b = (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0;
    if (nan_a || nan_b) {
        return nan_a && nan_b;
    }
    if ((a & 0x7fffffffu) == 0 && (b & 0x7fffffffu) == 0) {
        return true;
    }
    if ((a ^ b) & 0x80000000u) {
        return false;
    }
    const u32 d = a > b ? a - b : b - a;
    return d <= ulps;
}
bool ShadowF16Close(u32 a, u32 b, u32 ulps) {
    if (a == b) {
        return true;
    }
    const bool nan_a = (a & 0x7c00u) == 0x7c00u && (a & 0x03ffu) != 0;
    const bool nan_b = (b & 0x7c00u) == 0x7c00u && (b & 0x03ffu) != 0;
    if (nan_a || nan_b) {
        return nan_a && nan_b;
    }
    if ((a & 0x7fffu) == 0 && (b & 0x7fffu) == 0) {
        return true;
    }
    if ((a ^ b) & 0x8000u) {
        return false;
    }
    const u32 d = a > b ? a - b : b - a;
    return d <= ulps;
}
} // namespace

void Rasterizer::SkinBatchShadowCompare() {
    auto& sb = skin_batch;
    auto& sh = sb.shadow;
    sh.armed = false;
    if (sh.clobbered || sb.ring->WrapGeneration() != sh.wrap) {
        ++sb.st.shadow_skip;
        LOG_CRITICAL(Render_Vulkan, "[CSBATCH-SHADOW] skip: {} (frame {}, out={:#x} n={})",
                     sh.clobbered ? "guest output rewritten before the compare" : "ring wrapped",
                     sh.frame, sh.out_base, sh.n);
        return;
    }
    const u64 bytes = u64(sh.n) * 28;
    sb.ring->HostInvalidate(sh.off, bytes);
    std::vector<u32> guest(bytes / sizeof(u32));
    if (!buffer_cache.PeekGuestBytes(sh.out_base, guest.data(), bytes)) {
        ++sb.st.shadow_skip;
        LOG_CRITICAL(Render_Vulkan, "[CSBATCH-SHADOW] skip: guest output {:#x}+{:#x} unreadable",
                     sh.out_base, bytes);
        return;
    }
    const u32* kern = reinterpret_cast<const u32*>(sb.ring->mapped_data.data() + sh.off);
    u32 bad = 0;
    u32 first_v = 0, first_dw = 0, first_g = 0, first_k = 0;
    for (u32 v = 0; v < sh.n; ++v) {
        for (u32 w = 0; w < 7; ++w) {
            const u32 g = guest[v * 7 + w];
            const u32 k = kern[v * 7 + w];
            // dwords 0-2: position f32 (<= 2 ULP: FMA contraction class); 3-6: packed halves
            // (<= 1 ULP each half); dword 6's high half is the tangent sign passed through.
            const bool ok = w < 3 ? ShadowF32Close(g, k, 2)
                                  : (ShadowF16Close(g & 0xffffu, k & 0xffffu, 1) &&
                                     ShadowF16Close(g >> 16, k >> 16, 1));
            if (!ok) {
                if (bad == 0) {
                    first_v = v;
                    first_dw = w;
                    first_g = g;
                    first_k = k;
                }
                ++bad;
            }
        }
    }
    if (bad == 0) {
        ++sb.st.shadow_ok;
    } else {
        ++sb.st.shadow_bad;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "[CSBATCH-SHADOW] job n={} bad={} first@v{} dword{} guest={:#x} kernel={:#x} "
                 "(frame {}, out={:#x}, dwords={})",
                 sh.n, bad, first_v, first_dw, first_g, first_k, sh.frame, sh.out_base,
                 u64(sh.n) * 7);
}

// [CSBATCH] 5 s census (CsSplitAccount's timer): window counters, reset after each print.
void Rasterizer::SkinBatchReport() {
    auto& sb = skin_batch;
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds(5);
    auto& s = sb.st;
    const u64 b = std::max<u64>(s.batches, 1);
    LOG_CRITICAL(Render_Vulkan,
                 "[CSBATCH] 5s: batches={} subs={} subs/batch={} (max {}) groups={} verts={} "
                 "barriers/batch={} count0={} null_out={} tick_splits={} straddle={} "
                 "deleted_owner={} ring_wait={} hook_offthread={} chained={} reupload={} "
                 "unstable={} orphan_bars={}",
                 s.batches, s.subs, s.subs / b, s.max_subs, s.groups, s.verts, s.barriers / b,
                 s.count0, s.null_out, s.tick_splits, s.straddle, s.deleted_owner, s.ring_wait,
                 s.hook_offthread.load(std::memory_order_relaxed), s.chained, s.reupload,
                 s.unstable, s.orphan_bars);
    std::string fl;
    for (u32 r = 0; r < u32(SkinBatch::Reason::N); ++r) {
        fl += fmt::format("{}{}={}", r ? " " : "", SkinBatch::ReasonName(SkinBatch::Reason(r)),
                          s.flush[r]);
    }
    LOG_CRITICAL(Render_Vulkan, "[CSBATCH] 5s: flush{{{}}}", fl);
    std::string dl;
    for (u32 d = 0; d < u32(SkinBatch::Decline::N); ++d) {
        dl += fmt::format("{}{}={}", d ? " " : "", SkinBatch::DeclineName(SkinBatch::Decline(d)),
                          s.decl[d]);
    }
    LOG_CRITICAL(Render_Vulkan, "[CSBATCH] 5s: decl{{{}}} shadow{{ok={} bad={} skip={}}}", dl,
                 s.shadow_ok, s.shadow_bad, s.shadow_skip);
    s.ResetWindow();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    // [SLOWPKT] vk phase (indirect dispatch).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhVk};
    RENDERER_TRACE;

    PopPendingTimed(scheduler);
    FlushSkinBatch(SkinBatch::Reason::DispatchIndirect); // [CSBATCH]

    const auto& cs_program = liverpool->GetCsRegs();
    // [CSSPLIT] the group count lives in a GPU buffer: never split, K=1 permutation
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(false);
    if (!pipeline) {
        return;
    }
    if (SkipComputeShader(pipeline->GetStage(Shader::LogicalStage::Compute).pgm_hash)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    {
        Parserprof::PhaseProbe p{Parserprof::PhSched};
        scheduler.EndRendering();
    }
    // [SKINVRAM] same hook as DispatchDirect: the ND job system may issue the skinning job
    // through either packet (DUMPDISP is a Warning line, absent from the stand's logs).
    {
        const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
        if (VideoCore::BufferCache::SkinVramWanted(cs.pgm_hash)) {
            buffer_cache.SkinVramRedirect(cs.pgm_hash, cs.ud_mask.mask, push_data.ud_regs.data(),
                                          buffer_barriers);
        }
    }
    {
        Parserprof::PhaseProbe p{Parserprof::PhBarr};
        Parserprof::t_pkt_barriers += static_cast<u32>(buffer_barriers.size());
        Parserprof::t_pkt_writes += static_cast<u32>(set_writes.size());
        Parserprof::g_bar_calls.fetch_add(1, std::memory_order_relaxed);
        Parserprof::g_bar_buffers.fetch_add(buffer_barriers.size(), std::memory_order_relaxed);
        Parserprof::g_desc_writes.fetch_add(set_writes.size(), std::memory_order_relaxed);
        CountRedundantDescriptors(set_writes);
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    {
        Parserprof::PhaseProbe p{Parserprof::PhBind};
        ++Parserprof::t_pkt_binds;
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    }
    RecordDispatch(pipeline->GetStage(Shader::LogicalStage::Compute).pgm_hash, 0, 0, 0);
    RecordDispatchTick(scheduler.CurrentTick(),
                       pipeline->GetStage(Shader::LogicalStage::Compute).pgm_hash, 0, 0, 0);
    if (instance.HasNvCheckpoints()) {
        cmdbuf.setCheckpointNV(reinterpret_cast<const void*>(
            pipeline->GetStage(Shader::LogicalStage::Compute).pgm_hash));
    }
    const u32 gpuprof_q =
        GpuProfInit(instance)
            ? GpuProfBegin(cmdbuf, scheduler.CurrentTick(),
                           pipeline->GetStage(Shader::LogicalStage::Compute).pgm_hash, 0, 3)
            : UINT32_MAX;
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    if (pipeline_cache.ComputeSplitListed()) {
        // [CSSPLIT] a listed shader that reaches the rasterizer through indirect args (real PM4
        // DispatchIndirect, or the ASC re-issue of a DMA'd DispatchDirect) is counted, not split
        CsSplitAccount(pipeline->GetStage(Shader::LogicalStage::Compute).pgm_hash, 0, 1, true);
    }
    buffer_cache.SkinVramStamp(scheduler.CurrentTick()); // [SKINVRAM]
    GpuProfEnd(cmdbuf, gpuprof_q);
    DebugState.IncDispatch();

    ResetBindings();
}

u64 Rasterizer::Flush() {
    // [эксперимент] SHADPS4_EXPO_POKE=<hex-адрес>: каждый сабмит пишет 1.0f
    // в буфер экспозиции (гостевая память + инвалидация кэша). Разрывает
    // спящую авто-экспозицию ND, из-за которой кадр умножается в ~0.01.
    static const u64 poke_addr = [] {
        const char* e = std::getenv("SHADPS4_EXPO_POKE");
        return e ? std::strtoull(e, nullptr, 16) : 0ULL;
    }();
    if (poke_addr != 0) {
        static float ones[1024];
        static const bool s_init = [] {
            std::fill_n(ones, 1024, 1.0f);
            return true;
        }();
        (void)s_init;
        Core::Memory::Instance()->TryWriteBacking(reinterpret_cast<u8*>(poke_addr), ones,
                                                  sizeof(ones));
        buffer_cache.InvalidateMemory(poke_addr, sizeof(ones));
    }

    FlushSkinBatch(SkinBatch::Reason::Submit); // [CSBATCH] explicit; the hook duplicates it
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    FlushSkinBatch(SkinBatch::Reason::Submit); // [CSBATCH]
    scheduler.Finish();
}

void Rasterizer::ProcessFaultBufferIfPending() {
    // [EOPRETIRE] also called at the EOP/ReleaseMem sites before the deferred fence: the
    // BDA-journal replay op must be queued (on this tick) ahead of the label, or the guest
    // reads its results before they land (2/2 loads died on "(memSize & 0xffff0000) == 0").
    if (fault_process_pending) {
        fault_process_pending = false;
        FlushSkinBatch(SkinBatch::Reason::Fault); // [CSBATCH] the replay op FindBuffers
        buffer_cache.ProcessFaultBuffer();
    }
}

void Rasterizer::OnSubmit() {
    // [CSBATCH] before the prefetch/GC steps (Create/DeleteBuffer) and the fault parser
    FlushSkinBatch(SkinBatch::Reason::Submit);
    // [FRAMECHAIN] Each step into g_fc_os_step_ns[0..6]; the whole call is timed by the caller
    // (liverpool.cpp, submit_done). FcTimer reads no clock unless SHADPS4_PARSERPROF is set.
    {
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[0]};
        GpuProfCollect(scheduler);
    }
    {
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[1]};
        buffer_cache.BackimpPrefetchStep();
    }
    {
        // [BYWRITE] Heuristic misses are corrected here, at a submit boundary. No-op in every
        // other mode: the queue only ever receives entries when SHADPS4_BACKIMP_MODE=bywrite.
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[2]};
        buffer_cache.SnaproRefreshStep();
        buffer_cache.BywriteCorrectionStep();
        buffer_cache.PromoteLabelRegions();
    }
    {
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[3]};
        ProcessFaultBufferIfPending();
    }
    {
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[4]};
        texture_cache.ProcessDownloadImages();
    }
    {
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[5]};
        texture_cache.RunGarbageCollector();
    }
    {
        Parserprof::FcTimer t{Parserprof::g_fc_os_step_ns[6]};
        buffer_cache.RunGarbageCollector();
    }
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    // [SLOWPKT] desc phase; the tex/buf work it calls into is subtracted by nested probes.
    Parserprof::PhaseProbe slow_probe{Parserprof::PhDesc};
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        SyncDmaZones(); // [CSBATCH] extracted (also run per append of the batched worker)
        fault_process_pending = true;
    }

    return true;
}

void Rasterizer::SyncDmaZones() {
    // [FRAMECHAIN] PhDma: the whole DMA block, nested in PhDesc (telescopic probe), so
    // `desc` no longer carries it. Armed only inside a phased packet with PARSERPROF on.
    Parserprof::PhaseProbe dma_probe{Parserprof::PhDma};
    // Every DMA-using draw used to re-synchronize *all* mapped guest memory, which
    // costs tens of thousands of page-protection syscalls per second in gameplay.
    // The sweep exists to publish CPU writes to shaders that read guest memory
    // directly; doing it once per frame is enough, since a write racing the GPU
    // inside a frame is unordered anyway. Set SHADPS4_DMASYNC=draw for the old
    // per-draw behaviour.
    static const bool sync_every_draw = [] {
        const char* e = std::getenv("SHADPS4_DMASYNC");
        return e && std::string_view{e} == "draw";
    }();
    const u64 flip_no = g_dbg_frame_no.load();
    // [DMASYNC-ZONE] SHADPS4_DMASYNC_ZONE=<hex base>:<hex size>[,...]: the job-system
    // ranges the GPU workers poll and consume INSIDE a frame (the CPU pushes a job or a
    // flag, a running worker waits for it). The once-per-frame sweep leaves such a write
    // invisible to the GPU until the next flip: the worker spins a whole frame (the "x2"
    // dispatch time) or reads last frame's input. Sync just those ranges on every DMA
    // dispatch; the full sweep stays once per frame.
    static const std::vector<std::pair<VAddr, u64>> dma_zone = [] {
        std::vector<std::pair<VAddr, u64>> out;
        const char* e = std::getenv("SHADPS4_DMASYNC_ZONE");
        const char* cursor = e ? e : "";
        while (*cursor != '\0') {
            unsigned long long b = 0, sz = 0;
            int consumed = 0;
            if (std::sscanf(cursor, "%llx:%llx%n", &b, &sz, &consumed) == 2 && b != 0 && sz != 0) {
                out.emplace_back(static_cast<VAddr>(b), static_cast<u64>(sz));
                cursor += consumed;
            } else {
                break;
            }
            if (*cursor == ',') {
                ++cursor;
            }
        }
        if (!out.empty()) {
            LOG_CRITICAL(Render_Vulkan, "DMASYNC-ZONE: armed, {} ranges, first {:#x}", out.size(),
                         out.front().first);
        }
        return out;
    }();
    if (sync_every_draw || dma_sync_stamp != flip_no) {
        dma_sync_stamp = flip_no;
        if (Parserprof::Enabled()) {
            Parserprof::g_dma_sweeps.fetch_add(1, std::memory_order_relaxed);
        }
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
    } else if (!dma_zone.empty()) {
        static u64 zone_syncs = 0;
        static auto zone_next = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (const auto& [zb, zs] : dma_zone) {
            buffer_cache.SynchronizeBuffersInRange(zb, zs);
        }
        ++zone_syncs;
        if (Parserprof::Enabled()) {
            Parserprof::g_dma_zone_runs.fetch_add(1, std::memory_order_relaxed);
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= zone_next) {
            zone_next = now + std::chrono::seconds(10);
            LOG_CRITICAL(Render_Vulkan, "DMASYNC-ZONE: {} zone syncs so far", zone_syncs);
        }
    }
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // [SKINDIRECT-STORES] a listed shader now carries WRITTEN descriptors, so this
                // heuristic can eat its dispatch. Counted only; behaviour unchanged.
                if (VideoCore::BufferCache::SkinDirectWanted(info.pgm_hash)) {
                    VideoCore::BufferCache::SkinDirectNoteMetaSkip(info.pgm_hash, address);
                }
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    texture_cache.SetGpuModified(dst_image); // [D1] stamps modified_tick
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    texture_cache.SetGpuModified(image1); // [D1] stamps modified_tick
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

// [LISTCOUNT] 0x43b8ee5e (8x8 tile compute, the shader every TDR names) loops
// `for (i = tid; i < count; i += 64)` with count = live dword at
// (user_data[0] | user_data[1] << 32) + 8. A garbage count is a 2-second hang and a device
// loss. Sample what the HOST sees in that dword at bind time plus the page's GPU home, into a
// ring that the DEVFAULT dump prints: a sane host value under a hang means the GPU read a
// different copy (stale VRAM), a garbage host value means the game itself holds garbage.
// SHADPS4_LISTCOUNT=<hash>[:<sgpr_lo>:<sgpr_hi>:<byte_off>], default 43b8ee5e:0:1:8.
namespace {
struct ListCountEntry {
    u64 tick;
    u64 base;
    u32 count;
    u32 home;
    u32 readable;
};
std::array<ListCountEntry, 64> g_listcount_ring{};
std::atomic<u32> g_listcount_pos{0};
std::atomic<u32> g_listcount_total{0};
u64 g_listcount_hash = 0;
u32 g_listcount_lo = 0;
u32 g_listcount_hi = 1;
u32 g_listcount_off = 8;

bool ListCountEnabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("SHADPS4_LISTCOUNT");
        if (!e || !e[0]) {
            return false;
        }
        unsigned long long h = 0, lo = 0, hi = 1, off = 8;
        std::sscanf(e, "%llx:%llu:%llu:%llu", &h, &lo, &hi, &off);
        g_listcount_hash = h;
        g_listcount_lo = static_cast<u32>(lo);
        g_listcount_hi = static_cast<u32>(hi);
        g_listcount_off = static_cast<u32>(off);
        LOG_CRITICAL(Render_Vulkan, "LISTCOUNT: armed for pgm {:#x}, base = ud[{}]|ud[{}]<<32, +{}",
                     g_listcount_hash, g_listcount_lo, g_listcount_hi, g_listcount_off);
        return h != 0;
    }();
    return enabled;
}

void ListCountSample(VideoCore::BufferCache& buffer_cache, const Shader::Info& stage, u64 tick) {
    if (!ListCountEnabled() || stage.pgm_hash != g_listcount_hash) {
        return;
    }
    if (stage.user_data.size() <= std::max(g_listcount_lo, g_listcount_hi)) {
        return;
    }
    const u64 base = static_cast<u64>(stage.user_data[g_listcount_lo]) |
                     (static_cast<u64>(stage.user_data[g_listcount_hi]) << 32);
    const u64 addr = base + g_listcount_off;
    u32 count = 0;
    u32 home = 0;
    // A protected (GPU-owned) page answers "not readable" instead of taking the fault path,
    // which would readback as a side effect of a diagnostic.
    const u32 readable = buffer_cache.PeekGuestDword(addr, &count, &home);
    const u32 pos =
        g_listcount_pos.fetch_add(1, std::memory_order_relaxed) % g_listcount_ring.size();
    g_listcount_ring[pos] = ListCountEntry{tick, base, count, home, readable};
    const u32 total = g_listcount_total.fetch_add(1, std::memory_order_relaxed) + 1;
    // Loud on anything absurd, quiet otherwise (one line per 2000 binds as a heartbeat).
    // home 1 (a VRAM copy) is the condition under which the GPU can read stale data: loud too.
    if ((readable && count > 0x10000) || !readable || home == 1 || total % 2000 == 1) {
        LOG_CRITICAL(Render_Vulkan,
                     "LISTCOUNT: pgm={:#x} bind#{} tick={} base={:#x} count={} ({:#x}) home={} "
                     "readable={}",
                     stage.pgm_hash, total, tick, base, count, count, home, readable);
    }
}
} // namespace

void DumpListCountRing() {
    if (!ListCountEnabled()) {
        return;
    }
    const u32 total = g_listcount_total.load(std::memory_order_relaxed);
    const u32 n = std::min<u32>(total, static_cast<u32>(g_listcount_ring.size()));
    LOG_CRITICAL(Render_Vulkan, "LISTCOUNT-RING: {} binds total, last {} (oldest first):", total,
                 n);
    const u32 end = g_listcount_pos.load(std::memory_order_relaxed);
    for (u32 i = 0; i < n; ++i) {
        const auto& e =
            g_listcount_ring[(end + g_listcount_ring.size() - n + i) % g_listcount_ring.size()];
        LOG_CRITICAL(Render_Vulkan,
                     "LISTCOUNT-RING[{}]: tick={} base={:#x} count={} ({:#x}) home={} readable={}",
                     i, e.tick, e.base, e.count, e.count, e.home, e.readable);
    }
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    buffer_bindings.clear();
    // [DESCDUMP] SHADPS4_DESCDUMP=<hex hash>[,...]: the ND workers read their buffer
    // descriptors from a block in guest memory (ud_0|ud_1<<32; the skinning worker 0x8f067540
    // takes V# bases from dwords 16/17, 20/21, 24/25 and its vertex count from ud_2|ud_3), so
    // the recompiler sees raw pointers. Dump the block and where each base page lives
    // (PageHomeKind: 0 none, 1 VRAM buffer, 2 import-backed, 3 backing-backed, 4 PT->guest RAM).
    {
        static const std::vector<u64> dd_wanted = [] {
            std::vector<u64> out;
            const char* e = std::getenv("SHADPS4_DESCDUMP");
            const char* cursor = e ? e : "";
            while (*cursor != '\0') {
                char* end = nullptr;
                const u64 v = std::strtoull(cursor, &end, 16);
                if (end == cursor) {
                    break;
                }
                if (v != 0) {
                    out.push_back(v);
                }
                cursor = (*end == ',') ? end + 1 : end;
            }
            return out;
        }();
        if (!dd_wanted.empty() && stage.user_data.size() >= 4 &&
            std::find(dd_wanted.begin(), dd_wanted.end(), stage.pgm_hash) != dd_wanted.end()) {
            static std::unordered_map<u64, u32> dd_n; // parser thread only
            const u32 k = dd_n[stage.pgm_hash]++;
            if (k < 24 || k % 2000 == 0) {
                const u64 blk = u64(stage.user_data[0]) | (u64(stage.user_data[1]) << 32);
                const u64 cnt = u64(stage.user_data[2]) | (u64(stage.user_data[3]) << 32);
                u32 d[32]{};
                u32 home_blk = 0;
                for (u32 i = 0; i < 32; i++) {
                    u32 home = 0;
                    buffer_cache.PeekGuestDword(blk + i * 4, &d[i], &home);
                    if (i == 0) {
                        home_blk = home;
                    }
                }
                u32 cnt_v = 0, cnt_home = 0;
                buffer_cache.PeekGuestDword(cnt, &cnt_v, &cnt_home);
                LOG_CRITICAL(Render_Vulkan,
                             "DESCDUMP #{} pgm={:#x} blk={:#x} home={} cnt@{:#x}={} home={} | "
                             "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} | "
                             "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} | "
                             "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} | "
                             "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
                             k, stage.pgm_hash, blk, home_blk, cnt, cnt_v, cnt_home, d[0], d[1],
                             d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12],
                             d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22],
                             d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]);
                for (const u32 w : {16u, 20u, 24u}) {
                    const u64 base = u64(d[w]) | (u64(d[w + 1] & 0xfffu) << 32);
                    const u32 stride = (d[w + 1] >> 16) & 0x3fffu;
                    const u32 records = d[w + 2];
                    LOG_CRITICAL(
                        Render_Vulkan,
                        "DESCDUMP #{} pgm={:#x} v#@{}: base={:#x} stride={} records={} "
                        "size={:#x} home={} home_end={}",
                        k, stage.pgm_hash, w, base, stride, records,
                        u64(records) * (stride ? stride : 1u), buffer_cache.PageHomeKind(base),
                        buffer_cache.PageHomeKind(base + u64(records) * (stride ? stride : 1u)));
                }
            }
        }
    }
    // [HOMEPOLICY] name the binder in CreateBuffer's union log (0 outside binds)
    VideoCore::g_bind_pgm_hash = stage.pgm_hash;
    struct BinderReset {
        ~BinderReset() {
            VideoCore::g_bind_pgm_hash = 0;
        }
    } binder_reset;
    ListCountSample(buffer_cache, stage, scheduler.CurrentTick());

    // [SKINDIRECT] SHADPS4_SKINDIRECT: the recompiler declined the BDA rewrite for this shader's
    // bounded skinning inputs, so they arrive here as ordinary descriptors. Nothing below
    // changes for them - this only classifies how each one resolved (bound / null / GPU-modified
    // / oversize) into the 5 s report, so a visual regression has a number attached.
    const bool skind_wanted = VideoCore::BufferCache::SkinDirectWanted(stage.pgm_hash);
    // [SKINDIRECT-STORES] first-pass verdict per descriptor, consumed by the second pass
    // (0 none, 1 converted read, 2 converted store). buffer_bindings gets exactly one entry per
    // stage.buffers element in both branches below, so index i is aligned across the passes.
    std::array<u8, Shader::NUM_BUFFERS> sd_kind{};
    u32 sd_i = 0;

    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
        u8 kind = 0;
        if (skind_wanted && !desc.IsSpecial()) {
            kind = buffer_cache.SkinDirectNote(stage.pgm_hash, desc.sharp_idx, vsharp.base_address,
                                               vsharp.GetSize(), desc.is_written);
        }
        sd_kind[sd_i++] = kind;
        // [debug] мусорный размер буфера из V#: ловим виновника чёрного экрана
        static const bool s_sus_log = std::getenv("SHADPS4_SUSLOG") != nullptr;
        if (s_sus_log && vsharp.base_address != 0 && vsharp.GetSize() >= 0xF0000000u) {
            LOG_WARNING(Render_Vulkan,
                        "SUS VSHARP: shader={:#x} base={:#x} num_records={:#x} stride={} "
                        "buffer_type={} storage={}",
                        stage.pgm_hash, vsharp.base_address, vsharp.num_records, vsharp.GetStride(),
                        (u32)desc.buffer_type, desc.IsStorage(vsharp));
        }
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            // [FRAMECHAIN] ClampRangeSize for the unbounded (>= 1 GiB) V#s only, to size
            // measure #10; the bounded ones are cheap and would only dilute the average.
            const bool fc_clamp = vsharp.GetSize() >= 0x40000000u && Parserprof::Enabled();
            const s64 fc_c0 = fc_clamp ? Parserprof::NowNs() : 0;
            u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            if (fc_clamp) {
                const s64 fc_c1 = Parserprof::NowNs();
                Parserprof::g_fc_clamp_ns.fetch_add(
                    fc_c1 > fc_c0 ? static_cast<u64>(fc_c1 - fc_c0) : 0, std::memory_order_relaxed);
                Parserprof::g_fc_clamp_n.fetch_add(1, std::memory_order_relaxed);
            }
            // ObtainBuffer takes the size as u32, so an unbounded descriptor spanning
            // more than 4 GiB silently wraps to a bogus (often tiny) size and the pages
            // past the wrap never get a page-table entry. Clamp before it can happen.
            if (size > std::numeric_limits<u32>::max()) {
                static bool s_warned = false;
                if (!s_warned) {
                    s_warned = true;
                    LOG_WARNING(Render_Vulkan, "SIZECLAMP: descriptor size {:#x} clamped to 4 GiB",
                                size);
                }
                size = std::numeric_limits<u32>::max() & ~u64{0xFFF};
            }
            // [PROBECOST 10.09] SLOWFIND used to read the clock twice for EVERY bound buffer
            // (thousands per frame in the jungle) just to catch a >3 ms outlier. Behind
            // SHADPS4_SLOWFIND=1 now; without it the bind path does no timing at all.
            static const bool s_slowfind = std::getenv("SHADPS4_SLOWFIND") != nullptr;
            const auto t0 = s_slowfind ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            if (s_slowfind) {
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
                if (us > 3000) {
                    LOG_WARNING(Render_Vulkan, "SLOWFIND: size={:#x} took {}us shader={:#x}", size,
                                us, stage.pgm_hash);
                }
            }
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            // Unbounded V# (num_records=0xffffffff = «указатель на всю память», паттерн
            // ND-движка): НЕ помечать весь клампнутый диапазон (гигабайты) как
            // GPU-modified и не гонять write-синк — это держит спинлоки сотен
            // региональных менеджеров на время аплоада (фолт-шторм на ~5 ядер),
            // отравляет IsRegionGpuModified для всего heap и убивает stream
            // fast-path остальных буферов. Чтения шейдера остаются корректными:
            // dirty-страницы по-прежнему заливаются read-синком.
            const bool unbounded_v = vsharp.num_records == 0xffffffffu && size >= 0x40000000u;
            // Compute dispatches poll ND's job queues through these unbounded
            // descriptors, and the queue payload can live anywhere in the
            // range: the bounded 16 MiB sync window that keeps draw-time
            // bindings cheap starves the GPU-side job workers of their input
            // (observed as an endless worker spin during level loads). For
            // compute, sync every dirty page of the range instead.
            const bool full_sync_unbounded = unbounded_v && stage.stage == Shader::Stage::Compute;
            // [UNBPERBASE] SHADPS4_UNBOUNDED_PERBASE=1: ObtainBuffer's 16 MiB window sync for
            // unbounded V#s is stamped ONCE PER FLIP for the whole cache (unbounded_sync_stamp),
            // so the first unbounded descriptor of a frame gets its window synced and every other
            // unbounded descriptor with a different base in the same frame is never synced -
            // CPU-simulated meshes (ND job output: water, cloth, debris) are then drawn from stale
            // VRAM copies: flat near water, stretched polygons (06.09). Sync each distinct base
            // once per flip here, via the same range sync DMASYNC_ZONE uses.
            static const bool unb_per_base = [] {
                const char* e = std::getenv("SHADPS4_UNBOUNDED_PERBASE");
                return e && e[0] == '1';
            }();
            if (unb_per_base && unbounded_v && !full_sync_unbounded) {
                static u64 unb_flip = ~0ull;
                static std::vector<u64> unb_bases;
                static u64 unb_syncs = 0;
                const u64 flip = g_dbg_frame_no.load(std::memory_order_relaxed);
                if (unb_flip != flip) {
                    unb_flip = flip;
                    unb_bases.clear();
                }
                if (std::find(unb_bases.begin(), unb_bases.end(), vsharp.base_address) ==
                    unb_bases.end()) {
                    unb_bases.push_back(vsharp.base_address);
                    buffer_cache.SynchronizeBuffersInRange(vsharp.base_address, 16u << 20);
                    ++unb_syncs;
                    if (unb_syncs <= 8 || unb_syncs % 4096 == 0) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "UNBPERBASE: synced base {:#x} (distinct bases this flip {}, "
                                     "total {})",
                                     vsharp.base_address, unb_bases.size(), unb_syncs);
                    }
                }
            }
            // [PROBECOST 10.09] read the clock only when SHADPS4_DRAWPROF asks for OBTPROF.
            static const bool s_drawprof_t0 = std::getenv("SHADPS4_DRAWPROF") != nullptr;
            const auto to0 = s_drawprof_t0 ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
            // Marking unbounded writes as GPU-modified was tried (with the marked
            // window capped inside ObtainBuffer) to get the GPU's results back into
            // guest memory. It changed nothing visually and cut gameplay survival
            // from ~6 minutes to ~2, so the descriptor stays unmarked.
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written && !unbounded_v, desc.is_formatted,
                buffer_id, unbounded_v && !full_sync_unbounded ? (16u << 20) : 0u);
            // [BUFDUMP] SHADPS4_BUFDUMP=<hex hash>[,...]: where do this shader's buffers live?
            // (stand diagnostic: import-backed = guest memory over PCIe, else VRAM)
            {
                static const std::vector<u64> bd_wanted = [] {
                    std::vector<u64> out;
                    const char* e = std::getenv("SHADPS4_BUFDUMP");
                    const char* cursor = e ? e : "";
                    while (*cursor != '\0') {
                        char* end = nullptr;
                        const u64 v = std::strtoull(cursor, &end, 16);
                        if (end == cursor) {
                            break;
                        }
                        if (v != 0) {
                            out.push_back(v);
                        }
                        cursor = (*end == ',') ? end + 1 : end;
                    }
                    return out;
                }();
                if (!bd_wanted.empty() && std::find(bd_wanted.begin(), bd_wanted.end(),
                                                    stage.pgm_hash) != bd_wanted.end()) {
                    static std::unordered_map<u64, u32>
                        bd_logged; // per shader (parser thread only)
                    if (bd_logged[stage.pgm_hash]++ < 120) {
                        LOG_CRITICAL(
                            Render_Vulkan,
                            "BUFDUMP: pgm={:#x} slot={} base={:#x} size={:#x} written={} "
                            "unbounded={} import_backed={} backing_backed={} buf_cpu={:#x} "
                            "buf_size={:#x}",
                            stage.pgm_hash, i, static_cast<u64>(vsharp.base_address), size,
                            desc.is_written, unbounded_v, vk_buffer->is_import_backed,
                            vk_buffer->is_backing_backed, vk_buffer->CpuAddr(),
                            vk_buffer->SizeBytes());
                    }
                }
            }
            {
                static const bool s_drawprof2 = std::getenv("SHADPS4_DRAWPROF") != nullptr;
                if (s_drawprof2) {
                    const auto us_ob = std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - to0)
                                           .count();
                    if (us_ob > 2000) {
                        LOG_WARNING(Render_Vulkan, "OBTPROF: size={:#x} written={} took {}us", size,
                                    desc.is_written, us_ob);
                    }
                }
            }
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            u32 adjust = offset - offset_aligned;
            if (adjust % 4 != 0) {
                // The guest may point a descriptor at a byte-aligned address, but the offset
                // handed to the shader counts dwords, so a remainder cannot be expressed.
                // Killing the run over it is worse than reading from a few bytes earlier:
                // report it once and carry on.
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    LOG_ERROR(Render_Vulkan,
                              "Unaligned buffer binding: offset={:#x} alignment={:#x} "
                              "remainder={} - rounding down",
                              offset, alignment, adjust % 4);
                }
                adjust &= ~3u;
            }
            push_data.AddOffset(binding.buffer, adjust);
            // A descriptor may never claim more bytes than the buffer actually holds: robustness
            // bounds-checks against the DECLARED range, so an inflated one lets the shader read
            // and write past the end of the allocation - which is exactly what the device fault
            // reports (an address just outside a live buffer).
            {
                const u64 declared_end = u64{offset_aligned} + u64{size} + u64{adjust};
                if (declared_end > vk_buffer->SizeBytes()) {
                    static u32 reported = 0;
                    if (reported < 16) {
                        ++reported;
                        LOG_CRITICAL(Render_Vulkan,
                                     "DESCOOB: guest={:#x} buffer_size={:#x} offset={:#x} "
                                     "range={:#x} past_end={:#x} unbounded={} written={}",
                                     vsharp.base_address, vk_buffer->SizeBytes(), offset_aligned,
                                     size + adjust, declared_end - vk_buffer->SizeBytes(),
                                     u32(unbounded_v), u32(desc.is_written));
                    }
                }
            }
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            // [PRECISE-STAGES] SHADPS4_PRECISE_STAGES=1: the barrier's destination is the stage
            // that will touch the buffer (compute or the graphics stages), not ALL_COMMANDS. The
            // hazard is covered exactly the same; what goes away is waiting for every earlier
            // command (a full drain, ~25 us) before each worker dispatch (liveloop104: 107-196
            // such barriers per frame = 2.8-4.6 ms of GPU time).
            static const bool precise_stages = [] {
                const bool v = VideoCore::Buffer::PreciseStages();
                if (v) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "PRECISE-STAGES: buffer barriers target the using stage");
                }
                return v;
            }();
            const vk::PipelineStageFlagBits2 dst_stage =
                !precise_stages                         ? vk::PipelineStageFlagBits2::eAllCommands
                : stage.stage == Shader::Stage::Compute ? vk::PipelineStageFlagBits2::eComputeShader
                                                        : vk::PipelineStageFlagBits2::eAllGraphics;
            const bool sd_store = sd_kind[i] == 2;
            auto barrier = vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                                 : vk::AccessFlagBits2::eShaderRead,
                                                 dst_stage);
            if (barrier) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (sd_store) {
                // [SKINDIRECT-STORES] where the converted output landed (owner flags, PT hit,
                // barrier). After ObtainBuffer, so the first touch already sees its owner.
                buffer_cache.SkinDirectNoteStoreBind(vsharp.base_address, size, *vk_buffer,
                                                     unbounded_v, barrier.has_value());
            }
            if (desc.is_written && desc.is_formatted && !unbounded_v) {
                // [SKINDIRECT-STORES] the vertex ring aliases no image; the range lookup is 3
                // calls per dispatch of the worker. Kept by default
                // (SHADPS4_SKINDIRECT_STORE_IMGINV=0 skips it, counted).
                if (sd_store && !VideoCore::BufferCache::SkinDirectStoreImgInv()) {
                    buffer_cache.SkinDirectNoteImgInvSkipped();
                } else {
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    // [FINDLOG] tag every FindImage of a watched shader (SHADPS4_BINDLOG) for the resolution
    // ring in the texture cache; cleared when this stage's binding is done.
    struct FindLogTag {
        VideoCore::TextureCache& tc;
        bool on;
        ~FindLogTag() {
            if (on) {
                tc.SetBindLogTag(0);
            }
        }
    } findlog_tag{texture_cache, BindLog::Wants(stage.pgm_hash)};
    if (findlog_tag.on) {
        texture_cache.SetBindLogTag(stage.pgm_hash);
    }
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            // The placeholder must carry the shader's storage/texture kind: the pipeline
            // layout is built from the shader, so a default-constructed (texture)
            // descriptor written into a storage slot is an invalid descriptor write.
            auto& [null_image_id, null_desc] =
                image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            null_desc.type = image_desc.is_written ? VideoCore::TextureCache::BindingType::Storage
                                                   : VideoCore::TextureCache::BindingType::Texture;
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            if (BindLog::Wants(stage.pgm_hash)) {
                BindLog::Note(BindLog::Entry{
                    .tick = scheduler.CurrentTick(),
                    .pgm = stage.pgm_hash,
                    .tsharp = desc.info.guest_address,
                    .guest = image.info.guest_address,
                    .guest_size = image.info.guest_size,
                    .img = reinterpret_cast<u64>(static_cast<VkImage>(image.GetImage())),
                    .view = reinterpret_cast<u64>(static_cast<VkImageView>(*image_view.image_view)),
                    .rt0 = g_rt_image[0],
                    .slot = image_id.index,
                    .w = image.info.size.width,
                    .h = image.info.size.height,
                    .idx = static_cast<u32>(bound_images.size() - 1),
                });
            }

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        if (cb < 8) {
            g_rt_image[cb] = reinterpret_cast<u64>(static_cast<VkImage>(image->GetImage()));
        }
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        // These flags are only ever set below, never cleared, so a draw that once had a
        // stencil buffer left every later depth-only draw claiming a stencil attachment -
        // handing the driver a stencil attachment whose view has no stencil aspect.
        attachment.has_depth = false;
        attachment.has_stencil = false;
        attachment.depth_clear = false;
        attachment.stencil_clear = false;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        // The guest register can claim a stencil buffer while the depth image has no stencil
        // aspect at all (D16 here). Attaching it anyway hands the driver a stencil attachment
        // whose view has no stencil, which the validation layer rejects and the driver may
        // fault on, so trust the image format rather than the register alone.
        // The guest register can claim a stencil buffer while the image carries none: the
        // requested depth+stencil format may be unsupported (D16_UNORM_S8_UINT is absent on
        // NVIDIA) and fall back to a depth-only one. Attaching a stencil attachment whose
        // view has no stencil aspect is invalid and can take the device down, so decide by
        // the format actually in use.
        // The attachment uses the VIEW, and a view can narrow a depth+stencil image to a
        // depth-only format; decide by what the attachment actually carries.
        const auto depth_format = image_view.info.format;
        const bool format_has_stencil = depth_format == vk::Format::eD16UnormS8Uint ||
                                        depth_format == vk::Format::eD24UnormS8Uint ||
                                        depth_format == vk::Format::eD32SfloatS8Uint ||
                                        depth_format == vk::Format::eS8Uint;
        if (regs.depth_buffer.StencilValid() && format_has_stencil) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    FlushSkinBatch(SkinBatch::Reason::Dma); // [CSBATCH]
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    FlushSkinBatch(SkinBatch::Reason::Dma); // [CSBATCH]
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    // [experiment] SHADPS4_HONESTFAULT=1: reporting a fault as handled when nothing was
    // unprotected turns a genuine guest access violation into an endless retry - the
    // instruction re-executes, faults again, and the game freezes at full frame rate instead
    // of crashing. Claim the fault only when a buffer really owns the page.
    static const bool honest = [] {
        const char* e = std::getenv("SHADPS4_HONESTFAULT");
        return e && e[0] == '1';
    }();
    // Both caches must be asked: a page tracked only by the texture cache has no buffer behind
    // it, and refusing that fault turns a legitimate readback into a crash (measured: survival
    // dropped from ~90 s to 11 s when only buffers were consulted).
    if (honest && !buffer_cache.HasBufferAt(addr) && !texture_cache.HasImageAt(addr)) {
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    FlushSkinBatch(SkinBatch::Reason::Readback); // [CSBATCH]
    texture_cache.ProcessDownloadImages();
}

void Rasterizer::DeferSignalToRetire(std::function<void()>&& func) {
    FlushSkinBatch(SkinBatch::Reason::Label); // [CSBATCH] the fence must retire after the batch
    scheduler.DeferPriorityOperation(std::move(func));
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
    buffer_cache.ImportHostRegion(addr, size);
    buffer_cache.OnGuestMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.ReleaseLabelRegion(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    buffer_cache.ReleaseHostRegion(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
