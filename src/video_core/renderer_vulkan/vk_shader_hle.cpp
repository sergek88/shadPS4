// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <string>
#include <string_view>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

// [CSBATCH] Defined in vk_shader_util.cpp next to Compile(): the same glslang path with the
// SPIR-V target selectable (1.3 like every host shader, or 1.5 as the buffer_reference fallback).
// Declared here because vk_shader_util.h is not part of the patch set.
vk::ShaderModule CompileWithSpvTarget(std::string_view code, vk::ShaderStageFlagBits stage,
                                      vk::Device device, std::vector<std::string> defines,
                                      bool spv15);

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;

// [CSBATCH] Host kernel for the ND skinning worker 0x8f067540 (Uncharted, CUSA02320). One
// dispatch per batch of guest dispatches; each host workgroup finds its job through group_map.
// Math transcribed 1:1 from spv/cs_8f067540.dis (%341-%819): operation order kept so results
// match the recompiled guest module up to FMA contraction. Defines from Vulkan::Compile:
// WG (64), ROBUST (1), PAL_VEC4 (1). Kept as a source string (compiled by glslang at start-up
// like every host shader) so WG/ROBUST/PAL_VEC4 are A/B knobs without a rebuild.
static constexpr std::string_view kSkinBatchComp = R"glsl(
#version 450
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_buffer_reference : require

layout(local_size_x = WG, local_size_y = 1, local_size_z = 1) in;

struct Job {                 // 88 B, std430; mirrors Vulkan::SkinJobGpu
    uint64_t out_pos;        // block dw 0/1  -> base + i*28 + {0,4,8}  f32 x3
    uint64_t out_nrm;        // block dw 4/5  -> base + i*28 + {0,4}    half2 x2 (= out_pos+12)
    uint64_t out_tan;        // block dw 8/9  -> base + i*28 + {0,4}    half2 x2 (= out_pos+20)
    uint64_t hdr;            // block dw 16/17: {K, byte_off} per vertex
    uint64_t infl;           // block dw 20/21: packed dwords (bone<<22 | w22)
    uint64_t pal;            // block dw 24/25: 48 B per bone = 3 rows of vec4 (row.w = translation)
    uint64_t pos;            // user-data V# 0: 12 B per vertex
    uint64_t nrm;            // user-data V# 1: 8 B (half4)
    uint64_t tan;            // user-data V# 2: 8 B (half4)
    uint     count;          // min(count, dim_x*1024)
    uint     infl_dw;        // dwords behind infl (ROBUST)
    uint     pal_dw;         // dwords behind pal  (ROBUST), >= 12
    uint     pad;
};
layout(std430, binding = 0) restrict readonly buffer Table {
    uint n_jobs; uint n_groups; uint tpad0; uint tpad1;
    Job jobs[];
};
layout(std430, binding = 1) restrict readonly buffer GroupMap {
    uint group_map[];        // (job << 20) | group_in_job
};
layout(buffer_reference, std430, buffer_reference_align = 4)  restrict readonly  buffer RoU32 { uint  d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4)  restrict readonly  buffer RoF32 { float f[]; };
#if PAL_VEC4
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly  buffer RoPal { uvec4 r[]; };
#else
layout(buffer_reference, std430, buffer_reference_align = 4)  restrict readonly  buffer RoPal { uint  d[]; };
#endif
layout(buffer_reference, std430, buffer_reference_align = 4)  restrict writeonly buffer WoU32 { uint  d[]; };

uint load_infl(RoU32 I, uint i, uint n) {
#if ROBUST
    const uint v = I.d[min(i, n - 1u)];   // in-bounds load, masked result: same as robustBufferAccess2 today
    return i < n ? v : 0u;
#else
    return I.d[i];
#endif
}

void load_rows(RoPal P, uint bone, uint nbones, out vec4 r0, out vec4 r1, out vec4 r2) {
#if ROBUST
    const bool ok = bone < nbones;
    bone = min(bone, nbones - 1u);
#endif
#if PAL_VEC4
    const uint b = bone * 3u;
    uvec4 u0 = P.r[b], u1 = P.r[b + 1u], u2 = P.r[b + 2u];
#else
    const uint b = bone * 12u;
    uvec4 u0 = uvec4(P.d[b],      P.d[b + 1u], P.d[b + 2u],  P.d[b + 3u]);
    uvec4 u1 = uvec4(P.d[b + 4u], P.d[b + 5u], P.d[b + 6u],  P.d[b + 7u]);
    uvec4 u2 = uvec4(P.d[b + 8u], P.d[b + 9u], P.d[b + 10u], P.d[b + 11u]);
#endif
#if ROBUST
    u0 = ok ? u0 : uvec4(0u); u1 = ok ? u1 : uvec4(0u); u2 = ok ? u2 : uvec4(0u);
#endif
    r0 = uintBitsToFloat(u0); r1 = uintBitsToFloat(u1); r2 = uintBitsToFloat(u2);
}

void main() {
    const uint g = gl_WorkGroupID.x;
    if (g >= n_groups) return;                          // uniform
    const uint m  = group_map[g];
    const uint j  = m >> 20;
    const uint gi = m & 0xFFFFFu;
    const Job job = jobs[j];                             // one uniform 88 B load per group
    const uint index = gi * WG + gl_LocalInvocationID.x;
    if (index >= job.count) return;                      // guest: count > index (%346)

    RoU32 H = RoU32(job.hdr);
    const uint K        = H.d[index * 2u];               // %369
    const uint byte_off = H.d[index * 2u + 1u];          // %371

    RoF32 P = RoF32(job.pos);
    const float px = P.f[index * 3u], py = P.f[index * 3u + 1u], pz = P.f[index * 3u + 2u];
    RoU32 N = RoU32(job.nrm);
    const vec2 nlo = unpackHalf2x16(N.d[index * 2u]), nhi = unpackHalf2x16(N.d[index * 2u + 1u]);
    const float nx = nlo.x, ny = nlo.y, nz = nhi.x;      // %406: (dw0.lo, dw0.hi, dw1.lo, dw1.hi)
    RoU32 T = RoU32(job.tan);
    const vec2 tlo = unpackHalf2x16(T.d[index * 2u]), thi = unpackHalf2x16(T.d[index * 2u + 1u]);
    const float tx = tlo.x, ty = tlo.y, tz = thi.x, tw = thi.y;

    RoU32 I   = RoU32(job.infl);
    RoPal PAL = RoPal(job.pal);
    const uint nbones = job.pal_dw / 12u;
    const uint i0 = byte_off >> 2;
    // prologue: bone of influence 0, read again in iteration 0 exactly like the guest (%442, %532)
    vec4 r0, r1, r2;
    load_rows(PAL, load_infl(I, i0, job.infl_dw) >> 22, nbones, r0, r1, r2);
    const float p0x = ((r0.x * px + r0.w) + py * r0.y) + r0.z * pz;   // %487 %498 %507
    const float p0y = ((r1.x * px + r1.w) + r1.y * py) + pz * r1.z;   // %491 %501 %510
    const float p0z = ((px * r2.x + r2.w) + py * r2.y) + r2.z * pz;   // %495 %504 %513

    vec3 accp = vec3(0.0), accn = vec3(0.0), acct = vec3(0.0);
    for (uint i = 0u; i < K; ++i) {
        const uint packed = load_infl(I, i0 + i, job.infl_dw);        // %532: infl[(byte_off+4i)>>2]
        vec4 m0, m1, m2;
        load_rows(PAL, packed >> 22, nbones, m0, m1, m2);
        const float w = 2.38418636e-07 * float(packed & 0x3FFFFFu);   // %561
        const float qx = ((px * m0.x + m0.w) + m0.y * py) + pz * m0.z; // %576 %583 %589
        const float qy = ((px * m1.x + m1.w) + m1.y * py) + pz * m1.z; // %580 %586 %599
        const float qz = ((px * m2.x + m2.w) + m2.y * py) + pz * m2.z; // %593 %596 %602
        const vec3 nn = vec3((ny * m0.y + m0.x * nx) + nz * m0.z,      // %612 %615
                             (ny * m1.y + m1.x * nx) + nz * m1.z,      // %632 %641
                             (ny * m2.y + m2.x * nx) + nz * m2.z);     // %659 %665
        const vec3 tt = vec3((m0.y * ty + m0.x * tx) + tz * m0.z,      // %622 %635
                             (ty * m1.y + m1.x * tx) + tz * m1.z,      // %629 %638
                             (ty * m2.y + m2.x * tx) + tz * m2.z);     // %656 %662
        accp = vec3(qx - p0x, qy - p0y, qz - p0z) * w + accp;          // %617-%649
        accn = nn * w + accn;                                          // %679-%689
        acct = tt * w + acct;                                          // %667-%677
    }
    const float rn = inversesqrt((accn.y * accn.y + accn.x * accn.x) + accn.z * accn.z); // %709 %710
    const vec3  Nn = accn * rn;                                                            // %714 %718 %712
    const vec3  c  = vec3(acct.z * Nn.y + Nn.z * (-acct.y),        // %739 = c.x
                          acct.x * Nn.z + Nn.x * (-acct.z),        // %722 = c.y
                          acct.y * Nn.x + Nn.y * (-acct.x));       // %729 = c.z
    const vec3  Tp = vec3(c.y * Nn.z + Nn.y * (-c.z),              // %734
                          c.z * Nn.x + Nn.z * (-c.x),              // %743
                          c.x * Nn.y + Nn.x * (-c.y));             // %749
    const float rt = inversesqrt((Tp.y * Tp.y + Tp.x * Tp.x) + Tp.z * Tp.z);             // %756 %761
    const vec3  Tn = Tp * rt;                                                              // %768-%770
    const vec3  outp = accp + vec3(p0x, p0y, p0z);                                         // %763-%767

    const uint rec = index * 7u;                                                           // %775
    WoU32 OP = WoU32(job.out_pos);
    OP.d[rec] = floatBitsToUint(outp.x); OP.d[rec + 1u] = floatBitsToUint(outp.y); OP.d[rec + 2u] = floatBitsToUint(outp.z);
    WoU32 ON = WoU32(job.out_nrm);
    ON.d[rec] = packHalf2x16(vec2(Nn.x, Nn.y)); ON.d[rec + 1u] = packHalf2x16(vec2(Nn.z, 0.0));   // %798 %800
    WoU32 OT = WoU32(job.out_tan);
    OT.d[rec] = packHalf2x16(vec2(Tn.x, Tn.y)); OT.d[rec + 1u] = packHalf2x16(vec2(Tn.z, tw));    // %814 %816
}
)glsl";

// ---------------------------------------------------------------------------------------------
// [CSBATCH] env. Every accessor parses once (static local) and returns a plain value; the hash
// list uses the SKIPCS grammar (SHADPS4_SKIPCS=<hex>[,<hex>...], vk_rasterizer.cpp).
// ---------------------------------------------------------------------------------------------
namespace {

u32 EnvU32(const char* name, u32 def) {
    const char* e = std::getenv(name);
    return (e != nullptr && *e != '\0') ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : def;
}

bool EnvBool(const char* name, bool def) {
    const char* e = std::getenv(name);
    return (e != nullptr && *e != '\0') ? e[0] == '1' : def;
}

const std::vector<u64>& WantedHashes() {
    static const std::vector<u64> wanted = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_CSBATCH");
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
    return wanted;
}

} // Anonymous namespace

bool SkinBatch::Wanted(u64 h) {
    const auto& wanted = WantedHashes();
    return !wanted.empty() && std::find(wanted.begin(), wanted.end(), h) != wanted.end();
}

const std::string& SkinBatch::Hashes() {
    static const std::string s = [] {
        std::string out;
        for (const u64 h : WantedHashes()) {
            out += fmt::format("{}{:x}", out.empty() ? "" : ",", h);
        }
        return out;
    }();
    return s;
}

u32 SkinBatch::Wg() {
    static const u32 v = std::clamp<u32>(EnvU32("SHADPS4_CSBATCH_WG", 64), 1u, 1024u);
    return v;
}

u32 SkinBatch::MaxJobs() {
    // (job << JobShift) must leave the job index below 1 << 12; the table must fit the ring.
    static const u32 v = std::clamp<u32>(EnvU32("SHADPS4_CSBATCH_MAX", 1024), 1u, 4095u);
    return v;
}

u32 SkinBatch::MaxGroups() {
    // group_in_job < 1 << JobShift; the map (4 B per group) must fit the ring with the table.
    static const u32 v = std::clamp<u32>(EnvU32("SHADPS4_CSBATCH_MAXGROUPS", 65536), 1u, 262144u);
    return v;
}

u32 SkinBatch::MaxHeldLabels() {
    static const u32 v = [] {
        const char* e = std::getenv("SHADPS4_CSBATCH_MAXLABELS");
        const u32 n = e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 64u;
        return std::clamp(n, 1u, 4096u);
    }();
    return v;
}

u32 SkinBatch::FlushEvery() {
    static const u32 v = EnvU32("SHADPS4_CSBATCH_EVERY", 0);
    return v;
}

bool SkinBatch::Robust() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_ROBUST", true);
    return v;
}

bool SkinBatch::PalVec4() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_PAL_VEC4", true);
    return v;
}

bool SkinBatch::PreBar() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_PREBAR", true);
    return v;
}

bool SkinBatch::PostBar() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_POSTBAR", false);
    return v;
}

bool SkinBatch::ImgInv() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_IMGINV", true);
    return v;
}

bool SkinBatch::ZoneSync() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_ZONESYNC", true);
    return v;
}

bool SkinBatch::CountsForChunk() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_CHUNK", true);
    return v;
}

u32 SkinBatch::Shadow() {
    static const u32 v = EnvU32("SHADPS4_CSBATCH_SHADOW", 0);
    return v;
}

bool SkinBatch::Log() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_LOG", false);
    return v;
}

bool SkinBatch::TableVram() {
    static const bool v = EnvBool("SHADPS4_CSBATCH_TABLE_VRAM", false);
    return v;
}

const char* SkinBatch::ReasonName(Reason r) {
    switch (r) {
    case Reason::Draw:
        return "draw";
    case Reason::DrawIndirect:
        return "drawind";
    case Reason::Dispatch:
        return "disp";
    case Reason::DispatchIndirect:
        return "dispind";
    case Reason::Dma:
        return "dma";
    case Reason::Sync:
        return "sync";
    case Reason::Submit:
        return "submit";
    case Reason::Hook:
        return "hook";
    case Reason::Chunk:
        return "chunk";
    case Reason::Fault:
        return "fault";
    case Reason::Readback:
        return "readback";
    case Reason::Label:
        return "label";
    case Reason::Cmd:
        return "cmd";
    case Reason::Full:
        return "full";
    case Reason::Every:
        return "every";
    case Reason::Fallback:
        return "fallback";
    case Reason::Create:
        return "create";
    case Reason::Chained:
        return "chained";
    case Reason::Reupload:
        return "reupload";
    case Reason::Shadow:
        return "shadow";
    default:
        return "?";
    }
}

const char* SkinBatch::DeclineName(Decline d) {
    switch (d) {
    case Decline::Dims:
        return "dims";
    case Decline::Unmapped:
        return "unmapped";
    case Decline::Unbounded:
        return "unbounded";
    case Decline::Oversize:
        return "oversize";
    case Decline::Layout:
        return "layout";
    case Decline::ShortSrc:
        return "shortsrc";
    case Decline::Empty:
        return "empty";
    case Decline::Align:
        return "align";
    case Decline::NoBda:
        return "nobda";
    case Decline::PtDiverge:
        return "pt";
    case Decline::Unstable:
        return "unstable";
    default:
        return "?";
    }
}

void SkinBatch::Stats::ResetWindow() {
    batches = subs = verts = groups = barriers = max_subs = count0 = null_out = tick_splits = 0;
    straddle = deleted_owner = ring_wait = chained = reupload = unstable = orphan_bars = 0;
    hook_offthread.store(0, std::memory_order_relaxed);
    std::fill(std::begin(flush), std::end(flush), u64{0});
    std::fill(std::begin(decl), std::end(decl), u64{0});
    shadow_ok = shadow_bad = shadow_skip = 0;
}

void SkinBatch::Create(const Instance& instance, Scheduler& scheduler, bool stream_bda) {
    const vk::Device device = instance.GetDevice();

    // Layout: binding 0 = job table, binding 1 = group map, both push descriptors - the
    // FaultManager pipeline pattern (fault_manager.cpp), no push constants.
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
    desc_layout = Vulkan::Check(device.createDescriptorSetLayoutUnique(desc_layout_ci));

    std::vector<std::string> defines{
        fmt::format("WG={}", Wg()),
        fmt::format("ROBUST={}", Robust() ? 1 : 0),
        fmt::format("PAL_VEC4={}", PalVec4() ? 1 : 0),
    };
    const char* spv = "1.3";
    vk::ShaderModule module =
        Vulkan::Compile(kSkinBatchComp, vk::ShaderStageFlagBits::eCompute, device, defines);
    if (!module) {
        // GL_EXT_buffer_reference under a SPIR-V 1.3 target was never exercised on the stand
        // (dual_writeback.comp is only compiled with DUALHOME); 1.5 carries the same feature
        // in core.
        LOG_CRITICAL(Render_Vulkan, "[CSBATCH] compile: retry spv1.5");
        module = CompileWithSpvTarget(kSkinBatchComp, vk::ShaderStageFlagBits::eCompute, device,
                                      defines, true);
        spv = "1.5";
    }
    if (!module) {
        LOG_CRITICAL(Render_Vulkan,
                     "[CSBATCH] compile FAILED (spv1.3, spv1.5): see the glslang log above - "
                     "feature off");
        return;
    }
    Vulkan::SetObjectName(device, module, "SkinBatch Kernel");

    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::PipelineLayoutCreateInfo layout_ci = {
        .setLayoutCount = 1U,
        .pSetLayouts = &(*desc_layout),
    };
    layout = Vulkan::Check(device.createPipelineLayoutUnique(layout_ci));
    const vk::ComputePipelineCreateInfo pipeline_ci = {
        .stage = shader_ci,
        .layout = *layout,
    };
    auto pipeline_result = device.createComputePipelineUnique({}, pipeline_ci);
    device.destroyShaderModule(module);
    if (pipeline_result.result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "[CSBATCH] pipeline creation FAILED ({}) - feature off",
                     vk::to_string(pipeline_result.result));
        return;
    }
    pipeline = std::move(pipeline_result.value);
    Vulkan::SetObjectName(device, *pipeline, "SkinBatch Pipeline");

    // The table ring: host-visible (no ReBAR on the stand -> system memory), read by the kernel
    // through push descriptors; the device address is for the shadow scratch only.
    ring = std::make_unique<VideoCore::StreamBuffer>(instance, scheduler,
                                                     VideoCore::MemoryUsage::Stream, RingBytes,
                                                     vk::BufferUsageFlagBits::eShaderDeviceAddress);
    Vulkan::SetObjectName(device, ring->Handle(), "SkinBatch Ring");
    if (TableVram()) {
        vram_table = std::make_unique<VideoCore::Buffer>(instance, scheduler,
                                                         VideoCore::MemoryUsage::DeviceLocal, 0,
                                                         VideoCore::AllFlags, RingBytes);
        Vulkan::SetObjectName(device, vram_table->Handle(), "SkinBatch VRAM Table");
    }
    // One vkCmdDispatch per batch: its group count is bounded by the device (the Vulkan
    // minimum is 65535; NVIDIA reports 2^31-1). The env knob can only lower it.
    const u32 dev_groups =
        instance.GetPhysicalDevice().getProperties().limits.maxComputeWorkGroupCount[0];
    max_groups = std::min(MaxGroups(), dev_groups);
    LOG_CRITICAL(Render_Vulkan,
                 "[CSBATCH] armed: hashes={} WG={} robust={} pal={} ring=8MiB stream_bda={} "
                 "pipeline=ok spv={} table={} max={}/{} (device {}) every={} prebar={} postbar={} "
                 "imginv={} zonesync={} chunk={} shadow={} log={}",
                 Hashes(), Wg(), Robust() ? 1 : 0, PalVec4() ? "vec4" : "scalar",
                 stream_bda ? "ok" : "MISSING", spv, TableVram() ? "vram" : "ring", MaxJobs(),
                 max_groups, dev_groups, FlushEvery(), PreBar() ? 1 : 0, PostBar() ? 1 : 0,
                 ImgInv() ? 1 : 0, ZoneSync() ? 1 : 0, CountsForChunk() ? 1 : 0, Shadow(),
                 Log() ? 1 : 0);
}

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& scheduler = rasterizer.GetScheduler();
    auto& buffer_cache = rasterizer.GetBufferCache();

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    scheduler.EndRendering();

    static constexpr vk::MemoryBarrier READ_BARRIER{
        .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
    };
    static constexpr vk::MemoryBarrier WRITE_BARRIER{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
    };
    scheduler.CommandBuffer().pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eByRegion, READ_BARRIER, {}, {});

    static constexpr vk::DeviceSize MaxDistanceForMerge = 64_MB;
    u32 batch_start = 0;
    u32 batch_end = 0;

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        // Obtain buffers for the total source and destination ranges.
        const auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_offset_min, src_offset_max - src_offset_min, false);
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_offset_min, dst_offset_max - dst_offset_min, true);

        // Apply found buffer base.
        const auto vk_copies = std::span{copies}.subspan(batch_start, batch_end - batch_start);
        for (auto& copy : vk_copies) {
            copy.srcOffset = copy.srcOffset - src_offset_min + src_buf_offset;
            copy.dstOffset = copy.dstOffset - dst_offset_min + dst_buf_offset;
        }

        // Execute buffer copies.
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                  src_offset_max - src_offset_min, dst_offset_max - dst_offset_min);
        scheduler.CommandBuffer().copyBuffer(src_buf->Handle(), dst_buf->Handle(), vk_copies);
        batch_start = batch_end;
    }

    scheduler.CommandBuffer().pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlagBits::eByRegion, WRITE_BARRIER, {}, {});

    return true;
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        // [CSBATCH] the copy HLE records transfer commands of its own: a pending batch of the
        // skinning worker has to land before them (same rule as every other GPU command).
        rasterizer.FlushSkinBatch(SkinBatch::Reason::Dispatch);
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    case SkinBatch::kHash:
        // [CSBATCH] env unset -> Wanted() is one empty() test and the dispatch takes the normal
        // path (CSSPLIT+SKINDIRECT); a decline inside Append flushes the batch and returns false.
        return SkinBatch::Wanted(info.pgm_hash) && rasterizer.SkinBatchAppend(info, cs_program);
    default:
        return false;
    }
}

} // namespace Vulkan
