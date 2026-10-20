// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit> // [CSSPLIT] has_single_bit / countr_zero
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ranges>
#include <vector>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/postmortem.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

// [CSSPLIT] SHADPS4_CSSPLIT=<hex pgm_hash>:<K>[,<hash>:<K>...] - K a power of two >= 2 that
// divides the guest LocalSize.x. Returns log2 K for a listed hash, 0 otherwise. Parsed once;
// a malformed entry is logged and skipped, never applied. The hash is the pgm_hash the logs
// print ({:#x}, e.g. 0x8f067540); an optional 0x prefix is accepted.
static u8 CsSplitRequestLog2(u64 pgm_hash) {
    static const std::vector<std::pair<u64, u8>> table = [] {
        std::vector<std::pair<u64, u8>> t;
        const char* e = std::getenv("SHADPS4_CSSPLIT");
        for (const char* p = e; p && *p;) {
            char* end = nullptr;
            const u64 hash = std::strtoull(p, &end, 16);
            u32 k = 0;
            if (end && *end == ':') {
                k = static_cast<u32>(std::strtoul(end + 1, &end, 10));
            }
            if (hash != 0 && k >= 2 && std::has_single_bit(k)) {
                t.emplace_back(hash, static_cast<u8>(std::countr_zero(k)));
                LOG_CRITICAL(Render_Vulkan, "[CSSPLIT] shader {:#x}: K={} requested", hash, k);
            } else {
                LOG_CRITICAL(Render_Vulkan, "[CSSPLIT] bad entry near '{}' (want <hex>:<pow2 K>)",
                             p);
            }
            p = end ? std::strchr(end, ',') : nullptr;
            if (p) {
                ++p;
            }
        }
        return t;
    }();
    for (const auto& [h, l] : table) {
        if (h == pgm_hash) {
            return l;
        }
    }
    return 0;
}

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Geometry));
        info.fs_info.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        // [CSSPLIT] set by RefreshComputeKey for this dispatch; part of the permutation key.
        // eff is stamped by GetProgram after the compile (Initialize() already zeroed both).
        info.cs_info.cssplit_req_log2 = cssplit_ask_log2;
        info.cs_info.cssplit_eff_log2 = 0;
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        // When binding a UBO, we calculate its size considering the offset in the larger buffer
        // cache underlying resource. In some cases, it may produce sizes exceeding the system
        // maximum allowed UBO range, so we need to reduce the threshold to prevent issues.
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
        // [Stage2 SPINGUARD] fold the flag into the Profile so the disk shader/pipeline cache
        // (profile.bin) invalidates when it toggles - PCACHE must miss/rebuild for a true A/B.
        .spinguard =
            [] {
                const char* e = std::getenv("SHADPS4_SPINGUARD");
                return e && e[0] == '1';
            }(),
    };

    // Seed the pipeline cache from disk. Compiling a level's worth of
    // pipelines takes minutes of GPU-thread time on a cold cache, and the
    // guest's GPU-side job workers spin the whole time — long enough for the
    // host GPU watchdog to fire mid-load. A warm cache removes the stall.
    std::vector<u8> cache_blob;
    const auto cache_path =
        Common::FS::GetUserPath(Common::FS::PathType::ShaderDir) / "vk_pipeline_cache.bin";
    if (std::filesystem::exists(cache_path)) {
        // Plain binary read: a text-mode read mangles the blob on Windows and
        // the driver then silently rejects it, discarding the warm cache.
        std::ifstream in(cache_path, std::ios::binary | std::ios::ate);
        if (in) {
            cache_blob.resize(static_cast<size_t>(in.tellg()));
            in.seekg(0);
            in.read(reinterpret_cast<char*>(cache_blob.data()), cache_blob.size());
        }
        LOG_WARNING(Render_Vulkan, "PCACHE: loaded {} KB from disk", cache_blob.size() >> 10);
    }
    const vk::PipelineCacheCreateInfo cache_info = {
        .initialDataSize = cache_blob.size(),
        .pInitialData = cache_blob.empty() ? nullptr : cache_blob.data(),
    };
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique(cache_info);
    if (cache_result != vk::Result::eSuccess && !cache_blob.empty()) {
        // A blob from another driver version is rejected - start clean.
        auto [retry_result, retry_cache] = instance.GetDevice().createPipelineCacheUnique({});
        cache_result = retry_result;
        cache = std::move(retry_cache);
    }
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    // [Stage3] WarmUp() now runs AFTER pipeline_cache exists: preload builds Load*Pipeline
    // through *pipeline_cache, so with WarmUp before creation it dereferenced a null cache and
    // rebuilt hundreds of pipelines with no driver blob (tens of seconds). Only the moment
    // moved - the blob logic is untouched.
    WarmUp();
    Postmortem::MarkStartup("pipeline_cache");

    // [Stage1 PARSERPROF] partial-window parser profile in the crash dump.
    Postmortem::RegisterPostmortemDump([] {
        LOG_CRITICAL(Render_Vulkan,
                     "POSTMORTEM-PARSER: compiles={} specmiss={} prof_pipe_us={} eop_gap_max_ms={}",
                     Parserprof::g_compile_count.load(std::memory_order_relaxed),
                     Parserprof::g_specmiss_count.load(std::memory_order_relaxed),
                     Parserprof::g_prof_pipe_us.load(std::memory_order_relaxed),
                     Postmortem::g_eop_gap_max_ms.load(std::memory_order_relaxed));
    });
}

void PipelineCache::MaybeSavePipelineCache() {
    // Flush every N compiles: a cold-cache level load can crash (or be killed)
    // before the destructor runs, and losing the whole batch means the next
    // attempt is just as slow.
    static constexpr u32 CompilesPerSave = 64;
    if (++compiles_since_save < CompilesPerSave) {
        return;
    }
    compiles_since_save = 0;
    SavePipelineCache();
}

void PipelineCache::SavePipelineCache() {
    if (!pipeline_cache) {
        return;
    }
    auto [result, blob] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result != vk::Result::eSuccess || blob.empty()) {
        LOG_WARNING(Render_Vulkan, "PCACHE: no data to save (result={}, size={})",
                    vk::to_string(result), blob.size());
        return;
    }
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::ShaderDir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / "vk_pipeline_cache.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        LOG_WARNING(Render_Vulkan, "PCACHE: cannot open {} for write",
                    Common::FS::PathToUTF8String(path));
        return;
    }
    out.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    LOG_WARNING(Render_Vulkan, "PCACHE: saved {} KB", blob.size() >> 10);
}

PipelineCache::~PipelineCache() {
    SavePipelineCache();
}

// [PIPESTAT] Field-group diff of two pipeline keys. The key is compared with memcmp by its own
// operator==, so the groups are addressed the same way - offsetof plus the field's size. Fields
// that are already DYNAMIC (viewport, scissor, blend constants, stencil ref, depth bias) are not
// part of the key at all and cannot appear here.
static void AccountPipelineSwitch(const GraphicsPipelineKey& prev, const GraphicsPipelineKey& cur) {
    const auto* a = reinterpret_cast<const u8*>(&prev);
    const auto* b = reinterpret_cast<const u8*>(&cur);
    const auto differs = [a, b](size_t off, size_t len) {
        return std::memcmp(a + off, b + off, len) != 0;
    };
#define KEY_FIELD(f) offsetof(GraphicsPipelineKey, f), sizeof(GraphicsPipelineKey::f)
    bool any = false;
    if (differs(KEY_FIELD(stage_hashes))) {
        Parserprof::g_pipe_d_shaders.fetch_add(1, std::memory_order_relaxed);
        any = true;
    }
    if (differs(KEY_FIELD(blend_controls)) || differs(KEY_FIELD(write_masks)) ||
        differs(KEY_FIELD(cb_shader_mask)) || differs(KEY_FIELD(logic_op))) {
        Parserprof::g_pipe_d_blend.fetch_add(1, std::memory_order_relaxed);
        any = true;
    }
    if (differs(KEY_FIELD(num_color_attachments)) || differs(KEY_FIELD(color_buffers)) ||
        differs(KEY_FIELD(mrt_mask)) || differs(KEY_FIELD(num_samples)) ||
        differs(KEY_FIELD(color_samples))) {
        Parserprof::g_pipe_d_rtfmt.fetch_add(1, std::memory_order_relaxed);
        any = true;
    }
    if (differs(KEY_FIELD(vertex_buffer_formats)) || differs(KEY_FIELD(patch_control_points))) {
        Parserprof::g_pipe_d_vertex.fetch_add(1, std::memory_order_relaxed);
        any = true;
    }
    if (differs(KEY_FIELD(depth_samples))) {
        Parserprof::g_pipe_d_depth.fetch_add(1, std::memory_order_relaxed);
        any = true;
    }
#undef KEY_FIELD
    // The two anonymous bitfield structs (z/stencil format, depth clamp/clip, primitive type,
    // polygon mode, clip space, provoking vertex) have no name to take an offset of, so they
    // land here: "the keys differ, but in none of the groups above".
    if (!any) {
        Parserprof::g_pipe_d_other.fetch_add(1, std::memory_order_relaxed);
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    // [SLOWPKT] compile phase: key refresh + shader compiles + vkCreateGraphicsPipelines.
    Parserprof::PhaseProbe slow_probe{Parserprof::PhCompile};
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    {
        static GraphicsPipelineKey prev_key{};
        static bool prev_valid = false;
        if (prev_valid && !(prev_key == graphics_key)) {
            Parserprof::g_pipe_switches.fetch_add(1, std::memory_order_relaxed);
            AccountPipelineSwitch(prev_key, graphics_key);
        }
        prev_key = graphics_key;
        prev_valid = true;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        Parserprof::g_pipe_created.fetch_add(1, std::memory_order_relaxed);
    }
    Parserprof::g_pipe_unique.store(graphics_pipelines.size(), std::memory_order_relaxed);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader.reset();
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline(bool allow_cssplit) {
    // [SLOWPKT] compile phase (dispatch packets).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhCompile};
    // [CSSPLIT] the key already differs per permutation (HashCombine(hash, perm_idx)), so the
    // split and unsplit variants of one shader get separate ComputePipeline objects.
    if (!RefreshComputeKey(allow_cssplit)) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        const auto tcp0 = std::chrono::steady_clock::now();
        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        {
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - tcp0)
                                .count();
            if (ms > 100) {
                LOG_WARNING(Render_Vulkan, "COMPILEPROF: compute pipeline {:#x} took {}ms",
                            pipeline_hash, ms);
            }
        }
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = std::nullopt;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(Stage::Fragment, LogicalStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey(bool allow_cssplit) {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    // [CSSPLIT] decided before GetProgram() -> BuildRuntimeInfo() copies it into cs_info
    cssplit_listed_log2 = CsSplitRequestLog2(cs_params.hash);
    cssplit_ask_log2 = allow_cssplit ? cssplit_listed_log2 : 0;
    cssplit_eff_log2 = 0;
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");
    MaybeSavePipelineCache();

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    // [Stage2] Register (which moves-from spv) only AFTER CollectShader has read spv - the
    // previous order left CollectShader with an emptied vector when shader-collect was on.
    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(stage, l_stage);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto pp0 = Parserprof::Enabled() ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        if (Parserprof::Enabled()) {
            Parserprof::g_prof_pipe_us.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - pp0)
                    .count(),
                std::memory_order_relaxed);
            Parserprof::g_compile_count.fetch_add(1, std::memory_order_relaxed);
        }
        if (stage == Stage::Compute) {
            // [CSSPLIT] the split the backend applied travels with the permutation's spec (and
            // its serialized meta), so every later dispatch - this run or preloaded - scales the
            // grid by the factor this very module was compiled for.
            runtime_info.cs_info.cssplit_eff_log2 = program->info.cssplit_eff_log2;
            cssplit_eff_log2 = program->info.cssplit_eff_log2;
        }
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        return std::make_tuple(&program->info, module, program->modules[0].spec.fetch_shader_data,
                               perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    // [Stage2] Treat a found-but-null module as a miss: InsertPermut() resizes the
    // permutation list and leaves default-spec holes with a VK_NULL_HANDLE module; a
    // stray spec match on such a hole would otherwise return a null shader module.
    if (it == program->modules.end() || it->module == VK_NULL_HANDLE) {
        auto new_info = Shader::Info(stage, l_stage, params);
        const auto pp0 = Parserprof::Enabled() ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);
        if (Parserprof::Enabled()) {
            Parserprof::g_prof_pipe_us.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - pp0)
                    .count(),
                std::memory_order_relaxed);
            Parserprof::g_compile_count.fetch_add(1, std::memory_order_relaxed);
            // [Stage1] SPECMISS: an already-cached pgm_hash recompiled at a new permutation
            // index - the "invisible field jumps" case (triple-compile of one hash).
            const u64 n = Parserprof::g_specmiss_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 32 || n % 64 == 0) {
                LOG_CRITICAL(Render_Vulkan,
                             "SPECMISS #{}: pgm_hash={:#x} stage={} perm_idx={} existing_perms={}",
                             n, params.hash, static_cast<u32>(stage), perm_idx,
                             program->modules.size());
            }
        }

        if (stage == Stage::Compute) {
            // [CSSPLIT] stamp eff into the spec before it is serialized and stored; eff is not
            // part of spec equality, so the request built above still matches this record.
            spec.runtime_info.cs_info.cssplit_eff_log2 = new_info.cssplit_eff_log2;
            cssplit_eff_log2 = new_info.cssplit_eff_log2;
        }
        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
        if (stage == Stage::Compute) {
            // [CSSPLIT] persisted with the spec: valid for modules compiled this run and for
            // permutations preloaded from the disk cache (0 in blobs written before this patch)
            cssplit_eff_log2 = it->spec.runtime_info.cs_info.cssplit_eff_log2;
        }
    }
    return std::make_tuple(&program->info, module,
                           program->modules[perm_idx].spec.fetch_shader_data, perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    // [SHADERDUMP] SHADPS4_SHADERDUMP=<hex hash>[,<hex hash>...] dumps ONLY the named shaders,
    // both the source GCN binary (.bin) and the emitted SPIR-V (.spv). Dumping everything
    // (dumpShaders) writes thousands of files during a level load and is not usable while
    // chasing one hash. The path is logged once per file so it can be collected afterwards.
    static const std::vector<u64> wanted = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_SHADERDUMP");
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
    const bool targeted =
        !wanted.empty() && std::find(wanted.begin(), wanted.end(), hash) != wanted.end();
    if (!EmulatorSettings.IsDumpShaders() && !targeted) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
    if (targeted) {
        LOG_CRITICAL(Render_Vulkan, "SHADERDUMP: wrote {} ({} dwords) into the shader dumps dir",
                     filename, code.size());
    }
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
