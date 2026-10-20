// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdlib>
#include "common/assert.h"
#include "common/div_ceil.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/buffer_cache/buffer_cache.h"

#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include <numbers>
#include <string_view>

namespace Shader::Backend::SPIRV {
namespace {

std::string_view StageName(Stage stage) {
    switch (stage) {
    case Stage::Vertex:
        return "vs";
    case Stage::Local:
        return "ls";
    case Stage::Export:
        return "es";
    case Stage::Hull:
        return "hs";
    case Stage::Geometry:
        return "gs";
    case Stage::Fragment:
        return "fs";
    case Stage::Compute:
        return "cs";
    }
    UNREACHABLE_MSG("Invalid hw stage {}", u32(stage));
}

static constexpr u32 NumVertices(AmdGpu::PrimitiveType type) {
    switch (type) {
    case AmdGpu::PrimitiveType::PointList:
        return 1u;
    case AmdGpu::PrimitiveType::LineList:
    case AmdGpu::PrimitiveType::LineStrip:
        return 2u;
    case AmdGpu::PrimitiveType::TriangleList:
    case AmdGpu::PrimitiveType::TriangleStrip:
    case AmdGpu::PrimitiveType::RectList:
        return 3u;
    case AmdGpu::PrimitiveType::AdjTriangleList:
        return 6u;
    case AmdGpu::PrimitiveType::AdjLineList:
        return 4u;
    default:
        UNREACHABLE();
    }
}

template <typename... Args>
void Name(EmitContext& ctx, Id object, std::string_view format_str, Args&&... args) {
    ctx.Name(object, fmt::format(fmt::runtime(format_str), StageName(ctx.stage),
                                 std::forward<Args>(args)...)
                         .c_str());
}

} // Anonymous namespace

EmitContext::EmitContext(const Profile& profile_, const RuntimeInfo& runtime_info_, Info& info_,
                         Bindings& binding_)
    : Sirit::Module(profile_.supported_spirv), info{info_}, runtime_info{runtime_info_},
      profile{profile_}, stage{info.stage}, l_stage{info.l_stage}, binding{binding_} {
    if (info.uses_dma) {
        SetMemoryModel(spv::AddressingModel::PhysicalStorageBuffer64, spv::MemoryModel::GLSL450);
    } else {
        SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    }
    String(fmt::format("{:#x}", info.pgm_hash));
    // [CSSPLIT] decided before DefineInterfaces(): DefineInputs needs to know whether the real
    // group id must exist for the local-id rewrite. Emits no SPIR-V itself.
    SetupCsSplit();

    AddCapability(spv::Capability::Shader);
    DefineArithmeticTypes();
    DefineInterfaces();
    DefineSharedMemory();
    DefineBuffers();
    DefineImagesAndSamplers();
    DefineFunctions();
}

EmitContext::~EmitContext() = default;

Id EmitContext::Def(const IR::Value& value) {
    if (!value.IsImmediate()) {
        return value.InstRecursive()->Definition<Id>();
    }
    switch (value.Type()) {
    case IR::Type::Void:
        return Id{};
    case IR::Type::U1:
        return value.U1() ? true_value : false_value;
    case IR::Type::U32:
        return ConstU32(value.U32());
    case IR::Type::U64:
        return Constant(U64, value.U64());
    case IR::Type::F32:
        return ConstF32(value.F32());
    case IR::Type::F64:
        return Constant(F64[1], value.F64());
    case IR::Type::StringLiteral:
        return String(value.StringLiteral());
    default:
        UNREACHABLE_MSG("Immediate type {}", value.Type());
    }
}

void EmitContext::DefineArithmeticTypes() {
    void_id = Name(TypeVoid(), "void_id");
    U1[1] = Name(TypeBool(), "bool_id");
    U8 = Name(TypeUInt(8), "u8_id");
    S8 = Name(TypeSInt(8), "i8_id");
    U16 = Name(TypeUInt(16), "u16_id");
    S16 = Name(TypeSInt(16), "i16_id");
    if (info.uses_fp16) {
        F16[1] = Name(TypeFloat(16), "f16_id");
    }
    if (info.uses_fp64) {
        F64[1] = Name(TypeFloat(64), "f64_id");
    }
    F32[1] = Name(TypeFloat(32), "f32_id");
    S32[1] = Name(TypeSInt(32), "i32_id");
    U32[1] = Name(TypeUInt(32), "u32_id");
    U64 = Name(TypeUInt(64), "u64_id");

    for (u32 i = 2; i <= 4; i++) {
        if (info.uses_fp16) {
            F16[i] = Name(TypeVector(F16[1], i), fmt::format("f16vec{}_id", i));
        }
        if (info.uses_fp64) {
            F64[i] = Name(TypeVector(F64[1], i), fmt::format("f64vec{}_id", i));
        }
        F32[i] = Name(TypeVector(F32[1], i), fmt::format("f32vec{}_id", i));
        S32[i] = Name(TypeVector(S32[1], i), fmt::format("i32vec{}_id", i));
        U32[i] = Name(TypeVector(U32[1], i), fmt::format("u32vec{}_id", i));
        U1[i] = Name(TypeVector(U1[1], i), fmt::format("bvec{}_id", i));
    }

    true_value = ConstantTrue(U1[1]);
    false_value = ConstantFalse(U1[1]);
    u8_one_value = Constant(U8, 1U);
    u8_zero_value = Constant(U8, 0U);
    u16_zero_value = Constant(U16, 0U);
    u32_one_value = ConstU32(1U);
    u32_zero_value = ConstU32(0U);
    f32_zero_value = ConstF32(0.0f);
    u64_one_value = Constant(U64, 1ULL);
    u64_zero_value = Constant(U64, 0ULL);

    pi_x2 = ConstF32(2.0f * float{std::numbers::pi});

    input_f32 = Name(TypePointer(spv::StorageClass::Input, F32[1]), "input_f32");
    input_u32 = Name(TypePointer(spv::StorageClass::Input, U32[1]), "input_u32");
    input_s32 = Name(TypePointer(spv::StorageClass::Input, S32[1]), "input_s32");

    output_f32 = Name(TypePointer(spv::StorageClass::Output, F32[1]), "output_f32");
    output_u32 = Name(TypePointer(spv::StorageClass::Output, U32[1]), "output_u32");
    output_s32 = Name(TypePointer(spv::StorageClass::Output, S32[1]), "output_s32");

    full_result_i32x2 = Name(TypeStruct(S32[1], S32[1]), "full_result_i32x2");
    full_result_u32x2 = Name(TypeStruct(U32[1], U32[1]), "full_result_u32x2");
    frexp_result_f32 = Name(TypeStruct(F32[1], S32[1]), "frexp_result_f32");
    if (info.uses_fp64) {
        frexp_result_f64 = Name(TypeStruct(F64[1], S32[1]), "frexp_result_f64");
    }
    if (info.uses_dma) {
        physical_pointer_type_u32 = TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[1]);
    }
}

void EmitContext::DefineInterfaces() {
    DefinePushDataBlock();
    DefineInputs();
    DefineOutputs();
}

const VectorIds& GetAttributeType(EmitContext& ctx, AmdGpu::NumberFormat fmt) {
    switch (GetNumberClass(fmt)) {
    case AmdGpu::NumberClass::Float:
        return ctx.F32;
    case AmdGpu::NumberClass::Sint:
        return ctx.S32;
    case AmdGpu::NumberClass::Uint:
        return ctx.U32;
    default:
        break;
    }
    UNREACHABLE_MSG("Invalid attribute type {}", fmt);
}

EmitContext::SpirvAttribute EmitContext::GetAttributeInfo(AmdGpu::NumberFormat fmt, Id id,
                                                          u32 num_components, bool output,
                                                          bool loaded, bool array) {
    switch (GetNumberClass(fmt)) {
    case AmdGpu::NumberClass::Float:
        return {id, output ? output_f32 : input_f32, F32[1], num_components, false, loaded, array};
    case AmdGpu::NumberClass::Uint:
        return {id, output ? output_u32 : input_u32, U32[1], num_components, true, loaded, array};
    case AmdGpu::NumberClass::Sint:
        return {id, output ? output_s32 : input_s32, S32[1], num_components, true, loaded, array};
    default:
        break;
    }
    UNREACHABLE_MSG("Invalid attribute type {}", fmt);
}

Id EmitContext::GetBufferSize(const u32 sharp_idx) {
    // Can this be done with memory access? Like we do now with ReadConst
    const auto& srt_flatbuf = buffers[flatbuf_index];
    ASSERT(srt_flatbuf.buffer_type == BufferType::Flatbuf);
    const auto [id, pointer_type] = srt_flatbuf.Alias(PointerType::U32);

    const auto rsrc1{
        OpLoad(U32[1], OpAccessChain(pointer_type, id, u32_zero_value, ConstU32(sharp_idx + 1)))};
    const auto rsrc2{
        OpLoad(U32[1], OpAccessChain(pointer_type, id, u32_zero_value, ConstU32(sharp_idx + 2)))};

    const auto stride{OpBitFieldUExtract(U32[1], rsrc1, ConstU32(16u), ConstU32(14u))};
    const auto num_records{rsrc2};

    const auto stride_zero{OpIEqual(U1[1], stride, u32_zero_value)};
    const auto stride_size{OpIMul(U32[1], num_records, stride)};
    return OpSelect(U32[1], stride_zero, num_records, stride_size);
}

void EmitContext::DefineBufferProperties() {
    if (!profile.needs_buffer_offsets) {
        return;
    }
    for (u32 i = 0; i < buffers.size(); i++) {
        auto& buffer = buffers[i];
        const auto& desc = info.buffers[i];
        const u32 binding = buffer.binding;
        if (buffer.buffer_type != BufferType::Guest) {
            continue;
        }

        const u32 half = PushData::BufOffsetIndex + (binding >> 4);
        const u32 comp = (binding & 0xf) >> 2;
        const u32 offset = (binding & 0x3) << 3;
        const Id ptr{OpAccessChain(TypePointer(spv::StorageClass::PushConstant, U32[1]),
                                   push_data_block, ConstU32(half), ConstU32(comp))};
        const Id value{OpLoad(U32[1], ptr)};

        const Id buf_offset{OpBitFieldUExtract(U32[1], value, ConstU32(offset), ConstU32(8U))};
        Name(buf_offset, fmt::format("buf{}_off", binding));
        buffer.Offset(PointerSize::B8) = buf_offset;

        if (True(desc.used_types & IR::Type::U16)) {
            const Id buf_word_offset{OpShiftRightLogical(U32[1], buf_offset, ConstU32(1U))};
            Name(buf_word_offset, fmt::format("buf{}_word_off", binding));
            buffer.Offset(PointerSize::B16) = buf_word_offset;
        }
        if (True(desc.used_types & IR::Type::U32)) {
            const Id buf_dword_offset{OpShiftRightLogical(U32[1], buf_offset, ConstU32(2U))};
            Name(buf_dword_offset, fmt::format("buf{}_dword_off", binding));
            buffer.Offset(PointerSize::B32) = buf_dword_offset;
        }
        if (True(desc.used_types & IR::Type::U64)) {
            const Id buf_qword_offset{OpShiftRightLogical(U32[1], buf_offset, ConstU32(3U))};
            Name(buf_qword_offset, fmt::format("buf{}_qword_off", binding));
            buffer.Offset(PointerSize::B64) = buf_qword_offset;
        }
    }
}

void EmitContext::DefineAmdPerVertexAttribs() {
    if (!profile.supports_amd_shader_explicit_vertex_parameter) {
        return;
    }
    for (s32 i = 0; i < runtime_info.fs_info.num_inputs; i++) {
        const auto& input = runtime_info.fs_info.inputs[i];
        if (input.IsDefault() || info.fs_interpolation[i].primary != Qualifier::PerVertex) {
            continue;
        }
        auto& param = input_params[i];
        const Id pointer = param.id;
        param.id_array[0] =
            OpInterpolateAtVertexAMD(F32[param.num_components], pointer, ConstU32(0U));
        param.id_array[1] =
            OpInterpolateAtVertexAMD(F32[param.num_components], pointer, ConstU32(1U));
        param.id_array[2] =
            OpInterpolateAtVertexAMD(F32[param.num_components], pointer, ConstU32(2U));
        param.is_loaded = true;
    }
}

void EmitContext::DefineWorkgroupIndex() {
    const Id workgroup_id_val{OpLoad(U32[3], workgroup_id)};
    const Id workgroup_x{OpCompositeExtract(U32[1], workgroup_id_val, 0)};
    const Id workgroup_y{OpCompositeExtract(U32[1], workgroup_id_val, 1)};
    const Id workgroup_z{OpCompositeExtract(U32[1], workgroup_id_val, 2)};
    const Id num_workgroups{OpLoad(U32[3], num_workgroups_id)};
    const Id num_workgroups_x{OpCompositeExtract(U32[1], num_workgroups, 0)};
    const Id num_workgroups_y{OpCompositeExtract(U32[1], num_workgroups, 1)};
    workgroup_index_id =
        OpIAdd(U32[1], OpIAdd(U32[1], workgroup_x, OpIMul(U32[1], workgroup_y, num_workgroups_x)),
               OpIMul(U32[1], workgroup_z, OpIMul(U32[1], num_workgroups_x, num_workgroups_y)));
    Name(workgroup_index_id, "workgroup_index");
}

// [CSSPLIT] Decide once per module whether the split the pipeline cache asked for can be applied.
// Anything that depends on the workgroup shape is a reason to decline; the flags come from
// CollectShaderInfoPass (which ran last), not from a disassembly. The applied value is reported
// back through info.cssplit_eff_log2 so the dispatch side multiplies the grid by the same factor
// this module was compiled for. With req == 0 (env unset, shader not listed, or the unsplit
// fallback permutation) nothing is logged and every later check reduces to the old code path.
void EmitContext::SetupCsSplit() {
    cssplit_log2 = 0;
    cssplit_local = 0;
    info.cssplit_eff_log2 = 0;
    if (l_stage != LogicalStage::Compute) {
        return;
    }
    const auto& cs = runtime_info.cs_info;
    const u32 req = cs.cssplit_req_log2;
    if (req == 0) {
        return;
    }
    const u32 k = 1u << req;
    const u32 s = cs.workgroup_size[0];
    const char* why = nullptr;
    if (cs.workgroup_size[1] != 1 || cs.workgroup_size[2] != 1) {
        why = "workgroup is not 1-D";
    } else if (s == 0 || (s % k) != 0) {
        why = "LocalSize.x not divisible by K";
    } else if (info.shared_types != IR::Type::Void || info.uses_shared_int64_atomics) {
        why = "uses LDS";
    } else if (info.uses_barrier) {
        why = "uses s_barrier / workgroup barrier";
    } else if (info.uses_group_quad || info.uses_group_ballot || info.uses_lane_id) {
        why = "uses subgroup ops";
    } else if (info.loads.GetAny(IR::Attribute::WorkgroupIndex)) {
        why = "uses WorkgroupIndex";
    }
    if (why) {
        LOG_CRITICAL(Render_Recompiler, "[CSSPLIT] shader {:#x}: declined K={} S={} ({})",
                     info.pgm_hash, k, s, why);
        return;
    }
    cssplit_log2 = req;
    cssplit_local = s / k;
    info.cssplit_eff_log2 = static_cast<u8>(req);
    LOG_CRITICAL(Render_Recompiler,
                 "[CSSPLIT] shader {:#x}: S={} K={} -> LocalSize {}x1x1, grid x{}", info.pgm_hash,
                 s, k, cssplit_local, k);
}

void EmitContext::DefineInputs() {
    if (info.uses_lane_id) {
        subgroup_local_invocation_id = DefineVariable(
            U32[1], spv::BuiltIn::SubgroupLocalInvocationId, spv::StorageClass::Input);
        Decorate(subgroup_local_invocation_id, spv::Decoration::Flat);
    }
    switch (l_stage) {
    case LogicalStage::Vertex: {
        vertex_index = DefineVariable(U32[1], spv::BuiltIn::VertexIndex, spv::StorageClass::Input);
        base_vertex = DefineVariable(U32[1], spv::BuiltIn::BaseVertex, spv::StorageClass::Input);
        instance_id = DefineVariable(U32[1], spv::BuiltIn::InstanceIndex, spv::StorageClass::Input);

        const auto fetch_shader = Gcn::ParseFetchShader(info);
        if (!fetch_shader) {
            break;
        }
        for (const auto& attrib : fetch_shader->attributes) {
            ASSERT(attrib.semantic < IR::NumParams);
            const auto sharp = attrib.GetSharp(info);
            const Id type{GetAttributeType(*this, sharp.GetNumberFmt())[4]};
            Id id{DefineInput(type, attrib.semantic)};
            if (attrib.GetStepRate() != Gcn::VertexAttribute::InstanceIdType::None) {
                Name(id, fmt::format("vs_instance_attr{}", attrib.semantic));
            } else {
                Name(id, fmt::format("vs_in_attr{}", attrib.semantic));
            }
            input_params[attrib.semantic] = GetAttributeInfo(sharp.GetNumberFmt(), id, 4, false);
        }
        break;
    }
    case LogicalStage::Fragment: {
        if (info.loads.GetAny(IR::Attribute::FragCoord)) {
            frag_coord = DefineVariable(F32[4], spv::BuiltIn::FragCoord, spv::StorageClass::Input);
        }
        if (info.loads.Get(IR::Attribute::IsFrontFace)) {
            front_facing =
                DefineVariable(U1[1], spv::BuiltIn::FrontFacing, spv::StorageClass::Input);
        }
        if (info.loads.GetAny(IR::Attribute::RenderTargetIndex)) {
            output_layer = DefineVariable(U32[1], spv::BuiltIn::Layer, spv::StorageClass::Input);
            Decorate(output_layer, spv::Decoration::Flat);
        }
        if (info.loads.Get(IR::Attribute::SampleIndex)) {
            sample_index = DefineVariable(U32[1], spv::BuiltIn::SampleId, spv::StorageClass::Input);
            Decorate(sample_index, spv::Decoration::Flat);
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordSmooth)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_smooth = DefineVariable(F32[2], spv::BuiltIn::BaryCoordSmoothAMD,
                                                   spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_smooth =
                    DefineVariable(F32[3], spv::BuiltIn::BaryCoordKHR, spv::StorageClass::Input);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordSmoothCentroid)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_smooth_centroid = DefineVariable(
                    F32[2], spv::BuiltIn::BaryCoordSmoothCentroidAMD, spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_smooth_centroid =
                    DefineVariable(F32[3], spv::BuiltIn::BaryCoordKHR, spv::StorageClass::Input);
                // Decorate(bary_coord_smooth_centroid, spv::Decoration::Centroid);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordSmoothSample)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_smooth_sample = DefineVariable(
                    F32[2], spv::BuiltIn::BaryCoordSmoothSampleAMD, spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_smooth_sample =
                    DefineVariable(F32[3], spv::BuiltIn::BaryCoordKHR, spv::StorageClass::Input);
                // Decorate(bary_coord_smooth_sample, spv::Decoration::Sample);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordNoPersp)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_nopersp = DefineVariable(F32[2], spv::BuiltIn::BaryCoordNoPerspAMD,
                                                    spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_nopersp = DefineVariable(F32[3], spv::BuiltIn::BaryCoordNoPerspKHR,
                                                    spv::StorageClass::Input);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordNoPerspSample)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_nopersp_sample = DefineVariable(
                    F32[2], spv::BuiltIn::BaryCoordNoPerspSampleAMD, spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_nopersp_sample = DefineVariable(
                    F32[3], spv::BuiltIn::BaryCoordNoPerspKHR, spv::StorageClass::Input);
                // Decorate(bary_coord_nopersp_sample, spv::Decoration::Sample);
            }
        }

        const bool has_clip_distance_inputs = runtime_info.fs_info.clip_distance_emulation;
        // Clip distances attribute vector is the last in inputs array
        const auto num_inputs =
            runtime_info.fs_info.num_inputs - (has_clip_distance_inputs ? 1 : 0);

        for (s32 i = 0; i < num_inputs; i++) {
            const auto& input = runtime_info.fs_info.inputs[i];
            if (input.IsDefault()) {
                continue;
            }
            const IR::Attribute param = IR::Attribute::Param0 + i;
            const u32 num_components = info.loads.NumComponents(param);
            const auto [primary, auxiliary] = info.fs_interpolation[i];
            const Id type = F32[num_components];
            const Id attr_id = [&] {
                const auto bind_location = input.param_index + (has_clip_distance_inputs ? 1 : 0);
                if (primary == Qualifier::PerVertex &&
                    profile.supports_fragment_shader_barycentric) {
                    return Name(DefineInput(TypeArray(type, ConstU32(3U)), bind_location),
                                fmt::format("fs_in_attr{}_p", i));
                }
                return Name(DefineInput(type, bind_location), fmt::format("fs_in_attr{}", i));
            }();
            if (primary == Qualifier::PerVertex) {
                Decorate(attr_id, profile.supports_amd_shader_explicit_vertex_parameter
                                      ? spv::Decoration::ExplicitInterpAMD
                                      : spv::Decoration::PerVertexKHR);
            } else if (primary != Qualifier::Smooth) {
                Decorate(attr_id, primary == Qualifier::Flat ? spv::Decoration::Flat
                                                             : spv::Decoration::NoPerspective);
            }
            if (auxiliary != Qualifier::None) {
                Decorate(attr_id, auxiliary == Qualifier::Centroid ? spv::Decoration::Centroid
                                                                   : spv::Decoration::Sample);
            }
            input_params[i] = GetAttributeInfo(AmdGpu::NumberFormat::Float, attr_id, num_components,
                                               false, false, primary == Qualifier::PerVertex);
        }

        if (has_clip_distance_inputs) {
            const auto type = F32[MaxEmulatedClipDistances];
            const auto attr_id = Name(DefineInput(type, 0), fmt::format("cldist_attr{}", 0));
            input_params[num_inputs] = GetAttributeInfo(AmdGpu::NumberFormat::Float, attr_id,
                                                        MaxEmulatedClipDistances, false);
        }
        break;
    }
    case LogicalStage::Compute:
        // [CSSPLIT] the local-id rewrite reads the real group id even when the guest only
        // consumed the local id; with the split off the condition is the old one.
        if (info.loads.GetAny(IR::Attribute::WorkgroupIndex) ||
            info.loads.GetAny(IR::Attribute::WorkgroupId) ||
            (cssplit_log2 != 0 && info.loads.GetAny(IR::Attribute::LocalInvocationId))) {
            workgroup_id =
                DefineVariable(U32[3], spv::BuiltIn::WorkgroupId, spv::StorageClass::Input);
        }
        if (info.loads.GetAny(IR::Attribute::WorkgroupIndex)) {
            num_workgroups_id =
                DefineVariable(U32[3], spv::BuiltIn::NumWorkgroups, spv::StorageClass::Input);
        }
        if (info.loads.GetAny(IR::Attribute::LocalInvocationId)) {
            local_invocation_id =
                DefineVariable(U32[3], spv::BuiltIn::LocalInvocationId, spv::StorageClass::Input);
        }
        break;
    case LogicalStage::Geometry: {
        primitive_id = DefineVariable(U32[1], spv::BuiltIn::PrimitiveId, spv::StorageClass::Input);
        const auto gl_per_vertex =
            Name(TypeStruct(F32[4], F32[1], TypeArray(F32[1], ConstU32(1u))), "gl_PerVertex");
        MemberName(gl_per_vertex, 0, "gl_Position");
        MemberName(gl_per_vertex, 1, "gl_PointSize");
        MemberName(gl_per_vertex, 2, "gl_ClipDistance");
        MemberDecorate(gl_per_vertex, 0, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::Position));
        MemberDecorate(gl_per_vertex, 1, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::PointSize));
        MemberDecorate(gl_per_vertex, 2, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::ClipDistance));
        Decorate(gl_per_vertex, spv::Decoration::Block);
        const auto num_verts_in = NumVertices(runtime_info.gs_info.in_primitive);
        const auto vertices_in = TypeArray(gl_per_vertex, ConstU32(num_verts_in));
        gl_in = Name(DefineVar(vertices_in, spv::StorageClass::Input), "gl_in");
        interfaces.push_back(gl_in);

        const auto num_params = runtime_info.gs_info.in_vertex_data_size / 4 - 1u;
        for (int param_id = 0; param_id < num_params; ++param_id) {
            const Id type{TypeArray(F32[4], ConstU32(num_verts_in))};
            const Id id{DefineInput(type, param_id)};
            Name(id, fmt::format("gs_in_attr{}", param_id));
            input_params[param_id] =
                GetAttributeInfo(AmdGpu::NumberFormat::Float, id, 4, false, false, true);
        }
        break;
    }
    case LogicalStage::TessellationControl: {
        invocation_id =
            DefineVariable(U32[1], spv::BuiltIn::InvocationId, spv::StorageClass::Input);
        patch_vertices =
            DefineVariable(U32[1], spv::BuiltIn::PatchVertices, spv::StorageClass::Input);
        primitive_id = DefineVariable(U32[1], spv::BuiltIn::PrimitiveId, spv::StorageClass::Input);

        const u32 num_attrs = Common::AlignUp(runtime_info.hs_info.ls_stride, 16) >> 4;
        if (num_attrs > 0) {
            const Id per_vertex_type{TypeArray(F32[4], ConstU32(num_attrs))};
            // The input vertex count isn't statically known, so make length 32 (what
            // glslang does)
            const Id patch_array_type{TypeArray(per_vertex_type, ConstU32(32u))};
            input_attr_array = DefineInput(patch_array_type, 0);
            Name(input_attr_array, "in_attrs");
        }
        break;
    }
    case LogicalStage::TessellationEval: {
        tess_coord = DefineInput(F32[3], std::nullopt, spv::BuiltIn::TessCoord);
        primitive_id = DefineVariable(U32[1], spv::BuiltIn::PrimitiveId, spv::StorageClass::Input);

        const u32 num_attrs =
            Common::AlignUp(runtime_info.hs_es_vs_info.hs_output_cp_stride, 16) >> 4;
        if (num_attrs > 0) {
            const Id per_vertex_type{TypeArray(F32[4], ConstU32(num_attrs))};
            // The input vertex count isn't statically known, so make length 32 (what
            // glslang does)
            const Id patch_array_type{TypeArray(per_vertex_type, ConstU32(32u))};
            input_attr_array = DefineInput(patch_array_type, 0);
            Name(input_attr_array, "in_attrs");
        }

        const u32 patch_base_location = num_attrs;
        for (size_t index = 0; index < 30; ++index) {
            if (!(info.uses_patches & (1U << index))) {
                continue;
            }
            const Id id{DefineInput(F32[4], patch_base_location + index)};
            Decorate(id, spv::Decoration::Patch);
            Name(id, fmt::format("patch_in{}", index));
            patches[index] = id;
        }
        break;
    }
    default:
        break;
    }
}

void EmitContext::DefineVertexBlock() {
    const std::array<Id, 8> zero{f32_zero_value, f32_zero_value, f32_zero_value, f32_zero_value,
                                 f32_zero_value, f32_zero_value, f32_zero_value, f32_zero_value};
    output_position = DefineVariable(F32[4], spv::BuiltIn::Position, spv::StorageClass::Output);
    const bool needs_clip_distance_emulation = l_stage == LogicalStage::Vertex &&
                                               stage == Stage::Vertex &&
                                               profile.needs_clip_distance_emulation;
    const auto has_clip_distance_outputs = info.stores.GetAny(IR::Attribute::ClipDistance);
    if (has_clip_distance_outputs && !needs_clip_distance_emulation) {
        const Id type{TypeArray(F32[1], ConstU32(8U))};
        const Id initializer{ConstantComposite(type, zero)};
        clip_distances = DefineVariable(type, spv::BuiltIn::ClipDistance, spv::StorageClass::Output,
                                        initializer);
    }
    if (info.stores.GetAny(IR::Attribute::CullDistance)) {
        const Id type{TypeArray(F32[1], ConstU32(8U))};
        const Id initializer{ConstantComposite(type, zero)};
        cull_distances = DefineVariable(type, spv::BuiltIn::CullDistance, spv::StorageClass::Output,
                                        initializer);
    }
    if (info.stores.GetAny(IR::Attribute::PointSize)) {
        output_point_size =
            DefineVariable(F32[1], spv::BuiltIn::PointSize, spv::StorageClass::Output);
    }
    if (info.stores.GetAny(IR::Attribute::RenderTargetIndex)) {
        output_layer = DefineVariable(U32[1], spv::BuiltIn::Layer, spv::StorageClass::Output);
    }
    if (info.stores.GetAny(IR::Attribute::ViewportIndex)) {
        output_viewport_index =
            DefineVariable(U32[1], spv::BuiltIn::ViewportIndex, spv::StorageClass::Output);
    }
}

void EmitContext::DefineOutputs() {
    switch (l_stage) {
    case LogicalStage::Vertex: {
        DefineVertexBlock();
        if (stage == Shader::Stage::Local) {
            const u32 num_attrs = Common::AlignUp(runtime_info.ls_info.ls_stride, 16) >> 4;
            if (num_attrs > 0) {
                const Id type{TypeArray(F32[4], ConstU32(num_attrs))};
                output_attr_array = DefineOutput(type, 0);
                Name(output_attr_array, "out_attrs");
            }
        } else {
            const bool needs_clip_distance_emulation =
                stage == Stage::Vertex && profile.needs_clip_distance_emulation &&
                info.stores.GetAny(IR::Attribute::ClipDistance);
            u32 num_attrs = 0u;
            for (u32 i = 0; i < IR::NumParams; i++) {
                const IR::Attribute param{IR::Attribute::Param0 + i};
                if (!info.stores.GetAny(param)) {
                    continue;
                }
                const u32 num_components = info.stores.NumComponents(param);
                const Id id{
                    DefineOutput(F32[num_components], i + (needs_clip_distance_emulation ? 1 : 0))};
                Name(id, fmt::format("out_attr{}", i));
                output_params[i] =
                    GetAttributeInfo(AmdGpu::NumberFormat::Float, id, num_components, true);
                ++num_attrs;
            }

            if (needs_clip_distance_emulation) {
                clip_distances = Id{DefineOutput(F32[MaxEmulatedClipDistances], 0)};
                output_params[num_attrs] = GetAttributeInfo(
                    AmdGpu::NumberFormat::Float, clip_distances, MaxEmulatedClipDistances, true);
                Name(clip_distances, fmt::format("cldist_attr{}", 0));
            }
        }
        break;
    }
    case LogicalStage::TessellationControl: {
        if (info.stores_tess_level_outer) {
            const Id type{TypeArray(F32[1], ConstU32(4U))};
            output_tess_level_outer =
                DefineOutput(type, std::nullopt, spv::BuiltIn::TessLevelOuter);
            Decorate(output_tess_level_outer, spv::Decoration::Patch);
        }
        if (info.stores_tess_level_inner) {
            const Id type{TypeArray(F32[1], ConstU32(2U))};
            output_tess_level_inner =
                DefineOutput(type, std::nullopt, spv::BuiltIn::TessLevelInner);
            Decorate(output_tess_level_inner, spv::Decoration::Patch);
        }

        const u32 num_attrs =
            Common::AlignUp(runtime_info.hs_es_vs_info.hs_output_cp_stride, 16) >> 4;
        if (num_attrs > 0) {
            const Id per_vertex_type{TypeArray(F32[4], ConstU32(num_attrs))};
            // The input vertex count isn't statically known, so make length 32 (what
            // glslang does)
            const Id patch_array_type{TypeArray(
                per_vertex_type, ConstU32(runtime_info.hs_info.NumOutputControlPoints()))};
            output_attr_array = DefineOutput(patch_array_type, 0);
            Name(output_attr_array, "out_attrs");
        }

        const u32 patch_base_location = num_attrs;
        for (size_t index = 0; index < 30; ++index) {
            if (!(info.uses_patches & (1U << index))) {
                continue;
            }
            const Id id{DefineOutput(F32[4], patch_base_location + index)};
            Decorate(id, spv::Decoration::Patch);
            Name(id, fmt::format("patch_out{}", index));
            patches[index] = id;
        }
        break;
    }
    case LogicalStage::TessellationEval: {
        DefineVertexBlock();
        for (u32 i = 0; i < IR::NumParams; i++) {
            const IR::Attribute param{IR::Attribute::Param0 + i};
            if (!info.stores.GetAny(param)) {
                continue;
            }
            const u32 num_components = info.stores.NumComponents(param);
            const Id id{DefineOutput(F32[num_components], i)};
            Name(id, fmt::format("out_attr{}", i));
            output_params[i] =
                GetAttributeInfo(AmdGpu::NumberFormat::Float, id, num_components, true);
        }
        break;
    }
    case LogicalStage::Fragment: {
        if (info.stores.Get(IR::Attribute::Depth)) {
            frag_depth = DefineVariable(F32[1], spv::BuiltIn::FragDepth, spv::StorageClass::Output);
        }
        if (info.stores.Get(IR::Attribute::SampleMask)) {
            sample_mask = DefineVariable(TypeArray(U32[1], u32_one_value), spv::BuiltIn::SampleMask,
                                         spv::StorageClass::Output);
        }
        if (info.stores.Get(IR::Attribute::StencilRef) && profile.supports_shader_stencil_export) {
            stencil_ref =
                DefineVariable(S32[1], spv::BuiltIn::FragStencilRefEXT, spv::StorageClass::Output);
        }
        u32 num_render_targets = 0;
        for (u32 i = 0; i < IR::NumRenderTargets; i++) {
            const IR::Attribute mrt{IR::Attribute::RenderTarget0 + i};
            if (!info.stores.GetAny(mrt)) {
                continue;
            }
            const u32 num_components = info.stores.NumComponents(mrt);
            const AmdGpu::NumberFormat num_format{runtime_info.fs_info.color_buffers[i].num_format};
            const Id type{GetAttributeType(*this, num_format)[num_components]};
            Id id;
            if (runtime_info.fs_info.dual_source_blending) {
                id = DefineOutput(type, 0);
                Decorate(id, spv::Decoration::Index, i);
            } else {
                id = DefineOutput(type, i);
            }
            Name(id, fmt::format("frag_color{}", i));
            frag_outputs[i] = GetAttributeInfo(num_format, id, num_components, true);
            ++num_render_targets;
        }
        // Dual source blending allows at most 2 render targets, one for each source.
        // Fewer targets are allowed but the missing blending source values will be
        // undefined.
        ASSERT_MSG(!runtime_info.fs_info.dual_source_blending || num_render_targets <= 2,
                   "Dual source blending enabled, there must be at most two MRT exports");
        break;
    }
    case LogicalStage::Geometry: {
        DefineVertexBlock();
        for (u32 attr_id = 0; attr_id < info.gs_copy_data.num_attrs; attr_id++) {
            const Id id{DefineOutput(F32[4], attr_id)};
            Name(id, fmt::format("out_attr{}", attr_id));
            output_params[attr_id] = GetAttributeInfo(AmdGpu::NumberFormat::Float, id, 4, true);
        }
        break;
    }
    case LogicalStage::Compute:
        break;
    default:
        UNREACHABLE();
    }
}

void EmitContext::DefinePushDataBlock() {
    // Create push constants block for instance steps rates
    const Id struct_type{Name(TypeStruct(F32[1], F32[1], F32[1], F32[1], U32[4], U32[4], U32[4],
                                         U32[4], U32[4], U32[4], U32[2]),
                              "AuxData")};
    Decorate(struct_type, spv::Decoration::Block);
    MemberName(struct_type, PushData::XOffsetIndex, "xoffset");
    MemberName(struct_type, PushData::YOffsetIndex, "yoffset");
    MemberName(struct_type, PushData::XScaleIndex, "xscale");
    MemberName(struct_type, PushData::YScaleIndex, "yscale");
    MemberName(struct_type, PushData::UdRegsIndex + 0, "ud_regs0");
    MemberName(struct_type, PushData::UdRegsIndex + 1, "ud_regs1");
    MemberName(struct_type, PushData::UdRegsIndex + 2, "ud_regs2");
    MemberName(struct_type, PushData::UdRegsIndex + 3, "ud_regs3");
    MemberName(struct_type, PushData::BufOffsetIndex + 0, "buf_offsets0");
    MemberName(struct_type, PushData::BufOffsetIndex + 1, "buf_offsets1");
    MemberName(struct_type, PushData::BufOffsetIndex + 2, "buf_offsets2");
    MemberDecorate(struct_type, PushData::XOffsetIndex, spv::Decoration::Offset, 0U);
    MemberDecorate(struct_type, PushData::YOffsetIndex, spv::Decoration::Offset, 4U);
    MemberDecorate(struct_type, PushData::XScaleIndex, spv::Decoration::Offset, 8U);
    MemberDecorate(struct_type, PushData::YScaleIndex, spv::Decoration::Offset, 12U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 0, spv::Decoration::Offset, 16U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 1, spv::Decoration::Offset, 32U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 2, spv::Decoration::Offset, 48U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 3, spv::Decoration::Offset, 64U);
    MemberDecorate(struct_type, PushData::BufOffsetIndex + 0, spv::Decoration::Offset, 80U);
    MemberDecorate(struct_type, PushData::BufOffsetIndex + 1, spv::Decoration::Offset, 96U);
    MemberDecorate(struct_type, PushData::BufOffsetIndex + 2, spv::Decoration::Offset, 112U);
    push_data_block = DefineVar(struct_type, spv::StorageClass::PushConstant);
    Name(push_data_block, "push_data");
    interfaces.push_back(push_data_block);
}

EmitContext::BufferSpv EmitContext::DefineBuffer(bool is_storage, bool is_written, u32 elem_shift,
                                                 BufferType buffer_type, Id data_type) {
    // Define array type.
    const Id max_num_items = ConstU32(u32(profile.max_ubo_size) >> elem_shift);
    const Id record_array_type{is_storage ? TypeRuntimeArray(data_type)
                                          : TypeArray(data_type, max_num_items)};
    // Define block struct type. Don't perform decorations twice on the same Id.
    const Id struct_type{TypeStruct(record_array_type)};
    if (std::ranges::find(buf_type_ids, record_array_type.value, &Id::value) ==
        buf_type_ids.end()) {
        Decorate(record_array_type, spv::Decoration::ArrayStride, 1 << elem_shift);
        Decorate(struct_type, spv::Decoration::Block);
        MemberName(struct_type, 0, "data");
        MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);
        buf_type_ids.push_back(record_array_type);
    }
    // Define buffer binding interface.
    const auto storage_class =
        is_storage ? spv::StorageClass::StorageBuffer : spv::StorageClass::Uniform;
    const Id struct_pointer_type{TypePointer(storage_class, struct_type)};
    const Id pointer_type = TypePointer(storage_class, data_type);
    const Id id{AddGlobalVariable(struct_pointer_type, storage_class)};
    Decorate(id, spv::Decoration::Binding, binding.unified);
    Decorate(id, spv::Decoration::DescriptorSet, 0U);
    if (is_storage && !is_written) {
        Decorate(id, spv::Decoration::NonWritable);
    }
    switch (buffer_type) {
    case BufferType::GdsBuffer:
        Name(id, "gds_buffer");
        break;
    case BufferType::Flatbuf:
        Name(id, "srt_flatbuf");
        break;
    case BufferType::ClipPlanes:
        Name(id, "clip_planes");
        break;
    case BufferType::BdaPagetable:
        Name(id, "bda_pagetable");
        break;
    case BufferType::FaultBuffer:
        Name(id, "fault_buffer");
        break;
    case BufferType::SharedMemory:
        Name(id, "ssbo_shmem");
        break;
    default:
        Name(id, fmt::format("{}_{}", is_storage ? "ssbo" : "ubo", binding.buffer));
        break;
    }
    interfaces.push_back(id);
    return {id, pointer_type};
};

void EmitContext::DefineBuffers() {
    for (const auto& desc : info.buffers) {
        const auto buf_sharp = desc.GetSharp(info);
        const bool is_storage = desc.IsStorage(buf_sharp);

        // Set indexes for special buffers.
        if (desc.buffer_type == BufferType::Flatbuf) {
            flatbuf_index = buffers.size();
        } else if (desc.buffer_type == BufferType::BdaPagetable) {
            bda_pagetable_index = buffers.size();
        } else if (desc.buffer_type == BufferType::FaultBuffer) {
            fault_buffer_index = buffers.size();
        }

        // Define aliases depending on the shader usage.
        auto& spv_buffer = buffers.emplace_back(binding.buffer++, desc.buffer_type);
        if (True(desc.used_types & IR::Type::U64)) {
            spv_buffer.Alias(PointerType::U64) =
                DefineBuffer(is_storage, desc.is_written, 3, desc.buffer_type, U64);
        }
        if (True(desc.used_types & IR::Type::U32)) {
            spv_buffer.Alias(PointerType::U32) =
                DefineBuffer(is_storage, desc.is_written, 2, desc.buffer_type, U32[1]);
        }
        if (True(desc.used_types & IR::Type::F32)) {
            spv_buffer.Alias(PointerType::F32) =
                DefineBuffer(is_storage, desc.is_written, 2, desc.buffer_type, F32[1]);
        }
        if (True(desc.used_types & IR::Type::U16)) {
            spv_buffer.Alias(PointerType::U16) =
                DefineBuffer(is_storage, desc.is_written, 1, desc.buffer_type, U16);
        }
        if (True(desc.used_types & IR::Type::U8)) {
            spv_buffer.Alias(PointerType::U8) =
                DefineBuffer(is_storage, desc.is_written, 0, desc.buffer_type, U8);
        }
        ++binding.unified;
    }
}

spv::ImageFormat GetFormat(const AmdGpu::Image& image) {
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::R32ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Sint) {
        return spv::ImageFormat::R32i;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::R32f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rg32f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::Rg32ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32_32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::Rgba32ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::R16f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::R16ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rg16f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Snorm) {
        return spv::ImageFormat::Rg16Snorm;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8_8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::Rg8;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16_16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rgba16f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16_16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::Rgba16;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::R8;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::Rgba8;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::Rgba8ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format10_11_11 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::R11fG11fB10f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32_32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rgba32f;
    }
    UNREACHABLE_MSG("Unknown storage format data_format={}, num_format={}", image.GetDataFmt(),
                    image.GetNumberFmt());
}

Id ImageType(EmitContext& ctx, const ImageResource& desc, Id sampled_type) {
    const auto image = desc.GetSharp(ctx.info);
    const auto format = desc.is_atomic ? GetFormat(image) : spv::ImageFormat::Unknown;
    const auto type = image.GetViewType(desc.is_array);
    const u32 sampled = desc.is_written ? 2 : 1;
    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim1D, false, false, false, sampled, format);
    case AmdGpu::ImageType::Color1DArray:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim1D, false, true, false, sampled, format);
    case AmdGpu::ImageType::Color2D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, false, false, sampled, format);
    case AmdGpu::ImageType::Color2DArray:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, true, false, sampled, format);
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, false, true, sampled, format);
    case AmdGpu::ImageType::Color3D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim3D, false, false, false, sampled, format);
    default:
        break;
    }
    UNREACHABLE_MSG("Invalid texture type {}", type);
}

void EmitContext::DefineImagesAndSamplers() {
    for (const auto& image_desc : info.images) {
        const auto sharp = image_desc.GetSharp(info);
        const auto nfmt = sharp.GetNumberFmt();
        const bool is_integer = AmdGpu::IsInteger(nfmt);
        const bool is_storage = image_desc.is_written;
        const MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const VectorIds& data_types = GetAttributeType(*this, nfmt);
        const Id sampled_type = data_types[1];
        const Id image_type{ImageType(*this, image_desc, sampled_type)};

        const u32 num_bindings = image_desc.NumBindings(info);
        Id pointee_type = image_type;
        if (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex) {
            pointee_type = TypeArray(pointee_type, ConstU32(num_bindings));
        }

        const Id pointer_type{TypePointer(spv::StorageClass::UniformConstant, pointee_type)};
        const Id id{AddGlobalVariable(pointer_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding.unified);
        binding.unified += num_bindings;
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        // TODO better naming for resources (flattened sharp_idx is not informative)
        Name(id, fmt::format("{}_{}{}", stage, "img", image_desc.sharp_idx));
        images.push_back({
            .data_types = &data_types,
            .id = id,
            .sampled_type = is_storage ? sampled_type : TypeSampledImage(image_type),
            .image_type = image_type,
            .view_type = sharp.GetViewType(image_desc.is_array),
            .is_integer = is_integer,
            .is_storage = is_storage,
            .mip_fallback_mode = mip_fallback_mode,
        });
        interfaces.push_back(id);
    }
    if (std::ranges::any_of(info.images, &ImageResource::is_atomic)) {
        image_u32 = TypePointer(spv::StorageClass::Image, U32[1]);
        image_f32 = TypePointer(spv::StorageClass::Image, F32[1]);
    }
    if (info.samplers.empty()) {
        return;
    }
    sampler_type = TypeSampler();
    sampler_pointer_type = TypePointer(spv::StorageClass::UniformConstant, sampler_type);
    for (const auto& samp_desc : info.samplers) {
        const Id id{AddGlobalVariable(sampler_pointer_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding.unified++);
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        const auto sharp_desc =
            samp_desc.is_inline_sampler
                ? fmt::format("inline:{:#x}:{:#x}", samp_desc.inline_sampler.raw0,
                              samp_desc.inline_sampler.raw1)
                : fmt::format("sgpr:{}", samp_desc.sharp_idx);
        Name(id, fmt::format("{}_{}{}", stage, "samp", sharp_desc));
        samplers.push_back(id);
        interfaces.push_back(id);
    }
}

void EmitContext::DefineSharedMemory() {
    const auto num_types = std::popcount(static_cast<u32>(info.shared_types));
    if (num_types == 0) {
        return;
    }
    ASSERT(info.stage == Stage::Compute);
    const u32 shared_memory_size = runtime_info.cs_info.shared_memory_size;

    const auto make_type = [&](IR::Type type, Id element_type, u32 element_size,
                               std::string_view name) {
        if (False(info.shared_types & type)) {
            // Skip unused shared memory types.
            return std::make_tuple(Id{}, Id{}, Id{});
        }

        const u32 num_elements{Common::DivCeil(shared_memory_size, element_size)};
        const Id array_type{TypeArray(element_type, ConstU32(num_elements))};

        const auto mem_type = [&] {
            if (num_types > 1) {
                const Id struct_type{TypeStruct(array_type)};
                Decorate(struct_type, spv::Decoration::Block);
                MemberDecorate(struct_type, 0u, spv::Decoration::Offset, 0u);
                return struct_type;
            } else {
                return array_type;
            }
        }();

        const Id pointer = TypePointer(spv::StorageClass::Workgroup, mem_type);
        const Id element_pointer = TypePointer(spv::StorageClass::Workgroup, element_type);
        const Id variable = AddGlobalVariable(pointer, spv::StorageClass::Workgroup);
        Name(variable, name);
        interfaces.push_back(variable);

        if (num_types > 1) {
            Decorate(array_type, spv::Decoration::ArrayStride, element_size);
            Decorate(variable, spv::Decoration::Aliased);
        }

        return std::make_tuple(variable, element_pointer, pointer);
    };
    std::tie(shared_memory_u16, shared_u16, shared_memory_u16_type) =
        make_type(IR::Type::U16, U16, 2u, "shared_mem_u16");
    std::tie(shared_memory_u32, shared_u32, shared_memory_u32_type) =
        make_type(IR::Type::U32, U32[1], 4u, "shared_mem_u32");
    std::tie(shared_memory_u64, shared_u64, shared_memory_u64_type) =
        make_type(IR::Type::U64, U64, 8u, "shared_mem_u64");
}

Id EmitContext::DefineFloat32ToUfloatM5(u32 mantissa_bits, const std::string_view name) {
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/util/format_r11g11b10f.h
    const auto func_type{TypeFunction(U32[1], F32[1])};
    const auto func{OpFunction(U32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto value{OpFunctionParameter(F32[1])};
    Name(func, name);
    AddLabel();

    const auto raw_value{OpBitcast(U32[1], value)};
    const auto exponent{
        OpBitcast(S32[1], OpBitFieldSExtract(U32[1], raw_value, ConstU32(23U), ConstU32(8U)))};
    const auto sign{OpBitFieldUExtract(U32[1], raw_value, ConstU32(31U), ConstU32(1U))};

    const auto is_zero{OpLogicalOr(U1[1], OpIEqual(U1[1], raw_value, ConstU32(0U)),
                                   OpIEqual(U1[1], sign, ConstU32(1U)))};
    const auto is_nan{OpIsNan(U1[1], value)};
    const auto is_inf{OpIsInf(U1[1], value)};
    const auto is_denorm{OpSLessThanEqual(U1[1], exponent, ConstS32(-15))};

    const auto denorm_mantissa{OpConvertFToU(
        U32[1],
        OpRoundEven(F32[1], OpFMul(F32[1], value,
                                   ConstF32(static_cast<float>(1 << (mantissa_bits + 14))))))};
    const auto denorm_overflow{
        OpINotEqual(U1[1], OpShiftRightLogical(U32[1], denorm_mantissa, ConstU32(mantissa_bits)),
                    ConstU32(0U))};
    const auto denorm{
        OpSelect(U32[1], denorm_overflow, ConstU32(1U << mantissa_bits), denorm_mantissa)};

    const auto norm_mantissa{OpConvertFToU(
        U32[1],
        OpRoundEven(F32[1],
                    OpLdexp(F32[1], value,
                            OpISub(S32[1], ConstS32(static_cast<int>(mantissa_bits)), exponent))))};
    const auto norm_overflow{
        OpUGreaterThanEqual(U1[1], norm_mantissa, ConstU32(2U << mantissa_bits))};
    const auto norm_final_mantissa{OpBitwiseAnd(
        U32[1],
        OpSelect(U32[1], norm_overflow, OpShiftRightLogical(U32[1], norm_mantissa, ConstU32(1U)),
                 norm_mantissa),
        ConstU32((1U << mantissa_bits) - 1))};
    const auto norm_final_exponent{OpBitcast(
        U32[1],
        OpIAdd(S32[1],
               OpSelect(S32[1], norm_overflow, OpIAdd(S32[1], exponent, ConstS32(1)), exponent),
               ConstS32(15)))};
    const auto norm{OpBitFieldInsert(U32[1], norm_final_mantissa, norm_final_exponent,
                                     ConstU32(mantissa_bits), ConstU32(5U))};

    const auto result{OpSelect(U32[1], is_zero, ConstU32(0U),
                               OpSelect(U32[1], is_nan, ConstU32(31u << mantissa_bits | 1U),
                                        OpSelect(U32[1], is_inf, ConstU32(31U << mantissa_bits),
                                                 OpSelect(U32[1], is_denorm, denorm, norm))))};

    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineUfloatM5ToFloat32(u32 mantissa_bits, const std::string_view name) {
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/util/format_r11g11b10f.h
    const auto func_type{TypeFunction(F32[1], U32[1])};
    const auto func{OpFunction(F32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto value{OpFunctionParameter(U32[1])};
    Name(func, name);
    AddLabel();

    const Id exponent{OpBitFieldUExtract(U32[1], value, ConstU32(mantissa_bits), ConstU32(5U))};
    const Id mantissa{OpBitFieldUExtract(U32[1], value, ConstU32(0U), ConstU32(mantissa_bits))};
    const Id mantissa_f{OpConvertUToF(F32[1], mantissa)};
    const Id a{OpSelect(F32[1], OpINotEqual(U1[1], mantissa, u32_zero_value),
                        OpFMul(F32[1], ConstF32(1.f / (1 << (14 + mantissa_bits))), mantissa_f),
                        f32_zero_value)};
    const Id b{OpBitcast(F32[1], OpBitwiseOr(U32[1], mantissa, ConstU32(0x7f800000U)))};
    const Id exponent_c{OpISub(U32[1], exponent, ConstU32(15U))};
    const Id scale_a{
        OpFDiv(F32[1], ConstF32(1.f),
               OpConvertUToF(F32[1], OpShiftLeftLogical(U32[1], u32_one_value,
                                                        OpSNegate(U32[1], exponent_c))))};
    const Id scale_b{OpConvertUToF(F32[1], OpShiftLeftLogical(U32[1], u32_one_value, exponent_c))};
    const Id scale{
        OpSelect(F32[1], OpSLessThan(U1[1], exponent_c, u32_zero_value), scale_a, scale_b)};
    const Id c{OpFMul(F32[1], scale,
                      OpFAdd(F32[1], ConstF32(1.f),
                             OpFDiv(F32[1], mantissa_f, ConstF32(f32(1 << mantissa_bits)))))};
    const Id result{OpSelect(F32[1], OpIEqual(U1[1], exponent, u32_zero_value), a,
                             OpSelect(F32[1], OpIEqual(U1[1], exponent, ConstU32(31U)), b, c))};
    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

// Flagging raw-address writes makes the host pull whole 16 KiB pages back into guest
// memory, which clobbers everything else the CPU wrote on those pages - the guest dies
// during boot. Kept behind SHADPS4_BDATRACK=1 until the download is made byte-accurate.
static bool BdaWriteTrackingEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BDATRACK");
        return e && e[0] == '1';
    }();
    return on;
}
// [PAGEJOURNAL step 2] SHADPS4_BDATRACK_DWORDS=0 drops the per-dword journal (atomic + 3 stores
// per dword, 2.7M entries/frame, 97% lost) while the page journal carries visibility.
// [DUALHOME] SHADPS4_DUALHOME=<ranges>: every raw store is issued a second time into the guest-
// memory home read from the window behind the page table (a sink page when the page is not
// armed). Straight-line - no new block - so the hot function keeps its shape.
static bool DualHomeEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DUALHOME");
        return e && e[0] != '\0';
    }();
    return on;
}
// Mode 1 = every raw store is duplicated into the guest home from the shader (measured: +19 ms
// per frame of uncoalesced PCIe writes); mode 2 (default) = stores stay in VRAM and a writeback
// kernel copies the journaled pages at the end of the batch.
static bool DualHomeStoreEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DUALHOME_MODE");
        return DualHomeEnabled() && e && e[0] == '1';
    }();
    return on;
}

// Mode 4: the store itself is unconditional - its address is the primary (VRAM owner) when
// the walk resolved, otherwise the guest home from the window (an import piece for armed
// ranges, the sink elsewhere). A first-touch write into a not-yet-resident page therefore lands
// in guest memory instead of being dropped (the 0x7FFFFFFF filler / assert 3543 root cause),
// and no basic block is added: OpSelect on the address, the store moves into the entry block.
static bool DualHomeFaultStoreEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DUALHOME_MODE");
        return DualHomeEnabled() && e && e[0] == '4';
    }();
    return on;
}

static bool DwordJournalEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BDATRACK_DWORDS");
        return !(e && e[0] == '0');
    }();
    return BdaWriteTrackingEnabled() && on;
}

// [DROPLOG] SHADPS4_DROPLOG=1: writes the walk could not place go to the dropped-write log
// (BufferCache::BDA_DROP_LOG_*) so the host lands them in guest memory instead of losing them.
static bool DropLogEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DROPLOG");
        return e && e[0] == '1';
    }();
    return on;
}

void EmitContext::EmitDropLog(Id addr, Id value) {
    if (!BdaWriteTrackingEnabled() || !DropLogEnabled()) {
        return;
    }
    // Stray job-system pointers (0x0, 0x10000, 0x8000000000...) made 28k dropped writes per 5 s
    // and drowned the real ones; only the guest user range is worth carrying.
    const auto in_guest{
        OpLogicalAnd(U1[1], OpUGreaterThanEqual(U1[1], addr, Constant(U64, u64{0x1000000000ULL})),
                     OpULessThan(U1[1], addr, Constant(U64, u64{0x1200000000ULL})))};
    const auto guest_label{OpLabel()};
    const auto guest_merge_label{OpLabel()};
    OpSelectionMerge(guest_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(in_guest, guest_label, guest_merge_label);
    AddLabel(guest_label);
    const auto& log_buffer{buffers[fault_buffer_index]};
    const auto [log_id, log_ptr_type] = log_buffer.Alias(PointerType::U32);
    const auto dlog_base{
        ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES / 32 +
                                  VideoCore::BufferCache::BDA_WRITE_LOG_BYTES / 4))};
    const auto count_ptr{OpAccessChain(log_ptr_type, log_id, u32_zero_value, dlog_base)};
    const auto slot{OpAtomicIAdd(U32[1], count_ptr, ConstU32(1U), ConstU32(0U), u32_one_value)};
    const auto capacity{ConstU32(VideoCore::BufferCache::BDA_DROP_LOG_ENTRIES)};
    const auto has_room{OpULessThan(U1[1], slot, capacity)};
    const auto store_label{OpLabel()};
    const auto merge_label{OpLabel()};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(has_room, store_label, merge_label);
    AddLabel(store_label);
    const auto entry_base{OpIAdd(
        U32[1], OpIAdd(U32[1], dlog_base, u32_one_value),
        OpIMul(U32[1], slot, ConstU32(VideoCore::BufferCache::BDA_WRITE_LOG_WORDS_PER_ENTRY)))};
    const auto addr_lo{OpUConvert(U32[1], OpBitwiseAnd(U64, addr, Constant(U64, u64{0xFFFFFFFF})))};
    const auto addr_hi{OpUConvert(U32[1], OpShiftRightLogical(U64, addr, Constant(U64, u64{32})))};
    OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value, entry_base), addr_lo);
    OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value,
                          OpIAdd(U32[1], entry_base, ConstU32(1U))),
            addr_hi);
    OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value,
                          OpIAdd(U32[1], entry_base, ConstU32(2U))),
            value);
    OpBranch(merge_label);
    AddLabel(merge_label);
    OpBranch(guest_merge_label);
    AddLabel(guest_merge_label);
}

// [PAGEJOURNAL] SHADPS4_PAGEJOURNAL=1: after a landed store, set the page's bit and, if the bit
// was clear, append the page index once - thousands of entries per batch instead of millions.
static bool PageJournalEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_PAGEJOURNAL");
        return e && (e[0] == '1' || e[0] == '2');
    }();
    return on;
}

// [PJCACHE] SHADPS4_PJCACHE=1: the page-journal AtomicOr is issued only when the store lands on
// a page other than the one this invocation flagged last. The skinning worker stores 7 dwords
// per vertex into one 28-byte record (liveloop215 dump + SPIR-V read 08.09), so 6 of its 7
// atomics per vertex hit a bit that the same thread has just set. Safe: the host clears the
// bitmap only between batches, never while a shader runs.
static bool PjCacheEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_PJCACHE");
        const bool v = e && e[0] == '1';
        if (v) {
            LOG_CRITICAL(Render_Recompiler,
                         "PJCACHE: page-journal atomic skipped while the invocation stays on the "
                         "page it flagged last");
        }
        return v;
    }();
    return on;
}

Id EmitContext::DefineWriteConst() {
    if (BdaWriteTrackingEnabled() && PageJournalEnabled() && PjCacheEnabled()) {
        pj_last_page = DefineVar(U32[1], spv::StorageClass::Private, ConstU32(0xFFFFFFFFU));
        Name(pj_last_page, "pj_last_page");
        interfaces.push_back(pj_last_page);
    }
    const auto func_type{TypeFunction(void_id, U32[2], U32[1], U32[1])};
    const auto func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type)};
    const auto base{OpFunctionParameter(U32[2])};
    const auto offset{OpFunctionParameter(U32[1])};
    const auto value{OpFunctionParameter(U32[1])};
    Name(func, "write_const_live");
    AddLabel();

    const auto base_lo{OpUConvert(U64, OpCompositeExtract(U32[1], base, 0))};
    const auto base_hi{OpUConvert(U64, OpCompositeExtract(U32[1], base, 1))};
    const auto base_shift{OpShiftLeftLogical(U64, base_hi, ConstU32(32U))};
    const auto base_addr{OpBitwiseOr(U64, base_lo, base_shift)};
    const auto offset_bytes{OpShiftLeftLogical(U32[1], offset, ConstU32(2U))};
    const auto addr{OpIAdd(U64, base_addr, OpUConvert(U64, offset_bytes))};

    if (DualHomeStoreEnabled()) {
        const auto& dual_buffer{buffers[bda_pagetable_index]};
        const auto [dual_buffer_id, dual_pointer_type] = dual_buffer.Alias(PointerType::U64);
        const auto page{OpShiftRightLogical(
            U64, addr, Constant(U64, u64{VideoCore::BufferCache::CACHING_PAGEBITS}))};
        const auto rel{
            OpISub(U64, page, Constant(U64, VideoCore::BufferCache::DUAL_WINDOW_BASE_PAGE))};
        const auto idx{OpUMin(U64, rel, Constant(U64, VideoCore::BufferCache::DUAL_WINDOW_PAGES))};
        const auto idx32{OpUConvert(U32[1], idx)};
        const auto tidx{OpIAdd(
            U32[1], idx32, ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES)))};
        const auto eptr{OpAccessChain(dual_pointer_type, dual_buffer_id, u32_zero_value, tidx)};
        const auto e2{
            OpBitwiseAnd(U64, OpLoad(U64, eptr), Constant(U64, u64{0x7FFFFFFFFFFFFFFFULL}))};
        const auto off{
            OpBitwiseAnd(U64, addr, Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1))};
        const auto home{OpIAdd(U64, e2, off)};
        const auto p2{OpConvertUToPtr(physical_pointer_type_u32, home)};
        OpStore(p2, value, spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned, 4u);
    }

    const Id available_label = OpLabel();
    const Id merge_label = OpLabel();

    const Id bda = OpFunctionCall(U64, get_bda_pointer, addr);
    const Id is_available = OpINotEqual(U1[1], bda, u64_zero_value);
    const bool fault_store = DualHomeFaultStoreEnabled();
    if (fault_store) {
        const auto& dual_buffer{buffers[bda_pagetable_index]};
        const auto [dual_buffer_id, dual_pointer_type] = dual_buffer.Alias(PointerType::U64);
        const auto page{OpShiftRightLogical(
            U64, addr, Constant(U64, u64{VideoCore::BufferCache::CACHING_PAGEBITS}))};
        const auto rel{
            OpISub(U64, page, Constant(U64, VideoCore::BufferCache::DUAL_WINDOW_BASE_PAGE))};
        const auto idx{OpUMin(U64, rel, Constant(U64, VideoCore::BufferCache::DUAL_WINDOW_PAGES))};
        const auto idx32{OpUConvert(U32[1], idx)};
        const auto tidx{OpIAdd(
            U32[1], idx32, ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES)))};
        const auto eptr{OpAccessChain(dual_pointer_type, dual_buffer_id, u32_zero_value, tidx)};
        const auto e2{
            OpBitwiseAnd(U64, OpLoad(U64, eptr), Constant(U64, u64{0x7FFFFFFFFFFFFFFFULL}))};
        const auto off{
            OpBitwiseAnd(U64, addr, Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1))};
        const auto home{OpIAdd(U64, e2, off)};
        const auto dst{OpSelect(U64, is_available, bda, home)};
        const Id dst_ptr = OpConvertUToPtr(physical_pointer_type_u32, dst);
        OpStore(dst_ptr, value, spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned,
                4u);
    }
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    // [DROPLOG] Only with the log on does the fault side get a block of its own: the extra
    // block alone (even empty) made every worker dispatch ~2x slower on NVIDIA (0903e:
    // disp 16 -> 30 ms/frame with the env unset), so the default keeps the two-block shape.
    if (BdaWriteTrackingEnabled() && DropLogEnabled()) {
        const Id drop_label = OpLabel();
        OpBranchConditional(is_available, available_label, drop_label);
        AddLabel(drop_label);
        EmitDropLog(addr, value);
        OpBranch(merge_label);
    } else {
        OpBranchConditional(is_available, available_label, merge_label);
    }

    // Available: volatile store so concurrent readers (CPU pollers, other
    // dispatches) see it immediately. Unmapped pages drop the write, matching
    // hardware behaviour for stray job-system pointers.
    AddLabel(available_label);
    if (!fault_store) {
        const Id addr_ptr = OpConvertUToPtr(physical_pointer_type_u32, bda);
        OpStore(addr_ptr, value, spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned,
                4u);
    }
    // Nothing on the host can observe a write made through a raw device address, so the
    // address and value are appended to a log that the host replays into guest memory.
    // Page-granular flagging was tried first and corrupted the guest: the host then had to
    // copy back whole 16 KiB pages, wiping everything the CPU had written on them.
    if (BdaWriteTrackingEnabled() && PageJournalEnabled()) {
        const auto& pj_buffer{buffers[fault_buffer_index]};
        const auto [pj_id, pj_ptr_type] = pj_buffer.Alias(PointerType::U32);
        const auto pj_bitmap_base{
            ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES / 32 +
                                      VideoCore::BufferCache::BDA_LOGS_BYTES / 4))};
        const auto pj_list_base{
            ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES / 32 +
                                      VideoCore::BufferCache::BDA_LOGS_BYTES / 4 +
                                      VideoCore::BufferCache::PAGEJ_BITMAP_BYTES / 4))};
        const auto page32{OpUConvert(
            U32[1], OpShiftRightLogical(
                        U64, addr, Constant(U64, u64{VideoCore::BufferCache::CACHING_PAGEBITS})))};
        const auto word_idx{
            OpIAdd(U32[1], pj_bitmap_base, OpShiftRightLogical(U32[1], page32, ConstU32(5U)))};
        const auto bit{
            OpShiftLeftLogical(U32[1], u32_one_value, OpBitwiseAnd(U32[1], page32, ConstU32(31U)))};
        const auto word_ptr{OpAccessChain(pj_ptr_type, pj_id, u32_zero_value, word_idx)};
        // [PJCACHE] outer branch: skip everything when the invocation already flagged this page
        const bool pj_cache = PjCacheEnabled();
        Id pj_skip_label{}, pj_do_label{};
        if (pj_cache) {
            pj_skip_label = OpLabel();
            pj_do_label = OpLabel();
            const auto last{OpLoad(U32[1], pj_last_page)};
            const auto same{OpIEqual(U1[1], last, page32)};
            OpSelectionMerge(pj_skip_label, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(same, pj_skip_label, pj_do_label);
            AddLabel(pj_do_label);
            OpStore(pj_last_page, page32);
        }
        const auto old{OpAtomicOr(U32[1], word_ptr, ConstU32(1U), ConstU32(0U), bit)};
        const auto is_new{OpIEqual(U1[1], OpBitwiseAnd(U32[1], old, bit), u32_zero_value)};
        const auto pj_store_label{OpLabel()};
        const auto pj_merge_label{OpLabel()};
        OpSelectionMerge(pj_merge_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(is_new, pj_store_label, pj_merge_label);
        AddLabel(pj_store_label);
        const auto count_ptr{OpAccessChain(pj_ptr_type, pj_id, u32_zero_value, pj_list_base)};
        const auto slot{OpAtomicIAdd(U32[1], count_ptr, ConstU32(1U), ConstU32(0U), u32_one_value)};
        // overflow: the last slot is overwritten (the host reads min(count, capacity))
        const auto slot_c{
            OpUMin(U32[1], slot, ConstU32(VideoCore::BufferCache::PAGEJ_LIST_ENTRIES - 1))};
        OpStore(OpAccessChain(pj_ptr_type, pj_id, u32_zero_value,
                              OpIAdd(U32[1], OpIAdd(U32[1], pj_list_base, u32_one_value), slot_c)),
                page32);
        OpBranch(pj_merge_label);
        AddLabel(pj_merge_label);
        if (pj_cache) {
            OpBranch(pj_skip_label);
            AddLabel(pj_skip_label);
        }
    }
    if (DwordJournalEnabled()) {
        const auto& log_buffer{buffers[fault_buffer_index]};
        const auto [log_id, log_ptr_type] = log_buffer.Alias(PointerType::U32);
        // The log sits directly after the fault bitmap in the same buffer.
        const auto log_base{
            ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES / 32))};
        const auto count_ptr{OpAccessChain(log_ptr_type, log_id, u32_zero_value, log_base)};
        const auto slot{OpAtomicIAdd(U32[1], count_ptr, ConstU32(1U), ConstU32(0U), u32_one_value)};
        const auto capacity{ConstU32(VideoCore::BufferCache::BDA_WRITE_LOG_ENTRIES)};
        const auto has_room{OpULessThan(U1[1], slot, capacity)};
        const auto store_label{OpLabel()};
        const auto log_merge_label{OpLabel()};
        OpSelectionMerge(log_merge_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(has_room, store_label, log_merge_label);

        AddLabel(store_label);
        const auto entry_base{OpIAdd(
            U32[1], OpIAdd(U32[1], log_base, u32_one_value),
            OpIMul(U32[1], slot, ConstU32(VideoCore::BufferCache::BDA_WRITE_LOG_WORDS_PER_ENTRY)))};
        const auto addr_lo{
            OpUConvert(U32[1], OpBitwiseAnd(U64, addr, Constant(U64, u64{0xFFFFFFFF})))};
        const auto addr_hi{
            OpUConvert(U32[1], OpShiftRightLogical(U64, addr, Constant(U64, u64{32})))};
        OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value, entry_base), addr_lo);
        OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value,
                              OpIAdd(U32[1], entry_base, ConstU32(1U))),
                addr_hi);
        OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value,
                              OpIAdd(U32[1], entry_base, ConstU32(2U))),
                value);
        OpBranch(log_merge_label);

        AddLabel(log_merge_label);
    }
    OpBranch(merge_label);

    AddLabel(merge_label);
    OpReturn();
    OpFunctionEnd();
    return func;
}

// [BDACACHE] SHADPS4_BDACACHE=1|2: remember the last page translation of this invocation.
// Every BDA read is two dependent memory accesses: load the 64-bit page-table entry for
// page = addr >> 14, then load the data at entry + (addr & 16383). The ND skinning worker
// 0x8f067540 reads inside very few pages (bone palette, influence list, 8-byte header) yet pays
// the entry load ~32 times per vertex - and the driver cannot hoist it: bda_pagetable is a
// written SSBO without Restrict and the walk itself stores into the fault bitmap, so every
// call must re-read the entry (see the twelve identical %450..%482 loads in cs_8f067540.dis).
// A Private pair (page id, entry) is provably not aliased by any StorageBuffer store, so a hit
// is plain register arithmetic. Same shape as [POLLMAP-ONCE], which already caches "the page
// this invocation last touched" for the poll bitmap.
//
// Level 1: read_const, read_const_dynamic, read_const_live_nv, read_const_live_x4_nv - all
//          non-volatile by construction, so remembering a translation weakens nothing that
//          the non-volatile load did not already give up.
// Level 2: + read_const_live_x4 (volatile record loads in the worker's inner loop). The 16-byte
//          load stays Volatile; only the translation is remembered. This is the real experiment:
//          VEC4STRAIGHT already collapsed the straight-line part and moved nothing.
// Never:   read_const_live (scalar) and read_const_live_poll are the poll loads that must see
//          host writes made while the dispatch runs (emit_spirv_context_get_set.cpp: liveloop44
//          hung when they went non-volatile), and every write_const flavour - a store through a
//          stale translation corrupts memory instead of just reading stale bytes.
// A zero entry is NEVER cached: a page can be published mid-dispatch, so an unresolved read must
// keep walking (and keep marking the fault bitmap) until the host repairs it.
// Off (unset/0) emits byte-identical SPIR-V: nothing below runs.
static int BdaCacheLevel() {
    static const int v = [] {
        const char* e = std::getenv("SHADPS4_BDACACHE");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}

// [BDACACHE-ONLY] SHADPS4_BDACACHE_ONLY=<hex hash>[,...]: apply the cache to these shaders only.
// Without it the switch rewrites every DMA shader in the game, so a frame-time delta cannot be
// attributed to the one worker under test and the blast radius is the whole renderer.
static bool BdaCacheWants(u64 pgm_hash) {
    static const std::vector<u64> only = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_BDACACHE_ONLY");
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
    return only.empty() || std::find(only.begin(), only.end(), pgm_hash) != only.end();
}

Id EmitContext::DefineGetBdaPointer() {
    const auto caching_pagebits{
        Constant(U64, static_cast<u64>(VideoCore::BufferCache::CACHING_PAGEBITS))};
    const auto caching_pagemask{Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1)};

    const auto func_type{TypeFunction(U64, U64)};
    const auto func{OpFunction(U64, spv::FunctionControlMask::MaskNone, func_type)};
    const auto address{OpFunctionParameter(U64)};
    Name(func, "get_bda_pointer");
    const auto entry_label{AddLabel()};

    const auto fault_label{OpLabel()};
    const auto available_label{OpLabel()};
    const auto merge_label{OpLabel()};
    const auto range_ok_label{OpLabel()};
    const auto range_merge_label{OpLabel()};

    // Get page BDA
    const auto page{OpShiftRightLogical(U64, address, caching_pagebits)};

    // A garbage guest pointer (job-system structures read before they are
    // seeded) can exceed the 40-bit guest space. Real hardware returns junk
    // for such reads; walking the page table with the raw index instead
    // performs out-of-bounds reads of the table and out-of-bounds writes of
    // the fault bitmap - the driver kills the device (ReadInvalid fault,
    // queue catatonia). Out-of-range pages resolve to null (read as zero)
    // without touching the fault machinery.
    const auto num_pages{Constant(U64, u64{VideoCore::BufferCache::CACHING_NUMPAGES})};
    const auto page_in_range{OpULessThan(U1[1], page, num_pages)};
    OpSelectionMerge(range_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(page_in_range, range_ok_label, range_merge_label);

    AddLabel(range_ok_label);
    const auto page32{OpUConvert(U32[1], page)};
    const auto& bda_buffer{buffers[bda_pagetable_index]};
    const auto [bda_buffer_id, bda_pointer_type] = bda_buffer.Alias(PointerType::U64);
    const auto bda_ptr{OpAccessChain(bda_pointer_type, bda_buffer_id, u32_zero_value, page32)};
    const auto bda{OpLoad(U64, bda_ptr)};

    // Check if page is GPU cached.
    // [bisect] SHADPS4_BDA_NODEREF=1 forces every lookup down the fault path, so the shader
    // still reads the page table, still marks the page and still makes the host cache it -
    // it just never dereferences the resolved address. Everything else about direct memory
    // access stays identical, which splits the mechanism cleanly in half.
    static const bool no_deref = [] {
        const char* e = std::getenv("SHADPS4_BDA_NODEREF");
        return e && e[0] == '1';
    }();
    // [bisect] SHADPS4_BDATAG=1: the host stamps the page index into the top 16 bits of every
    // entry. Refusing to dereference an entry whose stamp does not match the page we asked for
    // turns "correct-looking address that points at nothing" into a plain cache miss.
    static const u32 tag_mode = [] {
        const char* e = std::getenv("SHADPS4_BDATAG");
        return e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    const bool tag_entries = tag_mode != 0;
    const auto bda_addr_mask{Constant(U64, u64{0x0000FFFFFFFFFFFFULL})};
    const auto bda_real{tag_entries ? OpBitwiseAnd(U64, bda, bda_addr_mask) : bda};
    Id tag_bad{};
    if (tag_entries) {
        const auto want_tag{tag_mode == 2 ? Constant(U64, u64{1})
                                          : OpBitwiseAnd(U64, page, Constant(U64, u64{0xFFFF}))};
        const auto have_tag{OpShiftRightLogical(U64, bda, Constant(U64, u64{48}))};
        tag_bad = OpINotEqual(U1[1], have_tag, want_tag);
    }
    const auto zero_or_bad{no_deref ? OpIEqual(U1[1], u64_zero_value, u64_zero_value)
                                    : OpIEqual(U1[1], bda_real, u64_zero_value)};
    const auto is_fault{tag_entries && !no_deref ? OpLogicalOr(U1[1], zero_or_bad, tag_bad)
                                                 : zero_or_bad};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(is_fault, fault_label, available_label);

    // First time acces, mark as fault
    AddLabel(fault_label);
    const auto& fault_buffer{buffers[fault_buffer_index]};
    const auto [fault_buffer_id, fault_pointer_type] = fault_buffer.Alias(PointerType::U32);
    if (tag_entries && !no_deref) {
        // Count how often the stamp actually disagreed. Without this number a run that simply
        // died earlier for another reason would look like proof that the check helped.
        const auto counter_index{
            ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES / 32) + 1u)};
        const auto counter_ptr{
            OpAccessChain(fault_pointer_type, fault_buffer_id, u32_zero_value, counter_index)};
        const auto bump{OpSelect(U32[1], tag_bad, u32_one_value, u32_zero_value)};
        OpAtomicIAdd(U32[1], counter_ptr, ConstU32(1U), ConstU32(0U), bump);
    }
    const auto page_div32{OpShiftRightLogical(U32[1], page32, ConstU32(5U))};
    const auto page_mod32{OpBitwiseAnd(U32[1], page32, ConstU32(31U))};
    const auto page_mask{OpShiftLeftLogical(U32[1], u32_one_value, page_mod32)};
    const auto fault_ptr{
        OpAccessChain(fault_pointer_type, fault_buffer_id, u32_zero_value, page_div32)};
    const auto fault_value{OpLoad(U32[1], fault_ptr)};
    const auto fault_value_masked{OpBitwiseOr(U32[1], fault_value, page_mask)};
    OpStore(fault_ptr, fault_value_masked);

    // Return null pointer
    const auto fallback_result{u64_zero_value};
    OpBranch(merge_label);

    // Value is available, compute address
    AddLabel(available_label);
    const auto offset_in_bda{OpBitwiseAnd(U64, address, caching_pagemask)};
    const auto addr{OpIAdd(U64, bda_real, offset_in_bda)};
    OpBranch(merge_label);

    // Merge
    AddLabel(merge_label);
    const auto result{OpPhi(U64, addr, available_label, fallback_result, fault_label)};
    OpBranch(range_merge_label);

    AddLabel(range_merge_label);
    const auto final_result{OpPhi(U64, result, merge_label, u64_zero_value, entry_label)};
    OpReturnValue(final_result);
    OpFunctionEnd();
    return func;
}

// [BDACACHE] Wrapper around get_bda_pointer, not a change inside it: the honest walk stays the
// only place where the range check, the fault-bitmap marking, BDATAG and BDA_NODEREF live, and
// polls and stores keep calling it directly.
//
// Four basic blocks, and no memory access at all on a hit. The "remember it" selection that would
// have sat inside the miss path is folded into an OpSelect on purpose: write_const_live records
// (see above) that one extra basic block - even an empty one - measured ~2x on these workers.
//
// Bit-exact by construction: the slot holds result - offset (i.e. the entry itself), and a hit
// adds the new offset back, so whatever the entry encodes (BDATAG stamp masked off, landed bit)
// comes back exactly as the walk would have returned it while the entry is unchanged.
Id EmitContext::DefineGetBdaPointerCached() {
    const auto caching_pagebits{
        Constant(U64, static_cast<u64>(VideoCore::BufferCache::CACHING_PAGEBITS))};
    const auto caching_pagemask{Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1)};
    const auto num_pages{Constant(U64, u64{VideoCore::BufferCache::CACHING_NUMPAGES})};
    // Sentinel "nothing cached". Unreachable as a real page id: an in-range page is < 2^26.
    const auto no_page{ConstU32(0xFFFFFFFFU)};

    // Two Private slots and no more: this shader is register-starved (LocalSize 1024 leaves 64
    // registers per thread), so the page id is kept as u32 (3 dwords in total, not 4).
    bdacache_page = DefineVar(U32[1], spv::StorageClass::Private, no_page);
    Name(bdacache_page, "bdacache_page");
    interfaces.push_back(bdacache_page); // SPIR-V 1.4: every global is an interface
    bdacache_base = DefineVar(U64, spv::StorageClass::Private, u64_zero_value);
    Name(bdacache_base, "bdacache_base");
    interfaces.push_back(bdacache_base);

    const auto func_type{TypeFunction(U64, U64)};
    const auto func{OpFunction(U64, spv::FunctionControlMask::MaskNone, func_type)};
    const auto address{OpFunctionParameter(U64)};
    Name(func, "get_bda_pointer_cached");
    AddLabel();

    const auto hit_label{OpLabel()};
    const auto miss_label{OpLabel()};
    const auto merge_label{OpLabel()};

    const auto page{OpShiftRightLogical(U64, address, caching_pagebits)};
    const auto offset_in_page{OpBitwiseAnd(U64, address, caching_pagemask)};
    // Truncating the page id to 32 bits is exact only inside the guest space. The range check is
    // what stops a stray job-system pointer (page >= CACHING_NUMPAGES, which the walk resolves to
    // null) from matching a cached page by its low bits and getting data instead of zero.
    const auto in_range{OpULessThan(U1[1], page, num_pages)};
    const auto page32{OpUConvert(U32[1], page)};
    const auto same_page{OpIEqual(U1[1], page32, OpLoad(U32[1], bdacache_page))};
    const auto hit{OpLogicalAnd(U1[1], in_range, same_page)};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(hit, hit_label, miss_label);

    // Hit: the same arithmetic the walk ends with, straight out of registers.
    AddLabel(hit_label);
    const auto hit_result{OpIAdd(U64, OpLoad(U64, bdacache_base), offset_in_page)};
    OpBranch(merge_label);

    // Miss: the honest walk, then remember it - but only if it resolved. A zero entry means the
    // page is not resident yet; caching that would turn a page the host publishes a microsecond
    // later into a permanent zero for the rest of the dispatch, and would swallow the fault mark
    // the walk makes on every unresolved read.
    AddLabel(miss_label);
    const auto miss_result{OpFunctionCall(U64, get_bda_pointer, address)};
    const auto resolved{OpINotEqual(U1[1], miss_result, u64_zero_value)};
    OpStore(bdacache_base, OpISub(U64, miss_result, offset_in_page));
    OpStore(bdacache_page, OpSelect(U32[1], resolved, page32, no_page));
    OpBranch(merge_label);

    AddLabel(merge_label);
    const auto result{OpPhi(U64, hit_result, hit_label, miss_result, miss_label)};
    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineReadConst(bool dynamic, bool live, bool is_volatile) {
    const auto func_type{!dynamic ? TypeFunction(U32[1], U32[2], U32[1], U32[1])
                                  : TypeFunction(U32[1], U32[2], U32[1])};
    const auto func{OpFunction(U32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto base{OpFunctionParameter(U32[2])};
    const auto offset{OpFunctionParameter(U32[1])};
    const auto flatbuf_offset{!dynamic ? OpFunctionParameter(U32[1]) : Id{}};
    const char* const fname = live ? (is_volatile ? "read_const_live" : "read_const_live_nv")
                                   : (dynamic ? "read_const_dynamic" : "read_const");
    Name(func, fname);
    AddLabel();

    const auto base_lo{OpUConvert(U64, OpCompositeExtract(U32[1], base, 0))};
    const auto base_hi{OpUConvert(U64, OpCompositeExtract(U32[1], base, 1))};
    const auto base_shift{OpShiftLeftLogical(U64, base_hi, ConstU32(32U))};
    const auto base_addr{OpBitwiseOr(U64, base_lo, base_shift)};
    const auto offset_bytes{OpShiftLeftLogical(U32[1], offset, ConstU32(2U))};
    const auto addr{OpIAdd(U64, base_addr, OpUConvert(U64, offset_bytes))};

    // [BDACACHE] The volatile scalar live flavour is the poll: its loads are the loop-control
    // reads (queue index, flags) that must observe host writes made during the dispatch, so it
    // always walks honestly. The other three are non-volatile by construction and take the
    // cached walk from level 1 on.
    const bool volatile_load = live && is_volatile;
    const Id walker =
        (!volatile_load && BdaCacheLevel() >= 1 && Sirit::ValidId(get_bda_pointer_cached))
            ? get_bda_pointer_cached
            : Id{};
    if (Sirit::ValidId(walker)) {
        LOG_CRITICAL(Render_Recompiler, "BDACACHE: shader {:#x} {} -> get_bda_pointer_cached",
                     info.pgm_hash, fname);
    }

    const auto result = EmitDwordMemoryRead(
        addr,
        [&]() {
            if (dynamic) {
                return u32_zero_value;
            } else {
                return EmitFlatbufferLoad(flatbuf_offset);
            }
        },
        volatile_load, walker);

    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

// [POLLJ] SHADPS4_PAGETRACE_POLLJ=1: read_const_live plus, in the available block right after
// the volatile load, one store of the shader hash into the presence-window slot of the page that
// was read (BufferCache::POLLJ_*). Straight-line - a single extra basic block in a hot function
// measured 2x on the workers (see write_const_live) - and a separate function, so read_const_live
// and every module compiled with the env off stay byte-identical. Only scalar poll-loop reads
// (PollReadConstFlag) call it.
// [POLLMAP] SHADPS4_POLLMAP=1: the same function also marks the polled page in the host-visible
// poll bitmap (BufferCache::POLLMAP_*), whose device address sits in a slot of the fault buffer:
// page = addr >> 14; if the page's bit in the fine map is clear (a plain volatile load first, so
// a page pays its two atomics exactly once), OpAtomicOr it and the region's bit in the summary.
// Two nested selections in the available block, both skipped once the page is marked. Each
// feature is gated on its own env, so POLLJ alone emits exactly what it emitted before.
Id EmitContext::DefineReadConstLivePoll() {
    const auto func_type{TypeFunction(U32[1], U32[2], U32[1])};
    const auto func{OpFunction(U32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto base{OpFunctionParameter(U32[2])};
    const auto offset{OpFunctionParameter(U32[1])};
    Name(func, "read_const_live_poll");
    AddLabel();

    const auto base_lo{OpUConvert(U64, OpCompositeExtract(U32[1], base, 0))};
    const auto base_hi{OpUConvert(U64, OpCompositeExtract(U32[1], base, 1))};
    const auto base_shift{OpShiftLeftLogical(U64, base_hi, ConstU32(32U))};
    const auto base_addr{OpBitwiseOr(U64, base_lo, base_shift)};
    const auto offset_bytes{OpShiftLeftLogical(U32[1], offset, ConstU32(2U))};
    const auto address{OpIAdd(U64, base_addr, OpUConvert(U64, offset_bytes))};

    // Same shape as EmitDwordMemoryRead(address, zero fallback, /*is_volatile=*/true).
    const Id available_label = OpLabel();
    const Id fallback_label = OpLabel();
    const Id merge_label = OpLabel();

    const Id addr = OpFunctionCall(U64, get_bda_pointer, address);
    const Id is_available = OpINotEqual(U1[1], addr, u64_zero_value);
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(is_available, available_label, fallback_label);

    AddLabel(available_label);
    // The block the available path ends in: the POLLMAP selections below move it.
    Id available_end = available_label;
    const Id addr_ptr = OpConvertUToPtr(physical_pointer_type_u32, addr);
    const Id result = OpLoad(U32[1], addr_ptr,
                             spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned, 4u);
    // Presence: slot = page - DUAL_WINDOW_BASE_PAGE, clamped to the dummy slot for pages outside
    // the window (pages below the base wrap to a huge value and clamp the same way).
    if (VideoCore::BufferCache::PollTraceEnabled()) {
        const auto& presence_buffer{buffers[fault_buffer_index]};
        const auto [presence_id, presence_ptr_type] = presence_buffer.Alias(PointerType::U32);
        const auto page{OpShiftRightLogical(
            U64, address, Constant(U64, u64{VideoCore::BufferCache::CACHING_PAGEBITS}))};
        const auto rel{
            OpISub(U64, page, Constant(U64, VideoCore::BufferCache::DUAL_WINDOW_BASE_PAGE))};
        const auto idx{OpUMin(U64, rel, Constant(U64, VideoCore::BufferCache::POLLJ_WINDOW_PAGES))};
        const auto slot{
            OpIAdd(U32[1], ConstU32(static_cast<u32>(VideoCore::BufferCache::POLLJ_BASE_DWORDS)),
                   OpUConvert(U32[1], idx))};
        OpStore(OpAccessChain(presence_ptr_type, presence_id, u32_zero_value, slot),
                ConstU32(static_cast<u32>(info.pgm_hash) | 1u));
    }
    // [POLLMAP] mark the polled page in the host-visible bitmap (once per page).
    if (VideoCore::BufferCache::PollMapEnabled()) {
        // [POLLMAP-ONCE] 06.09: the volatile load of the fine word came from HOST memory on
        // every poll iteration; three stand runs hung within seconds of the workers getting
        // busy (0904r..r4). A per-invocation Private slot remembers the last page this
        // invocation marked, so a poll loop pays the host load once, not per iteration.
        const Id last_page_var{
            DefineVar(U64, spv::StorageClass::Private, Constant(U64, u64{0xFFFFFFFFFFFFFFFFull}))};
        Name(last_page_var, "pollmap_last_page");
        interfaces.push_back(last_page_var); // SPIR-V 1.4: every global is an interface
        const auto& slot_buffer{buffers[fault_buffer_index]};
        const auto [slot_id, slot_ptr_type] = slot_buffer.Alias(PointerType::U32);
        // Bitmap device address from the fault buffer slot: {lo, hi} at PollMapSlotDwords().
        const auto slot_dw{ConstU32(VideoCore::BufferCache::PollMapSlotDwords())};
        const auto map_lo{
            OpLoad(U32[1], OpAccessChain(slot_ptr_type, slot_id, u32_zero_value, slot_dw))};
        const auto map_hi{OpLoad(
            U32[1], OpAccessChain(slot_ptr_type, slot_id, u32_zero_value,
                                  ConstU32(VideoCore::BufferCache::PollMapSlotDwords() + 1)))};
        const auto map_base{
            OpBitwiseOr(U64, OpUConvert(U64, map_lo),
                        OpShiftLeftLogical(U64, OpUConvert(U64, map_hi), ConstU32(32U)))};
        const auto page{OpShiftRightLogical(
            U64, address, Constant(U64, u64{VideoCore::BufferCache::CACHING_PAGEBITS}))};
        // Outside the 40-bit guest space (stray job pointers) or no bitmap published: skip.
        const auto in_range{
            OpULessThan(U1[1], page, Constant(U64, u64{VideoCore::BufferCache::CACHING_NUMPAGES}))};
        const auto have_map{OpINotEqual(U1[1], map_base, u64_zero_value)};
        const auto last_page{OpLoad(U64, last_page_var)};
        const auto new_page{OpINotEqual(U1[1], page, last_page)};
        const auto do_mark{OpLogicalAnd(U1[1], OpLogicalAnd(U1[1], in_range, have_map), new_page)};
        const Id mark_label = OpLabel();
        const Id mark_merge_label = OpLabel();
        OpSelectionMerge(mark_merge_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(do_mark, mark_label, mark_merge_label);

        AddLabel(mark_label);
        OpStore(last_page_var, page);
        // fine word: (page >> 5) * 4 bytes from the base; bit: page & 31
        const auto fine_word_off{OpShiftLeftLogical(
            U64, OpShiftRightLogical(U64, page, Constant(U64, u64{5})), Constant(U64, u64{2}))};
        const auto fine_ptr{
            OpConvertUToPtr(physical_pointer_type_u32, OpIAdd(U64, map_base, fine_word_off))};
        // Volatile like the data load: a hoisted copy of the word would keep the atomics firing
        // on every iteration of the poll after the first mark.
        const auto fine_word{
            OpLoad(U32[1], fine_ptr,
                   spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned, 4u)};
        const auto fine_bit{OpShiftLeftLogical(
            U32[1], u32_one_value,
            OpUConvert(U32[1], OpBitwiseAnd(U64, page, Constant(U64, u64{31}))))};
        const auto unmarked{
            OpIEqual(U1[1], OpBitwiseAnd(U32[1], fine_word, fine_bit), u32_zero_value)};
        const Id atomic_label = OpLabel();
        const Id atomic_merge_label = OpLabel();
        OpSelectionMerge(atomic_merge_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(unmarked, atomic_label, atomic_merge_label);

        AddLabel(atomic_label);
        // Device scope, relaxed: the host only ever needs the bits eventually.
        OpAtomicOr(U32[1], fine_ptr, ConstU32(1U), ConstU32(0U), fine_bit);
        // summary word: (page >> 15) * 4 bytes behind the fine map; bit: (page >> 10) & 31
        const auto summary_word_off{OpShiftLeftLogical(
            U64, OpShiftRightLogical(U64, page, Constant(U64, u64{15})), Constant(U64, u64{2}))};
        const auto summary_ptr{OpConvertUToPtr(
            physical_pointer_type_u32,
            OpIAdd(U64,
                   OpIAdd(U64, map_base,
                          Constant(U64, u64{VideoCore::BufferCache::POLLMAP_FINE_BYTES})),
                   summary_word_off))};
        const auto summary_bit{OpShiftLeftLogical(
            U32[1], u32_one_value,
            OpUConvert(U32[1],
                       OpBitwiseAnd(U64, OpShiftRightLogical(U64, page, Constant(U64, u64{10})),
                                    Constant(U64, u64{31}))))};
        OpAtomicOr(U32[1], summary_ptr, ConstU32(1U), ConstU32(0U), summary_bit);
        OpBranch(atomic_merge_label);

        AddLabel(atomic_merge_label);
        OpBranch(mark_merge_label);

        AddLabel(mark_merge_label);
        available_end = mark_merge_label;
    }
    OpBranch(merge_label);

    AddLabel(fallback_label);
    OpBranch(merge_label);

    AddLabel(merge_label);
    const Id final_result = OpPhi(U32[1], u32_zero_value, fallback_label, result, available_end);
    OpReturnValue(final_result);
    OpFunctionEnd();
    return func;
}

// [LIVEVEC4] SHADPS4_LIVEVEC4=1: a live vec4 read costs one page-table walk instead of four.
// The 16 bytes must stay inside one caching page and be 16-byte aligned for the vector load;
// anything else takes four scalar live reads (same semantics, slower).
// Level: 1 = vector live loads, 2 = + vector live stores (one walk, per-dword journal),
// 3 = + one journal reservation per vector store instead of one per dword.
static int LiveVec4Level() {
    static const int v = [] {
        const char* e = std::getenv("SHADPS4_LIVEVEC4");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
static bool LiveVec4Enabled() {
    return LiveVec4Level() >= 1;
}

Id EmitContext::DefineWriteConstX(u32 count, bool batch_journal) {
    const auto func_type{TypeFunction(void_id, U32[2], U32[1], U32[4])};
    const auto func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type)};
    const auto base{OpFunctionParameter(U32[2])};
    const auto offset{OpFunctionParameter(U32[1])};
    const auto value{OpFunctionParameter(U32[4])};
    Name(func, fmt::format("write_const_live_x{}", count));
    AddLabel();

    const auto base_lo{OpUConvert(U64, OpCompositeExtract(U32[1], base, 0))};
    const auto base_hi{OpUConvert(U64, OpCompositeExtract(U32[1], base, 1))};
    const auto base_shift{OpShiftLeftLogical(U64, base_hi, ConstU32(32U))};
    const auto base_addr{OpBitwiseOr(U64, base_lo, base_shift)};
    const auto offset_bytes{OpShiftLeftLogical(U32[1], offset, ConstU32(2U))};
    const auto addr{OpIAdd(U64, base_addr, OpUConvert(U64, offset_bytes))};

    const Id vec_label = OpLabel();
    const Id scalar_label = OpLabel();
    const Id merge_label = OpLabel();
    const Id store_label = OpLabel();
    const Id vec_merge_label = OpLabel();

    // All dwords inside one caching page -> one walk
    const auto page_mask{Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1)};
    const auto page_off{OpBitwiseAnd(U64, addr, page_mask)};
    const auto in_page{OpULessThanEqual(
        U1[1], page_off, Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 4 * count))};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(in_page, vec_label, scalar_label);

    AddLabel(vec_label);
    std::array<Id, 4> comps{};
    for (u32 i = 0; i < count; ++i) {
        comps[i] = OpCompositeExtract(U32[1], value, i);
    }
    const Id bda = OpFunctionCall(U64, get_bda_pointer, addr);
    const Id is_available = OpINotEqual(U1[1], bda, u64_zero_value);
    OpSelectionMerge(vec_merge_label, spv::SelectionControlMask::MaskNone);
    if (BdaWriteTrackingEnabled() && DropLogEnabled()) {
        // [DROPLOG] no live entry: every dword of the slice goes to the dropped-write log
        const Id drop_label = OpLabel();
        OpBranchConditional(is_available, store_label, drop_label);
        AddLabel(drop_label);
        for (u32 i = 0; i < count; ++i) {
            EmitDropLog(OpIAdd(U64, addr, Constant(U64, u64{4} * i)), comps[i]);
        }
        OpBranch(vec_merge_label);
    } else {
        OpBranchConditional(is_available, store_label, vec_merge_label);
    }

    AddLabel(store_label);
    for (u32 i = 0; i < count; ++i) {
        const Id ptr =
            OpConvertUToPtr(physical_pointer_type_u32, OpIAdd(U64, bda, Constant(U64, u64{4} * i)));
        OpStore(ptr, comps[i], spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned,
                4u);
    }
    if (BdaWriteTrackingEnabled()) {
        // Same per-dword journal entries as write_const_live (the host replays dwords).
        const auto& log_buffer{buffers[fault_buffer_index]};
        const auto [log_id, log_ptr_type] = log_buffer.Alias(PointerType::U32);
        const auto log_base{
            ConstU32(static_cast<u32>(VideoCore::BufferCache::CACHING_NUMPAGES / 32))};
        const auto count_ptr{OpAccessChain(log_ptr_type, log_id, u32_zero_value, log_base)};
        const auto capacity{ConstU32(VideoCore::BufferCache::BDA_WRITE_LOG_ENTRIES)};
        const auto words_per_entry{ConstU32(VideoCore::BufferCache::BDA_WRITE_LOG_WORDS_PER_ENTRY)};
        const auto write_entry = [&](Id slot, u32 i) {
            const auto entry_base{OpIAdd(U32[1], OpIAdd(U32[1], log_base, u32_one_value),
                                         OpIMul(U32[1], slot, words_per_entry))};
            const auto addr_i{OpIAdd(U64, addr, Constant(U64, u64{4} * i))};
            const auto addr_lo{
                OpUConvert(U32[1], OpBitwiseAnd(U64, addr_i, Constant(U64, u64{0xFFFFFFFF})))};
            const auto addr_hi{
                OpUConvert(U32[1], OpShiftRightLogical(U64, addr_i, Constant(U64, u64{32})))};
            OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value, entry_base), addr_lo);
            OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value,
                                  OpIAdd(U32[1], entry_base, ConstU32(1U))),
                    addr_hi);
            OpStore(OpAccessChain(log_ptr_type, log_id, u32_zero_value,
                                  OpIAdd(U32[1], entry_base, ConstU32(2U))),
                    comps[i]);
        };
        if (batch_journal) {
            const auto slot{
                OpAtomicIAdd(U32[1], count_ptr, ConstU32(1U), ConstU32(0U), ConstU32(count))};
            const auto has_room{
                OpULessThanEqual(U1[1], OpIAdd(U32[1], slot, ConstU32(count)), capacity)};
            const auto jstore_label{OpLabel()};
            const auto jmerge_label{OpLabel()};
            OpSelectionMerge(jmerge_label, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(has_room, jstore_label, jmerge_label);
            AddLabel(jstore_label);
            for (u32 i = 0; i < count; ++i) {
                write_entry(OpIAdd(U32[1], slot, ConstU32(i)), i);
            }
            OpBranch(jmerge_label);
            AddLabel(jmerge_label);
        } else {
            for (u32 i = 0; i < count; ++i) {
                const auto slot{
                    OpAtomicIAdd(U32[1], count_ptr, ConstU32(1U), ConstU32(0U), u32_one_value)};
                const auto has_room{OpULessThan(U1[1], slot, capacity)};
                const auto jstore_label{OpLabel()};
                const auto jmerge_label{OpLabel()};
                OpSelectionMerge(jmerge_label, spv::SelectionControlMask::MaskNone);
                OpBranchConditional(has_room, jstore_label, jmerge_label);
                AddLabel(jstore_label);
                write_entry(slot, i);
                OpBranch(jmerge_label);
                AddLabel(jmerge_label);
            }
        }
    }
    OpBranch(vec_merge_label);
    AddLabel(vec_merge_label);
    OpBranch(merge_label);

    // Page-crossing: the scalar path, dword by dword
    AddLabel(scalar_label);
    for (u32 i = 0; i < count; ++i) {
        const Id off_i = i == 0 ? offset : OpIAdd(U32[1], offset, ConstU32(i));
        OpFunctionCall(void_id, write_const_live, base, off_i,
                       OpCompositeExtract(U32[1], value, i));
    }
    OpBranch(merge_label);

    AddLabel(merge_label);
    OpReturn();
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineReadConstLiveX4(bool is_volatile) {
    const auto func_type{TypeFunction(U32[4], U32[2], U32[1])};
    const auto func{OpFunction(U32[4], spv::FunctionControlMask::MaskNone, func_type)};
    const auto base{OpFunctionParameter(U32[2])};
    const auto offset{OpFunctionParameter(U32[1])};
    const char* const fname = is_volatile ? "read_const_live_x4" : "read_const_live_x4_nv";
    Name(func, fname);
    AddLabel();

    // [BDACACHE] The non-volatile flavour caches from level 1; the volatile one - the record
    // loads in the worker's inner loop, three per iteration on the same bone-palette page - only
    // at level 2, which is the actual experiment. The 16-byte load keeps its Volatile mask in
    // both cases: only the translation is remembered, never the data.
    const bool use_cache =
        Sirit::ValidId(get_bda_pointer_cached) && (!is_volatile || BdaCacheLevel() >= 2);
    if (use_cache) {
        LOG_CRITICAL(Render_Recompiler, "BDACACHE: shader {:#x} {} -> get_bda_pointer_cached",
                     info.pgm_hash, fname);
    }

    const auto base_lo{OpUConvert(U64, OpCompositeExtract(U32[1], base, 0))};
    const auto base_hi{OpUConvert(U64, OpCompositeExtract(U32[1], base, 1))};
    const auto base_shift{OpShiftLeftLogical(U64, base_hi, ConstU32(32U))};
    const auto base_addr{OpBitwiseOr(U64, base_lo, base_shift)};
    const auto offset_bytes{OpShiftLeftLogical(U32[1], offset, ConstU32(2U))};
    const auto addr{OpIAdd(U64, base_addr, OpUConvert(U64, offset_bytes))};

    const Id vec_label = OpLabel();
    const Id scalar_label = OpLabel();
    const Id merge_label = OpLabel();
    const Id load_label = OpLabel();
    const Id null_label = OpLabel();
    const Id vec_merge_label = OpLabel();

    const auto page_mask{Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1)};
    const auto page_off{OpBitwiseAnd(U64, addr, page_mask)};
    const auto in_page{OpULessThanEqual(
        U1[1], page_off, Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 16))};
    const auto aligned{
        OpIEqual(U1[1], OpBitwiseAnd(U64, addr, Constant(U64, u64{15})), u64_zero_value)};
    const auto vec_ok{OpLogicalAnd(U1[1], in_page, aligned)};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(vec_ok, vec_label, scalar_label);

    // One walk, one 16-byte volatile load (unmapped page: the walk marked the fault, read zero)
    AddLabel(vec_label);
    const Id bda = OpFunctionCall(U64, use_cache ? get_bda_pointer_cached : get_bda_pointer, addr);
    const Id is_available = OpINotEqual(U1[1], bda, u64_zero_value);
    OpSelectionMerge(vec_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(is_available, load_label, null_label);
    AddLabel(load_label);
    const Id ptr_type = TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[4]);
    const Id addr_ptr = OpConvertUToPtr(ptr_type, bda);
    const Id vec =
        OpLoad(U32[4], addr_ptr,
               is_volatile ? (spv::MemoryAccessMask::Volatile | spv::MemoryAccessMask::Aligned)
                           : spv::MemoryAccessMask::Aligned,
               16u);
    OpBranch(vec_merge_label);
    AddLabel(null_label);
    const Id zero4 = ConstantNull(U32[4]);
    OpBranch(vec_merge_label);
    AddLabel(vec_merge_label);
    const Id vec_result = OpPhi(U32[4], vec, load_label, zero4, null_label);
    OpBranch(merge_label);

    // Page-crossing or unaligned: four scalar live reads
    AddLabel(scalar_label);
    std::array<Id, 4> dw{};
    for (u32 i = 0; i < 4; ++i) {
        const Id off_i = i == 0 ? offset : OpIAdd(U32[1], offset, ConstU32(i));
        dw[i] =
            OpFunctionCall(U32[1], is_volatile ? read_const_live : read_const_live_nv, base, off_i);
    }
    const Id scalar_result = OpCompositeConstruct(U32[4], dw[0], dw[1], dw[2], dw[3]);
    OpBranch(merge_label);

    AddLabel(merge_label);
    const Id result = OpPhi(U32[4], vec_result, vec_merge_label, scalar_result, scalar_label);
    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

// [LIVENV] Live BDA reads are volatile so polling loops re-read memory; the ND job workers'
// data loops (for i < n: index, 12 dwords of a record) only pay for it. Listed shaders get
// non-volatile live reads: same page-table walk, but the compiler may pipeline and coalesce.
bool EmitContext::LiveReadsNonVolatile() const {
    static const std::vector<u64> list = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_LIVE_NONVOLATILE");
        if (!e || !e[0]) {
            return out;
        }
        if (std::string_view{e} == "all") {
            out.push_back(~0ull);
            return out;
        }
        std::string_view sv{e};
        while (!sv.empty()) {
            const auto comma = sv.find(',');
            const auto tok = sv.substr(0, comma);
            if (!tok.empty()) {
                out.push_back(std::strtoull(std::string{tok}.c_str(), nullptr, 16));
            }
            if (comma == std::string_view::npos) {
                break;
            }
            sv.remove_prefix(comma + 1);
        }
        return out;
    }();
    if (list.empty()) {
        return false;
    }
    if (list.front() == ~0ull) {
        return true;
    }
    return std::find(list.begin(), list.end(), info.pgm_hash) != list.end();
}

void EmitContext::DefineFunctions() {
    if (info.uses_pack_10_11_11) {
        f32_to_uf11 = DefineFloat32ToUfloatM5(6, "f32_to_uf11");
        f32_to_uf10 = DefineFloat32ToUfloatM5(5, "f32_to_uf10");
    }
    if (info.uses_unpack_10_11_11) {
        uf11_to_f32 = DefineUfloatM5ToFloat32(6, "uf11_to_f32");
        uf10_to_f32 = DefineUfloatM5ToFloat32(5, "uf10_to_f32");
    }
    if (info.uses_dma) {
        // Which shaders actually reach memory through a raw device address? The device fault
        // names the shader by checkpoint, so this list says whether the faulting one is even
        // on the DMA path.
        // [Stage1] Full key surface: visible fields (stage/readconst_types) matched across
        // the triple-compile of 0x853608ae, so log every codegen-affecting field of Info to
        // reveal which "invisible" one jumps between recompilations.
        LOG_CRITICAL(Render_Recompiler,
                     "DMASHADER: {:#x} stage={} l_stage={} readconst_types={:#x} fp16={} fp64={} "
                     "lane_id={} patches={:#x} tess_in={} tess_out={} bufs={} imgs={} smps={} "
                     "shared_types={:#x} pollj={}",
                     info.pgm_hash, static_cast<u32>(stage), static_cast<u32>(info.l_stage),
                     static_cast<u32>(info.readconst_types), static_cast<u32>(info.uses_fp16),
                     static_cast<u32>(info.uses_fp64), static_cast<u32>(info.uses_lane_id),
                     static_cast<u32>(info.uses_patches),
                     static_cast<u32>(info.stores_tess_level_inner),
                     static_cast<u32>(info.stores_tess_level_outer), info.buffers.size(),
                     info.images.size(), info.samplers.size(), static_cast<u32>(info.shared_types),
                     static_cast<u32>(info.uses_poll_read));
        get_bda_pointer = DefineGetBdaPointer();
        // [BDACACHE] One wrapper shared by every flavour allowed to remember a translation, so a
        // straight-line run of reads on one page (bone palette, SRT page) pays a single walk.
        // Defined right after the honest walk and before any flavour, which then routes itself.
        if (BdaCacheLevel() >= 1 && BdaCacheWants(info.pgm_hash)) {
            LOG_CRITICAL(Render_Recompiler,
                         "BDACACHE: shader {:#x} defines get_bda_pointer_cached (level {})",
                         info.pgm_hash, BdaCacheLevel());
            get_bda_pointer_cached = DefineGetBdaPointerCached();
        }
    }

    if (True(info.readconst_types & Info::ReadConstType::Immediate)) {
        LOG_DEBUG(Render_Recompiler, "Shader {:#x} uses immediate ReadConst", info.pgm_hash);
        read_const = DefineReadConst(false);
    }
    if (True(info.readconst_types & Info::ReadConstType::Dynamic)) {
        LOG_DEBUG(Render_Recompiler, "Shader {:#x} uses dynamic ReadConst", info.pgm_hash);
        read_const_dynamic = DefineReadConst(true);
    }
    if (True(info.readconst_types & Info::ReadConstType::Live)) {
        LOG_DEBUG(Render_Recompiler, "Shader {:#x} uses live ReadConst", info.pgm_hash);
        read_const_live = DefineReadConst(true, true);
        // [POLLMAP] the poll flavour is shared: POLLJ stamps the presence window, POLLMAP marks
        // the poll bitmap; either env alone defines it, with only its own part inside.
        if (info.uses_poll_read && (VideoCore::BufferCache::PollTraceEnabled() ||
                                    VideoCore::BufferCache::PollMapEnabled())) {
            if (VideoCore::BufferCache::PollTraceEnabled()) {
                LOG_CRITICAL(Render_Recompiler, "POLLJ: shader {:#x} defines read_const_live_poll",
                             info.pgm_hash);
            }
            if (VideoCore::BufferCache::PollMapEnabled()) {
                LOG_CRITICAL(Render_Recompiler,
                             "POLLMAP: shader {:#x} marks polled pages in read_const_live_poll",
                             info.pgm_hash);
            }
            read_const_live_poll = DefineReadConstLivePoll();
        }
        if (DualHomeStoreEnabled()) {
            LOG_CRITICAL(Render_Recompiler, "DUALHOME: shader {:#x} dual store on", info.pgm_hash);
        }
        if (DualHomeFaultStoreEnabled()) {
            LOG_CRITICAL(Render_Recompiler, "DUALHOME: shader {:#x} fault-path store to home on",
                         info.pgm_hash);
        }
        if (LiveVec4Enabled()) {
            LOG_CRITICAL(Render_Recompiler, "LIVEVEC4: shader {:#x} defines read_const_live_x4",
                         info.pgm_hash);
            read_const_live_x4 = DefineReadConstLiveX4();
        }
        if (LiveReadsNonVolatile()) {
            LOG_CRITICAL(Render_Recompiler, "LIVENV: shader {:#x} live reads non-volatile",
                         info.pgm_hash);
            read_const_live_nv = DefineReadConst(true, true, false);
            if (LiveVec4Enabled()) {
                read_const_live_x4_nv = DefineReadConstLiveX4(false);
            }
        }
    }
    if (info.uses_live_write) {
        LOG_DEBUG(Render_Recompiler, "Shader {:#x} uses live WriteConst", info.pgm_hash);
        write_const_live = DefineWriteConst();
        if (LiveVec4Level() >= 2) {
            LOG_CRITICAL(Render_Recompiler,
                         "LIVEVEC4: shader {:#x} defines write_const_live_x2..4 (batch={})",
                         info.pgm_hash, LiveVec4Level() >= 3);
            for (u32 n = 2; n <= 4; ++n) {
                write_const_live_x[n] = DefineWriteConstX(n, LiveVec4Level() >= 3);
            }
        }
    }
}

} // namespace Shader::Backend::SPIRV
