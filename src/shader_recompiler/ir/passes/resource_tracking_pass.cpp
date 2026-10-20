// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <boost/container/static_vector.hpp>

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/operand_helper.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reinterpret.h"
#include "shader_recompiler/profile.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/buffer_cache.h"

namespace Shader::Optimization {
namespace {

using SharpLocation = u32;

bool IsBufferAtomic(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::BufferAtomicIAdd32:
    case IR::Opcode::BufferAtomicIAdd64:
    case IR::Opcode::BufferAtomicISub32:
    case IR::Opcode::BufferAtomicSMin32:
    case IR::Opcode::BufferAtomicSMin64:
    case IR::Opcode::BufferAtomicUMin32:
    case IR::Opcode::BufferAtomicUMin64:
    case IR::Opcode::BufferAtomicFMin32:
    case IR::Opcode::BufferAtomicSMax32:
    case IR::Opcode::BufferAtomicSMax64:
    case IR::Opcode::BufferAtomicUMax32:
    case IR::Opcode::BufferAtomicUMax64:
    case IR::Opcode::BufferAtomicFMax32:
    case IR::Opcode::BufferAtomicInc32:
    case IR::Opcode::BufferAtomicDec32:
    case IR::Opcode::BufferAtomicAnd32:
    case IR::Opcode::BufferAtomicOr32:
    case IR::Opcode::BufferAtomicXor32:
    case IR::Opcode::BufferAtomicSwap32:
    case IR::Opcode::BufferAtomicCmpSwap32:
    case IR::Opcode::BufferAtomicFCmpSwap32:
        return true;
    default:
        return false;
    }
}

bool IsBufferStore(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::StoreBufferU8:
    case IR::Opcode::StoreBufferU16:
    case IR::Opcode::StoreBufferU32:
    case IR::Opcode::StoreBufferU32x2:
    case IR::Opcode::StoreBufferU32x3:
    case IR::Opcode::StoreBufferU32x4:
    case IR::Opcode::StoreBufferU64:
    case IR::Opcode::StoreBufferF32:
    case IR::Opcode::StoreBufferF32x2:
    case IR::Opcode::StoreBufferF32x3:
    case IR::Opcode::StoreBufferF32x4:
    case IR::Opcode::StoreBufferFormatF32:
        return true;
    default:
        return IsBufferAtomic(inst);
    }
}

bool IsBufferInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadBufferU8:
    case IR::Opcode::LoadBufferU16:
    case IR::Opcode::LoadBufferU32:
    case IR::Opcode::LoadBufferU32x2:
    case IR::Opcode::LoadBufferU32x3:
    case IR::Opcode::LoadBufferU32x4:
    case IR::Opcode::LoadBufferU64:
    case IR::Opcode::LoadBufferF32:
    case IR::Opcode::LoadBufferF32x2:
    case IR::Opcode::LoadBufferF32x3:
    case IR::Opcode::LoadBufferF32x4:
    case IR::Opcode::LoadBufferFormatF32:
    case IR::Opcode::ReadConstBuffer:
        return true;
    default:
        return IsBufferStore(inst);
    }
}

bool IsDataRingInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::DataAppend:
    case IR::Opcode::DataConsume:
        return true;
    case IR::Opcode::LoadSharedU16:
    case IR::Opcode::LoadSharedU32:
    case IR::Opcode::LoadSharedU64:
    case IR::Opcode::WriteSharedU16:
    case IR::Opcode::WriteSharedU32:
    case IR::Opcode::WriteSharedU64:
    case IR::Opcode::SharedAtomicIAdd32:
    case IR::Opcode::SharedAtomicIAdd64:
    case IR::Opcode::SharedAtomicUMin32:
    case IR::Opcode::SharedAtomicUMin64:
    case IR::Opcode::SharedAtomicSMin32:
    case IR::Opcode::SharedAtomicSMin64:
    case IR::Opcode::SharedAtomicUMax32:
    case IR::Opcode::SharedAtomicUMax64:
    case IR::Opcode::SharedAtomicSMax32:
    case IR::Opcode::SharedAtomicSMax64:
    case IR::Opcode::SharedAtomicAnd32:
    case IR::Opcode::SharedAtomicAnd64:
    case IR::Opcode::SharedAtomicOr32:
    case IR::Opcode::SharedAtomicOr64:
    case IR::Opcode::SharedAtomicXor32:
    case IR::Opcode::SharedAtomicXor64:
    case IR::Opcode::SharedAtomicISub32:
    case IR::Opcode::SharedAtomicISub64:
    case IR::Opcode::SharedAtomicInc32:
    case IR::Opcode::SharedAtomicInc64:
    case IR::Opcode::SharedAtomicDec32:
    case IR::Opcode::SharedAtomicDec64:
        return inst.Flags<bool>(); // is_gds
    default:
        return false;
    }
}

IR::Type BufferDataType(const IR::Inst& inst, const Profile& profile,
                        AmdGpu::NumberFormat num_format) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadBufferU8:
    case IR::Opcode::StoreBufferU8:
        return IR::Type::U8;
    case IR::Opcode::LoadBufferU16:
    case IR::Opcode::StoreBufferU16:
        return IR::Type::U16;
    case IR::Opcode::LoadBufferU64:
    case IR::Opcode::StoreBufferU64:
    case IR::Opcode::BufferAtomicIAdd64:
    case IR::Opcode::BufferAtomicSMax64:
    case IR::Opcode::BufferAtomicSMin64:
    case IR::Opcode::BufferAtomicUMax64:
    case IR::Opcode::BufferAtomicUMin64:
        return IR::Type::U64;
    case IR::Opcode::BufferAtomicFMax32:
    case IR::Opcode::BufferAtomicFMin32:
        return profile.supports_buffer_fp32_atomic_min_max ? IR::Type::F32 : IR::Type::U32;
    case IR::Opcode::LoadBufferFormatF32:
    case IR::Opcode::StoreBufferFormatF32:
        // Formatted buffer loads can use a variety of types.
        return IR::Type::U32 | IR::Type::F32 | IR::Type::U16 | IR::Type::U8;
    default:
        return IR::Type::U32;
    }
}

u32 BufferAddressShift(const IR::Inst& inst, AmdGpu::DataFormat data_format) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadBufferU8:
    case IR::Opcode::StoreBufferU8:
        return 0;
    case IR::Opcode::LoadBufferU16:
    case IR::Opcode::StoreBufferU16:
        return 1;
    case IR::Opcode::LoadBufferU64:
    case IR::Opcode::StoreBufferU64:
    case IR::Opcode::BufferAtomicIAdd64:
    case IR::Opcode::BufferAtomicSMax64:
    case IR::Opcode::BufferAtomicSMin64:
    case IR::Opcode::BufferAtomicUMax64:
    case IR::Opcode::BufferAtomicUMin64:
        return 3;
    case IR::Opcode::LoadBufferFormatF32:
    case IR::Opcode::StoreBufferFormatF32: {
        switch (data_format) {
        case AmdGpu::DataFormat::Format8:
            return 0;
        case AmdGpu::DataFormat::Format8_8:
        case AmdGpu::DataFormat::Format16:
            return 1;
        case AmdGpu::DataFormat::Format8_8_8_8:
        case AmdGpu::DataFormat::Format16_16:
        case AmdGpu::DataFormat::Format10_11_11:
        case AmdGpu::DataFormat::Format2_10_10_10:
        case AmdGpu::DataFormat::Format16_16_16_16:
        case AmdGpu::DataFormat::Format32:
        case AmdGpu::DataFormat::Format32_32:
        case AmdGpu::DataFormat::Format32_32_32:
        case AmdGpu::DataFormat::Format32_32_32_32:
            return 2;
        default:
            return 0;
        }
        break;
    }
    case IR::Opcode::ReadConstBuffer:
        // Provided address is already in dwords
        return 0;
    default:
        return 2;
    }
}

bool IsImageAtomicInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::ImageAtomicIAdd32:
    case IR::Opcode::ImageAtomicSMin32:
    case IR::Opcode::ImageAtomicUMin32:
    case IR::Opcode::ImageAtomicSMax32:
    case IR::Opcode::ImageAtomicUMax32:
    case IR::Opcode::ImageAtomicFMax32:
    case IR::Opcode::ImageAtomicFMin32:
    case IR::Opcode::ImageAtomicInc32:
    case IR::Opcode::ImageAtomicDec32:
    case IR::Opcode::ImageAtomicAnd32:
    case IR::Opcode::ImageAtomicOr32:
    case IR::Opcode::ImageAtomicXor32:
    case IR::Opcode::ImageAtomicExchange32:
    case IR::Opcode::ImageAtomicCmpSwap32:
        return true;
    default:
        return false;
    }
}

bool IsImageInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::ImageRead:
    case IR::Opcode::ImageWrite:
    case IR::Opcode::ImageQueryDimensions:
    case IR::Opcode::ImageQueryLod:
    case IR::Opcode::ImageSampleRaw:
        return true;
    default:
        return IsImageAtomicInstruction(inst);
    }
}

class Descriptors {
public:
    explicit Descriptors(Info& info_)
        : info{info_}, buffer_resources{info_.buffers}, image_resources{info_.images},
          sampler_resources{info_.samplers}, fmask_resources(info_.fmasks) {}

    u32 Add(const BufferResource& desc) {
        const u32 index{Add(buffer_resources, desc, [&desc](const auto& existing) {
            return desc.sharp_idx == existing.sharp_idx &&
                   desc.inline_cbuf == existing.inline_cbuf &&
                   desc.buffer_type == existing.buffer_type;
        })};
        auto& buffer = buffer_resources[index];
        buffer.used_types |= desc.used_types;
        buffer.is_written |= desc.is_written;
        buffer.is_formatted |= desc.is_formatted;
        return index;
    }

    u32 Add(const ImageResource& desc) {
        const u32 index{Add(image_resources, desc, [&desc](const auto& existing) {
            return desc.sharp_idx == existing.sharp_idx && desc.is_array == existing.is_array &&
                   desc.mip_fallback_mode == existing.mip_fallback_mode &&
                   desc.constant_mip_index == existing.constant_mip_index;
        })};
        auto& image = image_resources[index];
        image.is_atomic |= desc.is_atomic;
        image.is_written |= desc.is_written;
        return index;
    }

    u32 Add(const SamplerResource& desc) {
        const u32 index{Add(sampler_resources, desc, [this, &desc](const auto& existing) {
            return desc.sharp_idx == existing.sharp_idx &&
                   desc.is_inline_sampler == existing.is_inline_sampler &&
                   desc.inline_sampler == existing.inline_sampler;
        })};
        return index;
    }

    u32 Add(const FMaskResource& desc) {
        u32 index = Add(fmask_resources, desc, [&desc](const auto& existing) {
            return desc.sharp_idx == existing.sharp_idx;
        });
        return index;
    }

private:
    template <typename Descriptors, typename Descriptor, typename Func>
    static u32 Add(Descriptors& descriptors, const Descriptor& desc, Func&& pred) {
        const auto it{std::ranges::find_if(descriptors, pred)};
        if (it != descriptors.end()) {
            return static_cast<u32>(std::distance(descriptors.begin(), it));
        }
        descriptors.push_back(desc);
        return static_cast<u32>(descriptors.size()) - 1;
    }

    const Info& info;
    BufferResourceList& buffer_resources;
    ImageResourceList& image_resources;
    SamplerResourceList& sampler_resources;
    FMaskResourceList& fmask_resources;
};

} // Anonymous namespace

std::pair<const IR::Inst*, bool> TryDisableAnisoLod0(const IR::Inst* inst) {
    std::pair not_found{inst, false};

    // Assuming S# is in UD s[12:15] and T# is in s[4:11]
    // The next pattern:
    //  s_bfe_u32     s0, s7,  $0x0008000c
    //  s_and_b32     s1, s12, $0xfffff1ff
    //  s_cmp_eq_u32  s0, 0
    //  s_cselect_b32 s0, s1, s12
    // is used to disable anisotropy in the sampler if the sampled texture doesn't have mips

    if (inst->GetOpcode() != IR::Opcode::SelectU32) {
        return not_found;
    }

    // Select should be based on zero check
    const auto* prod0 = inst->Arg(0).InstRecursive();
    if (prod0->GetOpcode() != IR::Opcode::IEqual32 ||
        !(prod0->Arg(1).IsImmediate() && prod0->Arg(1).U32() == 0u)) {
        return not_found;
    }

    // The bitfield extract might be hidden by phi sometimes
    auto* prod0_arg0 = prod0->Arg(0).InstRecursive();
    if (prod0_arg0->GetOpcode() == IR::Opcode::Phi) {
        auto arg0 = prod0_arg0->Arg(0);
        auto arg1 = prod0_arg0->Arg(1);
        if (!arg0.IsImmediate() &&
            arg0.InstRecursive()->GetOpcode() == IR::Opcode::BitFieldUExtract) {
            prod0_arg0 = arg0.InstRecursive();
        } else if (!arg1.IsImmediate() &&
                   arg1.InstRecursive()->GetOpcode() == IR::Opcode::BitFieldUExtract) {
            prod0_arg0 = arg1.InstRecursive();
        }
    }

    // The bits range is for lods (note that constants are changed after constant propagation pass)
    if (prod0_arg0->GetOpcode() != IR::Opcode::BitFieldUExtract ||
        !(prod0_arg0->Arg(1).IsImmediate() && prod0_arg0->Arg(1).U32() == 12) ||
        !(prod0_arg0->Arg(2).IsImmediate() && prod0_arg0->Arg(2).U32() == 8)) {
        return not_found;
    }

    // Make sure mask is masking out anisotropy
    const auto* prod1 = inst->Arg(1).InstRecursive();
    if (prod1->GetOpcode() != IR::Opcode::BitwiseAnd32 || prod1->Arg(1).U32() != 0xfffff1ff) {
        return not_found;
    }

    // We're working on the first dword of s#
    const auto* prod2 = inst->Arg(2).InstRecursive();
    if (prod2->GetOpcode() != IR::Opcode::GetUserData &&
        prod2->GetOpcode() != IR::Opcode::ReadConst && prod2->GetOpcode() != IR::Opcode::Phi) {
        return not_found;
    }

    return {prod2, true};
}

using SharpSources = boost::container::small_vector<const IR::Inst*, 4>;

bool IsSharpSource(const IR::Inst* inst) {
    return inst->GetOpcode() == IR::Opcode::GetUserData ||
           inst->GetOpcode() == IR::Opcode::ReadConst;
}

SharpSources FindSharpSources(const IR::Inst* handle, u32 pc) {
    SharpSources sources;
    if (IsSharpSource(handle)) {
        sources.push_back(handle);
        return sources;
    }

    bool found_read_const_buffer = false;

    boost::container::small_vector<const IR::Inst*, 8> visited;
    std::queue<const IR::Inst*> queue;
    queue.push(handle);

    while (!queue.empty()) {
        const IR::Inst* inst{queue.front()};
        queue.pop();
        if (IsSharpSource(inst)) {
            sources.push_back(inst);
            continue;
        }
        found_read_const_buffer |= inst->GetOpcode() == IR::Opcode::ReadConstBuffer;
        if (inst->GetOpcode() != IR::Opcode::Phi) {
            continue;
        }
        for (size_t arg = inst->NumArgs(); arg--;) {
            const IR::Value arg_value = inst->Arg(arg);
            if (arg_value.IsImmediate()) {
                continue;
            }
            const IR::Inst* arg_inst = arg_value.InstRecursive();
            if (std::ranges::find(visited, arg_inst) == visited.end()) {
                visited.push_back(arg_inst);
                queue.push(arg_inst);
            }
        }
    }
    if (sources.empty()) {
        if (found_read_const_buffer) {
            UNREACHABLE_MSG("Bindless sharp access detected pc={:#x}", pc);
        } else {
            UNREACHABLE_MSG("Unable to find sharp sources pc={:#x}", pc);
        }
    }
    return sources;
}

bool IsCfgBlockDominatedBy(const Shader::Gcn::Block* maybe_dominator,
                           const Shader::Gcn::Block* block, const Shader::Gcn::Block* dest_block) {
    if (block == maybe_dominator) {
        return true;
    }

    boost::container::small_vector<const Shader::Gcn::Block*, 8> visited;
    std::queue<const Shader::Gcn::Block*> queue;
    queue.push(block);

    while (!queue.empty()) {
        const Shader::Gcn::Block* block{queue.front()};
        queue.pop();
        if (block == dest_block) {
            return false;
        }
        if (block == maybe_dominator) {
            continue;
        }
        if (block->branch_false && !std::ranges::contains(visited, block->branch_false)) {
            visited.push_back(block->branch_false);
            queue.push(block->branch_false);
        }
        if (block->branch_true && !std::ranges::contains(visited, block->branch_true)) {
            visited.push_back(block->branch_true);
            queue.push(block->branch_true);
        }
    }

    return true;
}

SharpLocation SharpLocationFromSource(const IR::Inst* inst) {
    if (inst->GetOpcode() == IR::Opcode::GetUserData) {
        return static_cast<SharpLocation>(inst->Arg(0).ScalarReg());
    } else {
        return inst->Flags<u32>();
    }
}

SharpLocation TrackSharp(const IR::Inst* inst, const IR::Block& current_parent, u32 pc = 0) {
    auto sources = FindSharpSources(inst, pc);
    size_t num_sources = sources.size();
    ASSERT(current_parent.cfg_block);

    // Perform dominance analysis on found sources and eliminate ones that don't pass
    // If a sharp source is dominated by another, the former can be eliminated.
    for (s32 i = 0; i < num_sources;) {
        const IR::Block* block = sources[i]->GetParent();
        ASSERT(block->cfg_block);
        bool was_removed = false;
        for (s32 j = 0; j < num_sources;) {
            const IR::Block* dominator = sources[j]->GetParent();
            ASSERT(dominator->cfg_block);
            if (i != j && IsCfgBlockDominatedBy(dominator->cfg_block, block->cfg_block,
                                                current_parent.cfg_block)) {
                std::swap(sources[i], sources[num_sources - 1]);
                --num_sources;
                sources.pop_back();
                was_removed = true;
                break;
            } else {
                ++j;
            }
        }
        if (!was_removed) {
            ++i;
        }
    }

    ASSERT_MSG(sources.size() == 1, "Unable to deduce sharp source");
    return SharpLocationFromSource(sources[0]);
}


IR::U32 CalculateBufferAddress(IR::IREmitter& ir, const IR::Inst& inst, const Info& info,
                               const AmdGpu::Buffer& buffer, u32 stride);

static bool DmaEligibleDwords(const IR::Inst& inst, u32& num_dwords, bool& is_float) {
    is_float = false;
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadBufferF32:
        is_float = true;
        [[fallthrough]];
    case IR::Opcode::ReadConstBuffer:
    case IR::Opcode::LoadBufferU32:
        num_dwords = 1;
        return true;
    case IR::Opcode::LoadBufferF32x2:
        is_float = true;
        [[fallthrough]];
    case IR::Opcode::LoadBufferU32x2:
        num_dwords = 2;
        return true;
    case IR::Opcode::LoadBufferF32x3:
        is_float = true;
        [[fallthrough]];
    case IR::Opcode::LoadBufferU32x3:
        num_dwords = 3;
        return true;
    case IR::Opcode::LoadBufferF32x4:
        is_float = true;
        [[fallthrough]];
    case IR::Opcode::LoadBufferU32x4:
        num_dwords = 4;
        return true;
    default:
        return false;
    }
}

// [LIVELOOP] SHADPS4_LIVELOOP=1: bounded buffer LOADS inside loop blocks of compute shaders are
// rewritten into live ReadConst (BDA, volatile) so a polling loop observes CPU writes made while
// the dispatch runs. Straight-line accesses keep their fast VRAM bindings. This is the targeted
// version of live DMA: seeds mode without it spins forever in the ND job worker (0x8f067540),
// whose poll loop reads the job queue through bounded descriptors.
static int LiveLoopMode() {
    static const int v = [] {
        const char* e = std::getenv("SHADPS4_LIVELOOP");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return v;
}
static bool LiveLoopEnabled() {
    return LiveLoopMode() >= 1;
}
// [LIVEVEC4] SHADPS4_LIVEVEC4=1: an in-loop x4 live load becomes one ReadConstX4 (one
// page-table walk and one 16-byte load instead of four of each). Same volatile semantics.
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
// [VEC4STRAIGHT] SHADPS4_VEC4STRAIGHT=1: coalesce 4-dword reads OUTSIDE loops too. The skinning
// worker's peeled first influence reads the same 48-byte bone matrix as twelve separate
// page-table walks (each an inlined get_bda_pointer with branches and a fault-bitmap store the
// driver may not reorder), while the loop reads it as three. 9 of 32 walks per vertex are here.
static bool Vec4Straight() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_VEC4STRAIGHT");
        return e && e[0] == '1';
    }();
    return v;
}
// >=2: live stores of 2..4 dwords go through one WriteConstX4 (one page-table walk).
static bool LiveVec4Stores() {
    return LiveVec4Level() >= 2;
}
// =2: additionally route every bounded STORE of a compute shader through write_const_live, so job
// results land in guest memory (host) where the CPU reads them (BDATRACK replays them into the
// guest and invalidates VRAM copies). Loads outside loops keep their VRAM bindings.
static bool LiveStoreEnabled() {
    return LiveLoopMode() >= 2;
}
// SHADPS4_LIVEWORKERS=<hex>[,<hex>...]: explicit list of job-worker shaders for LIVELOOP=2. When
// set, only these become live-DMA workers (the loop census alone also catches screen-space passes
// such as 0x43b8ee5e, whose out-of-loop stores then flood the write log: 9.8M entries per 5 s).
static const std::vector<u64>& LiveWorkerList() {
    static const std::vector<u64> list = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_LIVEWORKERS");
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
    return list;
}
// [POLLJ] SHADPS4_PAGETRACE_POLLJ=1: scalar live loads inside census poll loops (store-free,
// 1..8 loads) carry PollReadConstFlag, so the backend stamps the polled page into the presence
// window behind the page journal. Off by default: every flag below then stays exactly as today.
static bool PollTraceEnabled() {
    // [POLLMAP] the same census (poll_blocks -> PollReadConstFlag) feeds the dynamic page-table
    // redirect; without it no load would ever reach read_const_live_poll with POLLJ off.
    return VideoCore::BufferCache::PollTraceEnabled() || VideoCore::BufferCache::PollMapEnabled();
}

// [SKINDIRECT] SHADPS4_SKINDIRECT=<hex hash>[,...]: for the listed shaders, a read whose V# is
// one of the ND skinning worker's BOUNDED input descriptors keeps its ordinary buffer binding
// instead of being rewritten into a BDA page-table walk. The worker loads those V#s from its
// descriptor block at dwords 16/20/24 behind ud_0|ud_1 - the same block DESCDUMP dumps
// (vk_rasterizer.cpp) and SkinVramRedirect copies (buffer_cache.cpp, kWords{16,20,24}) - so the
// shape ReadConst(CompositeConstructU32x2(ud_0, ud_1), Imm32(W)) is recognisable here even
// though the base itself is only known at run time. Declining lets control fall through to
// descriptors.Add(BufferResource{...}) in the caller, and the read is emitted as
// OpAccessChain+OpLoad on a bounds-checked SSBO - ONE memory access instead of two dependent
// ones, no branch, no Volatile - which strictly REMOVES basic blocks. That matters: these
// worker shaders once measured ~2x slower from a single extra block, so nothing may be added.
// Reads only; stores keep write_const_live so job results still land in guest memory.
// Empty env -> SkinDirectMatch returns false on its first line and the SPIR-V is unchanged.
static const std::vector<u64>& SkinDirectHashes() {
    return VideoCore::BufferCache::SkinDirectHashes();
}
// SHADPS4_SKINDIRECT_WORDS=<dec>[,...]: which block dwords may be converted. Default 20,24 =
// influence list + bone matrix palette, the two ranges read inside the per-vertex loop. Dword 16
// (the per-vertex influence headers) also carries the LOOP BOUND, and the LIVELOOP note above
// records that this worker spins forever when its loop control goes stale - so the header is a
// separate, later stage that has to be asked for explicitly.
static const std::vector<u32>& SkinDirectWords() {
    static const std::vector<u32> words = [] {
        std::vector<u32> out;
        const char* e = std::getenv("SHADPS4_SKINDIRECT_WORDS");
        if (e == nullptr) {
            // [SKINDIRECT] 07.09: default is dword 20 (the influence list) ALONE. The bone
            // palette (24) lives in the arena that HOMEPOLICY_PT deliberately points at guest
            // RAM, so a buffer binding would read the cache's copy instead - a correctness
            // change, not a speed one. Ask for 24 explicitly once 20 is proven.
            return std::vector<u32>{20u};
        }
        const char* cursor = e;
        while (*cursor != '\0') {
            char* end = nullptr;
            const unsigned long v = std::strtoul(cursor, &end, 10);
            if (end == cursor) {
                break;
            }
            out.push_back(static_cast<u32>(v));
            cursor = (*end == ',') ? end + 1 : end;
        }
        return out;
    }();
    return words;
}
// [SKINDIRECT-STORES] SHADPS4_SKINDIRECT_STORES=1: the OUTPUT stores of a listed shader whose V#
// comes from block dword W in SHADPS4_SKINDIRECT_STORE_WORDS (default 0,4,8: the skinning
// worker's position/normal/tangent, 28-byte records) keep an ordinary WRITTEN SSBO binding
// instead of write_const_live: N OpStore on a bounds-checked SSBO instead of N page-table
// walks + PAGEJOURNAL atomics + dword journal - strictly REMOVES blocks (see the 2x note above).
// The match is STRUCTURAL and never reads the V# contents: every permutation's pipeline binds
// through program->info.buffers from the first compile while its SPIR-V comes from its own
// Info (vk_pipeline_cache.cpp GetProgram), so a V#-dependent decision would give one
// permutation a different descriptor count and shift every binding after it (PERMLAYOUT).
// Both values are parsed once in buffer_cache.cpp so the report prints what the recompiler saw.
static bool SkinDirectStoresEnabled() {
    return VideoCore::BufferCache::SkinDirectStoresEnabled();
}
static const std::vector<u32>& SkinDirectStoreWords() {
    return VideoCore::BufferCache::SkinDirectStoreWords();
}
// SHADPS4_SKINDIRECT_STAGE: 1 (default) declines the rewrite - the feature proper. 0 is the
// isolation probe: keep the page-table walk but drop the live/volatile flag from the matched
// reads, so the scalar in-loop chain head becomes read_const_dynamic. That read is the one every
// earlier experiment left untouched (SKINVRAM only moved the data, LIVENV keeps scalar live
// reads volatile, BDACACHE never caches a volatile walk, VEC4STRAIGHT only touched straight-line
// x4 loads), so stage 0 asks "is the chain head the cost?" without changing one descriptor.
static int SkinDirectStage() {
    static const int v = [] {
        const char* e = std::getenv("SHADPS4_SKINDIRECT_STAGE");
        return e != nullptr ? std::atoi(e) : 1;
    }();
    return v;
}
// The three ranges have fixed shapes in the block: SkinVramRedirect enforces the same
// kStride{8,0,0} / kMinRecords{1,4,48}, and DESCDUMP measured stride=8 records=2125,
// stride=0 records=8500 and stride=0 records=48. Requiring the shape is what keeps the decision
// honest - anything the compile-time V# cannot prove falls through to today's BDA path.
static bool SkinDirectShape(u32 word, const AmdGpu::Buffer& buffer) {
    u32 want_stride = 0;
    u32 min_records = 0;
    switch (word) {
    case 16u: // per-vertex influence headers
        want_stride = 8u;
        min_records = 1u;
        break;
    case 20u: // influence list
        want_stride = 0u;
        min_records = 4u;
        break;
    case 24u: // bone matrix palette
        want_stride = 0u;
        min_records = 48u;
        break;
    default:
        return false;
    }
    if (buffer.num_records == std::numeric_limits<u32>::max()) {
        return false; // the "pointer to all of memory" window: it has to keep the walk
    }
    if (buffer.base_address == 0 || buffer.GetSize() == 0 || buffer.num_records < min_records ||
        static_cast<u32>(buffer.stride) != want_stride) {
        return false;
    }
    return !buffer.swizzle_enable && !buffer.add_tid_enable && (buffer.base_address & 3) == 0;
}
// True when this read goes through one of the selected block dwords of a listed shader AND the
// V# sitting there is bindable. `out_word` is that block dword on success.
static bool SkinDirectMatch(const Info& info, const IR::Inst& inst, IR::Inst* handle,
                            const AmdGpu::Buffer& buffer, u32 sharp_idx, u32& out_word) {
    const auto& hashes = SkinDirectHashes();
    if (hashes.empty()) {
        return false; // off, checked before anything else: identical to today
    }
    if (std::find(hashes.begin(), hashes.end(), info.pgm_hash) == hashes.end()) {
        return false;
    }
    if (IsBufferStore(inst)) {
        return false; // stores keep write_const_live: results must reach guest memory
    }
    // V# word 0 must come from ReadConst(<block>, Imm32(W)). An inline V# (all args immediate)
    // has no producing instruction and stops here.
    IR::Inst* lo = handle->Arg(0).TryInstRecursive();
    if (lo == nullptr || lo->GetOpcode() != IR::Opcode::ReadConst || !lo->Arg(1).IsImmediate()) {
        return false;
    }
    const u32 word = lo->Arg(1).U32();
    const auto& words = SkinDirectWords();
    if (std::find(words.begin(), words.end(), word) == words.end()) {
        return false;
    }
    // The V# checked below is flat-buffer dword `sharp_idx`; make sure that is the one THIS read
    // loads. TrackSharp does dominance analysis and could have landed on another source.
    if (lo->Flags<u32>() != sharp_idx) {
        return false;
    }
    // <block> must be (ud_0, ud_1) - the descriptor-block pointer DESCDUMP and SkinVramRedirect
    // both decode, so all three agree on what a "block dword" means by construction.
    IR::Inst* blk = lo->Arg(0).TryInstRecursive();
    if (blk == nullptr || blk->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return false;
    }
    for (u32 i = 0; i < 2; ++i) {
        IR::Inst* ud = blk->Arg(i).TryInstRecursive();
        if (ud == nullptr || ud->GetOpcode() != IR::Opcode::GetUserData ||
            static_cast<u32>(ud->Arg(0).ScalarReg()) != i) {
            return false;
        }
    }
    if (!SkinDirectShape(word, buffer)) {
        // One of ours, but the descriptor is not bindable: keep the BDA path and say so, so a
        // run where nothing was converted cannot be mistaken for a measured result.
        VideoCore::BufferCache::SkinDirectNoteDecline();
        static std::atomic<u32> rej_logged{0};
        if (rej_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_CRITICAL(Render_Recompiler,
                         "[SKINDIRECT] shader {:#x} block dword {} (sharp_idx={}) NOT converted: "
                         "base={:#x} stride={} records={:#x} swizzle={} tid={} - keeping the BDA "
                         "walk",
                         info.pgm_hash, word, sharp_idx, static_cast<u64>(buffer.base_address),
                         static_cast<u32>(buffer.stride), static_cast<u32>(buffer.num_records),
                         static_cast<u32>(buffer.swizzle_enable),
                         static_cast<u32>(buffer.add_tid_enable));
        }
        return false;
    }
    out_word = word;
    return true;
}

// Rewrites a read through an unbounded V# (num_records == 0xFFFFFFFF, the
// "pointer to all of memory" pattern used by Naughty Dog's GPU job system)
// into a ReadConst chain. ReadConst is the DMA primitive: it walks the BDA
// page table and reads live guest memory, so the shader observes CPU writes
// made while the dispatch is executing. The snapshot copy a regular buffer
// binding would get is stale by definition and deadlocks the job system.
// Returns true if the access was rewritten (no buffer binding is created).
static bool TryPatchUnboundedBufferAccess(IR::Block& block, IR::Inst& inst, Info& info,
                                          const AmdGpu::Buffer& buffer, IR::Inst* handle,
                                          u32 sharp_idx, bool require_unbounded = true) {
    // [SKINDIRECT] first, ahead of the require_unbounded test: the decision has to win over
    // info.uses_live_dma and over the in-loop live rule, which are exactly what force these
    // bounded reads into a walk today (:997 passes require_unbounded=false for this shader).
    u32 sd_word = 0;
    const bool sd_match = SkinDirectMatch(info, inst, handle, buffer, sharp_idx, sd_word);
    const bool sd_nonlive = sd_match && SkinDirectStage() == 0;
    if (sd_match && !sd_nonlive) {
        if (VideoCore::BufferCache::SkinDirectRegister(info.pgm_hash, sharp_idx, sd_word)) {
            LOG_CRITICAL(Render_Recompiler,
                         "[SKINDIRECT] shader {:#x} block dword {} (sharp_idx={}) -> buffer "
                         "binding instead of BDA: base={:#x} stride={} records={} size={:#x}",
                         info.pgm_hash, sd_word, sharp_idx,
                         static_cast<u64>(buffer.base_address), static_cast<u32>(buffer.stride),
                         static_cast<u32>(buffer.num_records), buffer.GetSize());
        }
        return false; // -> descriptors.Add(BufferResource{...}) in the caller
    }
    if (require_unbounded && buffer.num_records != std::numeric_limits<u32>::max()) {
        return false;
    }
    if (!EmulatorSettings.IsDirectMemoryAccessEnabled() ||
        IsDmaMutedByBisect(info.pgm_hash)) {
        LOG_WARNING(Render_Recompiler,
                    "Unbounded V# sharp_idx={} but DMA disabled, keeping snapshot path",
                    sharp_idx);
        return false;
    }
    if (buffer.swizzle_enable || buffer.add_tid_enable || (buffer.base_address & 3) != 0) {
        LOG_WARNING(Render_Recompiler,
                    "Unbounded V# sharp_idx={} has swizzle/tid/unaligned base, keeping "
                    "snapshot path",
                    sharp_idx);
        return false;
    }
    u32 num_dwords{};
    bool is_float{};
    if (!DmaEligibleDwords(inst, num_dwords, is_float)) {
        LOG_WARNING(Render_Recompiler,
                    "Unbounded V# sharp_idx={} used by opcode {}, keeping snapshot path",
                    sharp_idx, inst.GetOpcode());
        return false;
    }

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    // Reconstruct the 44-bit guest base address from V# words 0-1 at runtime
    // so the rewrite stays valid if the game rebases the buffer.
    const IR::U32 base_lo{handle->Arg(0)};
    const IR::U32 base_hi{ir.BitwiseAnd(IR::U32{handle->Arg(1)}, ir.Imm32(0xFFFU))};
    const IR::Value base{ir.CompositeConstruct(base_lo, base_hi)};
    // Dword-sized opcodes get their offset already scaled to dwords.
    IR::U32 dword_offset;
    if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
        dword_offset = IR::U32{inst.Arg(1)};
    } else {
        dword_offset = CalculateBufferAddress(ir, inst, info, buffer, buffer.stride);
    }

    std::array<IR::Value, 4> dw{};
    // [SKINDIRECT] stage 0 (isolation probe only): the matched read keeps its walk but loses the
    // live flag, i.e. read_const_dynamic - non-volatile, BDACACHE-eligible, coalescable. It also
    // leaves the vec4 path below, because ReadConstX4 has no non-live flavour at all.
    if (sd_nonlive) {
        static std::atomic<u32> sd0_logged{0};
        if (sd0_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_CRITICAL(Render_Recompiler,
                         "[SKINDIRECT] stage 0: shader {:#x} block dword {} (sharp_idx={}) keeps "
                         "the BDA walk without the live/volatile flag ({} dwords)",
                         info.pgm_hash, sd_word, sharp_idx, num_dwords);
        }
    }
    const bool in_loop = info.loop_blocks.contains(&block);
    // [POLLJ] a read from a census poll loop also stamps its page into the presence window
    const bool poll_read = in_loop && PollTraceEnabled() && info.poll_blocks.contains(&block);
    if (poll_read && num_dwords == 4 && LiveVec4Enabled()) {
        // ReadConstX4 has no stamping flavour, so a poll whose control read is one 16-byte
        // load would be invisible to the trace: it takes the scalar path below instead (four
        // stamped walks; diagnostic mode only - with the env off the x4 shape is unchanged).
        LOG_CRITICAL(Render_Recompiler,
                     "POLLJ: shader {:#x} sharp_idx={} in-loop x4 poll load -> 4 stamped scalar "
                     "reads",
                     info.pgm_hash, sharp_idx);
    }
    if ((in_loop || Vec4Straight()) && num_dwords == 4 && LiveVec4Enabled() && !poll_read &&
        !sd_nonlive) {
        // [LIVEVEC4] one page-table walk for the whole 16-byte slice of the record
        const IR::Value vec = ir.ReadConstX4(base, dword_offset, Info::LiveReadConstFlag);
        for (u32 i = 0; i < 4; ++i) {
            const IR::U32 value{ir.CompositeExtract(vec, i)};
            dw[i] = is_float ? IR::Value{ir.BitCast<IR::F32>(value)} : IR::Value{value};
        }
        LOG_CRITICAL(Render_Recompiler,
                     "LIVEVEC4: shader {:#x} sharp_idx={} in-loop x4 load -> one live vec4",
                     info.pgm_hash, sharp_idx);
    } else
    for (u32 i = 0; i < num_dwords; ++i) {
        IR::U32 off = dword_offset;
        if (i != 0) {
            off = IR::U32{ir.IAdd(dword_offset, ir.Imm32(i))};
        }
        // Polling loops must observe memory writes made while the dispatch is
        // executing — their loads stay volatile. Straight-line reads keep
        // cacheable loads: volatile everywhere collapses PCIe throughput.
        // Faults populate the page table; unmapped pages read as zero.
        const u32 live_flag = poll_read ? Info::PollReadConstFlag : Info::LiveReadConstFlag;
        const IR::U32 value = ir.ReadConst(base, off, (in_loop && !sd_nonlive) ? live_flag : 0u);
        dw[i] = is_float ? IR::Value{ir.BitCast<IR::F32>(value)} : IR::Value{value};
    }
    switch (num_dwords) {
    case 1:
        inst.ReplaceUsesWith(dw[0]);
        break;
    case 2:
        inst.ReplaceUsesWith(ir.CompositeConstruct(dw[0], dw[1]));
        break;
    case 3:
        inst.ReplaceUsesWith(ir.CompositeConstruct(dw[0], dw[1], dw[2]));
        break;
    case 4:
        inst.ReplaceUsesWith(ir.CompositeConstruct(dw[0], dw[1], dw[2], dw[3]));
        break;
    }
    LOG_INFO(Render_Recompiler,
             "Unbounded V# access -> DMA: sharp_idx={}, base={:#x}, stride={}, dwords={}",
             sharp_idx, static_cast<u64>(buffer.base_address), static_cast<u32>(buffer.stride),
             num_dwords);
    return true;
}

// Store counterpart of DmaEligibleDwords.
static bool LiveStoreDwords(const IR::Inst& inst, u32& num_dwords, bool& is_float) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::StoreBufferU32:
        num_dwords = 1;
        is_float = false;
        return true;
    case IR::Opcode::StoreBufferU32x2:
        num_dwords = 2;
        is_float = false;
        return true;
    case IR::Opcode::StoreBufferU32x3:
        num_dwords = 3;
        is_float = false;
        return true;
    case IR::Opcode::StoreBufferU32x4:
        num_dwords = 4;
        is_float = false;
        return true;
    case IR::Opcode::StoreBufferF32:
        num_dwords = 1;
        is_float = true;
        return true;
    case IR::Opcode::StoreBufferF32x2:
        num_dwords = 2;
        is_float = true;
        return true;
    case IR::Opcode::StoreBufferF32x3:
        num_dwords = 3;
        is_float = true;
        return true;
    case IR::Opcode::StoreBufferF32x4:
        num_dwords = 4;
        is_float = true;
        return true;
    default:
        return false;
    }
}

// [SKINDIRECT-STORES] Expected output V# (DESCDUMP 08.09, 879/879 samples): stride 28, no
// swizzle/tid, records = vertex count, base dword-aligned, dword3 0x1806f3ac (32_32_32 Float) /
// 0x18067fac (16_16_16_16 Float). nullptr when the compile-time V# is that, else WHY not.
// Reported and counted only - NEVER a veto (layout invariance, see SkinDirectStoreMatch).
static const char* SkinDirectStoreShapeOdd(const AmdGpu::Buffer& buffer, const IR::Inst& inst) {
    if (buffer.base_address == 0 && buffer.num_records == 0) {
        // Valid() is type==0, so an all-zero V# passes GetSharp as is -> data_format 0 -> the
        // FMT0 guard in LowerBufferFormatToRaw drops the store (empty job slot permutation).
        return "zero-vsharp(FMT0 permutation)";
    }
    if (!buffer.Valid()) {
        return "type!=0(GetSharp->Null,FMT0)";
    }
    if (buffer.num_records == std::numeric_limits<u32>::max()) {
        return "unbounded";
    }
    if (buffer.base_address == 0 || buffer.GetSize() == 0) {
        return "null-base-or-size";
    }
    if (static_cast<u32>(buffer.stride) != 28u) {
        return "stride!=28";
    }
    if (buffer.swizzle_enable || buffer.add_tid_enable) {
        return "swizzle/tid";
    }
    if ((buffer.base_address & 3) != 0) {
        return "unaligned-base";
    }
    if (inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32) {
        const auto flags = inst.Flags<IR::BufferInstInfo>();
        const bool typed = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid;
        const auto dfmt = typed ? AmdGpu::RemapDataFormat(flags.inst_data_fmt.Value())
                                : buffer.GetDataFmt();
        switch (dfmt) { // the dword-multiple set LIVESTORE-FMT accepts (below)
        case AmdGpu::DataFormat::Format32:
        case AmdGpu::DataFormat::Format32_32:
        case AmdGpu::DataFormat::Format32_32_32:
        case AmdGpu::DataFormat::Format32_32_32_32:
        case AmdGpu::DataFormat::Format8_8_8_8:
        case AmdGpu::DataFormat::Format16_16:
        case AmdGpu::DataFormat::Format10_11_11:
        case AmdGpu::DataFormat::Format2_10_10_10:
        case AmdGpu::DataFormat::Format16_16_16_16:
            break;
        default:
            return "sub-dword-or-invalid-dfmt";
        }
    }
    return nullptr;
}

// [SKINDIRECT-STORES] STRUCTURAL match: listed hash, dword store opcode, V# word 0 =
// ReadConst(CompositeConstructU32x2(GetUserData(s0), GetUserData(s1)), Imm32(W)) with W in the
// store words and Flags == sharp_idx (the TrackSharp dominance guard SkinDirectMatch uses).
// Nothing here reads the V# contents: the pipeline of every permutation binds through
// program->info.buffers (the first compile) while its SPIR-V comes from its own Info, so a
// V#-dependent verdict would hand one permutation a shifted descriptor layout. SkinDirectMatch
// itself stays untouched (reads only), so the read-side A/B is byte-identical with this feature
// off or on. `out_word` is the block dword on success.
static bool SkinDirectStoreMatch(const Info& info, const IR::Inst& inst, IR::Inst* handle,
                                 u32 sharp_idx, u32& out_word) {
    if (!SkinDirectStoresEnabled()) {
        return false; // off: identical to today, checked before anything else
    }
    const auto& hashes = SkinDirectHashes();
    if (hashes.empty() || std::find(hashes.begin(), hashes.end(), info.pgm_hash) == hashes.end()) {
        return false;
    }
    u32 nd{};
    bool is_f{};
    if (inst.GetOpcode() != IR::Opcode::StoreBufferFormatF32 && !LiveStoreDwords(inst, nd, is_f)) {
        return false; // U8/U16/U64/atomics: not this worker's shape, today's paths decide
    }
    IR::Inst* lo = handle->Arg(0).TryInstRecursive();
    if (lo == nullptr || lo->GetOpcode() != IR::Opcode::ReadConst || !lo->Arg(1).IsImmediate()) {
        return false;
    }
    const u32 word = lo->Arg(1).U32();
    const auto& words = SkinDirectStoreWords();
    if (std::find(words.begin(), words.end(), word) == words.end()) {
        return false;
    }
    if (lo->Flags<u32>() != sharp_idx) {
        return false;
    }
    IR::Inst* blk = lo->Arg(0).TryInstRecursive();
    if (blk == nullptr || blk->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return false;
    }
    for (u32 i = 0; i < 2; ++i) {
        IR::Inst* ud = blk->Arg(i).TryInstRecursive();
        if (ud == nullptr || ud->GetOpcode() != IR::Opcode::GetUserData ||
            static_cast<u32>(ud->Arg(0).ScalarReg()) != i) {
            return false;
        }
    }
    out_word = word;
    return true;
}

// Rewrites a store in a live (job-system) shader into WriteConst dwords going
// through the BDA page table. The store then lands in imported host memory —
// the same home the CPU and live readers use — instead of a VRAM snapshot
// no one else can see. The original store is invalidated: this shader must
// not claim the job pages with a buffer binding.
static bool TryPatchLiveBufferStore(IR::Block& block, IR::Inst& inst, Info& info,
                                    const AmdGpu::Buffer& buffer, IR::Inst* handle,
                                    u32 sharp_idx) {
    if (!EmulatorSettings.IsDirectMemoryAccessEnabled() ||
        IsDmaMutedByBisect(info.pgm_hash)) {
        return false;
    }
    if (buffer.swizzle_enable || buffer.add_tid_enable || (buffer.base_address & 3) != 0) {
        LOG_CRITICAL(Render_Recompiler,
                    "Live store sharp_idx={} has swizzle/tid/unaligned base, keeping snapshot",
                    sharp_idx);
        return false;
    }
    // [LIVESTORE-FMT] Typed stores (buffer_store_format_*) are lowered to raw dwords later by
    // LowerBufferFormatToRaw; the job workers of Uncharted write their results exactly this way
    // (StoreBufferFormatF32 through sharps 16/20/24/28). Route them live too: pack the value the
    // same way that pass would and write whole dwords through the BDA page table. Sub-dword
    // formats would need a read-modify-write and stay on the snapshot path.
    u32 num_dwords{};
    bool is_float{};
    const bool is_raw_store = LiveStoreDwords(inst, num_dwords, is_float);
    const bool is_fmt_store = inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32;
    AmdGpu::DataFormat data_format{AmdGpu::DataFormat::FormatInvalid};
    AmdGpu::NumberFormat num_format{AmdGpu::NumberFormat::Unorm};
    AmdGpu::CompMapping swizzle{};
    AmdGpu::NumberConversion num_conversion{};
    u32 num_components{};
    if (is_fmt_store) {
        const auto flags = inst.Flags<IR::BufferInstInfo>();
        const bool is_inst_typed = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid;
        data_format = is_inst_typed ? AmdGpu::RemapDataFormat(flags.inst_data_fmt.Value())
                                    : buffer.GetDataFmt();
        num_format = is_inst_typed
                         ? AmdGpu::RemapNumberFormat(flags.inst_num_fmt.Value(), data_format)
                         : buffer.GetNumberFmt();
        swizzle = is_inst_typed
                      ? AmdGpu::RemapSwizzle(flags.inst_data_fmt.Value(), AmdGpu::IdentityMapping)
                      : buffer.DstSelect();
        num_conversion = is_inst_typed ? AmdGpu::MapNumberConversion(flags.inst_num_fmt.Value(),
                                                                     flags.inst_data_fmt.Value())
                                       : buffer.GetNumberConversion();
        num_components = AmdGpu::NumComponents(data_format);
        switch (data_format) {
        case AmdGpu::DataFormat::Format32:
        case AmdGpu::DataFormat::Format32_32:
        case AmdGpu::DataFormat::Format32_32_32:
        case AmdGpu::DataFormat::Format32_32_32_32:
        case AmdGpu::DataFormat::Format8_8_8_8:
        case AmdGpu::DataFormat::Format16_16:
        case AmdGpu::DataFormat::Format10_11_11:
        case AmdGpu::DataFormat::Format2_10_10_10:
        case AmdGpu::DataFormat::Format16_16_16_16:
            break;
        default:
            LOG_CRITICAL(Render_Recompiler,
                         "Live typed store sharp_idx={} fmt {} kept on snapshot path", sharp_idx,
                         data_format);
            return false;
        }
    } else if (!is_raw_store) {
        LOG_CRITICAL(Render_Recompiler, "Live store sharp_idx={} opcode {} kept on snapshot path",
                     sharp_idx, inst.GetOpcode());
        return false;
    }

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U32 base_lo{handle->Arg(0)};
    const IR::U32 base_hi{ir.BitwiseAnd(IR::U32{handle->Arg(1)}, ir.Imm32(0xFFFU))};
    const IR::Value base{ir.CompositeConstruct(base_lo, base_hi)};
    const IR::U32 dword_offset = CalculateBufferAddress(ir, inst, info, buffer, buffer.stride);
    const IR::Value value = inst.Arg(2);
    boost::container::static_vector<IR::U32, 4> dwords;
    if (is_raw_store) {
        for (u32 i = 0; i < num_dwords; ++i) {
            IR::Value comp = value;
            if (num_dwords > 1) {
                comp = ir.CompositeExtract(value, i);
            }
            if (is_float) {
                dwords.push_back(ir.BitCast<IR::U32>(IR::F32{comp}));
            } else {
                dwords.push_back(IR::U32{comp});
            }
        }
    } else {
        const auto swizzled = ApplySwizzle(ir, value, swizzle.Inverse());
        boost::container::static_vector<IR::Value, 4> comps;
        for (u32 i = 0; i < num_components; ++i) {
            comps.push_back(ApplyWriteNumberConversion(
                ir, IR::F32{ir.CompositeExtract(swizzled, i)}, num_conversion));
        }
        const auto comp = [&](u32 i) -> IR::Value {
            return i < comps.size() ? comps[i] : IR::Value{ir.Imm32(0.f)};
        };
        switch (data_format) {
        case AmdGpu::DataFormat::Format32:
        case AmdGpu::DataFormat::Format32_32:
        case AmdGpu::DataFormat::Format32_32_32:
        case AmdGpu::DataFormat::Format32_32_32_32:
            for (u32 i = 0; i < num_components; ++i) {
                dwords.push_back(ir.BitCast<IR::U32>(IR::F32{comps[i]}));
            }
            break;
        case AmdGpu::DataFormat::Format8_8_8_8:
            dwords.push_back(ir.Pack4x8(
                num_format, ir.CompositeConstruct(comp(0), comp(1), comp(2), comp(3))));
            break;
        case AmdGpu::DataFormat::Format16_16:
            dwords.push_back(ir.Pack2x16(num_format, ir.CompositeConstruct(comp(0), comp(1))));
            break;
        case AmdGpu::DataFormat::Format10_11_11:
            dwords.push_back(
                ir.Pack10_11_11(num_format, ir.CompositeConstruct(comp(0), comp(1), comp(2))));
            break;
        case AmdGpu::DataFormat::Format2_10_10_10:
            dwords.push_back(ir.Pack2_10_10_10(
                num_format, ir.CompositeConstruct(comp(0), comp(1), comp(2), comp(3))));
            break;
        case AmdGpu::DataFormat::Format16_16_16_16:
            dwords.push_back(ir.Pack2x16(num_format, ir.CompositeConstruct(comp(0), comp(1))));
            dwords.push_back(ir.Pack2x16(num_format, ir.CompositeConstruct(comp(2), comp(3))));
            break;
        default:
            UNREACHABLE();
        }
        LOG_CRITICAL(Render_Recompiler,
                     "LIVESTORE-FMT: shader {:#x} sharp_idx={} dfmt={} nfmt={} comps={} -> {} "
                     "live dwords",
                     info.pgm_hash, sharp_idx, data_format, num_format, num_components,
                     dwords.size());
    }
    if (LiveVec4Stores() && dwords.size() >= 2) {
        // [LIVEVEC4] one page-table walk for the whole record slice
        const u32 n = static_cast<u32>(dwords.size());
        const auto pad = [&](u32 i) -> IR::Value {
            return i < n ? IR::Value{dwords[i]} : IR::Value{ir.Imm32(0U)};
        };
        const IR::Value vec = ir.CompositeConstruct(pad(0), pad(1), pad(2), pad(3));
        ir.WriteConstX4(base, dword_offset, vec, n);
        LOG_CRITICAL(Render_Recompiler,
                     "LIVEVEC4-STORE: shader {:#x} sharp_idx={} {} dwords -> one live vector store",
                     info.pgm_hash, sharp_idx, n);
    } else
    for (u32 i = 0; i < dwords.size(); ++i) {
        IR::U32 off = dword_offset;
        if (i != 0) {
            off = IR::U32{ir.IAdd(dword_offset, ir.Imm32(i))};
        }
        ir.WriteConst(base, off, dwords[i], Info::LiveReadConstFlag);
    }
    inst.Invalidate();
    LOG_INFO(Render_Recompiler, "Live store -> DMA: sharp_idx={}, base={:#x}, dwords={}",
             sharp_idx, static_cast<u64>(buffer.base_address), dwords.size());
    return true;
}

void PatchBufferSharp(IR::Block& block, IR::Inst& inst, Info& info, Descriptors& descriptors,
                      const Profile& profile) {
    IR::Inst* handle = inst.Arg(0).InstRecursive();
    u32 buffer_binding = 0;
    if (handle->AreAllArgsImmediates()) {
        // Assuming V# is in UD s[32:35]
        // The next pattern:
        // s_getpc_b64     s[32:33]
        // s_add_u32       s32, <const>, s32
        // s_addc_u32      s33, 0, s33
        // s_mov_b32       s35, <const>
        // s_movk_i32      s34, <const>
        // buffer_load_format_xyz v[8:10], v1, s[32:35], 0 ...
        // is used to define an inline buffer resource
        std::array<u64, 2> raw;
        // Keep relative address, we'll do fixup of the address at buffer fetch later
        raw[0] = handle->Arg(0).U32() | u64(handle->Arg(1).U32()) << 32;
        raw[1] = handle->Arg(2).U32() | u64(handle->Arg(3).U32()) << 32;
        const auto buffer = std::bit_cast<AmdGpu::Buffer>(raw);
        LOG_INFO(Render_Recompiler,
                 "PBS: op={}, inline=1, base={:#x}, num_records={:#x}, stride={}",
                 inst.GetOpcode(), static_cast<u64>(buffer.base_address),
                 static_cast<u32>(buffer.num_records), static_cast<u32>(buffer.stride));
        // Inline V# with base 0 and num_records 0xFFFFFFFF is the ND job
        // system's "all of memory" window: the guest address lives entirely
        // in the index/offset operands, so there is no PC-relative base to
        // fix up and the access can go through the BDA page table directly.
        if (buffer.base_address == 0 &&
            TryPatchUnboundedBufferAccess(block, inst, info, buffer, handle,
                                          std::numeric_limits<u32>::max())) {
            return;
        }
        buffer_binding = descriptors.Add(BufferResource{
            .sharp_idx = std::numeric_limits<u32>::max(),
            .used_types = BufferDataType(inst, profile, buffer.GetNumberFmt()),
            .inline_cbuf = buffer,
            .buffer_type = BufferType::Guest,
        });
    } else {
        // Normal buffer resource.
        IR::Inst* buffer_handle = handle->Arg(0).InstRecursive();
        const auto inst_info = inst.Flags<IR::BufferInstInfo>();
        const auto sharp_idx = TrackSharp(buffer_handle, block, inst_info.pc);
        const auto buffer = info.ReadUdSharp<AmdGpu::Buffer>(sharp_idx);
        LOG_INFO(Render_Recompiler,
                 "PBS: op={}, inline=0, sharp_idx={}, base={:#x}, num_records={:#x}, stride={}",
                 inst.GetOpcode(), sharp_idx, static_cast<u64>(buffer.base_address),
                 static_cast<u32>(buffer.num_records), static_cast<u32>(buffer.stride));
        // [SKINDIRECT-STORES] decided FIRST: TryPatchLiveBufferStore is what turns these stores
        // into walks today, and TryPatchUnboundedBufferAccess never rewrites a store
        // (DmaEligibleDwords is load-only) - it would only print a misleading "keeping snapshot
        // path" warning, hence sd_store skips that call too. On a match NOTHING is rewritten:
        // the store falls through to descriptors.Add below as a WRITTEN + FORMATTED guest
        // binding - the path a SHADPS4_NODMA run and the 06.09 FMT0 permutation already take.
        u32 sd_word = 0;
        const bool sd_store = info.uses_live_dma && IsBufferStore(inst) &&
                              SkinDirectStoreMatch(info, inst, handle, sharp_idx, sd_word);
        if (sd_store) {
            const char* odd = SkinDirectStoreShapeOdd(buffer, inst);
            const bool needs_lowering = inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32;
            const bool first = VideoCore::BufferCache::SkinDirectRegisterStore(
                info.pgm_hash, sharp_idx, sd_word, odd != nullptr, needs_lowering);
            // Once per (hash, sharp_idx) plus the first 32 without dedup, so every permutation
            // (split/unsplit, SPECMISS, the zero-V# one) shows its own compile-time V#.
            static std::atomic<u32> sds_logged{0};
            if (first || sds_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                // dfmt/nfmt as raw numbers: the compile-time V# of a SPECMISS permutation is
                // whatever the flat buffer held, and fmt::formatter<DataFormat> goes through
                // NameOf(), which UNREACHABLE()s on a value without an enumerator (dfmt 15).
                LOG_CRITICAL(Render_Recompiler,
                             "[SKINDIRECT-STORE] shader {:#x} block dword {} (sharp_idx={}) op={} "
                             "-> bound WRITTEN SSBO instead of write_const_live: base={:#x} "
                             "stride={} records={} size={:#x} dfmt={} nfmt={} shape={}",
                             info.pgm_hash, sd_word, sharp_idx, inst.GetOpcode(),
                             static_cast<u64>(buffer.base_address),
                             static_cast<u32>(buffer.stride),
                             static_cast<u32>(buffer.num_records), buffer.GetSize(),
                             static_cast<u32>(buffer.GetDataFmt()),
                             static_cast<u32>(buffer.GetNumberFmt()), odd ? odd : "ok");
            }
        } else if (info.uses_live_dma && IsBufferStore(inst) &&
                   TryPatchLiveBufferStore(block, inst, info, buffer, handle, sharp_idx)) {
            // Job-system shaders (live DMA mode) route every eligible access
            // through the BDA page table so the job pages keep a single home.
            if (LiveStoreEnabled()) {
                static std::atomic<u32> ls_logged{0};
                if (ls_logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                    LOG_CRITICAL(Render_Recompiler, "LIVESTORE: shader={:#x} store -> live BDA (op={})",
                                 info.pgm_hash, inst.GetOpcode());
                }
            }
            return;
        }
        const bool live_loop_load = LiveLoopEnabled() && info.l_stage == LogicalStage::Compute &&
                                    !IsBufferStore(inst) && info.loop_blocks.contains(&block);
        if (live_loop_load) {
            static std::atomic<u32> ll_logged{0};
            if (ll_logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_CRITICAL(Render_Recompiler, "LIVELOOP: shader={:#x} in-loop load -> live BDA (op={})",
                             info.pgm_hash, inst.GetOpcode());
            }
        }
        if (!sd_store &&
            TryPatchUnboundedBufferAccess(block, inst, info, buffer, handle, sharp_idx,
                                          /*require_unbounded=*/!(info.uses_live_dma || live_loop_load))) {
            // The access now reads live guest memory through the BDA page
            // table; no buffer binding or snapshot copy is needed.
            return;
        }
        buffer_binding = descriptors.Add(BufferResource{
            .sharp_idx = sharp_idx,
            .used_types = BufferDataType(inst, profile, buffer.GetNumberFmt()),
            .buffer_type = BufferType::Guest,
            .is_written = IsBufferStore(inst),
            .is_formatted = inst.GetOpcode() == IR::Opcode::LoadBufferFormatF32 ||
                            inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32,
        });
    }

    // Replace handle with binding index in buffer resource list.
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    inst.SetArg(0, ir.Imm32(buffer_binding));
}

void PatchImageSharp(IR::Block& block, IR::Inst& inst, Info& info, Descriptors& descriptors,
                     const Profile& profile) {
    // Read image sharp.
    const auto inst_info = inst.Flags<IR::TextureInstInfo>();
    const IR::Inst* image_handle = inst.Arg(0).InstRecursive();
    const auto tsharp = TrackSharp(image_handle, block, inst_info.pc);
    const bool is_atomic = IsImageAtomicInstruction(inst);
    const bool is_written = inst.GetOpcode() == IR::Opcode::ImageWrite || is_atomic;
    const bool is_storage =
        inst.GetOpcode() == IR::Opcode::ImageRead || inst.GetOpcode() == IR::Opcode::ImageWrite;
    // ImageRead with !is_written gets emitted as OpImageFetch with LOD operand, doesn't
    // need fallback (TODO is this 100% true?)
    const bool needs_mip_storage_fallback =
        inst_info.has_lod && is_written && !profile.supports_image_load_store_lod;
    ImageResource image_res = {
        .sharp_idx = tsharp,
        .is_depth = bool(inst_info.is_depth),
        .is_atomic = is_atomic,
        .is_array = bool(inst_info.is_array),
        .is_written = is_written,
        .is_r128 = bool(inst_info.is_r128),
    };

    auto image = image_res.GetSharp(info);
    ASSERT(image.GetType() != AmdGpu::ImageType::Invalid);

    if (needs_mip_storage_fallback) {
        // If the mip level to IMAGE_(LOAD/STORE)_MIP is a constant, set up ImageResource
        // so that we will only bind a single level.
        // If index is dynamic, we will bind levels as an array
        const auto view_type = image.GetViewType(image_res.is_array);

        IR::Inst* body = inst.Arg(1).InstRecursive();
        const auto lod_arg = [&] -> IR::Value {
            switch (view_type) {
            case AmdGpu::ImageType::Color1D: // x, [lod]
                return body->Arg(1);
            case AmdGpu::ImageType::Color1DArray: // x, slice, [lod]
            case AmdGpu::ImageType::Color2D:      // x, y, [lod]
                return body->Arg(2);
            case AmdGpu::ImageType::Color2DArray: // x, y, slice, [lod]
            case AmdGpu::ImageType::Color3D:      // x, y, z, [lod]
                return body->Arg(3);
            case AmdGpu::ImageType::Color2DMsaa:
            case AmdGpu::ImageType::Color2DMsaaArray:
            default:
                UNREACHABLE_MSG("Invalid image type {}", view_type);
            }
        }();

        if (lod_arg.IsImmediate()) {
            image_res.mip_fallback_mode = MipStorageFallbackMode::ConstantIndex;
            image_res.constant_mip_index = lod_arg.U32();
        } else {
            image_res.mip_fallback_mode = MipStorageFallbackMode::DynamicIndex;
        }
    }

    // Patch image instruction if image is FMask.
    if (AmdGpu::IsFmask(image.GetDataFmt())) {
        ASSERT_MSG(!is_written, "FMask storage instructions are not supported");

        IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
        switch (inst.GetOpcode()) {
        case IR::Opcode::ImageRead:
        case IR::Opcode::ImageSampleRaw: {
            IR::F32 fmaskx = ir.BitCast<IR::F32>(ir.Imm32(0x76543210));
            IR::F32 fmasky = ir.BitCast<IR::F32>(ir.Imm32(0xfedcba98));
            inst.ReplaceUsesWith(ir.CompositeConstruct(fmaskx, fmasky));
            return;
        }
        case IR::Opcode::ImageQueryLod:
            inst.ReplaceUsesWith(ir.Imm32(1));
            return;
        case IR::Opcode::ImageQueryDimensions: {
            IR::Value dims = ir.CompositeConstruct(ir.Imm32(static_cast<u32>(image.width)),  // x
                                                   ir.Imm32(static_cast<u32>(image.height)), // y
                                                   ir.Imm32(1), ir.Imm32(1)); // depth, mip
            inst.ReplaceUsesWith(dims);

            // Track FMask resource to do specialization.
            descriptors.Add(FMaskResource{
                .sharp_idx = tsharp,
            });
            return;
        }
        default:
            UNREACHABLE_MSG("Can't patch fmask instruction {}", inst.GetOpcode());
        }
    }

    u32 image_binding = descriptors.Add(image_res);

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};

    if (inst.GetOpcode() == IR::Opcode::ImageSampleRaw) {
        u32 sampler_binding = 0;
        const IR::Inst* sampler = inst.Arg(1).InstRecursive();
        ASSERT(sampler && sampler->GetOpcode() == IR::Opcode::CompositeConstructU32x4);
        // Inline sampler resource.
        if (sampler->AreAllArgsImmediates()) {
            const auto inline_sampler = AmdGpu::Sampler{
                .raw0 = u64(sampler->Arg(1).U32()) << 32 | u64(sampler->Arg(0).U32()),
                .raw1 = u64(sampler->Arg(3).U32()) << 32 | u64(sampler->Arg(2).U32()),
            };
            sampler_binding = descriptors.Add(SamplerResource{
                .sharp_idx = std::numeric_limits<u32>::max(),
                .inline_sampler = inline_sampler,
                .is_inline_sampler = true,
            });
        } else {
            // Normal sampler resource.
            const auto& [sampler_handle, disable_aniso] =
                TryDisableAnisoLod0(sampler->Arg(0).InstRecursive());
            const auto ssharp = TrackSharp(sampler_handle, block, inst_info.pc);
            sampler_binding = descriptors.Add(SamplerResource{
                .sharp_idx = ssharp,
                .is_inline_sampler = false,
                .associated_image = image_binding,
                .disable_aniso = disable_aniso,
            });
        }
        inst.SetArg(0, ir.Imm32(image_binding | sampler_binding << 16));
    } else {
        inst.SetArg(0, ir.Imm32(image_binding));
    }
}

void PatchGlobalDataShareAccess(IR::Block& block, IR::Inst& inst, Info& info,
                                Descriptors& descriptors, const Profile& profile) {
    const u32 binding = descriptors.Add(BufferResource{
        .used_types = IR::Type::U32,
        .inline_cbuf = AmdGpu::Buffer::Null(),
        .buffer_type = BufferType::GdsBuffer,
        .is_written = true,
    });

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};

    // For data append/consume operations attempt to deduce the GDS address.
    if (inst.GetOpcode() == IR::Opcode::DataAppend || inst.GetOpcode() == IR::Opcode::DataConsume) {
        const auto pred = [](const IR::Inst* inst) -> std::optional<const IR::Inst*> {
            if (inst->GetOpcode() == IR::Opcode::GetUserData) {
                return inst;
            }
            return std::nullopt;
        };

        u32 gds_addr = 0;
        const IR::Value& gds_offset = inst.Arg(0);
        if (gds_offset.IsImmediate()) {
            // Nothing to do, offset is known.
            gds_addr = gds_offset.U32() & 0xFFFF;
        } else {
            const auto result = IR::BreadthFirstSearch(&inst, pred);
            ASSERT_MSG(result, "Unable to track M0 source");

            // M0 must be set by some user data register.
            const IR::Inst* prod = gds_offset.InstRecursive();
            const u32 ud_reg = u32(result.value()->Arg(0).ScalarReg());
            u32 m0_val = info.user_data[ud_reg] >> 16;
            if (prod->GetOpcode() == IR::Opcode::IAdd32) {
                m0_val += prod->Arg(1).U32();
            }
            gds_addr = m0_val & 0xFFFF;
        }

        // Patch instruction to GDS buffer atomic increment/decrement.
        const IR::U32 handle = ir.Imm32(binding);
        const IR::U32 index = ir.Imm32(gds_addr >> 2);
        const bool is_append = inst.GetOpcode() == IR::Opcode::DataAppend;
        const IR::Value prev = is_append ? ir.BufferAtomicInc(handle, index, {})
                                         : ir.BufferAtomicDec(handle, index, {});
        inst.ReplaceUsesWithAndRemove(prev);
    } else {
        // Convert shared memory opcode to storage buffer atomic to GDS buffer.
        auto& buffer = info.buffers[binding];
        const IR::U32 offset = IR::U32{inst.Arg(0)};
        const IR::U32 address_words = ir.ShiftRightLogical(offset, ir.Imm32(1));
        const IR::U32 address_dwords = ir.ShiftRightLogical(offset, ir.Imm32(2));
        const IR::U32 address_qwords = ir.ShiftRightLogical(offset, ir.Imm32(3));
        const IR::U32 handle = ir.Imm32(binding);
        switch (inst.GetOpcode()) {
        case IR::Opcode::SharedAtomicIAdd32:
            inst.ReplaceUsesWith(ir.BufferAtomicIAdd(handle, address_dwords, inst.Arg(1), {}));
            break;
        case IR::Opcode::SharedAtomicIAdd64:
            inst.ReplaceUsesWith(
                ir.BufferAtomicIAdd(handle, address_qwords, IR::U64{inst.Arg(1)}, {}));
            break;
        case IR::Opcode::SharedAtomicISub32:
            inst.ReplaceUsesWith(ir.BufferAtomicISub(handle, address_dwords, inst.Arg(1), {}));
            break;
        case IR::Opcode::SharedAtomicSMin32:
        case IR::Opcode::SharedAtomicUMin32: {
            const bool is_signed = inst.GetOpcode() == IR::Opcode::SharedAtomicSMin32;
            inst.ReplaceUsesWith(
                ir.BufferAtomicIMin(handle, address_dwords, inst.Arg(1), is_signed, {}));
            break;
        }
        case IR::Opcode::SharedAtomicSMax32:
        case IR::Opcode::SharedAtomicUMax32: {
            const bool is_signed = inst.GetOpcode() == IR::Opcode::SharedAtomicSMax32;
            inst.ReplaceUsesWith(
                ir.BufferAtomicIMax(handle, address_dwords, inst.Arg(1), is_signed, {}));
            break;
        }
        case IR::Opcode::SharedAtomicInc32:
            inst.ReplaceUsesWith(ir.BufferAtomicInc(handle, address_dwords, {}));
            break;
        case IR::Opcode::SharedAtomicDec32:
            inst.ReplaceUsesWith(ir.BufferAtomicDec(handle, address_dwords, {}));
            break;
        case IR::Opcode::SharedAtomicAnd32:
            inst.ReplaceUsesWith(ir.BufferAtomicAnd(handle, address_dwords, inst.Arg(1), {}));
            break;
        case IR::Opcode::SharedAtomicOr32:
            inst.ReplaceUsesWith(ir.BufferAtomicOr(handle, address_dwords, inst.Arg(1), {}));
            break;
        case IR::Opcode::SharedAtomicXor32:
            inst.ReplaceUsesWith(ir.BufferAtomicXor(handle, address_dwords, inst.Arg(1), {}));
            break;
        case IR::Opcode::LoadSharedU16: {
            inst.ReplaceUsesWith(ir.LoadBufferU16(handle, address_words, {}));
            buffer.used_types |= IR::Type::U16;
            break;
        }
        case IR::Opcode::LoadSharedU32:
            inst.ReplaceUsesWith(ir.LoadBufferU32(1, handle, address_dwords, {}));
            break;
        case IR::Opcode::LoadSharedU64: {
            inst.ReplaceUsesWith(ir.LoadBufferU64(handle, address_qwords, {}));
            buffer.used_types |= IR::Type::U64;
            break;
        }
        case IR::Opcode::WriteSharedU16: {
            ir.StoreBufferU16(handle, address_words, IR::U16{inst.Arg(1)}, {});
            inst.Invalidate();
            buffer.used_types |= IR::Type::U16;
            break;
        }
        case IR::Opcode::WriteSharedU32:
            ir.StoreBufferU32(1, handle, address_dwords, inst.Arg(1), {});
            inst.Invalidate();
            break;
        case IR::Opcode::WriteSharedU64: {
            ir.StoreBufferU64(handle, address_qwords, IR::U64{inst.Arg(1)}, {});
            inst.Invalidate();
            buffer.used_types |= IR::Type::U64;
            break;
        }
        default:
            UNREACHABLE();
        }
    }
}

IR::U32 CalculateBufferAddress(IR::IREmitter& ir, const IR::Inst& inst, const Info& info,
                               const AmdGpu::Buffer& buffer, u32 stride) {
    const auto inst_info = inst.Flags<IR::BufferInstInfo>();
    const u32 inst_offset = inst_info.inst_offset.Value();
    const auto is_inst_typed = inst_info.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid;
    const auto data_format = is_inst_typed
                                 ? AmdGpu::RemapDataFormat(inst_info.inst_data_fmt.Value())
                                 : buffer.GetDataFmt();
    const u32 shift = BufferAddressShift(inst, data_format);
    const u32 mask = (1 << shift) - 1;
    const IR::U32 soffset = IR::GetBufferSOffsetArg(&inst);

    // If address calculation is of the form "index * const_stride + offset" with offset constant
    // and both const_stride and offset are divisible with the element size, apply shift directly.
    if (inst_info.index_enable && !inst_info.voffset_enable && soffset.IsImmediate() &&
        !buffer.swizzle_enable && !buffer.add_tid_enable && (stride & mask) == 0) {
        const u32 total_offset = soffset.U32() + inst_offset;
        if ((total_offset & mask) == 0) {
            // buffer_offset = index * (const_stride >> shift) + (offset >> shift)
            const IR::U32 index = IR::GetBufferIndexArg(&inst);
            return ir.IAdd(ir.IMul(index, ir.Imm32(stride >> shift)),
                           ir.Imm32(total_offset >> shift));
        }
    }

    // index = (inst_idxen ? vgpr_index : 0) + (const_add_tid_enable ? thread_id[5:0] : 0)
    IR::U32 index = ir.Imm32(0U);
    if (inst_info.index_enable) {
        const IR::U32 vgpr_index = IR::GetBufferIndexArg(&inst);
        index = ir.IAdd(index, vgpr_index);
    }
    if (buffer.add_tid_enable) {
        ASSERT_MSG(info.l_stage == LogicalStage::Compute,
                   "Thread ID buffer addressing is not supported outside of compute.");
        const IR::U32 thread_id{ir.LaneId()};
        index = ir.IAdd(index, thread_id);
    }
    // offset = (inst_offen ? vgpr_offset : 0) + inst_offset
    IR::U32 offset = ir.Imm32(inst_offset);
    offset = ir.IAdd(offset, soffset);
    if (inst_info.voffset_enable) {
        const IR::U32 voffset = IR::GetBufferVOffsetArg(&inst);
        offset = ir.IAdd(offset, voffset);
    }
    const IR::U32 const_stride = ir.Imm32(stride);
    IR::U32 buffer_offset;
    if (buffer.swizzle_enable) {
        const IR::U32 const_index_stride = ir.Imm32(buffer.GetIndexStride());
        const IR::U32 const_element_size = ir.Imm32(buffer.GetElementSize());
        // index_msb = index / const_index_stride
        const IR::U32 index_msb{ir.IDiv(index, const_index_stride)};
        // index_lsb = index % const_index_stride
        const IR::U32 index_lsb{ir.IMod(index, const_index_stride)};
        // offset_msb = offset / const_element_size
        const IR::U32 offset_msb{ir.IDiv(offset, const_element_size)};
        // offset_lsb = offset % const_element_size
        const IR::U32 offset_lsb{ir.IMod(offset, const_element_size)};
        // buffer_offset =
        //     (index_msb * const_stride + offset_msb * const_element_size) * const_index_stride
        //     + index_lsb * const_element_size + offset_lsb
        const IR::U32 buffer_offset_msb = ir.IMul(
            ir.IAdd(ir.IMul(index_msb, const_stride), ir.IMul(offset_msb, const_element_size)),
            const_index_stride);
        const IR::U32 buffer_offset_lsb =
            ir.IAdd(ir.IMul(index_lsb, const_element_size), offset_lsb);
        buffer_offset = ir.IAdd(buffer_offset_msb, buffer_offset_lsb);
    } else {
        // buffer_offset = index * const_stride + offset
        buffer_offset = ir.IAdd(ir.IMul(index, const_stride), offset);
    }
    if (shift != 0) {
        buffer_offset = ir.ShiftRightLogical(buffer_offset, ir.Imm32(shift));
    }
    return buffer_offset;
}

void PatchBufferArgs(IR::Block& block, IR::Inst& inst, Info& info) {
    const auto handle = inst.Arg(0);
    const auto buffer_res = info.buffers[handle.U32()];
    const auto buffer = buffer_res.GetSharp(info);

    // Address of constant buffer reads can be calculated at IR emission time.
    if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
        return;
    }

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    inst.SetArg(IR::LoadBufferArgs::Address,
                CalculateBufferAddress(ir, inst, info, buffer, buffer.stride));
}

IR::Value FixCubeCoords(IR::IREmitter& ir, const AmdGpu::Image& image, const IR::Value& x,
                        const IR::Value& y, const IR::Value& face) {
    if (!image.IsCube()) {
        return ir.CompositeConstruct(x, y, face);
    }
    // AMD cube math results in coordinates in the range [1.0, 2.0]. We need
    // to convert this to the range [0.0, 1.0] to get correct results.
    const auto fixed_x = ir.FPSub(IR::F32{x}, ir.Imm32(1.f));
    const auto fixed_y = ir.FPSub(IR::F32{y}, ir.Imm32(1.f));
    const auto fixed_face =
        ir.FPFma(ir.FPFloor(ir.FPDiv(IR::F32{face}, ir.Imm32(8.f))), ir.Imm32(-2.f), IR::F32{face});
    return ir.CompositeConstruct(fixed_x, fixed_y, fixed_face);
}

void PatchImageSampleArgs(IR::Block& block, IR::Inst& inst, Info& info,
                          const ImageResource& image_res, const AmdGpu::Image& image) {
    const auto handle = inst.Arg(0);
    const auto& sampler_res = info.samplers[(handle.U32() >> 16) & 0xFFFF];
    const auto sampler = sampler_res.GetSharp(info);

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const auto inst_info = inst.Flags<IR::TextureInstInfo>();
    const auto view_type = image.GetViewType(image_res.is_array);

    IR::Inst* body1 = inst.Arg(2).InstRecursive();
    IR::Inst* body2 = inst.Arg(3).InstRecursive();
    IR::Inst* body3 = inst.Arg(4).InstRecursive();
    IR::F32 body4 = IR::F32{inst.Arg(5)};
    const auto get_addr_reg = [&](u32 index) -> IR::F32 {
        if (index <= 3) {
            return IR::F32{body1->Arg(index)};
        }
        if (index >= 4 && index <= 7) {
            return IR::F32{body2->Arg(index - 4)};
        }
        if (index >= 8 && index <= 11) {
            return IR::F32{body3->Arg(index - 8)};
        }
        if (index == 12) {
            return body4;
        }
        UNREACHABLE();
    };
    u32 addr_reg = 0;

    // Load first address components as denoted in 8.2.4 VGPR Usage Sea Islands Series Instruction
    // Set Architecture
    const IR::Value offset = [&] -> IR::Value {
        if (!inst_info.has_offset) {
            return IR::U32{};
        }

        // The offsets are six-bit signed integers: X=[5:0], Y=[13:8], and Z=[21:16].
        IR::Value arg = get_addr_reg(addr_reg++);
        if (const IR::Inst* offset_inst = arg.TryInstRecursive()) {
            ASSERT(offset_inst->GetOpcode() == IR::Opcode::BitCastF32U32);
            arg = offset_inst->Arg(0);
        }

        const auto read = [&](u32 off) -> IR::U32 {
            if (arg.IsImmediate()) {
                const u32 imm =
                    arg.Type() == IR::Type::F32 ? std::bit_cast<u32>(arg.F32()) : arg.U32();
                const u16 comp = (imm >> off) & 0x3F;
                return ir.Imm32(s32(comp << 26) >> 26);
            }
            return ir.BitFieldExtract(IR::U32{arg}, ir.Imm32(off), ir.Imm32(6), true);
        };

        switch (view_type) {
        case AmdGpu::ImageType::Color1D:
        case AmdGpu::ImageType::Color1DArray:
            return read(0);
        case AmdGpu::ImageType::Color2D:
        case AmdGpu::ImageType::Color2DMsaa:
        case AmdGpu::ImageType::Color2DArray:
            return ir.CompositeConstruct(read(0), read(8));
        case AmdGpu::ImageType::Color3D:
            return ir.CompositeConstruct(read(0), read(8), read(16));
        default:
            UNREACHABLE();
        }
    }();
    const IR::F32 bias = inst_info.has_bias ? get_addr_reg(addr_reg++) : IR::F32{};
    const IR::F32 dref = inst_info.is_depth ? get_addr_reg(addr_reg++) : IR::F32{};
    const auto [derivatives_dx, derivatives_dy] = [&] -> std::pair<IR::Value, IR::Value> {
        if (!inst_info.has_derivatives) {
            return {};
        }
        switch (view_type) {
        case AmdGpu::ImageType::Color1D:
        case AmdGpu::ImageType::Color1DArray:
            // du/dx, du/dy
            addr_reg = addr_reg + 2;
            return {get_addr_reg(addr_reg - 2), get_addr_reg(addr_reg - 1)};
        case AmdGpu::ImageType::Color2D:
        case AmdGpu::ImageType::Color2DMsaa:
        case AmdGpu::ImageType::Color2DArray:
            // (du/dx, dv/dx), (du/dy, dv/dy)
            addr_reg = addr_reg + 4;
            return {ir.CompositeConstruct(get_addr_reg(addr_reg - 4), get_addr_reg(addr_reg - 3)),
                    ir.CompositeConstruct(get_addr_reg(addr_reg - 2), get_addr_reg(addr_reg - 1))};
        case AmdGpu::ImageType::Color3D:
            // (du/dx, dv/dx, dw/dx), (du/dy, dv/dy, dw/dy)
            addr_reg = addr_reg + 6;
            return {ir.CompositeConstruct(get_addr_reg(addr_reg - 6), get_addr_reg(addr_reg - 5),
                                          get_addr_reg(addr_reg - 4)),
                    ir.CompositeConstruct(get_addr_reg(addr_reg - 3), get_addr_reg(addr_reg - 2),
                                          get_addr_reg(addr_reg - 1))};
        default:
            UNREACHABLE();
        }
    }();

    const bool is_msaa = view_type == AmdGpu::ImageType::Color2DMsaa ||
                         view_type == AmdGpu::ImageType::Color2DMsaaArray;
    const bool unnormalized = sampler.force_unnormalized || inst_info.is_unnormalized;
    const bool needs_dimentions = (!is_msaa && unnormalized) || (is_msaa && !unnormalized);
    const auto dimensions =
        needs_dimentions ? ir.ImageQueryDimension(handle, ir.Imm32(0u), ir.Imm1(false), inst_info)
                         : IR::Value{};
    const auto get_coord = [&](u32 coord_idx, u32 dim_idx) -> IR::Value {
        const auto coord = get_addr_reg(coord_idx);
        if (is_msaa) {
            // For MSAA images preserve the unnormalized coord or manually unnormalize it
            if (unnormalized) {
                return ir.ConvertFToU(32, coord);
            } else {
                const auto dim =
                    ir.ConvertUToF(32, 32, IR::U32{ir.CompositeExtract(dimensions, dim_idx)});
                return ir.ConvertFToU(32, ir.FPMul(coord, dim));
            }
        }
        if (unnormalized) {
            // Normalize the coordinate for sampling, dividing by its corresponding dimension.
            const auto dim =
                ir.ConvertUToF(32, 32, IR::U32{ir.CompositeExtract(dimensions, dim_idx)});
            return ir.FPDiv(coord, dim);
        }
        return coord;
    };

    // Now we can load body components as noted in Table 8.9 Image Opcodes with Sampler
    const IR::Value coords = [&] -> IR::Value {
        switch (view_type) {
        case AmdGpu::ImageType::Color1D: // x
            addr_reg = addr_reg + 1;
            return get_coord(addr_reg - 1, 0);
        case AmdGpu::ImageType::Color1DArray: // x, slice
        case AmdGpu::ImageType::Color2D:      // x, y
        case AmdGpu::ImageType::Color2DMsaa:  // x, y
            addr_reg = addr_reg + 2;
            return ir.CompositeConstruct(get_coord(addr_reg - 2, 0), get_coord(addr_reg - 1, 1));
        case AmdGpu::ImageType::Color2DArray: // x, y, slice
            addr_reg = addr_reg + 3;
            // Note we can use FixCubeCoords with fallthrough cases since it checks for image type.
            return FixCubeCoords(ir, image, get_coord(addr_reg - 3, 0), get_coord(addr_reg - 2, 1),
                                 get_addr_reg(addr_reg - 1));
        case AmdGpu::ImageType::Color3D: // x, y, z
            addr_reg = addr_reg + 3;
            return ir.CompositeConstruct(get_coord(addr_reg - 3, 0), get_coord(addr_reg - 2, 1),
                                         get_coord(addr_reg - 1, 2));
        default:
            UNREACHABLE();
        }
    }();

    ASSERT(!inst_info.has_lod || !inst_info.has_lod_clamp);
    const bool explicit_lod = inst_info.has_lod || inst_info.force_level0;
    const IR::F32 lod = inst_info.has_lod        ? get_addr_reg(addr_reg++)
                        : inst_info.force_level0 ? ir.Imm32(0.0f)
                                                 : IR::F32{};
    const IR::F32 lod_clamp = inst_info.has_lod_clamp ? get_addr_reg(addr_reg++) : IR::F32{};

    auto texel = [&] -> IR::Value {
        if (is_msaa) {
            return ir.ImageRead(handle, coords, {}, ir.Imm32(0U), inst_info);
        }
        if (inst_info.is_gather) {
            if (inst_info.is_depth) {
                return ir.ImageGatherDref(handle, coords, offset, dref, inst_info);
            }
            return ir.ImageGather(handle, coords, offset, inst_info);
        }
        if (inst_info.has_derivatives) {
            return ir.ImageGradient(handle, coords, derivatives_dx, derivatives_dy, offset,
                                    lod_clamp, inst_info);
        }
        if (inst_info.is_depth) {
            if (explicit_lod) {
                return ir.ImageSampleDrefExplicitLod(handle, coords, dref, lod, offset, inst_info);
            }
            return ir.ImageSampleDrefImplicitLod(handle, coords, dref, bias, offset, inst_info);
        }
        if (explicit_lod) {
            return ir.ImageSampleExplicitLod(handle, coords, lod, offset, inst_info);
        }
        return ir.ImageSampleImplicitLod(handle, coords, bias, offset, inst_info);
    }();

    auto converted = ApplyReadNumberConversionVec4(ir, texel, image.GetNumberConversion());
    if (sampler.force_degamma && image.GetNumberFmt() != AmdGpu::NumberFormat::Srgb) {
        converted = ApplyForceDegamma(ir, texel);
    }
    inst.ReplaceUsesWith(converted);
}

void PatchImageArgs(IR::Block& block, IR::Inst& inst, Info& info) {
    // Nothing to patch for dimension query.
    if (inst.GetOpcode() == IR::Opcode::ImageQueryDimensions) {
        return;
    }

    const auto image_handle = inst.Arg(0);
    const auto binding_index = image_handle.U32() & 0xFFFF;
    const auto& image_res = info.images[binding_index];
    auto image = image_res.GetSharp(info);

    // Sample instructions must be handled separately using address register data.
    if (inst.GetOpcode() == IR::Opcode::ImageSampleRaw) {
        return PatchImageSampleArgs(block, inst, info, image_res, image);
    }

    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    auto inst_info = inst.Flags<IR::TextureInstInfo>();
    const auto view_type = image.GetViewType(image_res.is_array);

    // Now that we know the image type, adjust texture coordinate vector.
    IR::Inst* body = inst.Arg(1).InstRecursive();
    const auto [coords, arg] = [&] -> std::pair<IR::Value, IR::Value> {
        switch (view_type) {
        case AmdGpu::ImageType::Color1D: // x, [lod]
            return {body->Arg(0), body->Arg(1)};
        case AmdGpu::ImageType::Color1DArray: // x, slice, [lod]
        case AmdGpu::ImageType::Color2D:      // x, y, [lod]
            return {ir.CompositeConstruct(body->Arg(0), body->Arg(1)), body->Arg(2)};
        case AmdGpu::ImageType::Color2DMsaa: // x, y. (sample is passed on different argument)
        {
            auto skip_lod = false;
            if (inst_info.has_lod) {
                const auto mipid = body->Arg(2);
                if (mipid.IsImmediate() && mipid.U32() == 0) {
                    // if image_x_mip refers to a MSAA image, and mipid is 0, it is safe to be
                    // skipped and fragid is taken from the next arg
                    LOG_WARNING(Render_Recompiler, "Encountered a _mip instruction with MSAA "
                                                   "image, and mipid is 0, skipping LoD");
                    inst_info.has_lod.Assign(false);
                    skip_lod = true;
                } else {
                    UNREACHABLE_MSG(
                        "Encountered a _mip instruction with MSAA image, and mipid is non-zero");
                }
            }
            return {ir.CompositeConstruct(body->Arg(0), body->Arg(1)), body->Arg(skip_lod ? 3 : 2)};
        }
        case AmdGpu::ImageType::Color2DArray:     // x, y, slice, [lod]
        case AmdGpu::ImageType::Color2DMsaaArray: // x, y, slice. (sample is passed on different
                                                  // argument)
        case AmdGpu::ImageType::Color3D:          // x, y, z, [lod]
            return {ir.CompositeConstruct(body->Arg(0), body->Arg(1), body->Arg(2)), body->Arg(3)};
        default:
            UNREACHABLE_MSG("Unknown image type {}", view_type);
        }
    }();

    const auto has_ms = view_type == AmdGpu::ImageType::Color2DMsaa ||
                        view_type == AmdGpu::ImageType::Color2DMsaaArray;
    ASSERT(!inst_info.has_lod || !has_ms);
    // If we are binding a single mip level as fallback, drop the argument
    const auto lod =
        (inst_info.has_lod && image_res.mip_fallback_mode != MipStorageFallbackMode::ConstantIndex)
            ? IR::U32{arg}
            : IR::U32{};
    const auto ms = has_ms ? IR::U32{arg} : IR::U32{};

    const auto is_storage = image_res.is_written;
    if (inst.GetOpcode() == IR::Opcode::ImageRead) {
        auto texel = ir.ImageRead(image_handle, coords, lod, ms, inst_info);
        if (is_storage) {
            // Storage image requires shader swizzle.
            texel = ApplySwizzle(ir, texel, image.DstSelect());
        }
        const auto converted =
            ApplyReadNumberConversionVec4(ir, texel, image.GetNumberConversion());
        inst.ReplaceUsesWith(converted);
    } else {
        inst.SetArg(1, coords);
        if (inst.GetOpcode() == IR::Opcode::ImageWrite) {
            inst.SetArg(2, lod);
            inst.SetArg(3, ms);

            auto texel = inst.Arg(4);
            if (is_storage) {
                // Storage image requires shader swizzle.
                texel = ApplySwizzle(ir, texel, image.DstSelect().Inverse());
            }
            const auto converted =
                ApplyWriteNumberConversionVec4(ir, texel, image.GetNumberConversion());
            inst.SetArg(4, converted);
        }
    }
}

void ResourceTrackingPass(IR::Program& program, const Profile& profile) {
    // Iterate resource instructions and patch them after finding the sharp.
    auto& info = program.info;

    // Pass 0: a compute shader touching memory through the "all of memory"
    // inline V# (base 0, num_records max) is a job-system worker. Mark it so
    // Pass 1 routes ALL its eligible buffer accesses (loads and stores,
    // bounded or not) through the BDA page table: the job pages must have a
    // single coherent home shared with the CPU.
    // Live-DMA mode rides on the UMA host imports; without them the diverted
    // accesses would lose the binding-forced buffer creation they rely on.
    static const bool uma_env = [] {
        const char* e = std::getenv("SHADPS4_UMA");
        return e && e[0] == '1';
    }();
    // [Stage2 SPINGUARD] SHADPS4_SPINGUARD=1: mark loop blocks so in-loop BDA reads emit through
    // read_const_live (a Volatile data load). Without it the SASS compiler hoists a seeded-page
    // poll out of the spin loop as loop-invariant and the wave never re-reads memory. Runs for
    // UMA or SPINGUARD, and for ANY stage (a spin is possible in graphics waves too - not only
    // Compute). The uses_live_dma store routing below stays STRICTLY under uma_env.
    // 0 = off, 1 = coarse (every loop block), 2 = targeted (polling loops only).
    static const u32 spinguard_mode = [] {
        const char* e = std::getenv("SHADPS4_SPINGUARD");
        return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    static const bool spinguard_env = spinguard_mode == 1;
    // [SPINGUARD2] Targeted volatility. Mode 1 makes EVERY in-loop BDA read volatile, which
    // collapses PCIe throughput on shaders that merely stream memory in a loop. A poll is
    // structurally different from streaming: its body performs NO stores (it produces nothing;
    // it only observes) and reads a handful of addresses whose value is the sole reason the loop
    // can ever end. Loops matching that shape get volatile loads; every other loop keeps
    // cacheable ones.
    // [POLLJ] The same walk also collects poll_blocks (store-free, 1..8 loads) while the
    // presence trace is on; loop_blocks marking stays exactly under the modes above.
    const bool census_marks = spinguard_mode == 2 || spinguard_mode == 3 || LiveStoreEnabled();
    const bool poll_census = PollTraceEnabled();
    if ((census_marks || poll_census) && EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        // Blocks of each open loop, innermost last. A block is attributed to every enclosing
        // loop so an outer poll containing an inner loop is still recognised as one region.
        boost::container::small_vector<boost::container::small_vector<IR::Block*, 8>, 4> open;
        u32 marked_loops = 0, marked_loads = 0;
        // Census, so a miss reports WHY instead of leaving silence to be interpreted.
        u32 loops_seen = 0, stores_seen = 0, max_loads = 0;
        const auto finish_loop = [&] {
            if (open.empty()) {
                return;
            }
            auto body = std::move(open.back());
            open.pop_back();
            u32 loads = 0;
            bool has_store = false;
            for (IR::Block* const block : body) {
                for (const IR::Inst& inst : block->Instructions()) {
                    if (IsBufferStore(inst)) {
                        has_store = true;
                        break;
                    }
                    if (IsBufferInstruction(inst)) {
                        ++loads;
                    }
                }
                if (has_store) {
                    break;
                }
            }
            ++loops_seen;
            stores_seen += has_store ? 1 : 0;
            max_loads = std::max(max_loads, loads);
            // No stores at all, and a small number of observed addresses: this loop cannot
            // terminate on its own work, only on what it reads. That is a poll.
            // Mode 3 keeps the "no stores" rule but drops the load cap - a poll that reads a
            // whole descriptor's worth of state is still a poll.
            const bool load_ok =
                (spinguard_mode >= 3 || LiveStoreEnabled()) ? loads > 0 : (loads > 0 && loads <= 8);
            if (census_marks && !has_store && load_ok) {
                for (IR::Block* const block : body) {
                    info.loop_blocks.insert(block);
                }
                ++marked_loops;
                marked_loads += loads;
            }
            // [POLLJ] the strict poll shape, independent of the mode's load cap: LIVELOOP=2
            // drops the cap so data loops become live too, but only the polls get traced.
            if (poll_census && !has_store && loads > 0 && loads <= 8) {
                for (IR::Block* const block : body) {
                    info.poll_blocks.insert(block);
                }
            } else if (poll_census && !has_store && loads > 8) {
                // Live under the mode-3 shape (no cap) but NOT traced: say so, or a page missing
                // from the poll set cannot be told apart from a page nobody polls.
                LOG_CRITICAL(Render_Recompiler,
                             "POLLJ: shader {:#x} store-free loop with {} loads is above the "
                             "8-load poll cap - not traced",
                             info.pgm_hash, loads);
            }
        };
        for (const auto& node : program.syntax_list) {
            switch (node.type) {
            case IR::AbstractSyntaxNode::Type::Loop:
                open.emplace_back();
                break;
            case IR::AbstractSyntaxNode::Type::Repeat:
                finish_loop();
                break;
            case IR::AbstractSyntaxNode::Type::Block:
                for (auto& level : open) {
                    level.push_back(node.data.block);
                }
                break;
            default:
                break;
            }
        }
        while (!open.empty()) {
            finish_loop(); // unbalanced Repeat markers: evaluate what we collected anyway
        }
        if (!census_marks) {
            // [POLLJ]-only census: nothing was marked, so no SPINGUARD2 report either
        } else if (marked_loops != 0) {
            LOG_CRITICAL(Render_Recompiler, "SPINGUARD2: shader={:#x} marked {} loads in {} loops",
                         info.pgm_hash, marked_loads, marked_loops);
        } else if (loops_seen != 0) {
            // A shader with loops that none of them qualified for: name the reason so the
            // criterion can be tightened or loosened from data.
            static std::atomic<u32> miss_logged{0};
            if (miss_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_CRITICAL(Render_Recompiler,
                             "SPINGUARD2: shader={:#x} MISS loops={} with_stores={} max_loads={}",
                             info.pgm_hash, loops_seen, stores_seen, max_loads);
            }
        } else {
            static std::atomic<u32> noloop_logged{0};
            if (noloop_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_CRITICAL(Render_Recompiler, "SPINGUARD2: shader={:#x} NO LOOPS in IR",
                             info.pgm_hash);
            }
        }
    }
    if ((uma_env || spinguard_env || (LiveLoopEnabled() && !LiveStoreEnabled())) &&
        EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        // Blocks inside loops: live reads there are polls and stay volatile;
        // everything else keeps cacheable loads.
        u32 loop_depth = 0;
        for (const auto& node : program.syntax_list) {
            switch (node.type) {
            case IR::AbstractSyntaxNode::Type::Loop:
                ++loop_depth;
                break;
            case IR::AbstractSyntaxNode::Type::Repeat:
                if (loop_depth > 0) {
                    --loop_depth;
                }
                break;
            case IR::AbstractSyntaxNode::Type::Block:
                if (loop_depth > 0) {
                    info.loop_blocks.insert(node.data.block);
                }
                break;
            default:
                break;
            }
        }
    }
    // [LIVELOOP=2] A compute shader that loops over buffer loads is a job worker (0x8f067540 and
    // friends): make ALL its accesses live (loads and stores through the BDA page table) so its
    // pages have a single home in guest memory. Loop-free compute (the 2.4 MB fill 0xe8746089)
    // keeps VRAM bindings. The inline-V# detection below did not catch these workers (their
    // unbounded sharp comes from user data, not an immediate).
    // Under LIVELOOP=2 loop_blocks come from the poll census (loops with loads and NO stores),
    // so a screen-space shader that loops with stores (0x43b8ee5e: 9.8M logged writes/5 s) is
    // left alone.
    if (LiveStoreEnabled() && EmulatorSettings.IsDirectMemoryAccessEnabled() &&
        info.l_stage == LogicalStage::Compute && !info.loop_blocks.empty()) {
        bool loads_in_loop = false;
        for (IR::Block* const block : program.blocks) {
            if (!info.loop_blocks.contains(block)) {
                continue;
            }
            for (const IR::Inst& inst : block->Instructions()) {
                if (IsBufferInstruction(inst) && !IsBufferStore(inst)) {
                    loads_in_loop = true;
                    break;
                }
            }
            if (loads_in_loop) {
                break;
            }
        }
        const auto& allow = LiveWorkerList();
        const bool listed = std::find(allow.begin(), allow.end(), info.pgm_hash) != allow.end();
        if (loads_in_loop && (allow.empty() || listed)) {
            info.uses_live_dma = true;
            LOG_CRITICAL(Render_Recompiler, "LIVELOOP2: shader {:#x} -> live DMA worker (loop loads{})",
                         info.pgm_hash, listed ? ", listed" : "");
        }
    }
    // Live-DMA store routing (uses_live_dma) is UMA-only and Compute-only, unchanged.
    if ((uma_env || LiveStoreEnabled()) && EmulatorSettings.IsDirectMemoryAccessEnabled() &&
        info.l_stage == LogicalStage::Compute) {
        for (IR::Block* const block : program.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                if (!IsBufferInstruction(inst)) {
                    continue;
                }
                IR::Inst* handle = inst.Arg(0).InstRecursive();
                if (!handle->AreAllArgsImmediates()) {
                    continue;
                }
                std::array<u64, 2> raw;
                raw[0] = handle->Arg(0).U32() | u64(handle->Arg(1).U32()) << 32;
                raw[1] = handle->Arg(2).U32() | u64(handle->Arg(3).U32()) << 32;
                const auto pre_buffer = std::bit_cast<AmdGpu::Buffer>(raw);
                if (pre_buffer.base_address == 0 &&
                    pre_buffer.num_records == std::numeric_limits<u32>::max()) {
                    info.uses_live_dma = true;
                    LOG_INFO(Render_Recompiler,
                             "Shader {:#x}: live DMA mode (job-system worker)", info.pgm_hash);
                    break;
                }
            }
            if (info.uses_live_dma) {
                break;
            }
        }
    }

    // Pass 1: Track resource sharps
    Descriptors descriptors{info};
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (IsBufferInstruction(inst)) {
                PatchBufferSharp(*block, inst, info, descriptors, profile);
            } else if (IsImageInstruction(inst)) {
                PatchImageSharp(*block, inst, info, descriptors, profile);
            }
        }
    }

    // Pass 2: Patch instruction args
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (IsBufferInstruction(inst)) {
                PatchBufferArgs(*block, inst, info);
            } else if (IsImageInstruction(inst)) {
                PatchImageArgs(*block, inst, info);
            } else if (IsDataRingInstruction(inst)) {
                PatchGlobalDataShareAccess(*block, inst, info, descriptors, profile);
            }
        }
    }
}

} // namespace Shader::Optimization
