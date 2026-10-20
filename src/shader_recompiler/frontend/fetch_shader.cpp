// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {

/**
 * s_load_dwordx4 s[8:11], s[2:3], 0x00
 * s_load_dwordx4 s[12:15], s[2:3], 0x04
 * s_load_dwordx4 s[16:19], s[2:3], 0x08
 * s_waitcnt     lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
 * buffer_load_format_xyz v[8:10], v0, s[12:15], 0 idxen
 * buffer_load_format_xy v[12:13], v0, s[16:19], 0 idxen
 * s_waitcnt     0
 * s_setpc_b64   s[0:1]

 * s_load_dwordx4  s[4:7], s[2:3], 0x0
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen
 * s_load_dwordx4  s[4:7], s[2:3], 0x8
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[8:11], v0, s[4:7], 0 idxen
 * s_waitcnt       vmcnt(0) & expcnt(0) & lgkmcnt(0)
 * s_setpc_b64     s[0:1]

 * A normal fetch shader looks like the above, the instructions are generated
 * using input semantics on cpu side. Load instructions can either be separate or interleaved
 * We take the reverse way, extract the original input semantics from these instructions.
 **/

static bool IsTypedBufferLoad(const Gcn::GcnInst& inst) {
    return inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_X ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XY ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZ ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZW;
}

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base) {
    const u32* code;
    std::memcpy(&code, &info.user_data[sgpr_base], sizeof(code));
    return code;
}

// [FETCHMEMO 10.09] SHADPS4_FETCHMEMO=1: remember the last few parsed fetch shaders.
// ParseFetchShader runs a full GCN decode (plus a 104-entry scratch array) and is called from
// StageSpecialization's constructor - that is once per stage per DRAW, and the jungle scene
// issues 1877 draws a frame from 143 pipelines, so the same handful of fetch shaders is decoded
// thousands of times per frame. The key is (code pointer, first four code dwords): the game
// generates this code on the CPU at run time, so a pointer alone could alias a rewritten buffer;
// the four dwords make an alias detectable at the cost of four compares. Off by default.
namespace {
struct FetchMemoEntry {
    const u32* code{};
    std::array<u32, 4> head{};
    std::optional<FetchShaderData> data{};
    bool valid{};
};
constexpr size_t kFetchMemoSize = 16;
thread_local std::array<FetchMemoEntry, kFetchMemoSize> g_fetch_memo{};
thread_local size_t g_fetch_memo_next = 0;
std::atomic<u64> g_fetch_memo_hits{};
std::atomic<u64> g_fetch_memo_misses{};
bool FetchMemoEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_FETCHMEMO");
        const bool v = e != nullptr && e[0] == '1';
        if (v) {
            LOG_CRITICAL(Render_Recompiler, "FETCHMEMO: on (last {} fetch shaders per thread)",
                         kFetchMemoSize);
        }
        return v;
    }();
    return on;
}
} // namespace

u64 FetchMemoHits() {
    return g_fetch_memo_hits.load(std::memory_order_relaxed);
}
u64 FetchMemoMisses() {
    return g_fetch_memo_misses.load(std::memory_order_relaxed);
}

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info) {
    if (!info.has_fetch_shader) {
        return std::nullopt;
    }

    const auto* code = GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    const bool memo = FetchMemoEnabled();
    std::array<u32, 4> head{};
    if (memo) {
        std::memcpy(head.data(), code, sizeof(head));
        for (const auto& e : g_fetch_memo) {
            if (e.valid && e.code == code && e.head == head) {
                g_fetch_memo_hits.fetch_add(1, std::memory_order_relaxed);
                return e.data;
            }
        }
        g_fetch_memo_misses.fetch_add(1, std::memory_order_relaxed);
    }
    FetchShaderData data{};
    GcnCodeSlice code_slice(code, code + std::numeric_limits<u32>::max());
    GcnDecodeContext decoder;

    struct VsharpLoad {
        u32 dword_offset{};
        u32 base_sgpr{};
    };
    std::array<VsharpLoad, 104> loads{};

    u32 semantic_index = 0;
    while (!code_slice.atEnd()) {
        const auto inst = decoder.decodeInstruction(code_slice);
        data.size += inst.length;

        if (inst.opcode == Opcode::S_SETPC_B64) {
            break;
        }

        if (inst.inst_class == InstClass::ScalarMemRd) {
            loads[inst.dst[0].code] = VsharpLoad{inst.control.smrd.offset, inst.src[0].code * 2};
            continue;
        }

        if (inst.opcode == Opcode::V_ADD_I32) {
            const auto vgpr = inst.dst[0].code;
            const auto sgpr = s8(inst.src[0].code);
            switch (vgpr) {
            case 0: // V0 is always the vertex offset
                data.vertex_offset_sgpr = sgpr;
                break;
            case 3: // V3 is always the instance offset
                data.instance_offset_sgpr = sgpr;
                break;
            default:
                UNREACHABLE();
            }
        }

        if (inst.inst_class == InstClass::VectorMemBufFmt) {
            // SRSRC is in units of 4 SPGRs while SBASE is in pairs of SGPRs
            const u32 base_sgpr = inst.src[2].code * 4;
            auto& attrib = data.attributes.emplace_back();
            attrib.semantic = semantic_index++;
            attrib.dest_vgpr = inst.src[1].code;
            attrib.num_elements = inst.control.mubuf.count;
            attrib.sgpr_base = loads[base_sgpr].base_sgpr;
            attrib.dword_offset = loads[base_sgpr].dword_offset;
            attrib.inst_offset = inst.control.mtbuf.offset;
            attrib.instance_data = inst.src[0].code;
            if (IsTypedBufferLoad(inst)) {
                attrib.data_format = inst.control.mtbuf.dfmt;
                attrib.num_format = inst.control.mtbuf.nfmt;
            }
        }
    }

    if (memo) {
        auto& e = g_fetch_memo[g_fetch_memo_next];
        g_fetch_memo_next = (g_fetch_memo_next + 1) % kFetchMemoSize;
        e.code = code;
        e.head = head;
        e.data = data;
        e.valid = true;
    }
    return data;
}

} // namespace Shader::Gcn
