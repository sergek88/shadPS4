// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(scope)),
                        ctx.ConstU32(static_cast<u32>(semantics)));
}
} // Anonymous namespace

// [BARRIERDEV] SHADPS4_BARRIER_DEVICE=1: s_barrier on GCN is preceded by s_waitcnt vmcnt(0), so
// every lane's global stores are complete when the wave leaves the barrier; a later store by
// one lane (a job's "done" word) can never overtake another lane's data. The default
// Workgroup/WorkgroupMemory barrier orders only LDS; give it device scope + uniform memory.
static bool BarrierDeviceEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BARRIER_DEVICE");
        return e && e[0] == '1';
    }();
    return on;
}

void EmitBarrier(EmitContext& ctx) {
    const auto execution{spv::Scope::Workgroup};
    spv::Scope memory;
    spv::MemorySemanticsMask memory_semantics;
    if (ctx.l_stage == Shader::LogicalStage::TessellationControl) {
        memory = spv::Scope::Invocation;
        memory_semantics = spv::MemorySemanticsMask::MaskNone;
    } else if (BarrierDeviceEnabled()) {
        memory = spv::Scope::Device;
        memory_semantics = spv::MemorySemanticsMask::AcquireRelease |
                           spv::MemorySemanticsMask::WorkgroupMemory |
                           spv::MemorySemanticsMask::UniformMemory;
    } else {
        memory = spv::Scope::Workgroup;
        memory_semantics =
            spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::WorkgroupMemory;
    }
    ctx.OpControlBarrier(ctx.ConstU32(static_cast<u32>(execution)),
                         ctx.ConstU32(static_cast<u32>(memory)),
                         ctx.ConstU32(static_cast<u32>(memory_semantics)));
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Device);
}

} // namespace Shader::Backend::SPIRV
