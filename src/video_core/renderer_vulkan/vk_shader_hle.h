// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_set.hpp>
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace AmdGpu {
struct ComputeProgram;
union Regs;
} // namespace AmdGpu

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Rasterizer;
class Scheduler;

/// Attempts to execute a shader using HLE if possible.
bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer);

// [CSBATCH] SHADPS4_CSBATCH=<hex hash>[,...]: the ND skinning worker (0x8f067540) is not
// dispatched through its recompiled module at all. Every guest dispatch becomes an 88-byte job
// record (raw device addresses of its 7 ranges, resolved through the buffer cache at append
// time, in guest order) plus ceil(n/WG) entries of a group map; one host compute dispatch per
// batch (Rasterizer::FlushSkinBatch) computes every job with a hand-written kernel whose math
// mirrors the guest module operation by operation. The batch is recorded before any other GPU
// command (draw, other dispatch, DMA, label, submit), so the guest never observes the difference.
// Default OFF: without the env the only effects are one empty test per hook and one branch per
// submit. Everything below is GPU-thread state unless said otherwise.
struct alignas(8) SkinJobGpu {
    u64 out_pos, out_nrm, out_tan, hdr, infl, pal, pos, nrm, tan;
    u32 count, infl_dw, pal_dw, pad;
};
static_assert(sizeof(SkinJobGpu) == 88);

struct SkinTableHdr {
    u32 n_jobs, n_groups, pad0, pad1;
};
static_assert(sizeof(SkinTableHdr) == 16);

class SkinBatch {
public:
    static constexpr u64 kHash = 0x8f067540ull;
    static constexpr u64 RingBytes = 8_MB;
    // Group map entry: (job << JobShift) | group_in_job. MaxJobs() is clamped below 1 << 12.
    static constexpr u32 JobShift = 20;

    enum class Reason : u8 {
        Draw,
        DrawIndirect,
        Dispatch,
        DispatchIndirect,
        Dma,
        Sync,
        Submit,
        Hook,
        Chunk,
        Fault,
        Readback,
        Label,
        Cmd,
        Full,
        Every,
        Fallback,
        Create,
        Chained,
        Reupload,
        Shadow,
        N
    };
    enum class Decline : u8 {
        Dims,
        Unmapped,
        Unbounded,
        Oversize,
        Layout,
        ShortSrc,
        Empty,
        Align,
        NoBda,
        PtDiverge,
        Unstable,
        N
    };
    static const char* ReasonName(Reason r);
    static const char* DeclineName(Decline d);

    // env (parsed once; the hash list uses the SKIPCS grammar)
    static bool Wanted(u64 h);         // SHADPS4_CSBATCH=<hex>[,<hex>]
    static const std::string& Hashes(); // the list as given, for the armed line
    static u32 Wg();                    // SHADPS4_CSBATCH_WG (64)
    static u32 MaxJobs();               // SHADPS4_CSBATCH_MAX (1024)
    static u32 MaxGroups();
    /// [CSBATCH-DEFERLABEL] SHADPS4_CSBATCH_MAXLABELS (default 64): how many guest completion
    /// labels may wait behind one batch before it is recorded anyway.
    static u32 MaxHeldLabels();             // SHADPS4_CSBATCH_MAXGROUPS (65536); see max_groups
    static u32 FlushEvery();            // SHADPS4_CSBATCH_EVERY (0)
    static bool Robust();               // SHADPS4_CSBATCH_ROBUST (1)
    static bool PalVec4();              // SHADPS4_CSBATCH_PAL_VEC4 (1)
    static bool PreBar();               // SHADPS4_CSBATCH_PREBAR (1)
    static bool PostBar();              // SHADPS4_CSBATCH_POSTBAR (0)
    static bool ImgInv();               // SHADPS4_CSBATCH_IMGINV (1)
    static bool ZoneSync();             // SHADPS4_CSBATCH_ZONESYNC (1)
    static bool CountsForChunk();       // SHADPS4_CSBATCH_CHUNK (1): batch = 1 dispatch for CHUNK_DRAWS
    static u32 Shadow();                // SHADPS4_CSBATCH_SHADOW=N frames (0)
    static bool Log();                  // SHADPS4_CSBATCH_LOG (0)
    static bool TableVram();            // SHADPS4_CSBATCH_TABLE_VRAM (0)

    /// Compiles the kernel, creates the pipeline and the private table ring. Logs
    /// "[CSBATCH] armed ..." or "[CSBATCH] compile FAILED ... - feature off".
    void Create(const Instance& instance, Scheduler& scheduler, bool stream_bda);

    bool Ready() const {
        return bool(pipeline);
    }

    // accumulation state, GPU thread only
    std::vector<SkinJobGpu> jobs;
    std::vector<u32> group_map;
    boost::container::small_vector<vk::BufferMemoryBarrier2, 32> barriers;
    // Cache owners of the outputs of the pending jobs: ids, not pointers (SlotVector reallocates
    // on insert). A deleted owner keeps its slot until its tick retires, i.e. past our flush.
    boost::container::small_vector<Common::SlotId, 8> out_owners;
    boost::icl::interval_set<VAddr> pending_out; // outputs of jobs in THIS batch (chained detector)
    boost::icl::interval_set<VAddr> pending_in;  // inputs of jobs in THIS batch (WAR, reupload)
    u64 verts = 0;
    u64 first_tick = 0;
    bool flushing = false;
    std::thread::id gpu_thread{};
    u32 max_groups = 0; // MaxGroups() clamped to maxComputeWorkGroupCount[0] (Create)
    std::unique_ptr<VideoCore::StreamBuffer> ring; // 8 MiB, MemoryUsage::Stream, never waits
    std::unique_ptr<VideoCore::Buffer> vram_table;  // SHADPS4_CSBATCH_TABLE_VRAM=1: staging copy
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;

    // [SHADOW] one job per N-th frame is computed twice: by the kernel into a scratch area of the
    // ring and by the guest permutation into the real output; compared after the tick retires.
    struct ShadowState {
        bool armed = false;
        bool registered = false; // the compare is deferred (the guest dispatch was recorded)
        bool clobbered = false;  // a later job wrote over the guest output before the compare
        u64 frame = 0;
        u64 tick = 0;
        u64 wrap = 0;
        u64 off = 0;
        VAddr out_base = 0;
        u64 out_size = 0;
        u32 n = 0;
    } shadow;

    struct Stats {
        u64 batches, subs, verts, groups, barriers, max_subs, count0, null_out, tick_splits,
            straddle, deleted_owner, ring_wait, chained, reupload, unstable, orphan_bars;
        std::atomic<u64> hook_offthread{0};
        u64 flush[size_t(Reason::N)];
        u64 decl[size_t(Decline::N)];
        bool decl_logged[size_t(Decline::N)];
        u64 shadow_ok, shadow_bad, shadow_skip;
        void ResetWindow();
    } st{};

    void Reset() {
        jobs.clear();
        group_map.clear();
        barriers.clear();
        out_owners.clear();
        pending_out.clear();
        pending_in.clear();
        verts = 0;
    }
};

} // namespace Vulkan
