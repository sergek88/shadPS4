// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <boost/preprocessor/stringize.hpp>

#include <cstdlib>
#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/postmortem.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_platform.h" // [R8 ADDRBIND] unbind_shader_region
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifdef _WIN32
// [FRAMECHAIN] Thread id and thread CPU time without pulling <windows.h> into this TU (the
// SetThreadStackGuarantee import further down set the precedent). The prototypes are the exact
// processthreadsapi.h signatures (BOOL = int, HANDLE = void*, DWORD = unsigned long), so they
// remain benign redeclarations should the header ever arrive transitively; FILETIME reuses the
// SDK's own include guard for the same reason.
#ifndef _FILETIME_
#define _FILETIME_
typedef struct _FILETIME {
    unsigned long dwLowDateTime;
    unsigned long dwHighDateTime;
} FILETIME, *PFILETIME, *LPFILETIME;
#endif
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
extern "C"
    __declspec(dllimport) int __stdcall GetThreadTimes(void* thread, _FILETIME* creation_time,
                                                       _FILETIME* exit_time, _FILETIME* kernel_time,
                                                       _FILETIME* user_time);

extern "C" __declspec(dllimport) int __stdcall QueryThreadCycleTime(void* thread,
                                                                    unsigned long long* cycles);
#endif

namespace Vulkan {
extern bool g_cs_sync_pending; // [CSBARRIER] defined in vk_rasterizer.cpp
extern u32 g_cs_sync_mask;     // [CSBARRIER] 1 ACQUIRE_MEM, 2 WAIT_REG_MEM, 4 EVENT_WRITE
} // namespace Vulkan

namespace Vulkan {
extern std::atomic<u32> g_dbg_draws;
extern std::atomic<u32> g_dbg_dispatches;
extern std::atomic<u32> g_dbg_frame_no;
extern std::atomic<u64> g_dbg_upload_bytes;
} // namespace Vulkan

namespace AmdGpu {

std::atomic<u64> g_flips_materialized{};

// [EOPRETIRE] SHADPS4_EOPRETIRE=1: EOP/ReleaseMem labels and IRQs are delivered when the GPU
// retires the tick (priority-ops queue), after the deferred readback writeback queued before
// them. The parser never blocks in Finish for a readback; the guest sees hardware order.
// Levels: 1 = gfx EOP and compute ReleaseMem; 2 = gfx EOP only (ReleaseMem stays at parse:
// run 201 showed a Flush per compute fence costs ~60 ms/frame of parser time in the ASC queues).
static int EopRetireLevel() {
    static const int v = [] {
        const char* e = std::getenv("SHADPS4_EOPRETIRE");
        return e ? static_cast<int>(std::strtol(e, nullptr, 10)) : 0;
    }();
    return v;
}
static bool EopRetire() {
    return EopRetireLevel() != 0;
}

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

namespace {

// [FRAMECHAIN] Host thread id / CPU time (0 where the platform has no cheap answer).
u32 FcCurrentThreadId() {
#ifdef _WIN32
    return static_cast<u32>(GetCurrentThreadId());
#else
    return 0;
#endif
}

// [FRAMECHAIN] Cycles actually executed by this thread. GetThreadTimes below is sampled at the
// scheduler tick (15.6 ms), so on a thread that alternates short work with waits it can report a
// fraction of the truth - 10.09 the same window said gfx_resume=14.2 ms/frame and cpu=2.0 ms/frame
// on ONE thread, which cannot both be right. Cycles are exact; printing both settles whether the
// parser is computing or waiting.
u64 FcThreadCycles() {
#ifdef _WIN32
    unsigned long long c = 0;
    if (QueryThreadCycleTime(GetCurrentThread(), &c) == 0) {
        return 0;
    }
    return c;
#else
    return 0;
#endif
}

// Kernel + user time of the calling thread, 100 ns units (GetThreadTimes).
u64 FcThreadCpu100ns() {
#ifdef _WIN32
    _FILETIME ct{}, et{}, kt{}, ut{};
    if (GetThreadTimes(GetCurrentThread(), &ct, &et, &kt, &ut) == 0) {
        return 0;
    }
    const u64 k = (u64{kt.dwHighDateTime} << 32) | u64{kt.dwLowDateTime};
    const u64 u = (u64{ut.dwHighDateTime} << 32) | u64{ut.dwLowDateTime};
    return k + u;
#else
    return 0;
#endif
}

// [Stage1b PARSERPROF] Per-packet wall time - attributes the EOP gap that no TEXTIME/compile
// bucket covered. RAII: every break path in the switch (and a coroutine suspension inside a
// packet) lands in the same account when the iteration scope finally exits.
// [FRAMECHAIN] Hoisted out of ProcessGraphics so ProcessCompute can use it too. ASC packets
// (`asc`) stay out of everything the GFX forensics rely on - OpRing, the HOSTSTALL-PKT
// publication, the draw/wrm/evw/other buckets, PKTHIST and the SLOWPKT cap - and get their own
// totals (g_pkt_asc_*), phase sums (g_ph_win_asc_us) and slow-line cap instead.
struct PktTimer {
    u32 op;
    s64 t0;
    bool asc;
    bool phased = false;
    PktTimer(u32 op_, s64 t0_, bool asc_ = false) : op{op_}, t0{t0_}, asc{asc_} {
        // [SLOWWAIT p.3] Ring of the last 8 GFX packets for the crash handler,
        // pushed at packet START so the one being processed at death is newest.
        if (t0 != 0 && !asc) {
            Postmortem::OpRingPush(op);
            // [HOSTSTALL-PKT] Publish what we are about to chew on; cleared in the
            // destructor so "in a packet" is unambiguous.
            Parserprof::g_cur_op.store(op, std::memory_order_relaxed);
            Parserprof::g_cur_seq.store(Postmortem::g_opring_seq.load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
            Parserprof::g_cur_pkt_us.store(t0, std::memory_order_release);
        }
        // [SLOWPKT] Arm the exclusive phase clock for draw-class packets and the EOP packet
        // (GFX), or for dispatches (ASC). The probes down in pipeline/texture/buffer cache
        // and the rasterizer only fire while this is armed on this thread.
        // An ASC dispatch never touches a clock a parked GFX packet still owns (no phased GFX
        // packet yields today - YIELD_GFX sits in WaitRegMem/MemSemaphore/Rewind/IndirectBuffer
        // only - so this changes nothing now and keeps the GFX split intact if one ever does).
        if (t0 != 0 && (Parserprof::IsDrawClassOp(op) || (!asc && op == 0x47)) &&
            !(asc && Parserprof::t_ph_active)) {
            phased = true;
            for (u64& v : Parserprof::t_ph_us) {
                v = 0;
            }
            Parserprof::t_pkt_barriers = 0;
            Parserprof::t_pkt_writes = 0;
            Parserprof::t_pkt_binds = 0;
            Parserprof::t_ph_cur = Parserprof::PhOther;
            Parserprof::t_ph_stamp = t0;
            Parserprof::t_ph_active = true;
            Parserprof::t_pkt_asc = asc;
        }
    }
    PktTimer(const PktTimer&) = delete;
    PktTimer& operator=(const PktTimer&) = delete;
    ~PktTimer() {
        if (t0 == 0) {
            return;
        }
        const s64 now = Parserprof::NowUs();
        const u64 total = now > t0 ? static_cast<u64>(now - t0) : 0;
        if (asc) {
            // Wait-class packets include the time this queue spent suspended while other
            // tasks ran; an IndirectBuffer's packets account for themselves. Everything else
            // is CPU work of the ASC path.
            const bool wait_class = op == static_cast<u32>(PM4ItOpcode::WaitRegMem) ||
                                    op == static_cast<u32>(PM4ItOpcode::MemSemaphore) ||
                                    op == static_cast<u32>(PM4ItOpcode::Rewind);
            const bool nested = op == static_cast<u32>(PM4ItOpcode::IndirectBuffer);
            if (wait_class) {
                Parserprof::g_pkt_asc_wait_us.fetch_add(total, std::memory_order_relaxed);
                Parserprof::g_pkt_asc_wait_n.fetch_add(1, std::memory_order_relaxed);
            } else if (!nested) {
                Parserprof::g_pkt_asc_us.fetch_add(total, std::memory_order_relaxed);
                Parserprof::g_pkt_asc_n.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            Parserprof::g_cur_pkt_us.store(0, std::memory_order_release);
            if (now > t0) {
                Parserprof::AccountPacket(op, static_cast<u64>(now - t0));
            }
        }
        if (!phased) {
            return;
        }
        // Flush the tail into the current (top-level = Other) bucket, then read out.
        Parserprof::PhaseSwitch(Parserprof::PhOther);
        Parserprof::t_ph_active = false;
        Parserprof::t_pkt_asc = false;
        const u64* ph = Parserprof::t_ph_us;
        if (asc) {
            for (u32 i = 0; i < Parserprof::PhCount; ++i) {
                Parserprof::g_ph_win_asc_us[i].fetch_add(ph[i], std::memory_order_relaxed);
            }
            if (total <= 8000) {
                return;
            }
            // >8 ms ASC dispatch: own cap (16 per run) so the GFX SLOWPKT cap is untouched.
            if (Parserprof::g_slow_asc_lines.fetch_add(1, std::memory_order_relaxed) >= 16) {
                return;
            }
            LOG_CRITICAL(Render,
                         "SLOWPKT-ASC: op={:#x} total={}us | compile={}us tex={}us buf={}us "
                         "desc={}us dma={}us vk={}us other={}us | vk[bind={}us rp={}us "
                         "barr={}us(bar={} wr={}) draw={}us sched={}us] binds={}",
                         op, total, ph[Parserprof::PhCompile], ph[Parserprof::PhTex],
                         ph[Parserprof::PhBuf], ph[Parserprof::PhDesc], ph[Parserprof::PhDma],
                         ph[Parserprof::PhVk], ph[Parserprof::PhOther], ph[Parserprof::PhBind],
                         ph[Parserprof::PhRp], ph[Parserprof::PhBarr], Parserprof::t_pkt_barriers,
                         Parserprof::t_pkt_writes, ph[Parserprof::PhDraw], ph[Parserprof::PhSched],
                         Parserprof::t_pkt_binds);
            return;
        }
        if (Parserprof::IsDrawClassOp(op)) {
            for (u32 i = 0; i < Parserprof::PhCount; ++i) {
                Parserprof::g_ph_win_us[i].fetch_add(ph[i], std::memory_order_relaxed);
            }
            const u32 bucket = total < 100     ? 0
                               : total < 1000  ? 1
                               : total < 5000  ? 2
                               : total < 20000 ? 3
                                               : 4;
            Parserprof::g_pkt_hist[bucket].fetch_add(1, std::memory_order_relaxed);
        } else if (op == 0x47) {
            // [FRAMECHAIN] The EOP-only buckets: nothing else feeds them, so without this the
            // PHASES/frame "eop drain/signal/flip" fields would be 0 by construction. Only
            // these three - they sit outside win_total and the vk split, so the top-3 and
            // per-frame draw-class figures stay comparable with earlier windows.
            Parserprof::g_ph_win_us[Parserprof::PhDrain].fetch_add(ph[Parserprof::PhDrain],
                                                                   std::memory_order_relaxed);
            Parserprof::g_ph_win_us[Parserprof::PhSignal].fetch_add(ph[Parserprof::PhSignal],
                                                                    std::memory_order_relaxed);
            Parserprof::g_ph_win_us[Parserprof::PhFlip].fetch_add(ph[Parserprof::PhFlip],
                                                                  std::memory_order_relaxed);
        }
        if (total <= 8000) {
            return;
        }
        // >8 ms: immediate line, capped at 32 per run (then counted silently).
        if (Parserprof::g_slow_lines.fetch_add(1, std::memory_order_relaxed) >= 32) {
            Parserprof::g_slow_suppressed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const u64 seq = Postmortem::g_gfx_op_seq.load(std::memory_order_relaxed);
        if (op == 0x47) {
            LOG_CRITICAL(Render,
                         "SLOWEOP: total={}us | drain={}us signal={}us flip={}us other={}us "
                         "seq={}",
                         total, ph[Parserprof::PhDrain], ph[Parserprof::PhSignal],
                         ph[Parserprof::PhFlip],
                         ph[Parserprof::PhOther] + ph[Parserprof::PhCompile] +
                             ph[Parserprof::PhTex] + ph[Parserprof::PhBuf] +
                             ph[Parserprof::PhDesc] + ph[Parserprof::PhDma] + ph[Parserprof::PhVk],
                         seq);
        } else {
            LOG_CRITICAL(Render,
                         "SLOWPKT: op={:#x} total={}us | compile={}us tex={}us buf={}us "
                         "desc={}us dma={}us vk={}us other={}us seq={} | vk[bind={}us rp={}us "
                         "barr={}us(bar={} wr={}) draw={}us sched={}us] binds={}",
                         op, total, ph[Parserprof::PhCompile], ph[Parserprof::PhTex],
                         ph[Parserprof::PhBuf], ph[Parserprof::PhDesc], ph[Parserprof::PhDma],
                         ph[Parserprof::PhVk],
                         ph[Parserprof::PhOther] + ph[Parserprof::PhDrain] +
                             ph[Parserprof::PhSignal] + ph[Parserprof::PhFlip],
                         seq, ph[Parserprof::PhBind], ph[Parserprof::PhRp], ph[Parserprof::PhBarr],
                         Parserprof::t_pkt_barriers, Parserprof::t_pkt_writes,
                         ph[Parserprof::PhDraw], ph[Parserprof::PhSched], Parserprof::t_pkt_binds);
        }
    }
};

} // namespace

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // [CSBATCH] a queued command (non-EOP flip -> draw_scheduler.Flush, ReleaseLabelRegion ->
    // Finish, ReadMemory -> Finish, TryServeFaultFast -> Flush) ends the tick or deletes
    // buffers: the pending worker batch is recorded first. One atomic load per packet.
    if (num_commands != 0 && rasterizer) {
        rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Cmd);
    }
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        // [FRAMECHAIN] Callback wall, attributed to the loop or to the task it ran inside of
        // (task-side time is already inside gfx/asc_resume; only loop-side time adds up).
        const bool fc_on = Parserprof::Enabled();
        const bool fc_task = fc_on && Parserprof::t_fc_in_task;
        {
            Parserprof::FcTimer fc_t{
                fc_task ? Parserprof::g_fc_cmd_task_ns : Parserprof::g_fc_cmd_ns, fc_on};
            callback();
        }
        if (fc_on) {
            (fc_task ? Parserprof::g_fc_cmd_task_n : Parserprof::g_fc_cmd_n)
                .fetch_add(1, std::memory_order_relaxed);
        }
    }
}

#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall SetThreadStackGuarantee(unsigned long*);
#endif

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    // [FRAMECHAIN] Who the parser is, so page_manager can tell faults/protects served on this
    // very thread from the rest (one store at start-up; nothing on the hot path).
    Parserprof::g_parser_tid.store(FcCurrentThreadId(), std::memory_order_relaxed);
#ifdef _WIN32
    {
        // Shader recompilation recurses deep on this thread; a fault near stack
        // exhaustion kills the process before any handler can run - the same
        // silent-death mode guest threads had before RunThread got this.
        unsigned long guarantee = 128 * 1024;
        SetThreadStackGuarantee(&guarantee);
    }
#endif
    LOG_WARNING(Render, "ENVCHECK: STUCKLOG={} TRACEWAIT={} SUBDEPTH={}",
                std::getenv("SHADPS4_STUCKLOG") != nullptr,
                std::getenv("SHADPS4_TRACEWAIT") != nullptr,
                std::getenv("SHADPS4_SUBDEPTH") != nullptr);
    gpu_id = std::this_thread::get_id();

    while (!stoken.stop_requested()) {
        {
            // [SLOWWAIT] starve = the parser thread sleeping with nothing queued at all.
            const bool sw_prof = Parserprof::Enabled();
            const s64 sw_t0 = sw_prof ? Parserprof::NowUs() : 0;
            if (sw_prof) {
                Postmortem::g_gfx_state.store(2, std::memory_order_relaxed);
            }
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
            if (sw_prof) {
                Postmortem::g_gfx_state.store(0, std::memory_order_relaxed);
                const s64 sw_t1 = Parserprof::NowUs();
                if (sw_t1 > sw_t0) {
                    Parserprof::g_starve_us.fetch_add(static_cast<u64>(sw_t1 - sw_t0),
                                                      std::memory_order_relaxed);
                }
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();

        curr_qid = -1;

        // [FRAMECHAIN] Wall of the round-robin loop and what happens inside it. Locals
        // accumulate between resumes and are flushed at every resume and at loop exit, so a
        // spin (every mapped queue polled, nothing runnable) costs no atomics per iteration.
        const bool fc_on = Parserprof::Enabled();
        s64 fc_seg_t0 = fc_on ? Parserprof::NowNs() : 0;
        u64 fc_iters = 0;
        u64 fc_gfx_empty = 0;

        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];
            ++fc_iters;

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    if (fc_on && curr_qid == GfxQueueId && num_submits) {
                        // no GFX work, but parked ASC submits keep the thread awake
                        ++fc_gfx_empty;
                    }
                    continue;
                }
                task = queue.submits.front();
            }
            // [FRAMECHAIN] Time the resume. An ASC resume that advanced no packet (woke up in
            // WaitRegMem/Rewind, tested, yielded again) is the round-robin spin.
            const bool fc_asc = curr_qid != GfxQueueId;
            const u32 fc_vq = fc_asc ? static_cast<u32>(curr_qid - 1) : 0u;
            s64 fc_t0 = 0;
            u64 fc_seq0 = 0;
            if (fc_on) {
                fc_t0 = Parserprof::NowNs();
                Parserprof::g_fc_loop_ns.fetch_add(
                    fc_t0 > fc_seg_t0 ? static_cast<u64>(fc_t0 - fc_seg_t0) : 0,
                    std::memory_order_relaxed);
                fc_seg_t0 = fc_t0;
                Parserprof::g_fc_loop_iters.fetch_add(fc_iters, std::memory_order_relaxed);
                Parserprof::g_fc_gfx_empty_iters.fetch_add(fc_gfx_empty, std::memory_order_relaxed);
                fc_iters = 0;
                fc_gfx_empty = 0;
                if (fc_asc && fc_vq < Parserprof::kFcAscQueues) {
                    fc_seq0 = Parserprof::g_fc_asc_pkt_seq[fc_vq].load(std::memory_order_relaxed);
                }
                Parserprof::t_fc_in_task = true;
            }
            task.resume();
            if (fc_on) {
                Parserprof::t_fc_in_task = false;
                const s64 fc_t1 = Parserprof::NowNs();
                const u64 fc_dt = fc_t1 > fc_t0 ? static_cast<u64>(fc_t1 - fc_t0) : 0;
                if (!fc_asc) {
                    Parserprof::g_fc_gfx_resume_ns.fetch_add(fc_dt, std::memory_order_relaxed);
                } else {
                    Parserprof::g_fc_asc_resumes.fetch_add(1, std::memory_order_relaxed);
                    // A task that ran to completion advanced by definition: its last packet
                    // may have been the wait it woke up in, with no new packet to bump seq.
                    const bool advanced = task.done() || fc_vq >= Parserprof::kFcAscQueues ||
                                          Parserprof::g_fc_asc_pkt_seq[fc_vq].load(
                                              std::memory_order_relaxed) != fc_seq0;
                    if (advanced) {
                        Parserprof::g_fc_asc_resume_ns.fetch_add(fc_dt, std::memory_order_relaxed);
                    } else {
                        Parserprof::g_fc_asc_spin_resumes.fetch_add(1, std::memory_order_relaxed);
                        Parserprof::g_fc_asc_spin_ns.fetch_add(fc_dt, std::memory_order_relaxed);
                    }
                }
            }

            if (task.done()) {
                task.destroy();

                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                }
                --num_submits;
                {
                    std::scoped_lock lock2{submit_mutex};
                    if (curr_qid == GfxQueueId) {
                        --num_gfx_submits;
                    } else {
                        --num_asc_submits;
                    }
                    Postmortem::g_submits_done.fetch_add(1, std::memory_order_relaxed);
                    Postmortem::g_last_retire_ms.store(Postmortem::NowMs(),
                                                       std::memory_order_relaxed);
                    Postmortem::g_num_gfx_submits.store(num_gfx_submits.load(),
                                                        std::memory_order_relaxed);
                    Postmortem::g_num_asc_submits.store(num_asc_submits.load(),
                                                        std::memory_order_relaxed);
                    submit_cv.notify_all();
                }
                if (curr_qid == GfxQueueId && flip_signal_armed) {
                    // fallback: flip NOP without a trailing EOP in this submit
                    flip_signal_armed = false;
                    ++retired_flip_submits;
                    {
                        // [FRAMECHAIN] flip handler outside any task (adds to the loop sum)
                        Parserprof::FcTimer fc_t{Parserprof::g_fc_flip_ns, fc_on};
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    }
                    if (fc_on) {
                        Parserprof::g_fc_flip_n.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (curr_qid == GfxQueueId && pending_flip_signals) {
                    static const bool s_fc = std::getenv("SHADPS4_STUCKLOG") != nullptr;
                    while (pending_flip_signals) {
                        --pending_flip_signals;
                        ++retired_flip_submits;
                        if (s_fc) {
                            static u32 s_fc_no = 0;
                            LOG_WARNING(Render, "FLIPCHAIN sig #{}", ++s_fc_no);
                        }
                        {
                            // [FRAMECHAIN] retire-time flip handler (outside any task)
                            Parserprof::FcTimer fc_t{Parserprof::g_fc_flip_ns, fc_on};
                            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                        }
                        if (fc_on) {
                            Parserprof::g_fc_flip_n.fetch_add(1, std::memory_order_relaxed);
                        }
                        // [debug] настоящие кадры = флипы: статистика и скриншоты
                        static u32 s_flip_no = 0;
                        const u32 f = ++s_flip_no;
                        Vulkan::g_dbg_frame_no.store(f);
                        static const char* s_ae_env2 = std::getenv("SHADPS4_AUTOSHOT_EVERY");
                        static const u32 s_ae2 =
                            s_ae_env2 ? (u32)std::strtoul(s_ae_env2, nullptr, 10) : 0;
                        if (s_ae2 && (f % s_ae2) == 0) {
                            VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
                        }
                        if ((f % 100) == 0) {
                            const u32 d = Vulkan::g_dbg_draws.exchange(0);
                            const u32 c = Vulkan::g_dbg_dispatches.exchange(0);
                            const u64 u = Vulkan::g_dbg_upload_bytes.exchange(0);
                            LOG_WARNING(Render,
                                        "FLIPSTAT: flips={} draws(last100)={} "
                                        "dispatches(last100)={} upMB(last100)={} pend_gfx={} "
                                        "pend_asc={}",
                                        f, d, c, u >> 20, num_gfx_submits.load(),
                                        num_asc_submits.load());
                        }
                    }
                    std::scoped_lock lk{submit_mutex};
                    submit_cv.notify_all();
                }
            }
        }

        // [FRAMECHAIN] loop exit: flush the last segment and the iteration counters.
        if (fc_on) {
            const s64 fc_end = Parserprof::NowNs();
            Parserprof::g_fc_loop_ns.fetch_add(
                fc_end > fc_seg_t0 ? static_cast<u64>(fc_end - fc_seg_t0) : 0,
                std::memory_order_relaxed);
            Parserprof::g_fc_loop_iters.fetch_add(fc_iters, std::memory_order_relaxed);
            Parserprof::g_fc_gfx_empty_iters.fetch_add(fc_gfx_empty, std::memory_order_relaxed);
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                // [FRAMECHAIN] the submit_done boundary, step by step (OnSubmit's own steps
                // are split in vk_rasterizer.cpp into g_fc_os_step_ns).
                {
                    Parserprof::FcTimer fc_t{Parserprof::g_fc_onsubmit_ns, fc_on};
                    rasterizer->OnSubmit();
                }
                {
                    Parserprof::FcTimer fc_t{Parserprof::g_fc_flush_ns, fc_on};
                    rasterizer->Flush();
                }
                if (fc_on) {
                    Parserprof::g_fc_flush_n.fetch_add(1, std::memory_order_relaxed);
                }
            }
            submit_done = false;
        }

        // [FRAMECHAIN] Everything drained: stamp the moment (SubmitGfx on the guest measures
        // its build time from it; the seq lets it count each drain once) and time the GpuIdle
        // handler, which runs on this thread through IrqC::Signal.
        if (fc_on) {
            Parserprof::g_fc_gpuidle_stamp_ns.store(Parserprof::NowNs(), std::memory_order_relaxed);
            Parserprof::g_fc_gpuidle_seq.fetch_add(1, std::memory_order_release);
        }
        {
            Parserprof::FcTimer fc_t{Parserprof::g_fc_gpuidle_ns, fc_on};
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        }
        if (fc_on) {
            Parserprof::g_fc_gpuidle_n.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled();

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        {
            const u64 hb = gfx_parse_heartbeat.fetch_add(1, std::memory_order_relaxed) + 1;
            if (Parserprof::Enabled()) {
                Postmortem::g_parse_heartbeat.store(hb, std::memory_order_relaxed);
                Postmortem::g_heartbeat_ms.store(Postmortem::NowMs(), std::memory_order_relaxed);
            }
        }
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            // [Stage1b PARSERPROF] Per-packet wall time (PktTimer at file scope): every break
            // path in the switch (and a coroutine suspension inside a packet) lands in the
            // same account when the iteration scope finally exits.
            PktTimer pkt_timer{static_cast<u32>(opcode),
                               Parserprof::Enabled() ? Parserprof::NowUs() : 0};
            // [R6 D5] Parser GFX heartbeat under SHADPS4_RINGSENSE - separates the wedge
            // candidates (spin vs fence-starvation). Off => no atomic writes, bit-identical.
            {
                static const bool pm_ringsense = [] {
                    const char* e = std::getenv("SHADPS4_RINGSENSE");
                    return e && e[0] == '1';
                }();
                if (pm_ringsense) {
                    Postmortem::g_gfx_cur_op.store(static_cast<u32>(opcode),
                                                   std::memory_order_relaxed);
                    Postmortem::g_gfx_op_seq.fetch_add(1, std::memory_order_relaxed);
                }
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // Defer the flip signal until this submit retires: PatchFlipRequest
                    // appends the frame-tick EventWriteEop AFTER this NOP, so signaling
                    // here lets the present thread deliver the flip event to the guest
                    // before the tick is written (observed as a guest-side assert of a
                    // zero EOP tick in Naughty Dog titles once the submission lock is
                    // relaxed). On real hardware an eop-flip cannot precede the EOP of
                    // its own submit either.
                    // Signal on the next EOP of this submit: the frame tick fence is
                    // appended right after this NOP, executing in the same packet
                    // stream — so the tick is guaranteed written before the flip
                    // event, without waiting for the whole submit to retire.
                    flip_signal_armed = true;
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                if (Vulkan::g_cs_sync_mask & 4u) {
                    Vulkan::g_cs_sync_pending =
                        true; // [CSBARRIER] game sync (e.g. CS_PARTIAL_FLUSH)
                }
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer) {
                    rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Label); // [CSBATCH]
                    const s64 dl_t0 = Parserprof::Enabled()
                                          ? std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now().time_since_epoch())
                                                .count()
                                          : 0;
                    // [A-EOPONLY] SHADPS4_DLIMG_EOPONLY=1 skips the EOS-site drain WITHOUT
                    // clearing the queue: the set dedups, images ride to the next EOP/ReleaseMem
                    // drain. EOP and ReleaseMem sites are untouched.
                    static const bool eoponly = [] {
                        const char* e = std::getenv("SHADPS4_DLIMG_EOPONLY");
                        return e && e[0] == '1';
                    }();
                    if (!eoponly) {
                        Parserprof::g_dl_eos_calls.fetch_add(1, std::memory_order_relaxed);
                        rasterizer->ProcessDownloadImages();
                    } else {
                        Parserprof::g_dl_eos_skipped.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (dl_t0 != 0) {
                        const s64 dl_now = std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now().time_since_epoch())
                                               .count();
                        Parserprof::g_evw_dl_us.fetch_add(static_cast<u64>(dl_now - dl_t0),
                                                          std::memory_order_relaxed);
                    }
                }
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                        memcpy(address, &data, num_bytes);
                    }
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        const s64 fin_t0 =
                            Parserprof::Enabled()
                                ? std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count()
                                : 0;
                        rasterizer->Finish();
                        if (fin_t0 != 0) {
                            const s64 fin_now =
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                            Parserprof::g_evw_fin_us.fetch_add(static_cast<u64>(fin_now - fin_t0),
                                                               std::memory_order_relaxed);
                        }

                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (rasterizer) {
                    rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Label); // [CSBATCH]
                }
                static const bool s_trace_fence = std::getenv("SHADPS4_TRACEFENCE") != nullptr;
                if (s_trace_fence) {
                    LOG_CRITICAL(Render, "EOPW addr={:#x} val={:#x}",
                                 reinterpret_cast<u64>(event_eop->Address<void*>()),
                                 event_eop->DataQWord());
                }
                if (rasterizer) {
                    // [MINGAP] SHADPS4_DLIMG_MINGAP_MS=N: drain no more often than every
                    // N ms. R3 measured ~2000 EOP packets/s at the deck, each non-empty
                    // drain paying a full Finish; the queue is a dedup set, so skipped
                    // drains just accumulate and flush in one batch on the next allowed
                    // pass. Same freshness class as EOPONLY (stale window <= N ms).
                    static const u32 mingap_ms = [] {
                        const char* e = std::getenv("SHADPS4_DLIMG_MINGAP_MS");
                        return e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
                    }();
                    if (mingap_ms != 0) {
                        static std::atomic<s64> last_drain_ms{0};
                        const s64 now_ms = Postmortem::NowMs();
                        const s64 last = last_drain_ms.load(std::memory_order_relaxed);
                        if (now_ms - last < static_cast<s64>(mingap_ms)) {
                            Parserprof::g_dl_gap_skipped.fetch_add(1, std::memory_order_relaxed);
                            goto eop_drain_done;
                        }
                        last_drain_ms.store(now_ms, std::memory_order_relaxed);
                    }
                    {
                        // [SLOWEOP] drain phase of the 0x47 packet.
                        Parserprof::PhaseProbe eop_drain_probe{Parserprof::PhDrain};
                        const s64 dl_t0 =
                            Parserprof::Enabled()
                                ? std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count()
                                : 0;
                        Parserprof::g_dl_eop_calls.fetch_add(1, std::memory_order_relaxed);
                        rasterizer->ProcessDownloadImages();
                        if (dl_t0 != 0) {
                            const s64 dl_now =
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                            Parserprof::g_evw_dl_us.fetch_add(static_cast<u64>(dl_now - dl_t0),
                                                              std::memory_order_relaxed);
                        }
                    }
                eop_drain_done:;
                    if (EopRetire()) {
                        // [EOPRETIRE] queue this tick's journal replay BEFORE the deferred
                        // label below (same FIFO): data first, then the fence the guest polls.
                        rasterizer->ProcessFaultBufferIfPending();
                    }
                }
                // [fix-candidate] SHADPS4_EOPIRQRETIRE=1: the guest's EOP callback treats the
                // interrupt as "everything this frame submitted has executed" and publishes
                // its frame counter from it. Raising the interrupt at parse time let the
                // counter run ahead of GPU-produced data (frame ticks stored by shaders),
                // which the guest catches with its own assert (fg-draw-mgr.cpp:234) on
                // no-flip frames - level loads and cutscene skips. Hardware raises EOP after
                // the work; do the same by holding only the IRQ (label writes stay at parse,
                // the parser's WaitRegMem polls depend on them) until the recorded commands
                // retire.
                static const bool eop_irq_at_retire = [] {
                    const char* e = std::getenv("SHADPS4_EOPIRQRETIRE");
                    return e && e[0] == '1';
                }();
                const auto signal_gfx_eop = [] {
                    // [postmortem] Branch-A sensor: when the parser last raised GfxEop.
                    const s64 now_ms = Postmortem::NowMs();
                    // [Stage1 PARSERPROF] EOP silence window: gap since the previous signal.
                    // The frame-tick assert fires when this exceeds ~3 frames; target <= 100 ms.
                    const s64 prev =
                        Postmortem::g_eop_signal_ms.exchange(now_ms, std::memory_order_relaxed);
                    if (prev != 0 && now_ms > prev) {
                        Postmortem::AtomicMax(Postmortem::g_eop_gap_max_ms,
                                              static_cast<u64>(now_ms - prev));
                    }
                    if (Parserprof::Enabled()) {
                        static std::atomic<s64> next_ms{0};
                        s64 expected = next_ms.load(std::memory_order_relaxed);
                        if (now_ms >= expected &&
                            next_ms.compare_exchange_strong(expected, now_ms + 5000) &&
                            expected != 0) {
                            LOG_CRITICAL(
                                Render,
                                "PARSERPROF: eop_gap_max_ms={} compiles={} specmiss={} "
                                "prof_pipe_us={} | pkt draw={} wrm={} evw={} dma={} "
                                "other={} max={}us op={:#x} | dl={} fin={} fbw={} | "
                                "dlq eos={} eop={} eos_skip={} gap_skip={} addrunbind={}",
                                Postmortem::g_eop_gap_max_ms.load(std::memory_order_relaxed),
                                Parserprof::g_compile_count.load(std::memory_order_relaxed),
                                Parserprof::g_specmiss_count.load(std::memory_order_relaxed),
                                Parserprof::g_prof_pipe_us.load(std::memory_order_relaxed),
                                Parserprof::g_pkt_draw_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_pkt_wrm_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_pkt_evw_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_pkt_dma_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_pkt_other_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_pkt_max_us.load(std::memory_order_relaxed),
                                Parserprof::g_pkt_max_op.load(std::memory_order_relaxed),
                                Parserprof::g_evw_dl_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_evw_fin_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_fb_wait_us.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_dl_eos_calls.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_dl_eop_calls.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_dl_eos_skipped.exchange(0, std::memory_order_relaxed),
                                Parserprof::g_dl_gap_skipped.exchange(0, std::memory_order_relaxed),
                                Vulkan::AddrBind::unbind_shader_region.load(
                                    std::memory_order_relaxed));
                            // [SLOWPKT] Draw-class phase split over the same 5s window; names
                            // the top-3 phases so the fatal packet class is attributable at a
                            // glance even after the 32-line SLOWPKT cap is spent.
                            {
                                struct PhVal {
                                    const char* name;
                                    u64 us;
                                };
                                // [FRAMECHAIN] Every phase bucket into a local first (one
                                // exchange per bucket): the full per-frame phase lines below
                                // and the legacy top-3 / vk-split lines then read one window,
                                // and the EOP-only buckets drain with the rest.
                                u64 phw[Parserprof::PhCount];
                                for (u32 i = 0; i < Parserprof::PhCount; ++i) {
                                    phw[i] = Parserprof::g_ph_win_us[i].exchange(
                                        0, std::memory_order_relaxed);
                                }
                                PhVal ph[6] = {
                                    {"compile", phw[Parserprof::PhCompile]},
                                    {"tex", phw[Parserprof::PhTex]},
                                    {"buf", phw[Parserprof::PhBuf]},
                                    {"desc", phw[Parserprof::PhDesc]},
                                    {"vk", phw[Parserprof::PhVk]},
                                    {"other", phw[Parserprof::PhOther]},
                                };
                                LOG_CRITICAL(
                                    Render,
                                    "PARSERPROF: vk split bind={}us rp={}us barr={}us draw={}us "
                                    "sched={}us",
                                    phw[Parserprof::PhBind], phw[Parserprof::PhRp],
                                    phw[Parserprof::PhBarr], phw[Parserprof::PhDraw],
                                    phw[Parserprof::PhSched]);
                                // Per-frame figures need a frame count; flips are the only
                                // honest one we have in this window.
                                const u64 frames = std::max<u64>(
                                    1, Postmortem::g_flips_total.load(std::memory_order_relaxed) -
                                           Parserprof::g_frames_counted.exchange(
                                               Postmortem::g_flips_total.load(
                                                   std::memory_order_relaxed),
                                               std::memory_order_relaxed));
                                const u64 rp_starts =
                                    Parserprof::g_rp_starts.exchange(0, std::memory_order_relaxed);
                                const u64 bar_buffers = Parserprof::g_bar_buffers.exchange(
                                    0, std::memory_order_relaxed);
                                const u64 desc_writes = Parserprof::g_desc_writes.exchange(
                                    0, std::memory_order_relaxed);
                                LOG_CRITICAL(
                                    Render,
                                    "RPSTAT: calls={} starts={} (per_frame={}) attach_changed={} "
                                    "ends={}",
                                    Parserprof::g_rp_calls.exchange(0, std::memory_order_relaxed),
                                    rp_starts, rp_starts / frames,
                                    Parserprof::g_rp_attach.exchange(0, std::memory_order_relaxed),
                                    Parserprof::g_rp_ends.exchange(0, std::memory_order_relaxed));
                                LOG_CRITICAL(
                                    Render, "BARSTAT: barrier_calls={} buffers={} per_frame={}",
                                    Parserprof::g_bar_calls.exchange(0, std::memory_order_relaxed),
                                    bar_buffers, bar_buffers / frames);
                                {
                                    // Read into locals: mixing load() and exchange(0) as
                                    // arguments of one call left the order unspecified, the
                                    // exchanges won, and both loads reported 0.
                                    const u64 bc = Parserprof::g_bind_calls.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 br = Parserprof::g_bind_real.exchange(
                                        0, std::memory_order_relaxed);
                                    LOG_CRITICAL(Render,
                                                 "BINDSTAT: binds={} real_switches={} "
                                                 "redundant={} | BINDSKIP skipped={} issued={}",
                                                 bc, br, bc > br ? bc - br : 0,
                                                 Vulkan::g_bind_skipped.exchange(
                                                     0, std::memory_order_relaxed),
                                                 Vulkan::g_bind_issued.exchange(
                                                     0, std::memory_order_relaxed));
                                }
                                LOG_CRITICAL(
                                    Render,
                                    "DESCSTAT: writes={} redundant={} per_frame={} frames={}",
                                    desc_writes,
                                    Parserprof::g_desc_redundant.exchange(
                                        0, std::memory_order_relaxed),
                                    desc_writes / frames, frames);
                                {
                                    // Rule after the BINDSTAT incident: every value into a local
                                    // BEFORE the log call, never load() and exchange() mixed in
                                    // one argument list.
                                    const u64 sw = Parserprof::g_pipe_switches.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 created = Parserprof::g_pipe_created.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 uniq =
                                        Parserprof::g_pipe_unique.load(std::memory_order_relaxed);
                                    const u64 d_sh = Parserprof::g_pipe_d_shaders.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 d_bl = Parserprof::g_pipe_d_blend.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 d_rt = Parserprof::g_pipe_d_rtfmt.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 d_vx = Parserprof::g_pipe_d_vertex.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 d_dp = Parserprof::g_pipe_d_depth.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 d_ot = Parserprof::g_pipe_d_other.exchange(
                                        0, std::memory_order_relaxed);
                                    // [BININFO 11.09] запоминание заголовка шейдера: без него
                                    // SearchBinaryInfo сканирует до 64 КБ на стадию на каждый вызов
                                    const u64 bi_h = Parserprof::g_bininfo_hits.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 bi_m = Parserprof::g_bininfo_misses.exchange(
                                        0, std::memory_order_relaxed);
                                    LOG_CRITICAL(Render,
                                                 "PIPESTAT: unique={} created_total={} "
                                                 "switches={} | by_field: shaders={} blend={} "
                                                 "rt_format={} vertex={} depth={} other={} | "
                                                 "BININFO hits={} misses={}",
                                                 uniq, created, sw, d_sh, d_bl, d_rt, d_vx, d_dp,
                                                 d_ot, bi_h, bi_m);
                                }
                                {
                                    // Locals before the log call - the BINDSTAT rule.
                                    const u64 tb =
                                        Parserprof::g_ws_total_bufs.load(std::memory_order_relaxed);
                                    const u64 tby = Parserprof::g_ws_total_bytes.load(
                                        std::memory_order_relaxed);
                                    const u64 wb = Parserprof::g_ws_written_bufs.load(
                                        std::memory_order_relaxed);
                                    const u64 wby = Parserprof::g_ws_written_bytes.load(
                                        std::memory_order_relaxed);
                                    const u64 lb =
                                        Parserprof::g_ws_late_bufs.load(std::memory_order_relaxed);
                                    const u64 lby =
                                        Parserprof::g_ws_late_bytes.load(std::memory_order_relaxed);
                                    LOG_CRITICAL(
                                        Render,
                                        "WRITESTAT: written_bufs={} ({} MB) | readonly_bufs={} "
                                        "({} MB) | write_once_early={} ({} MB) | still_writing={} "
                                        "({} MB) | upMB_since_flipstat={} sync_us={}",
                                        wb, wby >> 20, tb > wb ? tb - wb : 0,
                                        tby > wby ? (tby - wby) >> 20 : 0, wb > lb ? wb - lb : 0,
                                        wby > lby ? (wby - lby) >> 20 : 0, lb, lby >> 20,
                                        Vulkan::g_dbg_upload_bytes.load(
                                            std::memory_order_relaxed) >>
                                            20,
                                        Postmortem::g_buf_sync_us.load(std::memory_order_relaxed) /
                                            1000); // [FRAMECHAIN] UsTimer sink: ns -> us
                                }
                                {
                                    // [TOUCHSTAT] What the GPU actually reaches for per frame.
                                    // Every value into a local BEFORE the log call - the BINDSTAT
                                    // incident was load() and exchange() mixed in one argument
                                    // list, where evaluation order is unspecified.
                                    const u64 ts_fr = frames != 0 ? frames : 1;
                                    const u64 b_ro = Parserprof::g_ts_bound_ro.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 b_rw = Parserprof::g_ts_bound_rw.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 u_ro = Parserprof::g_ts_unique_ro.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 u_rw = Parserprof::g_ts_unique_rw.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 u_bufs = Parserprof::g_ts_unique_bufs.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 s_by = Parserprof::g_ts_stream_bytes.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 s_calls = Parserprof::g_ts_stream_calls.exchange(
                                        0, std::memory_order_relaxed);
                                    LOG_CRITICAL(
                                        Render,
                                        "TOUCHSTAT/frame: bound_ro={}MB bound_rw={}MB | "
                                        "unique_ro={}MB unique_rw={}MB ({} bufs) | "
                                        "stream={}KB ({} calls) | reread_x100={} | frames={}",
                                        (b_ro / ts_fr) >> 20, (b_rw / ts_fr) >> 20,
                                        (u_ro / ts_fr) >> 20, (u_rw / ts_fr) >> 20, u_bufs / ts_fr,
                                        (s_by / ts_fr) >> 10, s_calls / ts_fr,
                                        (u_ro + u_rw) != 0 ? ((b_ro + b_rw) * 100) / (u_ro + u_rw)
                                                           : 0,
                                        frames);
                                }
                                LOG_CRITICAL(
                                    Render,
                                    "PKTHIST: <0.1ms={} 0.1-1ms={} 1-5ms={} 5-20ms={} "
                                    ">20ms={}",
                                    Parserprof::g_pkt_hist[0].exchange(0,
                                                                       std::memory_order_relaxed),
                                    Parserprof::g_pkt_hist[1].exchange(0,
                                                                       std::memory_order_relaxed),
                                    Parserprof::g_pkt_hist[2].exchange(0,
                                                                       std::memory_order_relaxed),
                                    Parserprof::g_pkt_hist[3].exchange(0,
                                                                       std::memory_order_relaxed),
                                    Parserprof::g_pkt_hist[4].exchange(0,
                                                                       std::memory_order_relaxed));
                                std::sort(
                                    std::begin(ph), std::end(ph),
                                    [](const PhVal& a, const PhVal& b) { return a.us > b.us; });
                                // [FRAMECHAIN] + dma: it used to sit inside desc, so the total
                                // stays comparable with pre-W0 windows.
                                const u64 win_total = ph[0].us + ph[1].us + ph[2].us + ph[3].us +
                                                      ph[4].us + ph[5].us + phw[Parserprof::PhDma];
                                // [SLOWWAIT] wait/starve split of the same window: busy
                                // (win_total) + wrm_wait + starve should account for the EOP
                                // gap; whatever is left is "other queues / unprobed".
                                const u64 win_wait = Parserprof::g_wrm_wait_us.exchange(
                                    0, std::memory_order_relaxed);
                                const u64 win_starve =
                                    Parserprof::g_starve_us.exchange(0, std::memory_order_relaxed);
                                if (win_total != 0 || win_wait != 0 || win_starve != 0) {
                                    LOG_CRITICAL(
                                        Render,
                                        "PARSERPROF: draw phases top3 {}={}us {}={}us {}={}us "
                                        "(win_total={}us wrm_wait={}us starve={}us "
                                        "slow_suppressed={} hashcap_skipped={} subfix_released={} "
                                        "gc_us={})",
                                        ph[0].name, ph[0].us, ph[1].name, ph[1].us, ph[2].name,
                                        ph[2].us, win_total, win_wait, win_starve,
                                        Parserprof::g_slow_suppressed.load(
                                            std::memory_order_relaxed),
                                        Postmortem::g_hashcap_skipped.load(
                                            std::memory_order_relaxed),
                                        Postmortem::g_subfix_released.load(
                                            std::memory_order_relaxed),
                                        Postmortem::g_gc_us.exchange(0, std::memory_order_relaxed));
                                }
                                // [FRAMECHAIN] All phases, per frame, GFX draw class then ASC
                                // dispatches, then the DMA block. Locals first, then the lines.
                                {
                                    const u64 fr = frames; // >= 1
                                    u64 pha[Parserprof::PhCount];
                                    u64 asc_phased_total = 0;
                                    for (u32 i = 0; i < Parserprof::PhCount; ++i) {
                                        pha[i] = Parserprof::g_ph_win_asc_us[i].exchange(
                                            0, std::memory_order_relaxed);
                                        asc_phased_total += pha[i];
                                    }
                                    const u64 asc_pk_us = Parserprof::g_pkt_asc_us.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_pk_n = Parserprof::g_pkt_asc_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_wt_us = Parserprof::g_pkt_asc_wait_us.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_wt_n = Parserprof::g_pkt_asc_wait_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 sweeps = Parserprof::g_dma_sweeps.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 zone_runs = Parserprof::g_dma_zone_runs.exchange(
                                        0, std::memory_order_relaxed);
                                    LOG_CRITICAL(
                                        Render,
                                        "PHASES/frame(us): gfx compile={} tex={} buf={} desc={} "
                                        "dma={} vk={} other={} | bind={} rp={} barr={} draw={} "
                                        "sched={} | eop drain={} signal={} flip={} hash={} | "
                                        "frames={}",
                                        phw[Parserprof::PhCompile] / fr,
                                        phw[Parserprof::PhTex] / fr, phw[Parserprof::PhBuf] / fr,
                                        phw[Parserprof::PhDesc] / fr, phw[Parserprof::PhDma] / fr,
                                        phw[Parserprof::PhVk] / fr, phw[Parserprof::PhOther] / fr,
                                        phw[Parserprof::PhBind] / fr, phw[Parserprof::PhRp] / fr,
                                        phw[Parserprof::PhBarr] / fr, phw[Parserprof::PhDraw] / fr,
                                        phw[Parserprof::PhSched] / fr,
                                        phw[Parserprof::PhDrain] / fr,
                                        phw[Parserprof::PhSignal] / fr,
                                        phw[Parserprof::PhFlip] / fr, phw[Parserprof::PhHash] / fr,
                                        fr);
                                    LOG_CRITICAL(
                                        Render,
                                        "PHASES-ASC/frame(us): compile={} tex={} buf={} desc={} "
                                        "dma={} vk={} other={} | bind={} rp={} barr={} draw={} "
                                        "sched={} | dispatch_total={} | asc pkts={}us(x{}) "
                                        "wait_pkts={}us(x{}, wall incl. suspended)",
                                        pha[Parserprof::PhCompile] / fr,
                                        pha[Parserprof::PhTex] / fr, pha[Parserprof::PhBuf] / fr,
                                        pha[Parserprof::PhDesc] / fr, pha[Parserprof::PhDma] / fr,
                                        pha[Parserprof::PhVk] / fr, pha[Parserprof::PhOther] / fr,
                                        pha[Parserprof::PhBind] / fr, pha[Parserprof::PhRp] / fr,
                                        pha[Parserprof::PhBarr] / fr, pha[Parserprof::PhDraw] / fr,
                                        pha[Parserprof::PhSched] / fr, asc_phased_total / fr,
                                        asc_pk_us / fr, asc_pk_n / fr, asc_wt_us / fr,
                                        asc_wt_n / fr);
                                    LOG_CRITICAL(
                                        Render,
                                        "DMASYNC/frame: sweeps={:.2f} zone_runs={:.1f} dma_us={} "
                                        "(+asc {}) | window: sweeps={} zone_runs={} dma_us={} "
                                        "asc_dma_us={} frames={}",
                                        static_cast<double>(sweeps) / static_cast<double>(fr),
                                        static_cast<double>(zone_runs) / static_cast<double>(fr),
                                        phw[Parserprof::PhDma] / fr, pha[Parserprof::PhDma] / fr,
                                        sweeps, zone_runs, phw[Parserprof::PhDma],
                                        pha[Parserprof::PhDma], fr);
                                }
                                // [FRAMECHAIN] Where the parser thread's wall goes per frame
                                // beyond the packet timers. Every value into a local first (the
                                // BINDSTAT rule), then three lines: parser side, guest side +
                                // small items, and the invariant against the window's wall.
                                // Sinks hold ns; printed as us/frame. starve/wrm_wait are the
                                // SLOWWAIT window figures above (us).
                                {
                                    const u64 fr = frames; // >= 1
                                    static s64 s_fc_wall_ns = 0;
                                    static u64 s_fc_cpu_100ns = 0;
                                    const s64 fc_now = Parserprof::NowNs();
                                    const u64 wall_ns =
                                        (s_fc_wall_ns != 0 && fc_now > s_fc_wall_ns)
                                            ? static_cast<u64>(fc_now - s_fc_wall_ns)
                                            : 0;
                                    s_fc_wall_ns = fc_now;
                                    // Thread CPU only when this really is the parser thread
                                    // (with SHADPS4_EOPIRQRETIRE=1 the EOP callback, and this
                                    // report with it, runs on the retire thread): else 0 = n/a.
                                    const u64 cpu_now =
                                        FcCurrentThreadId() == Parserprof::g_parser_tid.load(
                                                                   std::memory_order_relaxed)
                                            ? FcThreadCpu100ns()
                                            : 0;
                                    const u64 cpu_ns =
                                        (s_fc_cpu_100ns != 0 && cpu_now >= s_fc_cpu_100ns)
                                            ? (cpu_now - s_fc_cpu_100ns) * 100
                                            : 0;
                                    s_fc_cpu_100ns = cpu_now;
                                    // [FRAMECHAIN] the same window in CYCLES: GetThreadTimes is
                                    // sampled at the 15.6 ms scheduler tick and under-reports a
                                    // thread that alternates short work with waits, which is why
                                    // "cpu=2.0 ms/frame" sat next to "gfx_resume=14.2 ms/frame".
                                    static u64 s_fc_cycles = 0;
                                    const u64 cyc_now =
                                        FcCurrentThreadId() == Parserprof::g_parser_tid.load(
                                                                   std::memory_order_relaxed)
                                            ? FcThreadCycles()
                                            : 0;
                                    const u64 cyc_delta =
                                        (s_fc_cycles != 0 && cyc_now >= s_fc_cycles)
                                            ? (cyc_now - s_fc_cycles)
                                            : 0;
                                    s_fc_cycles = cyc_now;
                                    const u64 gfx_resume = Parserprof::g_fc_gfx_resume_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_resume = Parserprof::g_fc_asc_resume_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_resumes = Parserprof::g_fc_asc_resumes.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_spin_resumes =
                                        Parserprof::g_fc_asc_spin_resumes.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 asc_spin = Parserprof::g_fc_asc_spin_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_pkts = Parserprof::g_fc_asc_pkts.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 asc_disp = Parserprof::g_fc_asc_dispatches.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 kick_lat = Parserprof::g_fc_asc_kick_lat_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 kick_n = Parserprof::g_fc_asc_kick_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 gfx_empty = Parserprof::g_fc_gfx_empty_iters.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 loop_iters = Parserprof::g_fc_loop_iters.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 loop_ns = Parserprof::g_fc_loop_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    u64 asc_wrm = 0;
                                    for (u32 i = 0; i < Parserprof::kFcAscQueues; ++i) {
                                        asc_wrm += Parserprof::g_asc_wait_ns[i].exchange(
                                            0, std::memory_order_relaxed);
                                    }
                                    const u64 onsubmit = Parserprof::g_fc_onsubmit_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    u64 os[8];
                                    for (u32 i = 0; i < 8; ++i) {
                                        os[i] = Parserprof::g_fc_os_step_ns[i].exchange(
                                            0, std::memory_order_relaxed);
                                    }
                                    const u64 flush = Parserprof::g_fc_flush_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 flush_n = Parserprof::g_fc_flush_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 flip = Parserprof::g_fc_flip_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 flip_n = Parserprof::g_fc_flip_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 flip_task = Parserprof::g_fc_flip_task_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 flip_task_n = Parserprof::g_fc_flip_task_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 gpuidle = Parserprof::g_fc_gpuidle_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 gpuidle_n = Parserprof::g_fc_gpuidle_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 cmd = Parserprof::g_fc_cmd_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 cmd_n = Parserprof::g_fc_cmd_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 cmd_task = Parserprof::g_fc_cmd_task_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 cmd_task_n = Parserprof::g_fc_cmd_task_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 frame_wait = Parserprof::g_fc_frame_wait_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 frame_wait_n = Parserprof::g_fc_frame_wait_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 idle_sub =
                                        Parserprof::g_fc_idle_to_submit_ns.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 idle_sub_n =
                                        Parserprof::g_fc_idle_to_submit_n.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 sd_wait_us =
                                        Parserprof::g_fc_guest_sd_wait_us.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 sd_int_us =
                                        Parserprof::g_fc_guest_sd_interval_us.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 sd_n = Parserprof::g_fc_guest_sd_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 sub_wait_us =
                                        Parserprof::g_fc_guest_submit_wait_us.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 sub_n = Parserprof::g_fc_guest_submit_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 dd_wait_us =
                                        Parserprof::g_fc_guest_dd_wait_us.exchange(
                                            0, std::memory_order_relaxed);
                                    const u64 dd_n = Parserprof::g_fc_guest_dd_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 pop = Parserprof::g_fc_pop_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 pop_n = Parserprof::g_fc_pop_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 clamp = Parserprof::g_fc_clamp_ns.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 clamp_n = Parserprof::g_fc_clamp_n.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 faults = Parserprof::g_fc_parser_faults.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 protects = Parserprof::g_fc_parser_protects.exchange(
                                        0, std::memory_order_relaxed);
                                    // The invariant: everything the thread did outside the
                                    // tasks plus the tasks themselves plus the sleep, against
                                    // the window's wall. What is left is still unprobed.
                                    const u64 starve_ns = win_starve * 1000;
                                    const u64 in_loop =
                                        gfx_resume + asc_resume + asc_spin + cmd + flip;
                                    const u64 loop_over = loop_ns > in_loop ? loop_ns - in_loop : 0;
                                    const u64 sum_ns =
                                        in_loop + onsubmit + flush + gpuidle + starve_ns;
                                    const u64 covered = sum_ns + loop_over;
                                    const u64 unacc = wall_ns > covered ? wall_ns - covered : 0;
                                    const auto pf = [fr](u64 ns) { return ns / 1000 / fr; };
                                    const auto pm = [fr](u64 ns) {
                                        return static_cast<double>(ns) / 1e6 /
                                               static_cast<double>(fr);
                                    };
                                    const auto gm = [fr](u64 us) {
                                        return static_cast<double>(us) / 1e3 /
                                               static_cast<double>(fr);
                                    };
                                    LOG_CRITICAL(
                                        Render,
                                        "FRAMECHAIN/frame(us): gfx_resume={} asc_resume={} "
                                        "asc_spin={}({} of {} resumes) asc_pkts={} asc_disp={} "
                                        "asc_kick_lat={}us avg(x{}) asc_wrm_wait={} "
                                        "gfx_empty_iters={} loop_iters={} | onsubmit={} [prof={} "
                                        "prefetch={} label={} fault={} dlimg={} texgc={} bufgc={}] "
                                        "flush={}(x{}) flip={}(x{}) flip_in_task={}(x{}) "
                                        "gpuidle={}(x{}) cmds={}(x{}) cmds_in_task={}(x{}) "
                                        "frame_wait={}(x{}) starve={} wrm_wait={}",
                                        pf(gfx_resume), pf(asc_resume), pf(asc_spin),
                                        asc_spin_resumes / fr, asc_resumes / fr, asc_pkts / fr,
                                        asc_disp / fr, kick_n != 0 ? kick_lat / 1000 / kick_n : 0,
                                        kick_n / fr, pf(asc_wrm), gfx_empty / fr, loop_iters / fr,
                                        pf(onsubmit), pf(os[0]), pf(os[1]), pf(os[2]), pf(os[3]),
                                        pf(os[4]), pf(os[5]), pf(os[6]), pf(flush), flush_n / fr,
                                        pf(flip), flip_n / fr, pf(flip_task), flip_task_n / fr,
                                        pf(gpuidle), gpuidle_n / fr, pf(cmd), cmd_n / fr,
                                        pf(cmd_task), cmd_task_n / fr, pf(frame_wait),
                                        frame_wait_n / fr, win_starve / fr, win_wait / fr);
                                    LOG_CRITICAL(
                                        Render,
                                        "FRAMECHAIN/frame: idle_to_submit={}us avg(x{}) | guest: "
                                        "sd_interval={:.2f}ms sd_wait={:.2f}ms(x{}) "
                                        "submit_wait={:.2f}ms(x{}) dd_wait={:.2f}ms(x{}) | "
                                        "pop={}us(x{}) clamp={}us(x{}) | parser faults={} "
                                        "protects={} | frames={}",
                                        idle_sub_n != 0 ? idle_sub / 1000 / idle_sub_n : 0,
                                        idle_sub_n / fr, gm(sd_int_us), gm(sd_wait_us), sd_n / fr,
                                        gm(sub_wait_us), sub_n / fr, gm(dd_wait_us), dd_n / fr,
                                        pf(pop), pop_n / fr, pf(clamp), clamp_n / fr, faults / fr,
                                        protects / fr, fr);
                                    LOG_CRITICAL(
                                        Render,
                                        "FRAMECHAIN/sum: gfx_resume+asc_resume+asc_spin+cmds+flip"
                                        "+onsubmit+flush+gpuidle+starve={:.2f}ms "
                                        "loop_overhead={:.2f}ms => covered={:.2f}ms of "
                                        "wall={:.2f}ms/frame (unaccounted={:.2f}ms) | "
                                        "cpu={:.2f}ms/frame busy={}% cycles={}M/frame | "
                                        "window={}ms frames={}",
                                        pm(sum_ns), pm(loop_over), pm(covered), pm(wall_ns),
                                        pm(unacc), pm(cpu_ns),
                                        wall_ns != 0 ? cpu_ns * 100 / wall_ns : 0,
                                        fr != 0 ? cyc_delta / fr / 1000000 : 0, wall_ns / 1000000,
                                        fr);
                                }
                            }
                            // [BUFPROF/RESSTAT] Where the dominant `buf` phase actually goes,
                            // and whether the caches grow without bound.
                            {
                                // [FRAMECHAIN] UsTimer/UsTimerOpt sinks hold nanoseconds now
                                // (postmortem.h): converted to us here, once, into locals - the
                                // arithmetic below and the lines keep their old units.
                                const u64 ob = Postmortem::g_buf_obtain_us.exchange(
                                                   0, std::memory_order_relaxed) /
                                               1000;
                                const u64 cr = Postmortem::g_buf_create_us.exchange(
                                                   0, std::memory_order_relaxed) /
                                               1000;
                                const u64 last_streamcopy_us = Postmortem::g_buf_streamcopy_us.load(
                                                                   std::memory_order_relaxed) /
                                                               1000;
                                const u64 last_sync_us =
                                    Postmortem::g_buf_sync_us.load(std::memory_order_relaxed) /
                                    1000;
                                const u64 last_gpumod_us =
                                    Postmortem::g_buf_gpumod_us.load(std::memory_order_relaxed) /
                                    1000;
                                const u64 last_labelint_us =
                                    Postmortem::g_buf_labelint_us.load(std::memory_order_relaxed) /
                                    1000;
                                const u64 backing_off_us =
                                    Postmortem::g_buf_backing_off_us.exchange(
                                        0, std::memory_order_relaxed) /
                                    1000;
                                const u64 vk_create_us = Postmortem::g_buf_vk_create_us.exchange(
                                                             0, std::memory_order_relaxed) /
                                                         1000;
                                const u64 register_us = Postmortem::g_buf_register_us.exchange(
                                                            0, std::memory_order_relaxed) /
                                                        1000;
                                const u64 overlap_us = Postmortem::g_buf_overlap_us.exchange(
                                                           0, std::memory_order_relaxed) /
                                                       1000;
                                const u64 creates = Postmortem::g_buf_creates.exchange(
                                    0, std::memory_order_relaxed);
                                const u64 obtains = Postmortem::g_buf_obtains.exchange(
                                    0, std::memory_order_relaxed);
                                const u64 streamcopy_us = Postmortem::g_buf_streamcopy_us.exchange(
                                                              0, std::memory_order_relaxed) /
                                                          1000;
                                const u64 streamcopies = Postmortem::g_buf_streamcopies.exchange(
                                    0, std::memory_order_relaxed);
                                const u64 find_us = Postmortem::g_buf_find_us.exchange(
                                                        0, std::memory_order_relaxed) /
                                                    1000;
                                const u64 sync_us = Postmortem::g_buf_sync_us.exchange(
                                                        0, std::memory_order_relaxed) /
                                                    1000;
                                const u64 gpumod_us = Postmortem::g_buf_gpumod_us.exchange(
                                                          0, std::memory_order_relaxed) /
                                                      1000;
                                const u64 labelint_us = Postmortem::g_buf_labelint_us.exchange(
                                                            0, std::memory_order_relaxed) /
                                                        1000;
                                LOG_CRITICAL(
                                    Render,
                                    "BUFPROF: lookup={}us create={}us backing_off={}us "
                                    "vk_create={}us register={}us overlap={}us | creates={} "
                                    "obtains={} || streamcopy={}us({}) find={}us sync={}us "
                                    "gpumod={}us labelint={}us",
                                    ob > cr ? ob - cr : 0, cr, backing_off_us, vk_create_us,
                                    register_us, overlap_us, creates, obtains, streamcopy_us,
                                    streamcopies, find_us, sync_us, gpumod_us, labelint_us);
                                // [BUFPROF2] What no probe covered, computed here rather than
                                // eyeballed - including an estimate of what the probes
                                // themselves cost, which at this call rate is not negligible.
                                {
                                    const u64 lookup_us = ob > cr ? ob - cr : 0;
                                    const u64 entry_us = 0;
                                    // UsTimerOpt sinks: ns -> us (see BUFPROF above).
                                    const u64 lockwait = Postmortem::g_buf_lockwait_us.exchange(
                                                             0, std::memory_order_relaxed) /
                                                         1000;
                                    const u64 written = Postmortem::g_buf_written_us.exchange(
                                                            0, std::memory_order_relaxed) /
                                                        1000;
                                    const u64 ret = Postmortem::g_buf_ret_us.exchange(
                                                        0, std::memory_order_relaxed) /
                                                    1000;
                                    const u64 probes = Postmortem::g_buf_probes.exchange(
                                        0, std::memory_order_relaxed);
                                    const u64 est_probe_us =
                                        probes * Postmortem::ProbeClockNs() / 1000ull;
                                    const u64 covered = last_streamcopy_us + last_sync_us +
                                                        last_gpumod_us + last_labelint_us +
                                                        written + ret;
                                    LOG_CRITICAL(Render,
                                                 "BUFPROF2: entry={}us lockwait={}us written={}us "
                                                 "ret={}us covered={}us | probes={} clock_ns={} "
                                                 "est_probe={}us unaccounted={}us",
                                                 entry_us, lockwait, written, ret, covered, probes,
                                                 Postmortem::ProbeClockNs(), est_probe_us,
                                                 lookup_us > covered + est_probe_us
                                                     ? lookup_us - covered - est_probe_us
                                                     : 0);
                                }
                                LOG_CRITICAL(
                                    Render,
                                    "STREAMHIST: <256b={} 256-1K={} 1-4K={} =4K={} | "
                                    "CONSTCACHE hits={} misses={} | STREAMWRAP wraps={} "
                                    "clears={}(frame={} wrap={}) cpumod={} wrap_in_copy={}",
                                    Postmortem::g_stream_hist[0].exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_stream_hist[1].exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_stream_hist[2].exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_stream_hist[3].exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_hits.exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_misses.exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_clear_wrap.load(
                                        std::memory_order_relaxed),
                                    Postmortem::g_constcache_clears.exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_clear_frame.exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_clear_wrap.exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_cpumod.exchange(
                                        0, std::memory_order_relaxed),
                                    Postmortem::g_constcache_wrap_in_copy.exchange(
                                        0, std::memory_order_relaxed));
                                LOG_CRITICAL(
                                    Render,
                                    "RESSTAT-HIST: b0(<4K)={} b1(4-64K)={} b2(64K-1M)={} "
                                    "b3(>1M)={} | dedicated_allocs={} maxMemoryAllocationCount={}",
                                    Postmortem::g_backing_hist[0].load(std::memory_order_relaxed),
                                    Postmortem::g_backing_hist[1].load(std::memory_order_relaxed),
                                    Postmortem::g_backing_hist[2].load(std::memory_order_relaxed),
                                    Postmortem::g_backing_hist[3].load(std::memory_order_relaxed),
                                    Postmortem::g_live_dedicated.load(std::memory_order_relaxed),
                                    Postmortem::g_max_mem_allocs.load(std::memory_order_relaxed));
                                LOG_CRITICAL(
                                    Render,
                                    "RESSTAT: buffers={} images={} backing_bufs={} "
                                    "vram_MB={}",
                                    Postmortem::g_res_buffers.load(std::memory_order_relaxed),
                                    Postmortem::g_res_images.load(std::memory_order_relaxed),
                                    Postmortem::g_res_backing_bufs.load(std::memory_order_relaxed),
                                    Postmortem::g_res_vram_mb.load(std::memory_order_relaxed));
                            }
                            // [R8] one-shot verdict on the driver: by the first 5s report the
                            // pipeline-creation shower must have produced events if the driver
                            // emits them at all.
                            if (Vulkan::AddrBindEnabled()) {
                                static std::atomic<bool> ab_warned{false};
                                if (Vulkan::AddrBind::TotalEvents() == 0 &&
                                    !ab_warned.exchange(true, std::memory_order_relaxed)) {
                                    LOG_CRITICAL(Render, "ADDRBIND: feature on but no events");
                                }
                            }
                        }
                    }
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                };
                {
                    // [SLOWEOP] signal phase: fence write + GfxEop IRQ (guest EOP callback runs
                    // synchronously on this thread through IrqC::Signal).
                    Parserprof::PhaseProbe eop_signal_probe{Parserprof::PhSignal};
                    event_eop->SignalFence(
                        [this](void* address, u64 data, u32 num_bytes) {
                            const auto write = [address, data, num_bytes] {
                                auto* memory = Core::Memory::Instance();
                                if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                                    memcpy(address, &data, num_bytes);
                                }
                            };
                            // [EOPRETIRE] the label lands when the GPU really finished the
                            // frame - after the deferred readback writeback queued before it.
                            // The parser's own WaitRegMem flushes on a miss so it cannot wait
                            // for a tick it never submitted.
                            if (EopRetire() && rasterizer) {
                                rasterizer->DeferSignalToRetire(write);
                            } else {
                                write();
                            }
                        },
                        [this, &signal_gfx_eop] {
                            if ((eop_irq_at_retire || EopRetire()) && rasterizer) {
                                rasterizer->DeferSignalToRetire(signal_gfx_eop);
                            } else {
                                signal_gfx_eop();
                            }
                        });
                    if (EopRetire()) {
                        // [EOPRETIRE] submit the tick the writeback + label were registered on
                        // (deferred ops take CurrentTick = the cmdbuf being recorded, so the
                        // registration must precede this Flush). Hardware signals EOP here.
                        rasterizer->Flush();
                    }
                }
                if (flip_signal_armed) {
                    // [SLOWEOP] flip phase: GfxFlip IRQ -> guest flip callback + flipstat.
                    Parserprof::PhaseProbe eop_flip_probe{Parserprof::PhFlip};
                    flip_signal_armed = false;
                    ++retired_flip_submits;
                    static const bool s_fc = std::getenv("SHADPS4_STUCKLOG") != nullptr;
                    if (s_fc) {
                        static u32 s_fc_no = 0;
                        LOG_WARNING(Render, "FLIPCHAIN sig #{}", ++s_fc_no);
                    }
                    {
                        // [FRAMECHAIN] in-task flip handler (inside gfx_resume and PhFlip)
                        Parserprof::FcTimer fc_t{Parserprof::g_fc_flip_task_ns};
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    }
                    if (Parserprof::Enabled()) {
                        Parserprof::g_fc_flip_task_n.fetch_add(1, std::memory_order_relaxed);
                    }
                    static u32 s_flip_no2 = 0;
                    const u32 f2 = ++s_flip_no2;
                    Vulkan::g_dbg_frame_no.store(f2);
                    static const char* s_ae3 = std::getenv("SHADPS4_AUTOSHOT_EVERY");
                    static const u32 s_ae3v = s_ae3 ? (u32)std::strtoul(s_ae3, nullptr, 10) : 0;
                    if (s_ae3v && (f2 % s_ae3v) == 0) {
                        VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
                    }
                    if ((f2 % 100) == 0) {
                        const u32 d2 = Vulkan::g_dbg_draws.exchange(0);
                        const u32 c2 = Vulkan::g_dbg_dispatches.exchange(0);
                        const u64 u2 = Vulkan::g_dbg_upload_bytes.exchange(0);
                        LOG_WARNING(Render,
                                    "FLIPSTAT: flips={} draws(last100)={} dispatches(last100)={} "
                                    "upMB(last100)={} pend_gfx={} pend_asc={}",
                                    f2, d2, c2, u2 >> 20, num_gfx_submits.load(),
                                    num_asc_submits.load());
                    }
                }
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                    u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                if (rasterizer) {
                    rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Dma); // [CSBATCH]
                }
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value()) {
                    std::memcpy(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
                // [CSBARRIER] the game's sync point: the next dispatch keeps its barriers
                Vulkan::g_cs_sync_pending = true;
                // [ACQSTAT] what the game asks for: cp_coher_cntl histogram every 10 s
                {
                    static std::array<std::pair<u32, u64>, 8> hist{};
                    static u64 other = 0;
                    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                    const u32 cntl = acquire_mem->cp_coher_cntl;
                    bool hit = false;
                    for (auto& [v, n] : hist) {
                        if (n != 0 && v == cntl) {
                            ++n;
                            hit = true;
                            break;
                        }
                        if (n == 0) {
                            v = cntl;
                            n = 1;
                            hit = true;
                            break;
                        }
                    }
                    if (!hit)
                        ++other;
                    const auto now = std::chrono::steady_clock::now();
                    if (now >= next) {
                        next = now + std::chrono::seconds(10);
                        std::string line;
                        for (const auto& [v, n] : hist) {
                            if (n)
                                fmt::format_to(std::back_inserter(line), "{:#x}x{} ", v, n);
                        }
                        LOG_CRITICAL(Render_Vulkan,
                                     "ACQSTAT: gfx cp_coher_cntl {} other={} size={:#x}", line,
                                     other,
                                     (u64{acquire_mem->cp_coher_size_hi} << 32) |
                                         acquire_mem->cp_coher_size_lo);
                    }
                }
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // [CSBATCH] the guest is about to spin on a label; anything the batch owes must
                // be recorded and its held-back labels registered, or the wait never ends
                // (run 363 died during the load with the labels still queued).
                if (rasterizer) {
                    rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Sync);
                }
                if (Vulkan::g_cs_sync_mask & 2u) {
                    Vulkan::g_cs_sync_pending = true; // [CSBARRIER] game sync (label wait)
                }
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                {
                    static const bool s_stucklog = std::getenv("SHADPS4_STUCKLOG") != nullptr;
                    auto t_wait0 = std::chrono::steady_clock::now();
                    int s_logged = 0;
                    // A non-VO label can only be written by the guest; while we
                    // stall on it the frame-pacing wait must not hold the guest.
                    const bool guest_label = !vo_port->IsVoLabel(wait_addr);
                    bool stall_flagged = false;
                    // [SLOWWAIT] Publish the wait for the crash handler and account the
                    // SUSPENDED time per resume (so waits crossing 5s windows split right).
                    const bool sw_prof = Parserprof::Enabled();
                    bool sw_published = false;
                    bool eop_flushed = false;
                    while (!wait_reg_mem->Test(regs.reg_array)) {
                        if (EopRetire() && rasterizer && !eop_flushed) {
                            // [EOPRETIRE] the label may be a deferred EOP of this very queue:
                            // submit what is recorded so its tick can retire.
                            eop_flushed = true;
                            rasterizer->Flush();
                        }
                        if (guest_label && !stall_flagged) {
                            stall_flagged = true;
                            gfx_wrm_guest_stall.store(true, std::memory_order_release);
                        }
                        if (sw_prof && !sw_published) {
                            sw_published = true;
                            Postmortem::g_wrm_addr.store(reinterpret_cast<u64>(wait_addr),
                                                         std::memory_order_relaxed);
                            Postmortem::g_wrm_func.store((u32)wait_reg_mem->function.Value(),
                                                         std::memory_order_relaxed);
                            Postmortem::g_wrm_ref.store(wait_reg_mem->ref,
                                                        std::memory_order_relaxed);
                            Postmortem::g_wrm_mask.store(wait_reg_mem->mask,
                                                         std::memory_order_relaxed);
                            Postmortem::g_wrm_since_ms.store(Postmortem::NowMs(),
                                                             std::memory_order_relaxed);
                            Postmortem::g_gfx_state.store(1, std::memory_order_relaxed);
                        }
                        const s64 sw_t0 = sw_prof ? Parserprof::NowUs() : 0;
                        YIELD_GFX();
                        if (sw_prof) {
                            const s64 sw_t1 = Parserprof::NowUs();
                            if (sw_t1 > sw_t0) {
                                Parserprof::g_wrm_wait_us.fetch_add(static_cast<u64>(sw_t1 - sw_t0),
                                                                    std::memory_order_relaxed);
                            }
                            Postmortem::g_wrm_cur.store(
                                *reinterpret_cast<const volatile u32*>(wait_addr),
                                std::memory_order_relaxed);
                        }
                        if (s_stucklog && s_logged < 3) {
                            const auto sec = std::chrono::duration_cast<std::chrono::seconds>(
                                                 std::chrono::steady_clock::now() - t_wait0)
                                                 .count();
                            if (sec >= (s_logged + 1) * 5) {
                                ++s_logged;
                                LOG_WARNING(Render,
                                            "STUCK WaitRegMem: addr={:#x} func={} ref={:#x} "
                                            "mask={:#x} mem={:#x} vo_label={} waited={}s",
                                            reinterpret_cast<u64>(wait_addr),
                                            (u32)wait_reg_mem->function.Value(), wait_reg_mem->ref,
                                            wait_reg_mem->mask,
                                            *reinterpret_cast<const u32*>(wait_addr),
                                            vo_port->IsVoLabel(wait_addr), sec);
                            }
                        }
                        // [R6 D5] WRMSTUCK: independent of SHADPS4_STUCKLOG (that env kept the old
                        // sensor dead in the field). Own capped LOG_CRITICAL under RINGSENSE -
                        // fires inside the 40s wedge to prove/deny the WaitRegMem-spin candidate.
                        static const bool pm_ringsense = [] {
                            const char* e = std::getenv("SHADPS4_RINGSENSE");
                            return e && e[0] == '1';
                        }();
                        if (pm_ringsense) {
                            static std::atomic<u32> pm_wrm_logged{0};
                            if (pm_wrm_logged.load(std::memory_order_relaxed) < 3) {
                                const auto sec = std::chrono::duration_cast<std::chrono::seconds>(
                                                     std::chrono::steady_clock::now() - t_wait0)
                                                     .count();
                                if (sec >= 5 &&
                                    pm_wrm_logged.fetch_add(1, std::memory_order_relaxed) < 3) {
                                    LOG_CRITICAL(
                                        Render,
                                        "WRMSTUCK: addr={:#x} func={} ref={:#x} mask={:#x} "
                                        "mem={:#x} vo_label={} waited={}s",
                                        reinterpret_cast<u64>(wait_addr),
                                        (u32)wait_reg_mem->function.Value(), wait_reg_mem->ref,
                                        wait_reg_mem->mask,
                                        *reinterpret_cast<const u32*>(wait_addr),
                                        vo_port->IsVoLabel(wait_addr), sec);
                                }
                            }
                        }
                    }
                    if (stall_flagged) {
                        gfx_wrm_guest_stall.store(false, std::memory_order_release);
                    }
                    // [SLOWWAIT] Wait completed: back to running; >500ms gets a line (cap 32).
                    if (sw_published) {
                        Postmortem::g_gfx_state.store(0, std::memory_order_relaxed);
                    }
                    if (sw_prof) {
                        const s64 waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  std::chrono::steady_clock::now() - t_wait0)
                                                  .count();
                        if (waited_ms > 500 && Parserprof::g_slowwait_lines.fetch_add(
                                                   1, std::memory_order_relaxed) < 32) {
                            LOG_CRITICAL(Render,
                                         "SLOWWAIT: queue=gfx addr={:#x} func={} ref={:#x} "
                                         "mask={:#x} cur={:#x} waited={}ms seq={}",
                                         reinterpret_cast<u64>(wait_addr),
                                         (u32)wait_reg_mem->function.Value(), wait_reg_mem->ref,
                                         wait_reg_mem->mask,
                                         *reinterpret_cast<const volatile u32*>(wait_addr),
                                         waited_ms,
                                         Postmortem::g_opring_seq.load(std::memory_order_relaxed));
                        }
                    }
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (rasterizer) {
                    rasterizer->CpSync();
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (flip_signal_armed) {
        // The flip NOP sits near the end of the DCB; every packet after it
        // (frame tick EOP / label WriteData, any of the four PrepareFlip
        // variants) has executed by now, so the flip event can be delivered
        // without waiting for the submit coroutine to retire.
        flip_signal_armed = false;
        ++retired_flip_submits;
        static const bool s_fc = std::getenv("SHADPS4_STUCKLOG") != nullptr;
        if (s_fc) {
            static u32 s_fc_no = 0;
            LOG_WARNING(Render, "FLIPCHAIN sig #{}", ++s_fc_no);
        }
        {
            // [FRAMECHAIN] in-task flip handler (inside gfx_resume, outside every packet)
            Parserprof::FcTimer fc_t{Parserprof::g_fc_flip_task_ns};
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
        }
        if (Parserprof::Enabled()) {
            Parserprof::g_fc_flip_task_n.fetch_add(1, std::memory_order_relaxed);
        }
        static u32 s_flip_no3 = 0;
        const u32 f3 = ++s_flip_no3;
        Vulkan::g_dbg_frame_no.store(f3);
        static const char* s_ae4 = std::getenv("SHADPS4_AUTOSHOT_EVERY");
        static const u32 s_ae4v = s_ae4 ? (u32)std::strtoul(s_ae4, nullptr, 10) : 0;
        if (s_ae4v && (f3 % s_ae4v) == 0) {
            VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
        }
        if ((f3 % 100) == 0) {
            const u32 d3 = Vulkan::g_dbg_draws.exchange(0);
            const u32 c3 = Vulkan::g_dbg_dispatches.exchange(0);
            const u64 u3 = Vulkan::g_dbg_upload_bytes.exchange(0);
            LOG_WARNING(Render,
                        "FLIPSTAT: flips={} draws(last100)={} dispatches(last100)={} "
                        "upMB(last100)={} pend_gfx={} pend_asc={}",
                        f3, d3, c3, u3 >> 20, num_gfx_submits.load(), num_asc_submits.load());
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    // [FRAMECHAIN] Kick latency: SubmitAsc stamped the queue when it was empty and this is
    // the task's first resume (initial_suspend), so the delta is how long the guest's kick
    // waited for the round-robin. Consumed once (exchange); top-level tasks only.
    if constexpr (!is_indirect) {
        if (Parserprof::Enabled() && vqid < Parserprof::kFcAscQueues) {
            const s64 fc_stamp =
                Parserprof::g_asc_submit_stamp_ns[vqid].exchange(0, std::memory_order_relaxed);
            if (fc_stamp != 0) {
                const s64 fc_now = Parserprof::NowNs();
                if (fc_now > fc_stamp) {
                    Parserprof::g_fc_asc_kick_lat_ns.fetch_add(static_cast<u64>(fc_now - fc_stamp),
                                                               std::memory_order_relaxed);
                }
                Parserprof::g_fc_asc_kick_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    auto& queue = asc_queues[{vqid}];
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        // [SLOWWAIT p.3] Per-ASC-queue liveness for the crash handler (packets + last stamp).
        const bool fc_prof = Parserprof::Enabled();
        if (fc_prof && vqid < 8) {
            Postmortem::g_asc_pkts[vqid].fetch_add(1, std::memory_order_relaxed);
            Postmortem::g_asc_last_ms[vqid].store(Postmortem::NowMs(), std::memory_order_relaxed);
        }
        // [FRAMECHAIN] Progress marker for the spin detector in Process() + ASC packet count,
        // then the per-packet timer (PktTimer at file scope, `asc` flavour): dispatches get
        // the phase split, everything else its class total.
        if (fc_prof) {
            if (vqid < Parserprof::kFcAscQueues) {
                Parserprof::g_fc_asc_pkt_seq[vqid].fetch_add(1, std::memory_order_relaxed);
            }
            Parserprof::g_fc_asc_pkts.fetch_add(1, std::memory_order_relaxed);
        }
        PktTimer pkt_timer{static_cast<u32>(opcode), fc_prof ? Parserprof::NowUs() : 0, true};

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            Vulkan::g_cs_sync_pending = true; // [CSBARRIER]
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (fc_prof) {
                    Parserprof::g_fc_asc_dispatches.fetch_add(1, std::memory_order_relaxed);
                }
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (fc_prof) {
                    Parserprof::g_fc_asc_dispatches.fetch_add(1, std::memory_order_relaxed);
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                    rasterizer->DispatchDirect();
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchDirect();
                }
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (fc_prof) {
                    Parserprof::g_fc_asc_dispatches.fetch_add(1, std::memory_order_relaxed);
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                }
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            if (rasterizer) {
                rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Dma); // [CSBATCH]
            }
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            if (rasterizer) {
                rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Sync); // [CSBATCH]
            }
            if (Vulkan::g_cs_sync_mask & 2u) {
                Vulkan::g_cs_sync_pending = true; // [CSBARRIER] game sync on the compute queue
            }
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            // [SLOWWAIT] Completion-only line for compute queues (>500ms, shared 32-line cap).
            const auto sw_asc_t0 = Parserprof::Enabled() ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};
            bool eop_flushed = false;
            while (!wait_reg_mem->Test(regs.reg_array)) {
                if (EopRetire() && rasterizer && !eop_flushed) {
                    // [EOPRETIRE] same as the GFX wait: let the deferred label's tick retire.
                    eop_flushed = true;
                    rasterizer->Flush();
                }
                // [FRAMECHAIN] Suspended time of this queue inside the wait (mirror of the
                // GFX g_wrm_wait_us): wall between yield and resume, other tasks ran meanwhile.
                const s64 fc_w0 = fc_prof ? Parserprof::NowNs() : 0;
                YIELD_ASC(vqid);
                if (fc_prof && vqid < Parserprof::kFcAscQueues) {
                    const s64 fc_w1 = Parserprof::NowNs();
                    if (fc_w1 > fc_w0) {
                        Parserprof::g_asc_wait_ns[vqid].fetch_add(static_cast<u64>(fc_w1 - fc_w0),
                                                                  std::memory_order_relaxed);
                    }
                }
            }
            if (sw_asc_t0 != std::chrono::steady_clock::time_point{}) {
                const s64 waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - sw_asc_t0)
                                          .count();
                if (waited_ms > 500 &&
                    Parserprof::g_slowwait_lines.fetch_add(1, std::memory_order_relaxed) < 32) {
                    const u64* sw_addr = wait_reg_mem->Address<u64*>();
                    LOG_CRITICAL(Render,
                                 "SLOWWAIT: queue=asc{} addr={:#x} func={} ref={:#x} mask={:#x} "
                                 "cur={:#x} waited={}ms seq={}",
                                 vqid, reinterpret_cast<u64>(sw_addr),
                                 (u32)wait_reg_mem->function.Value(), wait_reg_mem->ref,
                                 wait_reg_mem->mask,
                                 *reinterpret_cast<const volatile u32*>(sw_addr), waited_ms,
                                 Postmortem::g_opring_seq.load(std::memory_order_relaxed));
                }
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            // [CSBATCH-DEFERLABEL] SHADPS4_CSBATCH_DEFERLABEL=1: the job system writes a label
            // after EVERY worker dispatch, and SignalFence writes it at parse time - so flushing
            // the batch here (the safe default) makes every batch one job (run 359: label=60695,
            // subs/batch=1) and the per-dispatch front-end cost stays. With the flag the label is
            // deferred to the retirement of the tick that carries the batch instead: the batch is
            // recorded into the current cmdbuf, the tick is submitted, and only after the GPU has
            // executed it does the guest see "done". A guest wait (WAIT_REG_MEM) still flushes the
            // batch first (Reason::Sync), so nobody can spin on a label whose work is unsubmitted.
            static const bool csb_defer_label = [] {
                const char* e = std::getenv("SHADPS4_CSBATCH_DEFERLABEL");
                const bool v = e && e[0] == '1';
                if (v) {
                    LOG_CRITICAL(Render, "[CSBATCH] DEFERLABEL on: RELEASE_MEM labels wait for the "
                                         "tick that carries the batch");
                }
                return v;
            }();
            bool csb_deferred_label = false;
            if (rasterizer) {
                if (csb_defer_label && release_mem->data_sel.Value() != DataSelect::GdsMemStore) {
                    const PM4CmdReleaseMem copy = *release_mem;
                    const u32 pipe_id = queue.pipe_id;
                    csb_deferred_label = rasterizer->SkinBatchDeferLabel([copy, pipe_id] {
                        copy.SignalFence(
                            [pipe_id] {
                                Platform::IrqC::Instance()->Signal(
                                    static_cast<Platform::InterruptId>(pipe_id));
                            },
                            [](VAddr, u16, u16) {});
                    });
                }
                if (!csb_deferred_label) {
                    rasterizer->FlushSkinBatch(Vulkan::SkinBatch::Reason::Label); // [CSBATCH]
                }
            }
            static const bool s_trace_fence2 = std::getenv("SHADPS4_TRACEFENCE") != nullptr;
            if (s_trace_fence2) {
                LOG_CRITICAL(Render, "RELM q={} addr={:#x} val={:#x}", queue.pipe_id,
                             release_mem->Address<u64>(), release_mem->DataQWord());
            }
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
            // Same reasoning as the gfx EOP above: compute completion interrupts must not
            // outrun the dispatches they announce.
            static const bool relm_irq_at_retire = [] {
                const char* e = std::getenv("SHADPS4_EOPIRQRETIRE");
                return e && e[0] == '1';
            }();
            if (EopRetireLevel() == 1 && rasterizer &&
                release_mem->data_sel.Value() != DataSelect::GdsMemStore) {
                // [EOPRETIRE] whole fence (label + IRQ) at retire, in queue order after the
                // readback writeback. The packet is copied: the ring may be reused meanwhile.
                // GDS stores keep the parse-time path (CopyBuffer is parser-thread only).
                rasterizer->ProcessFaultBufferIfPending(); // replay op ahead of the fence
                const PM4CmdReleaseMem copy = *release_mem;
                const u32 pipe_id = queue.pipe_id;
                rasterizer->DeferSignalToRetire([copy, pipe_id] {
                    copy.SignalFence(
                        [pipe_id] {
                            Platform::IrqC::Instance()->Signal(
                                static_cast<Platform::InterruptId>(pipe_id));
                        },
                        [](VAddr, u16, u16) {});
                });
                // Deferred ops take CurrentTick = the tick of the cmdbuf being recorded, so
                // register first, then submit that tick; a Flush before the registration would
                // make the fence wait for a tick nobody submits until the next packet.
                rasterizer->Flush();
                break;
            }
            if (csb_deferred_label) {
                break; // the fence is queued behind the batch (registered when it is recorded)
            }
            release_mem->SignalFence(
                [this, pipe_id = queue.pipe_id] {
                    const auto raise = [pipe_id] {
                        Platform::IrqC::Instance()->Signal(
                            static_cast<Platform::InterruptId>(pipe_id));
                    };
                    if (relm_irq_at_retire && rasterizer) {
                        rasterizer->DeferSignalToRetire(raise);
                    } else {
                        raise();
                    }
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            if (Vulkan::g_cs_sync_mask & 4u) {
                Vulkan::g_cs_sync_pending = true; // [CSBARRIER] game sync on the compute queue
            }
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    // [FRAMECHAIN] Guest half of the lock-step: how long after the parser drained (GpuIdle
    // stamp) did the next GFX submit arrive. Counted once per drain (seq), relaxed reads.
    if (Parserprof::Enabled() && num_gfx_submits.load() == 0) {
        const u64 fc_seq = Parserprof::g_fc_gpuidle_seq.load(std::memory_order_acquire);
        if (fc_seq != 0 &&
            fc_seq != Parserprof::g_fc_idle_seq_seen.load(std::memory_order_relaxed)) {
            Parserprof::g_fc_idle_seq_seen.store(fc_seq, std::memory_order_relaxed);
            const s64 fc_stamp = Parserprof::g_fc_gpuidle_stamp_ns.load(std::memory_order_relaxed);
            const s64 fc_now = Parserprof::NowNs();
            if (fc_stamp != 0 && fc_now > fc_stamp) {
                Parserprof::g_fc_idle_to_submit_ns.fetch_add(static_cast<u64>(fc_now - fc_stamp),
                                                             std::memory_order_relaxed);
                Parserprof::g_fc_idle_to_submit_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // PS4-подобный бэкпрешер: ограничение глубины конвейера гостевых
    // сабмитов закрывает гонку m_gfxEopTick — движок не может убежать
    // от парсера дальше, чем на N сабмитов, и EOP-тики старых кадров
    // успевают доехать до гостевой памяти до переиспользования слота.
    static const u32 submit_depth = [] {
        const char* e = std::getenv("SHADPS4_SUBDEPTH");
        return e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    if (submit_depth > 0) {
        WaitForSubmitDepth(submit_depth);
    }

    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    ++num_gfx_submits;
    Postmortem::g_submits_in.fetch_add(1, std::memory_order_relaxed);
    Postmortem::g_num_gfx_submits.store(num_gfx_submits.load(), std::memory_order_relaxed);
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        // [FRAMECHAIN] A kick into an idle queue: stamp it, the task's first resume reads it.
        if (queue.submits.empty() && Parserprof::Enabled() && vqid < Parserprof::kFcAscQueues) {
            Parserprof::g_asc_submit_stamp_ns[vqid].store(Parserprof::NowNs(),
                                                          std::memory_order_relaxed);
        }
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    ++num_asc_submits;
    Postmortem::g_submits_in.fetch_add(1, std::memory_order_relaxed);
    Postmortem::g_num_asc_submits.store(num_asc_submits.load(), std::memory_order_relaxed);
    submit_cv.notify_one();
}

} // namespace AmdGpu
