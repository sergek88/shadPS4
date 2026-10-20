// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// [postmortem] Crash-time telemetry for the fg-draw-mgr.cpp:234 frame-tick assert.
// Behaviour-neutral: only relaxed atomics and, in the crash handler, load()s - no
// exchanges (the live periodic reporters keep their own windows), no locks, no
// allocations. Lives next to the g_armed_watch_address watchpoint machinery.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include "common/types.h"

namespace Postmortem {

inline s64 NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_POSTMORTEM");
        return e && e[0] == '1';
    }();
    return on;
}

inline void AtomicMax(std::atomic<u64>& a, u64 v) {
    u64 cur = a.load(std::memory_order_relaxed);
    while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

// Tick-writer (guest thread "GPU Interrupts Handler GfxEop").
inline std::atomic<u32> g_tick_writer_tid{0};
inline std::atomic<s64> g_tick_last_write_ms{0}; // last NON-zero tick store
inline std::atomic<s64> g_tick_last_zero_ms{0};  // last legitimate zeroing (cleaner)

// AV wrapper of the tick-writer: one point covers ALL our blocking locks (fault
// entry is single-point - the only registrar is page_manager's Impl).
inline std::atomic<u64> g_tw_av_count{0}, g_tw_av_us{0}, g_tw_av_max_us{0};
inline std::atomic<s64> g_tw_inflight_since_ms{0}; // 0 = not currently in a fault

// Branch A (parser) and branch C (delivery).
inline std::atomic<s64> g_eop_signal_ms{0};        // moment Signal(GfxEop) left the parser
inline std::atomic<s64> g_eop_dispatch_ms{0};      // moment the handler loop finished
inline std::atomic<u64> g_eop_lock_wait_max_us{0}; // FifoMutex convoy, worst acquire
inline std::atomic<u64> g_eop_lock_wait_us{0};     // FifoMutex convoy, summed acquires
// [Stage1 PARSERPROF] worst EOP-signal gap in ms - the direct "silence window" metric
// (verdict target <= 100 ms). Always tracked (pure telemetry, no behaviour change).
inline std::atomic<u64> g_eop_gap_max_ms{0};
// [R6 D5] Parser GFX heartbeat: last opcode entering the DCB switch + a monotonic packet
// sequence. Written only under SHADPS4_RINGSENSE. In the 40s wedge: seq frozen + op=WaitRegMem
// => spin; seq frozen on another op / seq advancing => fence-starvation (not the parser).
inline std::atomic<u32> g_gfx_cur_op{0};
inline std::atomic<u64> g_gfx_op_seq{0};

// Postmortem hooks: fixed array, registered at start-up, invoked from the handler.
using DumpFn = void (*)();
inline std::atomic<DumpFn> g_dumps[8]{};

inline void RegisterPostmortemDump(DumpFn fn) {
    for (auto& s : g_dumps) {
        DumpFn expected = nullptr;
        if (s.compare_exchange_strong(expected, fn, std::memory_order_relaxed)) {
            return;
        }
    }
}

inline void RunPostmortemDumps() {
    for (auto& s : g_dumps) {
        if (DumpFn fn = s.load(std::memory_order_relaxed)) {
            fn();
        }
    }
}

} // namespace Postmortem

// [Stage1 PARSERPROF] SHADPS4_PARSERPROF=1: shader-compile-path profiling on the parser
// thread. All counters are relaxed atomics; when the flag is off nothing but a cached bool
// is read on the hot path (bit-identical to the reference build).
namespace Parserprof {
inline bool Enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_PARSERPROF");
        return e && e[0] == '1';
    }();
    return on;
}
inline std::atomic<u64> g_prof_pipe_us{0};   // total compile-path time on a permutation miss
inline std::atomic<u64> g_compile_count{0};  // number of shader-permutation compiles
inline std::atomic<u64> g_specmiss_count{0}; // recompiles of an already-cached pgm_hash

// [Stage1b] Per-packet wall time of the parser loop: the fatal EOP gap was not
// covered by any TEXTIME/compile bucket, so attribute it at the source. Buckets
// by opcode class + the single worst packet ever seen.
inline std::atomic<u64> g_pkt_draw_us{0};  // draws + dispatches (rasterizer work)
inline std::atomic<u64> g_pkt_wrm_us{0};   // WaitRegMem polls
inline std::atomic<u64> g_pkt_evw_us{0};   // EventWrite* (EOP/EOS paths)
inline std::atomic<u64> g_pkt_dma_us{0};   // DmaData
inline std::atomic<u64> g_pkt_other_us{0}; // everything else
inline std::atomic<u64> g_pkt_max_us{0};   // worst single packet
inline std::atomic<u32> g_pkt_max_op{0};   // its opcode
inline std::atomic<u64> g_evw_dl_us{0};    // ProcessDownloadImages inside EventWrite*
inline std::atomic<u64> g_evw_fin_us{0};   // rasterizer->Finish() on EOS GdsStore
inline std::atomic<u64> g_fb_wait_us{0};   // fault-buffer area wait (scheduler.Wait)
// [A-EOPONLY] download-drain call shares by drain site, to measure the EOS fraction
// before deciding the expected win, and to count EOS drains skipped under EOPONLY.
inline std::atomic<u64> g_dl_eos_calls{0}, g_dl_eop_calls{0}, g_dl_eos_skipped{0};
// [MINGAP] EOP drains skipped by the SHADPS4_DLIMG_MINGAP_MS throttle.
inline std::atomic<u64> g_dl_gap_skipped{0};
inline void AccountPacket(u32 op, u64 us) {
    switch (op) {
    case 0x15: // DispatchDirect
    case 0x16: // DispatchIndirect
    case 0x24: // DrawIndirect
    case 0x25: // DrawIndexIndirect
    case 0x27: // DrawIndex2
    case 0x2C: // DrawIndirectMulti
    case 0x2D: // DrawIndexAuto
    case 0x30: // DrawIndexMultiAuto
    case 0x35: // DrawIndexOffset2
    case 0x38: // DrawIndexIndirectMulti
        g_pkt_draw_us.fetch_add(us, std::memory_order_relaxed);
        break;
    case 0x3C: // WaitRegMem
        g_pkt_wrm_us.fetch_add(us, std::memory_order_relaxed);
        break;
    case 0x46: // EventWrite
    case 0x47: // EventWriteEop
    case 0x48: // EventWriteEos
        g_pkt_evw_us.fetch_add(us, std::memory_order_relaxed);
        break;
    case 0x50: // DmaData
        g_pkt_dma_us.fetch_add(us, std::memory_order_relaxed);
        break;
    default:
        g_pkt_other_us.fetch_add(us, std::memory_order_relaxed);
        break;
    }
    u64 prev = g_pkt_max_us.load(std::memory_order_relaxed);
    while (us > prev && !g_pkt_max_us.compare_exchange_weak(prev, us, std::memory_order_relaxed)) {
    }
    if (us > prev) {
        g_pkt_max_op.store(op, std::memory_order_relaxed);
    }
}

// [SLOWPKT] EXCLUSIVE per-phase attribution inside one parser packet: a probe switches the
// current bucket and its destructor switches back, so nested probes subtract naturally and the
// bucket sum always equals the packet total. thread_local + armed flag = zero effect on any
// other thread and (one cached-bool test per probe) near-zero when PARSERPROF is off or the
// packet is not a draw/EOP.
enum PktPhase : u32 {
    PhCompile = 0, // pipeline cache: shader compile + spec key + vkCreate*Pipelines
    PhTex,         // texture cache: FindImage/FindTexture/RefreshImage (upload, detile)
    PhBuf,         // buffer cache: ObtainBuffer/CreateBuffer/binds
    PhDesc,        // BindResources minus the nested tex/buf work
    PhVk,          // the rest of Rasterizer::Draw*/Dispatch* (vkCmd recording, barriers)
    PhDrain,       // 0x47 only: ProcessDownloadImages at the EOP site
    PhSignal,      // 0x47 only: SignalFence (fence write + GfxEop IRQ -> guest callback)
    PhFlip,        // 0x47 only: flip block (GfxFlip IRQ -> guest callback, flipstat)
    PhHash,        // XXH3 over guest image memory (texture content check)
    // [FRAMECHAIN] The DMA block of Rasterizer::BindResources (once-per-flip full sweep +
    // per-dispatch DMASYNC_ZONE syncs). Nested inside PhDesc, so `desc` is now exclusive of
    // it: compare with pre-W0 windows as desc + dma.
    PhDma,
    // Breakdown of what used to be one `vk` bucket. The remainder stays in PhVk.
    PhBind,  // vkCmdBindPipeline + dynamic state
    PhRp,    // BeginRendering/EndRendering, incl. any render-pass rebuild
    PhBarr,  // pipeline->BindResources: descriptor writes + vkCmdPipelineBarrier
    PhDraw,  // the vkCmdDraw*/vkCmdDispatch call itself
    PhSched, // scheduler Flush / command-buffer turnover
    PhOther, // remainder of the packet (decode, regs, everything unprobed)
    PhCount,
};
// [HOSTSTALL-PKT] What the gfx parser is chewing on right now. PktTimer publishes it; the
// watchdog reads it to turn "heartbeat stopped" into "stuck in opcode X for N ms".
inline std::atomic<u32> g_cur_op{0};
inline std::atomic<u64> g_cur_seq{0};
inline std::atomic<s64> g_cur_pkt_us{0}; // packet start, microseconds (0 = not in a packet)
inline std::atomic<u32> g_cur_phase{PhOther};
inline std::atomic<s64> g_cur_phase_us{0}; // when the current phase was entered

inline thread_local u64 t_ph_us[PhCount];
inline thread_local u32 t_ph_cur = PhOther;
inline thread_local s64 t_ph_stamp = 0;
inline thread_local bool t_ph_active = false;
// [FRAMECHAIN] The armed phase clock belongs to an ASC dispatch (ProcessCompute's PktTimer):
// its sums go to g_ph_win_asc_us and it stays out of the HOSTSTALL-PKT publication.
inline thread_local bool t_pkt_asc = false;
// Per-packet counts that a duration alone cannot explain: a slow packet with 400 barriers is a
// different bug from a slow packet with 2.
inline thread_local u32 t_pkt_barriers = 0;
inline thread_local u32 t_pkt_writes = 0;
inline thread_local u32 t_pkt_binds = 0;

// [WRITESTAT] Split buffers by whether the GUEST ever writes them, not by size. Counted
// incrementally on state transitions - no iteration over the cache, so it is safe to leave on in
// any mode. Bytes are the deciding figure: if what the guest writes is a few megabytes, those
// ranges can live in shared memory and everything else can stay in VRAM, and neither side then
// needs synchronising.
inline std::atomic<u64> g_ws_total_bufs{0};
inline std::atomic<u64> g_ws_total_bytes{0};
inline std::atomic<u64> g_ws_written_bufs{0};
inline std::atomic<u64> g_ws_written_bytes{0};
inline std::atomic<u64> g_ws_late_bufs{0}; // still written after the load phase
inline std::atomic<u64> g_ws_late_bytes{0};

// [TOUCHSTAT] What the GPU actually reaches for, per frame, split by binding direction.
//
// WARNING about the WRITESTAT block above: it does NOT measure guest writes. Its counter sits
// under `if (is_written)` in ObtainBuffer, and `is_written` is the SHADER DESCRIPTOR flag "bound
// for writing" - a write from the GPU side, not from the guest CPU. It also adds
// buffer.SizeBytes() once, in full, so its total is a FOOTPRINT, not per-frame traffic. Anything
// derived from "the guest rewrites N MB" is unfounded.
//
// These counters answer the question that actually decides buffer placement:
//   bound_*   - sum of the requested size over every ObtainBuffer call (with repeats)
//   unique_*  - sum of buffer sizes counted ONCE per frame, on the first touch of that buffer.
//               This is exactly what a "snapshot into VRAM on first bind" scheme would have to
//               copy per frame, so it is the upper bound on that idea's cost.
//   ro / rw   - is_written false / true, i.e. GPU reads vs GPU writes.
//   stream    - the <=16 KiB read-only path that already goes to the device-local stream buffer
//               and never touches the backing import.
// bound/unique is the re-read multiplier: high means many binds hit the same buffer in a frame.
inline std::atomic<u64> g_ts_bound_ro{0};
inline std::atomic<u64> g_ts_bound_rw{0};
inline std::atomic<u64> g_ts_unique_ro{0};
inline std::atomic<u64> g_ts_unique_rw{0};
inline std::atomic<u64> g_ts_unique_bufs{0};
inline std::atomic<u64> g_ts_stream_bytes{0};
inline std::atomic<u64> g_ts_stream_calls{0};

/// Frame after which a write counts as "live" rather than "load-time". SHADPS4_WRITESTAT_EARLY.
inline u64 WriteStatEarlyFrames() {
    static const u64 n = [] {
        const char* e = std::getenv("SHADPS4_WRITESTAT_EARLY");
        const u64 v = e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
        return v != 0 ? v : 600ull;
    }();
    return n;
}

// [PIPESTAT] Why the pipeline changes ~900 times a frame. Every switch is compared field group
// by field group against the previous key; ALL differing groups are counted, not just the first,
// so the totals can exceed the switch count - that is deliberate, a switch caused by three
// changes at once should show up in all three.
// [BININFO 11.09] попадания и промахи запоминания заголовка шейдера (SearchBinaryInfo)
inline std::atomic<u64> g_bininfo_hits{0};
inline std::atomic<u64> g_bininfo_misses{0};
inline std::atomic<u64> g_pipe_switches{0};
inline std::atomic<u64> g_pipe_created{0};
inline std::atomic<u64> g_pipe_d_shaders{0};
inline std::atomic<u64> g_pipe_d_blend{0};
inline std::atomic<u64> g_pipe_d_rtfmt{0};
inline std::atomic<u64> g_pipe_d_vertex{0};
inline std::atomic<u64> g_pipe_d_depth{0};
inline std::atomic<u64> g_pipe_d_other{0};
inline std::atomic<u64> g_pipe_unique{0};

// [RPSTAT/BARSTAT/BINDSTAT/DESCSTAT] Why state changes cost 9x more than the draws themselves.
// All plain relaxed counters on paths that already run; they answer "how many", which a
// microsecond total cannot.
inline std::atomic<u64> g_rp_calls{0};       // scheduler.BeginRendering called
inline std::atomic<u64> g_rp_starts{0};      // ...and actually began a pass (state differed)
inline std::atomic<u64> g_rp_attach{0};      // ...with a different attachment SET
inline std::atomic<u64> g_rp_ends{0};        // EndRendering that closed a live pass
inline std::atomic<u64> g_bar_calls{0};      // BindResources invocations that carried barriers
inline std::atomic<u64> g_bar_buffers{0};    // buffer barriers submitted
inline std::atomic<u64> g_bind_calls{0};     // vkCmdBindPipeline issued
inline std::atomic<u64> g_bind_real{0};      // ...where the pipeline actually differed
inline std::atomic<u64> g_desc_writes{0};    // descriptor writes handed to the driver
inline std::atomic<u64> g_desc_redundant{0}; // ...identical to what that slot already held
inline std::atomic<u64> g_frames_counted{0};

// [PKTHIST] Draw-class packet duration distribution over the 5s window. Hitches live in the
// tail, and an average hides them completely.
inline std::atomic<u64> g_pkt_hist[5]{}; // <0.1ms, 0.1-1, 1-5, 5-20, >20

inline s64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void PhaseSwitch(u32 next) {
    const s64 now = NowUs();
    if (t_ph_stamp != 0 && now > t_ph_stamp) {
        t_ph_us[t_ph_cur] += static_cast<u64>(now - t_ph_stamp);
    }
    t_ph_stamp = now;
    t_ph_cur = next;
    // [FRAMECHAIN] ASC dispatches keep their phases to themselves: this publication describes
    // the GFX packet named by g_cur_op, which may be parked in WaitRegMem meanwhile.
    if (!t_pkt_asc) {
        g_cur_phase.store(next, std::memory_order_relaxed);
        g_cur_phase_us.store(now, std::memory_order_relaxed);
    }
}

inline const char* PhaseName(u32 p) {
    switch (p) {
    case PhCompile:
        return "compile";
    case PhTex:
        return "tex";
    case PhBuf:
        return "buf";
    case PhDesc:
        return "desc";
    case PhVk:
        return "vk";
    case PhDrain:
        return "drain";
    case PhSignal:
        return "signal";
    case PhFlip:
        return "flip";
    case PhHash:
        return "hash";
    case PhDma:
        return "dma";
    case PhBind:
        return "bind";
    case PhRp:
        return "rp";
    case PhBarr:
        return "barr";
    case PhDraw:
        return "draw";
    case PhSched:
        return "sched";
    default:
        return "other";
    }
}

struct PhaseProbe {
    u32 prev = PhOther;
    bool armed = false;
    explicit PhaseProbe(u32 idx) {
        if (t_ph_active) {
            armed = true;
            prev = t_ph_cur;
            PhaseSwitch(idx);
        }
    }
    PhaseProbe(const PhaseProbe&) = delete;
    PhaseProbe& operator=(const PhaseProbe&) = delete;
    ~PhaseProbe() {
        if (armed) {
            PhaseSwitch(prev);
        }
    }
};

inline bool IsDrawClassOp(u32 op) {
    switch (op) {
    case 0x15:
    case 0x16:
    case 0x24:
    case 0x25:
    case 0x27:
    case 0x2C:
    case 0x2D:
    case 0x30:
    case 0x35:
    case 0x38:
        return true;
    default:
        return false;
    }
}

// 5-second-window phase sums for the draw class (exchanged by the PARSERPROF reporter).
inline std::atomic<u64> g_ph_win_us[PhCount]{};
// SLOWPKT/SLOWEOP line cap (32 per run), then a silent counter the summary can expose.
inline std::atomic<u64> g_slow_lines{0};
inline std::atomic<u64> g_slow_suppressed{0};

// [SLOWWAIT] Wall time the GFX stream spent SUSPENDED inside WaitRegMem (accumulated per
// resume, so a wait crossing the 5s window boundary lands in the right windows), and wall time
// the parser thread slept on the submit condvar with nothing queued at all.
inline std::atomic<u64> g_wrm_wait_us{0};
inline std::atomic<u64> g_starve_us{0};
inline std::atomic<u64> g_slowwait_lines{0};

// [FRAMECHAIN] W0: where the parser thread's wall goes beyond the packet timers (~26 ms/frame
// that no probe covered in run 149). Every counter is a relaxed atomic exchanged to 0 by the
// 5 s PARSERPROF reporter; everything that feeds them sits behind Parserprof::Enabled(), so
// with SHADPS4_PARSERPROF unset nothing here is touched. Durations are accumulated in
// NANOSECONDS (`_ns`): the resume/spin probes fire thousands of times per frame and truncating
// each to a microsecond would lose milliseconds per frame. The reporter prints us per frame.
inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline std::atomic<u32> g_parser_tid{0}; // host tid of GpuCommandProcessor (page_manager compares)
// Round-robin in Liverpool::Process.
inline std::atomic<u64> g_fc_gfx_resume_ns{0}; // GFX coroutine running (all packets, waits incl.)
inline std::atomic<u64> g_fc_asc_resume_ns{0}; // ASC coroutine running and advancing a packet
inline std::atomic<u64> g_fc_asc_resumes{0};   // all ASC resumes
inline std::atomic<u64> g_fc_asc_spin_resumes{0}; // ...that yielded again without a new packet
inline std::atomic<u64> g_fc_asc_spin_ns{0};      // wall of those spin resumes
inline std::atomic<u64> g_fc_asc_pkts{0};         // type-3 ASC packets processed
inline std::atomic<u64> g_fc_asc_dispatches{0};   // ASC dispatches handed to the rasterizer
inline std::atomic<u64> g_fc_asc_kick_lat_ns{0};  // SubmitAsc into an idle queue -> first resume
inline std::atomic<u64> g_fc_asc_kick_n{0};
inline std::atomic<u64> g_fc_gfx_empty_iters{
    0};                                     // GFX polled with nothing queued while submits pend
inline std::atomic<u64> g_fc_loop_iters{0}; // round-robin iterations (queue polls)
inline std::atomic<u64> g_fc_loop_ns{0};    // wall inside the round-robin loop
// Per ASC queue: the guest's kick stamp (SubmitAsc writes it when the queue was empty, the
// task's first resume consumes it), a packet sequence for the spin detector, and the time the
// queue spent suspended inside WaitRegMem (mirror of g_wrm_wait_us; wall, not CPU).
inline constexpr u32 kFcAscQueues = 64; // Liverpool::NumTotalQueues < 64
inline std::atomic<s64> g_asc_submit_stamp_ns[kFcAscQueues]{};
inline std::atomic<u64> g_fc_asc_pkt_seq[kFcAscQueues]{};
inline std::atomic<u64> g_asc_wait_ns[kFcAscQueues]{};
// submit_done boundary and the handlers that run on the parser outside any task.
inline std::atomic<u64> g_fc_onsubmit_ns{0};
inline std::atomic<u64>
    g_fc_os_step_ns[8]{}; // 0 prof 1 prefetch 2 label 3 fault 4 dlimg 5 texgc 6 bufgc
inline std::atomic<u64> g_fc_flush_ns{0}, g_fc_flush_n{0};
inline std::atomic<u64> g_fc_flip_ns{0}, g_fc_flip_n{0}; // Signal(GfxFlip) from the retire loop
inline std::atomic<u64> g_fc_flip_task_ns{0},
    g_fc_flip_task_n{0}; // ...from inside the GFX task (EOP, DCB end)
inline std::atomic<u64> g_fc_gpuidle_ns{0}, g_fc_gpuidle_n{0}; // Signal(GpuIdle) handler
inline std::atomic<u64> g_fc_cmd_ns{0}, g_fc_cmd_n{0}; // ProcessCommands callbacks from the loop
inline std::atomic<u64> g_fc_cmd_task_ns{0},
    g_fc_cmd_task_n{0}; // ...from inside a task (inside resume time)
inline std::atomic<u64> g_fc_frame_wait_ns{0},
    g_fc_frame_wait_n{0}; // presenter GetRenderFrame waits
// Guest half of the lock-step: how long after the parser drained did the next GFX submit come.
inline std::atomic<s64> g_fc_gpuidle_stamp_ns{
    0}; // a stamp, never zeroed; the guest reads it relaxed
inline std::atomic<u64> g_fc_gpuidle_seq{
    0}; // bumped with the stamp: the guest counts each drain once
inline std::atomic<u64> g_fc_idle_seq_seen{0}; // guest side: last drain already accounted
inline std::atomic<u64> g_fc_idle_to_submit_ns{0}, g_fc_idle_to_submit_n{0};
// Guest timers (gnmdriver.cpp: sceGnmSubmitDone / SubmitCommandBuffersForWorkload / DingDong).
// Ungated by design - 2 NowUs per call at 1-3 calls per frame - and therefore in microseconds.
inline std::atomic<u64> g_fc_guest_sd_wait_us{0}, g_fc_guest_sd_interval_us{0}, g_fc_guest_sd_n{0},
    g_fc_guest_submit_wait_us{0}, g_fc_guest_submit_n{0}, g_fc_guest_dd_wait_us{0},
    g_fc_guest_dd_n{0};
// Small items sized before anyone optimises them (measures #5 and #10).
inline std::atomic<u64> g_fc_pop_ns{0},
    g_fc_pop_n{0}; // scheduler.PopPendingOperations per draw/dispatch
inline std::atomic<u64> g_fc_clamp_ns{0}, g_fc_clamp_n{0}; // ClampRangeSize for V# >= 1 GiB
// Faults / VirtualProtect served on the parser thread itself (page_manager.cpp).
inline std::atomic<u64> g_fc_parser_faults{0}, g_fc_parser_protects{0};
// ASC packets, timed apart from the GFX buckets (AccountPacket stays GFX-only so `other` keeps
// its meaning): `asc` = packets that ran to completion in one go, `asc_wait` = the wait-class
// packets (WaitRegMem / MemSemaphore / Rewind) whose wall includes suspended time. Microseconds,
// like the GFX packet buckets they mirror.
inline std::atomic<u64> g_pkt_asc_us{0}, g_pkt_asc_n{0};
inline std::atomic<u64> g_pkt_asc_wait_us{0}, g_pkt_asc_wait_n{0};
inline std::atomic<u64> g_ph_win_asc_us[PhCount]{}; // phase sums of ASC dispatches (us)
// (t_pkt_asc lives next to t_ph_active above - PhaseSwitch reads it.)
inline thread_local bool t_fc_in_task = false; // inside task.resume() (ProcessCommands attribution)
inline std::atomic<u64> g_slow_asc_lines{0};   // SLOWPKT-ASC cap (16 per run)
// DMA block of BindResources.
inline std::atomic<u64> g_dma_sweeps{0};    // full mapped-range sweeps (once per flip)
inline std::atomic<u64> g_dma_zone_runs{0}; // DMASYNC_ZONE range syncs (per DMA dispatch)

/// [FRAMECHAIN] Scope timer that reads no clock at all unless SHADPS4_PARSERPROF is set (or the
/// caller passes its own already-evaluated gate). Nanoseconds into the sink.
struct FcTimer {
    std::atomic<u64>* sink;
    std::chrono::steady_clock::time_point t0;
    explicit FcTimer(std::atomic<u64>& s) : FcTimer{s, Enabled()} {}
    FcTimer(std::atomic<u64>& s, bool on) : sink{on ? &s : nullptr} {
        if (sink != nullptr) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    FcTimer(const FcTimer&) = delete;
    FcTimer& operator=(const FcTimer&) = delete;
    ~FcTimer() {
        if (sink == nullptr) {
            return;
        }
        sink->fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - t0)
                                             .count()),
                        std::memory_order_relaxed);
    }
};
} // namespace Parserprof

// [SLOWWAIT p.3] GFX parser state snapshot for the crash handler. The parser thread is the
// only writer (relaxed stores); the handler only loads - no exchanges, locks or allocations.
namespace Postmortem {
inline std::atomic<u32> g_gfx_state{0}; // 0=running 1=waiting(WaitRegMem) 2=starved(idle)
inline std::atomic<u64> g_wrm_addr{0};
inline std::atomic<u32> g_wrm_func{0};
inline std::atomic<u32> g_wrm_ref{0};
inline std::atomic<u32> g_wrm_mask{0};
inline std::atomic<u32> g_wrm_cur{0}; // refreshed on every resume - safe for the handler,
                                      // which must not touch guest memory itself
inline std::atomic<s64> g_wrm_since_ms{0};
// Last 8 processed GFX packets: [op:8][seq:32][start-ms low 24 bits]. Own monotonic seq (the
// RINGSENSE one is env-gated). Pushed at packet start so the packet being processed at death
// is the newest entry.
inline std::atomic<u64> g_opring_seq{0};
inline std::atomic<u64> g_opring[8]{};
inline void OpRingPush(u32 op) {
    const u64 seq = g_opring_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 packed = (u64{op & 0xFF} << 56) | ((seq & 0xFFFFFFFFull) << 24) |
                       (static_cast<u64>(NowMs()) & 0xFFFFFFull);
    g_opring[seq & 7].store(packed, std::memory_order_relaxed);
}
// Per-ASC-queue liveness: packets processed + last activity stamp.
inline std::atomic<u64> g_asc_pkts[8]{};
inline std::atomic<s64> g_asc_last_ms[8]{};

// [GUESTSTALL] Registry of live GUEST threads (tid + name), filled by RunThread. Needed to
// snapshot them during a stall without dbghelp: OpenThread/Suspend/GetThreadContext only.
// Fixed storage, no allocation, no locks - the name is written BEFORE the tid is published so
// a reader never sees a half-written name.
inline constexpr u32 kMaxGuestThreads = 96;
struct GuestThreadSlot {
    std::atomic<u32> tid{0};
    char name[48]{};
    // [GUESTWAIT] Which emulated kernel primitive this thread is parked on. A host RIP inside
    // ntdll says only "it is waiting"; this says WHAT for. Written by the thread itself, read by
    // the stall watchdog, so it has to live in the shared slot rather than a thread_local.
    std::atomic<const char*> wait_what{nullptr}; // static literal, never freed
    std::atomic<u64> wait_a0{0};
    std::atomic<u64> wait_a1{0};
    std::atomic<s64> wait_since_ms{0};
};
inline GuestThreadSlot g_guest_threads[kMaxGuestThreads];
inline thread_local GuestThreadSlot* t_my_slot = nullptr;

/// Registry of OUR OWN named threads (parser, presenter, schedulers...). The stall snapshot
/// used to cover guest threads only, so the gfx parser - the thread that actually wedged - was
/// never visible. Same slot layout, filled from Common::SetCurrentThreadName.
inline constexpr u32 kMaxHostThreads = 64;
inline GuestThreadSlot g_host_threads[kMaxHostThreads];
inline thread_local GuestThreadSlot* t_host_slot = nullptr;

inline void RegisterHostThread(u32 tid, const char* name) {
    if (t_host_slot != nullptr) {
        return; // renamed later on - keep the first registration
    }
    for (u32 i = 0; i < kMaxHostThreads; ++i) {
        u32 expected = 0;
        if (!g_host_threads[i].tid.compare_exchange_strong(expected, 0xFFFFFFFFu,
                                                           std::memory_order_acq_rel)) {
            continue;
        }
        u32 k = 0;
        for (; name != nullptr && name[k] != '\0' && k < sizeof(g_host_threads[i].name) - 1; ++k) {
            g_host_threads[i].name[k] = name[k];
        }
        g_host_threads[i].name[k] = '\0';
        g_host_threads[i].wait_what.store(nullptr, std::memory_order_relaxed);
        t_host_slot = &g_host_threads[i];
        g_host_threads[i].tid.store(tid, std::memory_order_release);
        return;
    }
}

/// Marks the calling thread as parked on `what` for the scope's lifetime. Nests (an inner wait
/// restores the outer one on exit) and no-ops when the thread has no slot.
struct WaitScopeOn {
    GuestThreadSlot* slot;
    const char* prev_what;
    u64 prev_a0, prev_a1;
    s64 prev_since;
    WaitScopeOn(GuestThreadSlot* s, const char* what, u64 a0, u64 a1) : slot{s} {
        if (slot == nullptr) {
            return;
        }
        prev_what = slot->wait_what.load(std::memory_order_relaxed);
        prev_a0 = slot->wait_a0.load(std::memory_order_relaxed);
        prev_a1 = slot->wait_a1.load(std::memory_order_relaxed);
        prev_since = slot->wait_since_ms.load(std::memory_order_relaxed);
        slot->wait_a0.store(a0, std::memory_order_relaxed);
        slot->wait_a1.store(a1, std::memory_order_relaxed);
        slot->wait_since_ms.store(NowMs(), std::memory_order_relaxed);
        slot->wait_what.store(what, std::memory_order_release);
    }
    WaitScopeOn(const WaitScopeOn&) = delete;
    WaitScopeOn& operator=(const WaitScopeOn&) = delete;
    ~WaitScopeOn() {
        if (slot == nullptr) {
            return;
        }
        slot->wait_what.store(prev_what, std::memory_order_release);
        slot->wait_a0.store(prev_a0, std::memory_order_relaxed);
        slot->wait_a1.store(prev_a1, std::memory_order_relaxed);
        slot->wait_since_ms.store(prev_since, std::memory_order_relaxed);
    }
};

struct GuestWaitScope : WaitScopeOn {
    explicit GuestWaitScope(const char* what, u64 a0 = 0, u64 a1 = 0)
        : WaitScopeOn{t_my_slot, what, a0, a1} {}
};

/// Same thing for OUR threads - notably the gfx parser, which can block deep inside the Vulkan
/// driver with nothing in the log to say so.
struct HostWaitScope : WaitScopeOn {
    explicit HostWaitScope(const char* what, u64 a0 = 0, u64 a1 = 0)
        : WaitScopeOn{t_host_slot, what, a0, a1} {}
};

inline void RegisterGuestThread(u32 tid, const char* name) {
    for (u32 i = 0; i < kMaxGuestThreads; ++i) {
        u32 expected = 0;
        if (!g_guest_threads[i].tid.compare_exchange_strong(expected, 0xFFFFFFFFu,
                                                            std::memory_order_acq_rel)) {
            continue; // slot taken
        }
        u32 k = 0;
        for (; name != nullptr && name[k] != '\0' && k < sizeof(g_guest_threads[i].name) - 1; ++k) {
            g_guest_threads[i].name[k] = name[k];
        }
        g_guest_threads[i].name[k] = '\0';
        g_guest_threads[i].wait_what.store(nullptr, std::memory_order_relaxed);
        t_my_slot = &g_guest_threads[i];
        g_guest_threads[i].tid.store(tid, std::memory_order_release);
        return;
    }
}
inline void UnregisterGuestThread(u32 tid) {
    for (u32 i = 0; i < kMaxGuestThreads; ++i) {
        if (g_guest_threads[i].tid.load(std::memory_order_acquire) == tid) {
            g_guest_threads[i].wait_what.store(nullptr, std::memory_order_relaxed);
            t_my_slot = nullptr;
            g_guest_threads[i].tid.store(0, std::memory_order_release);
            return;
        }
    }
}

// [GUESTSTALL-TIME] Ages of the links in the frame loop. Everything is a plain relaxed stamp on
// an already-existing code path; the watchdog turns them into ages at snapshot time.
inline std::atomic<s64> g_last_submit_ms{0};      // sceGnmSubmitCommandBuffers*
inline std::atomic<s64> g_last_submit_flip_ms{0}; // ...AndFlipCommandBuffers*
inline std::atomic<s64> g_last_submit_done_ms{0}; // sceGnmSubmitDone
inline std::atomic<s64> g_last_flip_req_ms{0};    // guest asked for a flip
inline std::atomic<s64> g_last_flip_ms{0};        // presenter actually flipped
inline std::atomic<s64> g_last_vblank_ms{0};      // a guest thread RETURNED from WaitVblank
// Separate from the above on purpose: the presenter ticks vblank unconditionally, whether or
// not anyone waits on it. "guest never asked" and "we stopped producing" are different bugs and
// were indistinguishable while one stamp covered both.
inline std::atomic<s64> g_vblank_tick_ms{0};
inline std::atomic<u64> g_vblank_ticks{0};
inline std::atomic<u64> g_submits_total{0};
inline std::atomic<u64> g_flips_total{0};
inline std::atomic<u64> g_flip_reqs_total{0};
inline std::atomic<s32> g_last_flip_result{0};
// Flip pipeline gauges mirrored from VideoOutPort::flip_status.
inline std::atomic<s32> g_flip_pending{0};   // flip_pending_num
inline std::atomic<s32> g_flip_gc_queue{0};  // gc_queue_num
inline std::atomic<u64> g_present_queued{0}; // requests sitting in the present queue

// [GUESTSTALL-GPU] Submission accounting mirrors. All are plain relaxed stores on paths that
// already existed; the watchdog only loads them.
inline std::atomic<u32> g_submission_lock{0}; // gnmdriver's submitDone gate (a FLAG, not a count)
inline std::atomic<u64> g_submits_in{0};      // rings handed to the parser
inline std::atomic<u64> g_submits_done{0};    // rings the parser retired
inline std::atomic<s64> g_last_retire_ms{0};
inline std::atomic<u64> g_num_gfx_submits{0};
inline std::atomic<u64> g_num_asc_submits{0};
inline std::atomic<u64> g_flips_requested{0};    // sceGnmSubmitAndFlip* asked for a flip
inline std::atomic<u64> g_flips_materialized{0}; // the flip reached the present queue
inline std::atomic<u64> g_parse_heartbeat{0};    // packets consumed by the gfx parser
inline std::atomic<s64> g_heartbeat_ms{0};       // when it last moved
// Per-call-site accounting for the submission_lock flag, so an unbalanced site is named rather
// than guessed. 0 = sceGnmSubmitDone raises it, 1 = the GpuIdle handler clears it.
inline constexpr u32 kSubSites = 4;
inline std::atomic<u64> g_sub_inc[kSubSites]{};
inline std::atomic<u64> g_sub_dec[kSubSites]{};

// [STARTUP] Cost of the phases that run before the first frame, so a black screen before the
// Sony logo is attributable instead of guessed at.
inline std::atomic<s64> g_startup_ms{0}; // first stamp taken, used as t0
inline std::atomic<u64> g_startup_backimp_ms{0};
inline std::atomic<u64> g_startup_pcache_ms{0};

// [STARTUP2] Named milestones on the way to the first frame. A fixed table so any call site can
// add one with a single line and no new atomic.
inline constexpr u32 kMaxStartupMarks = 16;
struct StartupMark {
    std::atomic<const char*> name{nullptr};
    std::atomic<s64> ms{0};
};
inline StartupMark g_startup_marks[kMaxStartupMarks];
inline std::atomic<u64> g_module_load_ms{0};
inline std::atomic<u64> g_module_count{0};

inline s64 StartupT0() {
    s64 expected = 0;
    const s64 now = NowMs();
    g_startup_ms.compare_exchange_strong(expected, now);
    return g_startup_ms.load(std::memory_order_relaxed);
}

/// Records "how long after process start did <name> happen". First writer per name wins, so a
/// call site on a hot path costs one relaxed load once it is filled.
inline void MarkStartup(const char* name) {
    const s64 t0 = StartupT0();
    for (u32 i = 0; i < kMaxStartupMarks; ++i) {
        const char* cur = g_startup_marks[i].name.load(std::memory_order_acquire);
        if (cur == name) {
            return; // already recorded
        }
        if (cur == nullptr) {
            const char* expected = nullptr;
            if (g_startup_marks[i].name.compare_exchange_strong(expected, name,
                                                                std::memory_order_acq_rel)) {
                g_startup_marks[i].ms.store(NowMs() - t0, std::memory_order_release);
                return;
            }
        }
    }
}

// [BUFPROF] Sub-buckets of the `buf` phase. Scopes are deliberately NON-OVERLAPPING except for
// obtain/create, which nest: `lookup` is reported as obtain-minus-create.
inline std::atomic<u64> g_buf_obtain_us{0};
inline std::atomic<u64> g_buf_create_us{0};
inline std::atomic<u64> g_buf_overlap_us{0};
inline std::atomic<u64> g_buf_backing_off_us{0};
inline std::atomic<u64> g_buf_vk_create_us{0};
inline std::atomic<u64> g_buf_register_us{0};
inline std::atomic<u64> g_buf_creates{0};
inline std::atomic<u64> g_buf_obtains{0};

// [BACKIMPGC] Time the collector spent inside frames, per 5s window.
inline std::atomic<u64> g_gc_us{0};

// [RESSTAT-HIST] Size distribution of BACKING buffers at creation: <4K / 4-64K / 64K-1M / >1M.
inline std::atomic<u64> g_backing_hist[4]{};
// Driver limits worth knowing before blaming ourselves.
inline std::atomic<u64> g_max_mem_allocs{0};
inline std::atomic<u64> g_live_dedicated{0}; // non-backing cache buffers = dedicated allocations

// [BUFPROF] The checks that guard the stream-copy branch. They run on every one of the ~568k
// small constant fetches per 5s, so they may well cost more than the copies they guard.
inline std::atomic<u64> g_buf_gpumod_us{0};
inline std::atomic<u64> g_buf_labelint_us{0};
// Size distribution of those copies: <256B / 256B-1K / 1-4K / exactly 4K.
inline std::atomic<u64> g_stream_hist[4]{};
// [CONSTCACHE] Memoised stream copies.
inline std::atomic<u64> g_constcache_clears{0};       // cache flushes, any reason
inline std::atomic<u64> g_constcache_clear_frame{0};  // ...because the frame changed
inline std::atomic<u64> g_constcache_clear_wrap{0};   // ...because the ring wrapped
inline std::atomic<u64> g_constcache_cpumod{0};       // entry dropped: guest rewrote the source
inline std::atomic<u64> g_constcache_wrap_in_copy{0}; // ring wrapped inside our own Copy
inline std::atomic<u64> g_constcache_hits{0};
inline std::atomic<u64> g_constcache_misses{0};
inline std::atomic<u64> g_constcache_saved_us{0};

// [BUFPROF2] Everything in ObtainBuffer that no probe covered yet, plus the cost of probing
// itself. With ~830k calls per 5s window, a handful of clock reads per call is not noise: it is
// the same order as the work being measured, so it has to be quantified rather than assumed
// away.
inline std::atomic<u64> g_buf_entry_us{0};
inline std::atomic<u64> g_buf_lockwait_us{0};
inline std::atomic<u64> g_buf_written_us{0};
inline std::atomic<u64> g_buf_ret_us{0};
inline std::atomic<u64> g_buf_probes{0}; // clock reads charged to ObtainBuffer this window

/// Cost of one steady_clock::now() in nanoseconds, measured once. Every probe pays two.
inline u64 ProbeClockNs() {
    static const u64 ns = [] {
        constexpr u32 kIters = 4096;
        const auto t0 = std::chrono::steady_clock::now();
        for (u32 i = 0; i < kIters; ++i) {
            [[maybe_unused]] volatile auto t = std::chrono::steady_clock::now();
        }
        const auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
        return static_cast<u64>(total) / kIters;
    }();
    return ns;
}

/// SHADPS4_BUFPROF_LEVEL=0 turns the INNER ObtainBuffer probes off (the outer total stays), so
/// the same run can be compared with and without the instrumentation.
/// Default OFF. Full instrumentation cost 7.07M clock reads (~156 ms) per 5s window - the same
/// order as the work it measures. Turn it on deliberately with SHADPS4_BUFPROF_LEVEL=1 when a
/// breakdown is actually needed.
inline bool BufProfInner() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BUFPROF_LEVEL");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

/// UsTimer that can be compiled in but switched off at runtime, and that counts itself.
/// [FRAMECHAIN] Accumulates NANOSECONDS like UsTimer below (same sinks, same reporter).
struct UsTimerOpt {
    std::atomic<u64>* sink;
    std::chrono::steady_clock::time_point t0;
    explicit UsTimerOpt(std::atomic<u64>& s) : sink{BufProfInner() ? &s : nullptr} {
        if (sink != nullptr) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    UsTimerOpt(const UsTimerOpt&) = delete;
    UsTimerOpt& operator=(const UsTimerOpt&) = delete;
    ~UsTimerOpt() {
        if (sink == nullptr) {
            return;
        }
        sink->fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - t0)
                                             .count()),
                        std::memory_order_relaxed);
        g_buf_probes.fetch_add(2, std::memory_order_relaxed);
    }
};

// [BUFPROF] Breakdown INSIDE ObtainBuffer. FindBuffer is already O(1) (one page-table lookup
// plus a bounds check), so the half-second per 5s window has to be somewhere else; these split
// it instead of guessing.
inline std::atomic<u64> g_buf_streamcopy_us{0};
inline std::atomic<u64> g_buf_find_us{0};
inline std::atomic<u64> g_buf_sync_us{0};
inline std::atomic<u64> g_buf_streamcopies{0};

// [RESSTAT] Live resource counts, to make monotonic growth visible instead of inferred.
inline std::atomic<u64> g_res_buffers{0};
inline std::atomic<u64> g_res_backing_bufs{0};
inline std::atomic<u64> g_res_images{0};
inline std::atomic<u64> g_res_vram_mb{0};

/// Adds its lifetime to a counter. Cheap enough for per-call use on the parser thread.
/// [FRAMECHAIN] Accumulates NANOSECONDS: truncating every call to whole microseconds hid
/// ~45 % of ObtainBuffer (BUFPROF `lookup` 231 ms against PhBuf 434 ms per window) and made the
/// BUFPROF2 `unaccounted` figure meaningless. The sinks keep their `_us` names (the value the
/// reports show); the PARSERPROF reporter in liverpool.cpp divides by 1000 when printing. Any
/// NEW reader of a UsTimer/UsTimerOpt sink must do the same.
struct UsTimer {
    std::atomic<u64>* sink;
    std::chrono::steady_clock::time_point t0;
    explicit UsTimer(std::atomic<u64>& s) : sink{&s}, t0{std::chrono::steady_clock::now()} {}
    UsTimer(const UsTimer&) = delete;
    UsTimer& operator=(const UsTimer&) = delete;
    ~UsTimer() {
        sink->fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - t0)
                                             .count()),
                        std::memory_order_relaxed);
    }
};

// [SUBFIX] Times frame pacing was released early to break the CPU<->GPU deadlock.
inline std::atomic<u64> g_subfix_released{0};

// [HASHCAP] Whole-image hashes skipped because the range was too large to read on the parser
// thread. Surfaced in the 5s PARSERPROF summary.
inline std::atomic<u64> g_hashcap_skipped{0};

// [RETWATCH] SHADPS4_RETWATCH=1: ring of the last HLE returns that were -1. The crash we are
// chasing dereferenced 0xFFFFFFFFFFFFFFFF - an error code used as a pointer - so the question is
// which of our implementations handed the guest a -1 it did not expect. Recording only the -1
// cases keeps this to a compare on the hot path.
inline constexpr u32 kRetRing = 32;
struct RetRec {
    std::atomic<const char*> name{nullptr};
    std::atomic<s64> ret{0};
    std::atomic<u64> a0{0};
    std::atomic<u64> a1{0};
    std::atomic<s64> ms{0};
};
inline std::atomic<u64> g_ret_seq{0};
inline RetRec g_ret_ring[kRetRing];

inline bool RetWatchEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_RETWATCH");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

/// Call with the value about to be returned. Returns it unchanged so it can wrap a return
/// expression directly.
inline void NoteRetRaw(const char* name, s64 ret, u64 a0, u64 a1) {
    const u64 seq = g_ret_seq.fetch_add(1, std::memory_order_relaxed);
    RetRec& r = g_ret_ring[seq % kRetRing];
    r.ret.store(ret, std::memory_order_relaxed);
    r.a0.store(a0, std::memory_order_relaxed);
    r.a1.store(a1, std::memory_order_relaxed);
    r.ms.store(NowMs(), std::memory_order_relaxed);
    r.name.store(name, std::memory_order_release);
}

inline s64 NoteRet(const char* name, s64 ret, u64 a0 = 0, u64 a1 = 0) {
    if (!RetWatchEnabled() || ret != -1) {
        return ret;
    }
    const u64 seq = g_ret_seq.fetch_add(1, std::memory_order_relaxed);
    RetRec& r = g_ret_ring[seq % kRetRing];
    r.ret.store(ret, std::memory_order_relaxed);
    r.a0.store(a0, std::memory_order_relaxed);
    r.a1.store(a1, std::memory_order_relaxed);
    r.ms.store(NowMs(), std::memory_order_relaxed);
    r.name.store(name, std::memory_order_release);
    return ret;
}

// [CONDTRACE] Condition-variable signal ledger. A thread parked in pthread_cond_wait tells us
// nothing on its own; what settles the case is whether ANYONE signalled that condvar. Three
// outcomes are distinguishable with this: no signal ever (producer is the one stuck), signals
// but all older than the wait started (lost wakeup), signals newer than the wait (wrong waiter
// woken). Fixed storage, no locks, no allocation.
inline constexpr u32 kCondRing = 64;
struct CondSignalRec {
    std::atomic<u64> cond{0};
    std::atomic<s64> ms{0};
    std::atomic<u32> tid{0};
    std::atomic<u32> bcast{0};
};
inline std::atomic<u64> g_cond_ring_seq{0};
inline CondSignalRec g_cond_ring[kCondRing];

inline constexpr u32 kCondStats = 512;
struct CondStatSlot {
    std::atomic<u64> key{0};
    std::atomic<u64> count{0};
    std::atomic<s64> last_ms{0};
};
inline CondStatSlot g_cond_stats[kCondStats];

/// Open-addressed lookup keyed by the guest's pthread_cond_t address (same value the wait
/// marker records as a0). Never evicts - a diagnostic table, not a cache.
inline CondStatSlot* CondSlot(u64 cond, bool create) {
    const u32 h = static_cast<u32>(((cond >> 4) * 0x9E3779B9u) & (kCondStats - 1));
    for (u32 i = 0; i < 16; ++i) {
        CondStatSlot& s = g_cond_stats[(h + i) & (kCondStats - 1)];
        const u64 k = s.key.load(std::memory_order_acquire);
        if (k == cond) {
            return &s;
        }
        if (k == 0) {
            if (!create) {
                return nullptr;
            }
            u64 expected = 0;
            if (s.key.compare_exchange_strong(expected, cond, std::memory_order_acq_rel) ||
                s.key.load(std::memory_order_acquire) == cond) {
                return &s;
            }
        }
    }
    return nullptr; // table full for this bucket - diagnostics only, drop it
}

/// Tid of the calling guest thread as recorded at registration (0 for host threads). Avoids
/// dragging windows.h into headers that have no business including it.
inline u32 SelfGuestTid() {
    return t_my_slot != nullptr ? t_my_slot->tid.load(std::memory_order_relaxed) : 0u;
}

inline void NoteCondSignal(u64 cond, u32 tid, bool broadcast) {
    const s64 now = NowMs();
    if (CondStatSlot* s = CondSlot(cond, true)) {
        s->count.fetch_add(1, std::memory_order_relaxed);
        s->last_ms.store(now, std::memory_order_relaxed);
    }
    const u64 seq = g_cond_ring_seq.fetch_add(1, std::memory_order_relaxed);
    CondSignalRec& r = g_cond_ring[seq & (kCondRing - 1)];
    r.cond.store(cond, std::memory_order_relaxed);
    r.tid.store(tid, std::memory_order_relaxed);
    r.bcast.store(broadcast ? 1u : 0u, std::memory_order_relaxed);
    r.ms.store(now, std::memory_order_release);
}

// [GUESTSTALL] Loaded guest modules, so a RIP can be reported as eboot+0x... without dbghelp
// (symbolisation wedged the process before and is opt-in only).
inline constexpr u32 kMaxGuestModules = 32;
struct GuestModuleSlot {
    std::atomic<u64> base{0};
    std::atomic<u64> size{0};
    char name[40]{};
};
inline GuestModuleSlot g_guest_modules[kMaxGuestModules];
inline void RegisterGuestModule(const char* name, u64 base, u64 size) {
    for (u32 i = 0; i < kMaxGuestModules; ++i) {
        u64 expected = 0;
        if (!g_guest_modules[i].base.compare_exchange_strong(expected, ~u64{0},
                                                             std::memory_order_acq_rel)) {
            continue;
        }
        u32 k = 0;
        for (; name != nullptr && name[k] != '\0' && k < sizeof(g_guest_modules[i].name) - 1; ++k) {
            g_guest_modules[i].name[k] = name[k];
        }
        g_guest_modules[i].name[k] = '\0';
        g_guest_modules[i].size.store(size, std::memory_order_relaxed);
        g_guest_modules[i].base.store(base, std::memory_order_release);
        return;
    }
}
} // namespace Postmortem
