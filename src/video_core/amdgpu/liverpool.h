// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include "common/logging/log.h"
#include "common/postmortem.h"
#include <coroutine>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include <functional>
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

extern std::atomic<u64> g_flips_materialized;


struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0; });
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    // Block until the number of in-flight GFX submissions drops to max_pending.
    // Unlike WaitGpuIdle this lets the GPU thread keep working while the
    // guest prepares the next frame (bounded pipelining instead of lockstep).
    // Only the gfx ring is counted: async-compute submissions may legitimately
    // stay parked in WaitRegMem waiting for guest CPU writes, so gating on
    // them would deadlock.
    void WaitForSubmitDepth(u32 max_pending) {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this, max_pending] { return num_gfx_submits <= max_pending; });
    }

    // Frame pacing: block until the number of flip-carrying submits still in
    // flight (requested by the guest but not yet retired) drops to max_pending.
    // Emulates console pacing where the CPU can never run more than the
    // render-frame ring ahead of the GPU. Does not require the ring to drain,
    // so guest-side helper threads keep making progress.
    // Bounded wait: give the GPU thread a chance to retire the current frame's
    // flip submit (so its EOP ticks are visible to the guest's FrameBegin), but
    // never deadlock — during loading phases the ring can legitimately stall on
    // WaitRegMem for labels the guest writes only after returning from here.
    void WaitForFlipBacklog(u64 requested, u32 max_pending) {
        using namespace std::chrono_literals;
        std::unique_lock lk{submit_mutex};
        submit_cv.wait_for(lk, 2000ms, [this, requested, max_pending] {
            return requested - retired_flip_submits.load() <= max_pending;
        });
    }

    // Bounded drain of the gfx ring: unlike flip-backlog pacing this also
    // covers frames submitted without a flip (menu transitions). The timeout
    // makes guest-dependent stalls (labels written after submitDone) safe.
    // Combined frame pacing: wake as soon as EITHER this frame's flip DCB has
    // fully executed (its fences are written - the fast path for scenes that
    // flip every frame) OR the gfx ring is fully drained (covers no-flip
    // transition frames whose fences live in ordinary submits). Bounded to
    // avoid deadlocks on guest-dependent ring stalls.
    void WaitForFramePacing(u64 requested_flips) {
        using namespace std::chrono_literals;
        // Pace against flip MATERIALIZATION (request placed into the present
        // queue by the flip handler), not against the raw signal: the handler
        // runs after this frame's fences are written, so returning here
        // guarantees the guest's next FrameBegin sees valid EOP ticks.
        // Polling quanta instead of a cv: the counter is written by the
        // videoout driver which has no access to our submit_cv; 1 ms quanta
        // keep added latency negligible against a frame's lifetime.
        // Depth 0 (full parse of the current flip DCB) is required: letting the
        // guest run even one flip ahead re-opens the zero-EOP-tick assert
        // within minutes — the guest's own frame tick for frame N-1 is only
        // guaranteed written once the DCB that carries flip N has been parsed
        // (observed: depth-1 experiment asserted in the load phase, depth-0
        // survived it).
        // NOTE: a progress-based deadline (bail when the parser stops consuming
        // packets) was tried and reverted — the parser legitimately spends
        // whole seconds inside the flip handler (presenter fence wait) without
        // touching a packet, so "no heartbeat" misfires every heavy frame and
        // voids the depth-0 guarantee.
        // No fixed deadline: a level-load frame legitimately runs for minutes
        // (tens of thousands of draws + pipeline compiles), and any constant
        // we picked (2s, 10s, 120s) eventually expired mid-load and voided
        // the depth-0 guarantee (guest assert on a zero EOP tick). Instead
        // wait while the parser shows a heartbeat; 30s of a completely silent
        // parser (no packet consumed) is a genuine deadlock — bail then.
        // The 30s grace also covers presenter-fence waits inside the flip
        // handler, during which the parser consumes nothing by design.
        using clock = std::chrono::steady_clock;
        // [GUESTWAIT] THE blocking point of sceGnmSubmitDone when SHADPS4_SUBDEPTH >= 0. It had
        // no marker, which is why the stall snapshot could only say "sceGnmSubmitDone".
        Postmortem::GuestWaitScope wait_scope{
            "WaitForFramePacing", requested_flips,
            g_flips_materialized.load(std::memory_order_relaxed)};
        u64 last_beat = gfx_parse_heartbeat.load(std::memory_order_relaxed);
        auto last_progress = clock::now();
        // [SUBFIX] SHADPS4_SUBFIX=1 (default OFF). Deadlock signature: the gfx ring is fully
        // drained - every DCB parsed - yet the flip counter is still short. Nothing remains
        // that COULD raise GfxFlip, so no amount of waiting changes the outcome; meanwhile the
        // engine's own 2s fuse blows long before the 300s heartbeat bail. A short grace lets an
        // in-flight flip IRQ finish before we call it.
        // 0 = off, 1 = release only on a fully drained ring, 2 = also release when the whole
        // pipeline is provably frozen.
        static const u32 subfix = [] {
            const char* e = std::getenv("SHADPS4_SUBFIX");
            return e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
        }();
        static const u64 subfix_ms = [] {
            const char* e = std::getenv("SHADPS4_SUBFIX_MS");
            const u64 v = e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
            return v != 0 ? v : 250ull;
        }();
        const auto wait_started = clock::now();
        clock::time_point drained_since{};
        // [SUBFIX=2] The deadlock is circular: this thread waits for the flip, the flip waits for
        // the parser, the parser waits on a GPU semaphore, and the shader holding that semaphore
        // waits for a job only this thread can produce. Mode 1 cannot see it - its ring-drained
        // test is false by construction while three submits are still in flight.
        // The signature is total stasis: the parser consumed no packet AND no submit retired for
        // subfix_ms. That distinguishes it from the two legitimate silences the comment above
        // warns about - a long parse still moves the heartbeat, and a presenter fence wait still
        // retires submits. Releasing on silence alone would fire during level loads, which is
        // precisely what voided the depth-0 guarantee in earlier experiments.
        u64 frozen_beat = gfx_parse_heartbeat.load(std::memory_order_relaxed);
        u64 frozen_submits = num_gfx_submits.load();
        auto frozen_since = clock::now();
        while (true) {
            // BOTH conditions, not either: a submit without a flip (the guest
            // does plenty of those while a level loads) leaves the flip
            // counter satisfied while its frame-tick EOP is still sitting in
            // an unparsed DCB. Returning on the flip counter alone let the
            // guest's render-frame ring run ahead of its own ticks, which is
            // exactly the zero-tick assert. Waiting for the ring to drain
            // covers flip and no-flip frames alike.
            if (g_flips_materialized.load(std::memory_order_acquire) >= requested_flips &&
                num_gfx_submits.load() == 0) {
                // [fix-candidate] SHADPS4_PACERETIRE=1: "ring drained" above means the
                // PARSER consumed every packet, not that the GPU executed them. Flip
                // frames are safe (materialization implies the flip submit retired), but
                // no-flip frames - level loads and cutscene skips - returned here with
                // Vulkan work still in flight; the guest then read frame ticks its GPU
                // stores had not produced yet and tripped fg-draw-mgr.cpp:234. Hold the
                // no-flip path until the scheduler reports every submitted tick retired.
                static const bool pace_retire = [] {
                    const char* e = std::getenv("SHADPS4_PACERETIRE");
                    return e && e[0] == '1';
                }();
                if (!pace_retire || !gpu_retire_query || gpu_retire_query()) {
                    return;
                }
            }
            // NOTE: the instant escape on a guest-label WaitRegMem stall was
            // removed: with GPU direct memory access on, labels are written by
            // guest worker threads (not the thread blocked here), so a stall
            // is just a silent parser — the 30s heartbeat bail below already
            // covers it. The instant escape fired mid level-load and voided
            // the depth-0 guarantee (guest assert on a zero EOP tick).
            if (subfix != 0) {
                const u64 now_beat = gfx_parse_heartbeat.load(std::memory_order_relaxed);
                const u64 now_submits = num_gfx_submits.load();
                if (now_beat != frozen_beat || now_submits != frozen_submits) {
                    frozen_beat = now_beat;
                    frozen_submits = now_submits;
                    frozen_since = clock::now();
                }
                const u64 frozen_ms = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() -
                                                                          frozen_since)
                        .count());
                if (now_submits == 0) {
                    if (drained_since == clock::time_point{}) {
                        drained_since = clock::now();
                    }
                } else {
                    drained_since = clock::time_point{};
                }
                const bool drained = drained_since != clock::time_point{} &&
                                     clock::now() - drained_since > 250ms;
                const bool frozen = subfix >= 2 && frozen_ms >= subfix_ms;
                if (drained || frozen) {
                    Postmortem::g_subfix_released.fetch_add(1, std::memory_order_relaxed);
                    LOG_CRITICAL(Lib_GnmDriver,
                                 "SUBFIX: released site=framepacing reason={} flips={}/{} "
                                 "gfx_inflight={} heartbeat_age={}ms waited={}ms "
                                 "tick_write_age={}ms",
                                 drained ? "drained" : "parser-stalled",
                                 g_flips_materialized.load(std::memory_order_acquire),
                                 requested_flips, now_submits, frozen_ms,
                                 std::chrono::duration_cast<std::chrono::milliseconds>(
                                     clock::now() - wait_started)
                                     .count(),
                                 Postmortem::g_tick_last_write_ms.load(
                                     std::memory_order_relaxed)
                                     ? Postmortem::NowMs() -
                                           Postmortem::g_tick_last_write_ms.load(
                                               std::memory_order_relaxed)
                                     : -1);
                    return;
                }
            }
            const u64 beat = gfx_parse_heartbeat.load(std::memory_order_relaxed);
            if (beat != last_beat) {
                last_beat = beat;
                last_progress = clock::now();
            } else if (clock::now() - last_progress > 300000ms) {
                // 300s: during a level load the parser legitimately goes
                // silent for minutes inside the presenter fence wait (the GPU
                // is chewing through the load frame); 30s bailed mid-load and
                // voided the guarantee. A genuine deadlock now costs one
                // 5-minute hang before the bail — acceptable for a debug
                // stack, and the only remaining exit without the guarantee.
                return;
            }
            std::this_thread::sleep_for(1ms);
        }
    }

    bool IsGfxStalledOnGuestLabel() const {
        return gfx_wrm_guest_stall.load(std::memory_order_acquire);
    }

    void SetGpuRetireQuery(std::function<bool()>&& q) {
        gpu_retire_query = std::move(q);
    }

    // Wait until the gfx ring is fully drained, tolerating minutes-long but
    // alive parses (pipeline compiles); bail only after 300s of a completely
    // silent parser. Used to keep DIRECT (non-EOP) flips from outrunning the
    // parser during level loads — see VideoOutDriver::SubmitFlip.
    void WaitForGfxRingIdlePulse() {
        using namespace std::chrono_literals;
        using clock = std::chrono::steady_clock;
        u64 last_beat = gfx_parse_heartbeat.load(std::memory_order_relaxed);
        auto last_progress = clock::now();
        while (num_gfx_submits.load() != 0) {
            const u64 beat = gfx_parse_heartbeat.load(std::memory_order_relaxed);
            if (beat != last_beat) {
                last_beat = beat;
                last_progress = clock::now();
            } else if (clock::now() - last_progress > 300000ms) {
                return;
            }
            std::this_thread::sleep_for(1ms);
        }
    }

    void WaitForGfxDrainBounded() {
        using namespace std::chrono_literals;
        std::unique_lock lk{submit_mutex};
        submit_cv.wait_for(lk, 2000ms, [this] { return num_gfx_submits.load() == 0; });
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if (std::this_thread::get_id() == gpu_id) {
            return func();
        }
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    void ProcessCommands();
    void Process(std::stop_token stoken);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_gfx_submits{};
    std::atomic<u32> num_asc_submits{};
    // Flip signals deferred until the containing gfx submit retires (GPU thread only).
    u32 pending_flip_signals{};
    bool flip_signal_armed{};
    // Count of flip-carrying gfx submits fully retired (all their packets,
    // including frame EOP ticks, executed).
    std::atomic<u64> retired_flip_submits{};
    // Gfx parser is stalled on a WaitRegMem label that only the guest can
    // write (not a VO label). Frame pacing must release the guest in this
    // state, otherwise every frame costs the full pacing timeout.
    std::atomic<bool> gfx_wrm_guest_stall{false};
    // Ticks once per parsed gfx packet — lets the frame-pacing wait
    // distinguish a busy parser (keep waiting) from a stuck one (bail out).
    std::atomic<u64> gfx_parse_heartbeat{};
    // Set by the renderer at init: returns true when every Vulkan tick submitted
    // so far has retired on the GPU. Used by the SHADPS4_PACERETIRE no-flip hold.
    std::function<bool()> gpu_retire_query;
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
    s32 curr_qid{-1};
};

} // namespace AmdGpu
