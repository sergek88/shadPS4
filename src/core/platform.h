// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/postmortem.h"
#include "common/singleton.h"
#include "common/types.h"

#include <magic_enum/magic_enum.hpp>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <queue>

namespace Platform {

enum class InterruptId : u32 {
    Compute0RelMem = 0x00,
    Compute1RelMem = 0x01,
    Compute2RelMem = 0x02,
    Compute3RelMem = 0x03,
    Compute4RelMem = 0x04,
    Compute5RelMem = 0x05,
    Compute6RelMem = 0x06,
    GfxEop = 0x40,
    GfxFlip = 0x08,
    GpuIdle = 0x09,

    InterruptIdMax = 0x40, ///< Max possible value (GfxEop)
};

using IrqHandler = std::function<void(InterruptId)>;

// FIFO (ticket) mutex. Drop-in replacement for std::mutex under std::unique_lock
// (satisfies BasicLockable/Lockable), but grants ownership strictly in arrival
// order.
//
// Why: std::mutex makes no fairness guarantees. The GPU thread signals IRQs in
// tight bursts (e.g. the pending-flip drain loop in liverpool.cpp), re-acquiring
// the per-context lock on every iteration. Because handlers run *under* that
// lock (intentionally - see IrqContext below), an unfair mutex lets the signaler
// win the re-acquire race indefinitely, and a guest thread parked in
// RegisterOnce() was observed starving for 30+ seconds during such storms.
//
// With a ticket lock the guest waits for AT MOST ONE handler invocation before
// getting its turn, while the uncontended path stays a couple of atomic-ish ops
// (mutex + counter), so inline signal delivery latency is unaffected.
//
// Deliberately NOT lock-free/spinning: waiters sleep on a condition_variable,
// and the waiter count here is tiny (signaler + guest + occasional
// register/unregister), so the notify_all() broadcast is cheap.
class FifoMutex {
public:
    FifoMutex() = default;
    FifoMutex(const FifoMutex&) = delete;
    FifoMutex& operator=(const FifoMutex&) = delete;

    void lock() {
        std::unique_lock lk{m};
        const u64 my_ticket = next_ticket++;
        cv.wait(lk, [&] { return my_ticket == now_serving; });
    }

    bool try_lock() {
        std::unique_lock lk{m};
        if (next_ticket != now_serving) {
            return false; // Held or someone is already queued ahead of us.
        }
        ++next_ticket;
        return true;
    }

    void unlock() {
        {
            std::scoped_lock lk{m};
            ++now_serving;
        }
        // Broadcast: each waiter waits on its own ticket predicate, so only the
        // next-in-line proceeds; the rest go back to sleep. Waiter count is
        // small (see class comment), thundering herd is not a concern here.
        cv.notify_all();
    }

private:
    std::mutex m;
    std::condition_variable cv;
    u64 next_ticket{0}; // Ticket handed to the next lock() caller.
    u64 now_serving{0}; // Ticket currently owning the mutex.
};

struct IrqController {
    void RegisterOnce(InterruptId irq, IrqHandler handler) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Invalid IRQ number");
        auto& ctx = irq_contexts.try_emplace(irq).first->second;
        std::unique_lock lock{ctx.m_lock};
        ctx.one_time_subscribers.emplace(std::move(handler));
    }

    void Register(InterruptId irq, IrqHandler handler, void* uid) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Invalid IRQ number");
        auto& ctx = irq_contexts.try_emplace(irq).first->second;

        std::unique_lock lock{ctx.m_lock};
        ASSERT_MSG(ctx.persistent_handlers.find(uid) == ctx.persistent_handlers.cend(),
                   "The handler is already registered!");
        ctx.persistent_handlers.emplace(uid, std::move(handler));
    }

    void Unregister(InterruptId irq, void* uid) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Invalid IRQ number");
        auto& ctx = irq_contexts.try_emplace(irq).first->second;
        std::unique_lock lock{ctx.m_lock};
        ctx.persistent_handlers.erase(uid);
    }

    void Signal(InterruptId irq) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Unexpected IRQ signaled");
        auto& ctx = irq_contexts.try_emplace(irq).first->second;
        // [postmortem] Branch-C sensor: the FifoMutex convoy is invisible to the
        // AV wrapper, so measure the EOP acquire and stamp dispatch completion.
        const bool pm_eop = irq == InterruptId::GfxEop;
        const auto pm_t0 = std::chrono::steady_clock::now();
        std::unique_lock lock{ctx.m_lock};
        if (pm_eop) {
            const u64 pm_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - pm_t0)
                                  .count();
            Postmortem::AtomicMax(Postmortem::g_eop_lock_wait_max_us, pm_us);
            Postmortem::g_eop_lock_wait_us.fetch_add(pm_us, std::memory_order_relaxed);
        }

        LOG_TRACE(Core, "IRQ signaled: {}", magic_enum::enum_name(irq));

        for (auto& [uid, h] : ctx.persistent_handlers) {
            h(irq);
        }

        if (!ctx.one_time_subscribers.empty()) {
            const auto& h = ctx.one_time_subscribers.front();
            h(irq);

            ctx.one_time_subscribers.pop();
        }
        if (pm_eop) {
            Postmortem::g_eop_dispatch_ms.store(Postmortem::NowMs(), std::memory_order_relaxed);
        }
    }

private:
    // Handlers are invoked while holding m_lock. This is intentional and
    // load-bearing, do not "optimize" it away:
    //
    // 1. Rendezvous / backpressure: returning from RegisterOnce() (and thus
    //    from sceGnmSubmitAndFlipCommandBuffers) proves every previously
    //    signaled flip has fully materialized (present request queued,
    //    flip_status/gcQueueNum accounted). Moving handler bodies outside the
    //    lock lets the guest run ahead of the flip pipeline: it then trips
    //    "Out of order flip IRQ" (buffer label reuse racing the label clear in
    //    VideoOutDriver::Flip) or reads a zero EOP timestamp after the 2s
    //    frame-pacing timeout. Both were reproduced within a minute when
    //    handlers ran unlocked.
    //
    // 2. Unregister() contract: after Unregister() returns, the handler for
    //    that uid is guaranteed to not be executing (callers rely on this
    //    before freeing the uid-associated context).
    //
    // The starvation this used to cause is fixed by making the lock FIFO
    // (see FifoMutex), not by shrinking the critical section: a waiter now
    // gets the lock after at most one handler invocation instead of losing
    // the re-acquire race for an entire signal storm.
    struct IrqContext {
        std::unordered_map<void*, IrqHandler> persistent_handlers{};
        std::queue<IrqHandler> one_time_subscribers{};
        FifoMutex m_lock{};
    };
    // NOTE: Pre-existing (inherited from upstream, not introduced here) race:
    // concurrent try_emplace() on irq_contexts from different threads is
    // unsynchronized. In practice every InterruptId is first touched during
    // single-threaded init, so the map is fully populated before concurrency
    // starts. Left as-is on purpose to keep this change minimal; fix separately.
    std::unordered_map<InterruptId, IrqContext> irq_contexts{};
};

using IrqC = Common::Singleton<IrqController>;

} // namespace Platform