// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/memory_patcher.h"
#include "common/postmortem.h"
#include "common/signal_context.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/signals.h"
#include "emulator.h"

#ifdef _WIN32
#include <chrono>
#include <thread>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <windows.h>
static constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

namespace VideoCore {
extern std::atomic<u64> g_crash_addr_hint;
}
namespace Core {

#if defined(_WIN32)

// A hardware write watchpoint on a guest address: the only way to see which code writes
// a structure when the writer cannot be found by reading the code statically. Debug
// registers are per-thread, so every existing thread has to be armed.
u64 g_armed_watch_address = 0;

void ArmWriteWatch(u64 address) {
    static u64 armed_address = 0;
    if (armed_address == address) {
        return;
    }
    armed_address = address;
    g_armed_watch_address = address;
    const DWORD pid = GetCurrentProcessId();
    const HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return;
    }
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    u32 armed = 0;
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID != pid || te.th32ThreadID == GetCurrentThreadId()) {
                continue;
            }
            const HANDLE th =
                OpenThread(THREAD_SET_CONTEXT | THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE,
                           te.th32ThreadID);
            if (th == nullptr) {
                continue;
            }
            SuspendThread(th);
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (GetThreadContext(th, &ctx)) {
                ctx.Dr0 = address;
                // local enable for slot 0, break on write, 4-byte length
                ctx.Dr7 =
                    (ctx.Dr7 & ~u64{0xF0003}) | u64{1} | (u64{0b01} << 16) | (u64{0b11} << 18);
                ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
                if (SetThreadContext(th, &ctx)) {
                    ++armed;
                }
            }
            ResumeThread(th);
            CloseHandle(th);
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    LOG_CRITICAL(Debug, "WATCHARM: hardware write watch on {:#x}, {} threads armed", address,
                 armed);
}

// [CRASHSTAT] Census of the SKIPNULLCALL crutch: how often it fired and where it returned to.
static std::atomic<u32> g_skipnull_count{0};
static std::atomic<u64> g_skipnull_ret[4]{};
static std::atomic<s64> g_skipnull_last_ms{0};

// [CRASHSTAT] Persistent crash census across runs. A single access violation says little; the
// same RIP appearing in four out of four runs names a place. Written with raw WinAPI (no logger,
// no allocation) because this runs inside the fatal-fault path, and appended to a file so the
// statistics survive the process.
static void AppendCrashStat(EXCEPTION_POINTERS* pExp) {
    if (pExp == nullptr || pExp->ContextRecord == nullptr || pExp->ExceptionRecord == nullptr) {
        return;
    }
    const auto* ctx = pExp->ContextRecord;
    const u64 rip = ctx->Rip;

    // rip -> "<module>+0x..." from the MODBASE table; no dbghelp (it has wedged the process).
    char loc[96];
    loc[0] = '\0';
    for (u32 i = 0; i < Postmortem::kMaxGuestModules; ++i) {
        const u64 mbase = Postmortem::g_guest_modules[i].base.load(std::memory_order_acquire);
        if (mbase == 0 || mbase == ~u64{0}) {
            continue;
        }
        const u64 msize = Postmortem::g_guest_modules[i].size.load(std::memory_order_relaxed);
        if (rip >= mbase && rip < mbase + msize) {
            std::snprintf(loc, sizeof(loc), "%s+%#llx", Postmortem::g_guest_modules[i].name,
                          static_cast<unsigned long long>(rip - mbase));
            break;
        }
    }
    if (loc[0] == '\0') {
        std::snprintf(loc, sizeof(loc), "%s", rip >= 0x700000000000ull ? "host" : "unknown");
    }

    const DWORD tid = GetCurrentThreadId();
    const char* tname = "?";
    for (u32 i = 0; i < Postmortem::kMaxGuestThreads; ++i) {
        if (Postmortem::g_guest_threads[i].tid.load(std::memory_order_acquire) == tid) {
            tname = Postmortem::g_guest_threads[i].name;
            break;
        }
    }

    char key[128];
    std::snprintf(key, sizeof(key), "rip=%s", loc);

    // Count how many times this exact site was already recorded.
    u32 seen = 1;
    {
        const HANDLE rf = CreateFileA("C:\\Users\\user\\crashes.log", GENERIC_READ,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
        if (rf != INVALID_HANDLE_VALUE) {
            static char buf[512 * 1024];
            DWORD got = 0;
            if (ReadFile(rf, buf, sizeof(buf) - 1, &got, nullptr) && got > 0) {
                buf[got] = '\0';
                const char* p = buf;
                while ((p = std::strstr(p, key)) != nullptr) {
                    // Only a full token match: "rip=eboot.bin+0x123" must not match "...+0x1234".
                    const char after = p[std::strlen(key)];
                    if (after == ' ' || after == '\0' || after == '\r' || after == '\n') {
                        ++seen;
                    }
                    p += 1;
                }
            }
            CloseHandle(rf);
        }
    }

    char line[512];
    const int len = std::snprintf(
        line, sizeof(line),
        "CRASHSTAT: code=%#lx rip=%s thread=%s tid=%lu seen=%u | raw_rip=%#llx rsp=%#llx "
        "fault_addr=%#llx rax=%#llx rcx=%#llx rdx=%#llx rbx=%#llx rsi=%#llx rdi=%#llx | "
        "skipnull=%u last_at=%lldms_ago rets=%#llx,%#llx,%#llx,%#llx\r\n",
        pExp->ExceptionRecord->ExceptionCode, loc, tname, tid, seen,
        static_cast<unsigned long long>(rip), static_cast<unsigned long long>(ctx->Rsp),
        pExp->ExceptionRecord->NumberParameters >= 2
            ? static_cast<unsigned long long>(pExp->ExceptionRecord->ExceptionInformation[1])
            : 0ull,
        static_cast<unsigned long long>(ctx->Rax), static_cast<unsigned long long>(ctx->Rcx),
        static_cast<unsigned long long>(ctx->Rdx), static_cast<unsigned long long>(ctx->Rbx),
        static_cast<unsigned long long>(ctx->Rsi), static_cast<unsigned long long>(ctx->Rdi),
        g_skipnull_count.load(std::memory_order_relaxed),
        g_skipnull_last_ms.load(std::memory_order_relaxed)
            ? static_cast<long long>(Postmortem::NowMs() -
                                     g_skipnull_last_ms.load(std::memory_order_relaxed))
            : -1LL,
        static_cast<unsigned long long>(g_skipnull_ret[0].load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_skipnull_ret[1].load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_skipnull_ret[2].load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_skipnull_ret[3].load(std::memory_order_relaxed)));

    // [RETWATCH] Append the last few -1 returns; the newest one is the prime suspect.
    if (Postmortem::RetWatchEnabled()) {
        const s64 now_ms = Postmortem::NowMs();
        char rw[512];
        int n = std::snprintf(rw, sizeof(rw), "CRASHSTAT-RET: last -1 returns (newest last):");
        const u64 seq = Postmortem::g_ret_seq.load(std::memory_order_relaxed);
        const u64 start = seq >= 8 ? seq - 8 : 0;
        for (u64 k = start; k < seq; ++k) {
            const Postmortem::RetRec& r = Postmortem::g_ret_ring[k % Postmortem::kRetRing];
            const char* nm = r.name.load(std::memory_order_acquire);
            if (nm == nullptr || n >= static_cast<int>(sizeof(rw)) - 72) {
                continue;
            }
            n += std::snprintf(
                rw + n, sizeof(rw) - n, " %s(a0=%#llx,a1=%#llx)@%lldms_ago", nm,
                static_cast<unsigned long long>(r.a0.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(r.a1.load(std::memory_order_relaxed)),
                static_cast<long long>(now_ms - r.ms.load(std::memory_order_relaxed)));
        }
        std::snprintf(rw + n, sizeof(rw) - n, "\r\n");
        const HANDLE rf = CreateFileA("C:\\Users\\user\\crashes.log", FILE_APPEND_DATA,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
        if (rf != INVALID_HANDLE_VALUE) {
            DWORD w = 0;
            WriteFile(rf, rw, static_cast<DWORD>(std::strlen(rw)), &w, nullptr);
            CloseHandle(rf);
        }
        LOG_CRITICAL(Debug, "{}", rw);
    }

    const HANDLE f = CreateFileA("C:\\Users\\user\\crashes.log", FILE_APPEND_DATA,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        WriteFile(f, line, len > 0 ? static_cast<DWORD>(len) : 0, &written, nullptr);
        CloseHandle(f);
    }
    LOG_CRITICAL(Debug, "{}", line);
}

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    // Re-entrancy guard: a fault raised while this handler is already running on
    // the same thread means the handler itself crashed - without the guard that
    // recurses until the process dies with NOTHING logged (observed as silent
    // 0xC0000005 exits). Leave a raw-file breadcrumb, no logger, no locks.
    static thread_local bool in_handler = false;
    if (in_handler) {
        char msg[192];
        const auto* rec = pExp ? pExp->ExceptionRecord : nullptr;
        const int len = std::snprintf(
            msg, sizeof(msg), "NESTED FAULT: code=%#lx rip=%#llx addr=%#llx thread=%lu\r\n",
            rec ? rec->ExceptionCode : 0ul,
            pExp && pExp->ContextRecord ? pExp->ContextRecord->Rip : 0ull,
            rec && rec->NumberParameters >= 2
                ? static_cast<unsigned long long>(rec->ExceptionInformation[1])
                : 0ull,
            GetCurrentThreadId());
        const HANDLE f = CreateFileA("C:\\Users\\user\\shad_nested_fault.txt", FILE_APPEND_DATA,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            WriteFile(f, msg, len > 0 ? static_cast<DWORD>(len) : 0, &written, nullptr);
            CloseHandle(f);
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    in_handler = true;
    const auto handler_guard = [](bool* flag) { *flag = false; };
    std::unique_ptr<bool, decltype(handler_guard)> guard_scope{&in_handler, handler_guard};
    using namespace Libraries::Kernel;
    const auto* signals = Signals::Instance();
    // Windows static guest red-zone protection
    const bool use_static_windows_guest_red_zone_protection =
        WindowsGuestRedZoneProtection::IsStaticPatchingEnabled();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    Ucontext guest_context{pExp->ContextRecord};
    Siginfo guest_info{
        ._si_signo = 0,
        ._si_errno = 0,
        ._si_code = POSIX_SI_NOINFO,
        ._si_addr = (void*)guest_context.uc_mcontext.mc_rip,
    };

    bool handled = false;
    bool static_protection_exception = false; // Windows static guest red-zone protection
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION: {
        guest_info._si_signo = POSIX_SIGSEGV;
        guest_info._si_code = POSIX_SEGV_MAPERR;
        static_protection_exception = true; // Windows static guest red-zone protection
        // [postmortem] Time the whole fault service when it runs on the tick-writer
        // thread: this single wrapper covers every blocking point of ours (fault entry
        // is single-point - the only registrar is page_manager's Impl).
        const u32 pm_tw_tid = Postmortem::g_tick_writer_tid.load(std::memory_order_relaxed);
        const bool pm_is_tw = pm_tw_tid != 0 && pm_tw_tid == GetCurrentThreadId();
        std::chrono::steady_clock::time_point pm_t0{};
        if (pm_is_tw) {
            pm_t0 = std::chrono::steady_clock::now();
            Postmortem::g_tw_inflight_since_ms.store(Postmortem::NowMs(),
                                                     std::memory_order_relaxed);
        }
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        if (pm_is_tw) {
            const u64 pm_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - pm_t0)
                                  .count();
            Postmortem::g_tw_av_count.fetch_add(1, std::memory_order_relaxed);
            Postmortem::g_tw_av_us.fetch_add(pm_us, std::memory_order_relaxed);
            Postmortem::AtomicMax(Postmortem::g_tw_av_max_us, pm_us);
            Postmortem::g_tw_inflight_since_ms.store(0, std::memory_order_relaxed);
        }
        break;
    }
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        guest_info._si_signo = POSIX_SIGILL;
        guest_info._si_code = POSIX_ILL_ILLOPC;
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case EXCEPTION_PRIV_INSTRUCTION: // Windows static guest red-zone protection
        if (use_static_windows_guest_red_zone_protection) {
            static_protection_exception = true;
            handled = signals->DispatchIllegalInstruction(pExp);
        }
        break;
    case EXCEPTION_IN_PAGE_ERROR:
        guest_info._si_signo = POSIX_SIGBUS;
        guest_info._si_code = POSIX_BUS_ADRALN;
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_INTDIV;
        break;
    case EXCEPTION_INT_OVERFLOW:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_INTOVF;
        break;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTDIV;
        break;
    case EXCEPTION_FLT_INVALID_OPERATION:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTINV;
        break;
    case EXCEPTION_FLT_OVERFLOW:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTOVF;
        break;
    case EXCEPTION_FLT_UNDERFLOW:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTUND;
        break;
    case EXCEPTION_FLT_DENORMAL_OPERAND:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTSUB; // i am not sure about this one
        break;
    case EXCEPTION_FLT_INEXACT_RESULT:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTRES;
        break;
    case EXCEPTION_FLT_STACK_CHECK:
        guest_info._si_signo = POSIX_SIGILL;
        guest_info._si_code = POSIX_ILL_BADSTK; // i am not sure about this one either
        break;
    case EXCEPTION_SINGLE_STEP:
        // Our hardware watchpoint fired: report the writer and let it continue.
        if ((pExp->ContextRecord->Dr6 & 0x1) != 0) {
            static u32 hits = 0;
            ++hits;
            // After the first dozen samples, keep only ZERO-writes: the hunt is
            // for whoever clears the frame tick right before the guest assert.
            const u64 watched_now = g_armed_watch_address != 0
                                        ? *reinterpret_cast<volatile u64*>(g_armed_watch_address)
                                        : ~0ull;
            // [postmortem] Unconditional (before any throttling/caps): a non-zero store
            // to the watched tick slot identifies the tick-writer thread; a zero store
            // is the legitimate FrameBegin cleaner - record the time, never the TID.
            if (watched_now != 0 && watched_now != ~0ull) {
                Postmortem::g_tick_writer_tid.store(GetCurrentThreadId(),
                                                    std::memory_order_relaxed);
                Postmortem::g_tick_last_write_ms.store(Postmortem::NowMs(),
                                                       std::memory_order_relaxed);
            } else if (watched_now == 0) {
                Postmortem::g_tick_last_zero_ms.store(Postmortem::NowMs(),
                                                      std::memory_order_relaxed);
            }
            if (hits <= 12 || (hits <= 400 && watched_now == 0)) {
                const auto rip = pExp->ContextRecord->Rip;
                LOG_CRITICAL(Debug,
                             "WATCHHIT #{}: writer rip={:#x} thread={} rax={:#x} rcx={:#x} "
                             "rdx={:#x} rbx={:#x} rsi={:#x} rdi={:#x}",
                             hits, rip, GetCurrentThreadId(), pExp->ContextRecord->Rax,
                             pExp->ContextRecord->Rcx, pExp->ContextRecord->Rdx,
                             pExp->ContextRecord->Rbx, pExp->ContextRecord->Rsi,
                             pExp->ContextRecord->Rdi);
                // The instruction that wrote is just before rip; dump the surrounding code
                // so it can be disassembled offline.
                if (hits <= 4) {
                    u8 code[96]{};
                    SIZE_T got = 0;
                    const DWORD64 from = rip - 64;
                    if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(from),
                                          code, sizeof(code), &got) &&
                        got > 0) {
                        std::string hex;
                        for (SIZE_T b = 0; b < got; ++b) {
                            fmt::format_to(std::back_inserter(hex), "{:02x}", code[b]);
                        }
                        LOG_CRITICAL(Debug, "WATCHCODE at {:#x}: {}", from, hex);
                    }
                }
            }
            pExp->ContextRecord->Dr6 = 0;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        guest_info._si_signo = POSIX_SIGTRAP;
        guest_info._si_code = POSIX_TRAP_BRKPT;
        break;
    case EXCEPTION_BREAKPOINT:
        guest_info._si_signo = POSIX_SIGTRAP;
        guest_info._si_code = POSIX_TRAP_BRKPT;
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    case MS_VC_EXCEPTION:
        LOG_DEBUG(Debug, "Pass MS_VC_EXCEPTION at {} to handler", address);
        return EXCEPTION_EXECUTE_HANDLER;
    default:
        break;
    }

    if (handled) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // The engine's job dispatcher occasionally calls an entry whose handler slot is still
    // empty - a producer/consumer race inside the game that real hardware wins on timing.
    // The call is already on the stack, so stepping over it is exactly what an empty
    // handler would have done, and the game keeps running instead of dying.
    if (code == EXCEPTION_ACCESS_VIOLATION && pExp->ContextRecord->Rip == 0) {
        static const bool skip_null_calls = [] {
            const char* e = std::getenv("SHADPS4_SKIPNULLCALL");
            return e && e[0] == '1';
        }();
        if (skip_null_calls) {
            DWORD64 ret_addr = 0;
            SIZE_T got = 0;
            if (pExp->ContextRecord->Rsp != 0 &&
                ReadProcessMemory(GetCurrentProcess(),
                                  reinterpret_cast<LPCVOID>(pExp->ContextRecord->Rsp), &ret_addr,
                                  sizeof(ret_addr), &got) &&
                got == sizeof(ret_addr) && ret_addr > 0x10000) {
                static std::atomic<u32> skipped{0};
                const u32 n = skipped.fetch_add(1, std::memory_order_relaxed) + 1;
                // [CRASHSTAT] Remember the last few sites. Stepping over a call and forcing
                // rax=0 hands the caller a null where it expected a result; if that is what
                // kills us minutes later, the crash line should show these growing.
                g_skipnull_count.store(n, std::memory_order_relaxed);
                g_skipnull_ret[(n - 1) & 3].store(ret_addr, std::memory_order_relaxed);
                g_skipnull_last_ms.store(Postmortem::NowMs(), std::memory_order_relaxed);
                if (n <= 5 || n % 100 == 0) {
                    LOG_WARNING(Debug,
                                "SKIPNULL #{}: stepping over empty handler, resuming at {:#x}", n,
                                ret_addr);
                }
                pExp->ContextRecord->Rip = ret_addr;
                pExp->ContextRecord->Rsp += 8;
                pExp->ContextRecord->Rax = 0;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
    }

    if (guest_info._si_signo != 0) {
        if (g_curthread &&
            g_curthread->DispatchSignal(guest_info._si_signo, &guest_info, &guest_context)) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }

    // Windows static guest red-zone protection
    const bool report_unhandled = use_static_windows_guest_red_zone_protection
                                      ? static_protection_exception
                                      : code != EXCEPTION_BREAKPOINT;
    if (report_unhandled) { // Windows static guest red-zone protection
        // A bare instruction pointer does not say WHAT the guest touched. Record the
        // access kind (0 read / 1 write / 8 execute) and the address it faulted on,
        // plus the registers, so the offending pointer can be traced to its source.
        {
            const bool have_info = pExp != nullptr && pExp->ExceptionRecord != nullptr &&
                                   pExp->ExceptionRecord->NumberParameters >= 2;
            const u64 fault_kind =
                have_info ? pExp->ExceptionRecord->ExceptionInformation[0] : ~0ull;
            const u64 fault_data =
                have_info ? pExp->ExceptionRecord->ExceptionInformation[1] : 0ull;
            LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {} kind={} data={:#x}", code,
                         address, fault_kind, fault_data);
            if (pExp != nullptr && pExp->ContextRecord != nullptr) {
                const auto& c = *pExp->ContextRecord;
                LOG_CRITICAL(Debug,
                             "CRASHREGS: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} "
                             "rdi={:#x} rbp={:#x} rsp={:#x} r8={:#x} r9={:#x} r10={:#x} "
                             "r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
                             c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.Rbp, c.Rsp, c.R8, c.R9,
                             c.R10, c.R11, c.R12, c.R13, c.R14, c.R15);
                // The registers alone do not say which of them the load used; the
                // instruction bytes do, and the process dies after this, so grab them now.
                // The trap at the crash site is preceded by the check that fired it, so
                // dump from well before RIP, plus the queue structures the registers name.
                const auto dump = [](const char* tag, u64 at, u32 len) {
                    u8 bytes[160]{};
                    len = std::min<u32>(len, sizeof(bytes));
                    SIZE_T got = 0;
                    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(at),
                                           bytes, len, &got) ||
                        got == 0) {
                        LOG_CRITICAL(Debug, "CRASHDUMP {} @ {:#x}: unreadable", tag, at);
                        return;
                    }
                    char hex[3 * sizeof(bytes) + 1]{};
                    for (SIZE_T bi = 0; bi < got; ++bi) {
                        std::snprintf(hex + bi * 3, 4, "%02x ", bytes[bi]);
                    }
                    LOG_CRITICAL(Debug, "CRASHDUMP {} @ {:#x}: {}", tag, at, hex);
                };
                dump("code-320", c.Rip >= 320 ? c.Rip - 320 : 0, 160);
                dump("code-160", c.Rip >= 160 ? c.Rip - 160 : 0, 160);
                dump("code", c.Rip >= 64 ? c.Rip - 64 : 0, 160);
                dump("r13", c.R13, 96);
                dump("r15", c.R15, 96);
                dump("slot", c.R15 + ((c.Rbx & 3) << 10) + 0x980, 96);
                // The frame-tick assert names the render-frame ring: 32 entries of
                // 0xA50 bytes at eboot+0x10CDB40, tick qword at +0xA08. RBX holds the
                // frame whose tick came up zero. Reading the slot right here separates
                // "the store arrived microseconds late" (non-zero now) from "the store
                // never happened" (still zero).
                if (MemoryPatcher::g_eboot_address != 0) {
                    const u64 ring = MemoryPatcher::g_eboot_address + 0x10CDB40;
                    const u64 tick_addr = ring + (c.Rbx & 0x1f) * 0xA50 + 0xA08;
                    dump("tick-slot", tick_addr - 0x20, 96);
                    const u64 next_addr = ring + ((c.Rbx + 1) & 0x1f) * 0xA50 + 0xA08;
                    dump("tick-next", next_addr, 16);
                }
                // [XFORM] The job-result validator (eboot+0x5682f0, assert line 3543) traps at
                // eboot+0x5683be (int 0x41) after eboot+0x360030(sret=[rbp-0x50], rsi=r12,
                // edx=[rbp-0x64], rcx=r13) returned a transform {float3 pos; float4 rot} with a
                // NaN/Inf component ((bits & 0x7f800000) == 0x7f800000). Dump the values and the
                // arguments so the bad dword and where it came from are named, not guessed.
                if (MemoryPatcher::g_eboot_address != 0 &&
                    c.Rip == MemoryPatcher::g_eboot_address + 0x5683be) {
                    u32 edx_arg = 0;
                    SIZE_T got_edx = 0;
                    ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(c.Rbp - 0x64),
                                      &edx_arg, sizeof(edx_arg), &got_edx);
                    float xf[8]{};
                    SIZE_T got_xf = 0;
                    ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(c.Rbp - 0x50),
                                      xf, sizeof(xf), &got_xf);
                    LOG_CRITICAL(Debug,
                                 "XFORM: obj(r14)={:#x} a1(r12)={:#x} a2(edx)={:#x} a3(r13)={:#x} "
                                 "rbx={:#x} rbp={:#x} | pos=({}, {}, {}, {}) rot=({}, {}, {}, {})",
                                 c.R14, c.R12, edx_arg, c.R13, c.Rbx, c.Rbp, xf[0], xf[1], xf[2],
                                 xf[3], xf[4], xf[5], xf[6], xf[7]);
                    dump("xform-raw", c.Rbp - 0x50, 48);
                    dump("xform-r12", c.R12, 96);
                    dump("xform-r13", c.R13, 96);
                    dump("xform-r14", c.R14, 96);
                    dump("getxf-code", MemoryPatcher::g_eboot_address + 0x360030, 160);
                    dump("getxf-code2", MemoryPatcher::g_eboot_address + 0x360030 + 160, 160);
                    // [XFORM-BT] The validator is particle.cpp:3543 (rsi=file, r8d=line): a
                    // particle spawn whose locator {pos, quat} has a NaN position. The locator
                    // is built by eboot+0x35fd60 from the caller's r12, and the caller
                    // (eboot+0x2d8360) got it as rsi from ITS caller. The validator's prologue
                    // pushes r15 r14 r13 r12 rbx, so the caller's r12/r13/r14/r15 sit at
                    // rbp-0x20..rbp-0x08. Walk the rbp chain for the return addresses: the game
                    // keeps frame pointers, so [rbp] is the caller's rbp and [rbp+8] its return.
                    {
                        u64 saved[4]{};
                        SIZE_T got_saved = 0;
                        ReadProcessMemory(GetCurrentProcess(),
                                          reinterpret_cast<LPCVOID>(c.Rbp - 0x20), saved,
                                          sizeof(saved), &got_saved);
                        LOG_CRITICAL(Debug,
                                     "XFORM-CALLER: src_locator(r12)={:#x} id(r13)={:#x} "
                                     "r14={:#x} owner(r15)={:#x}",
                                     saved[0], saved[1], saved[2], saved[3]);
                        VideoCore::g_crash_addr_hint.store(saved[0], std::memory_order_relaxed);
                        dump("xform-src", saved[0], 64);
                        dump("xform-owner", saved[3], 128);
                        const u64 eb = MemoryPatcher::g_eboot_address;
                        char bt[640]{};
                        int bn = std::snprintf(bt, sizeof(bt), "XFORM-BT:");
                        u64 fp = c.Rbp;
                        u64 frame_rbp[16]{};
                        int frames = 0;
                        for (int fi = 0; fi < 14 && fp != 0 && bn < 600; ++fi) {
                            u64 pair[2]{};
                            SIZE_T got_pair = 0;
                            if (!ReadProcessMemory(GetCurrentProcess(),
                                                   reinterpret_cast<LPCVOID>(fp), pair,
                                                   sizeof(pair), &got_pair) ||
                                got_pair != sizeof(pair)) {
                                break;
                            }
                            frame_rbp[frames++] = fp;
                            if (pair[1] >= eb && pair[1] < eb + 0x1400000) {
                                bn += std::snprintf(bt + bn, sizeof(bt) - bn, " eboot+%#llx",
                                                    static_cast<unsigned long long>(pair[1] - eb));
                            } else {
                                bn += std::snprintf(bt + bn, sizeof(bt) - bn, " %#llx",
                                                    static_cast<unsigned long long>(pair[1]));
                            }
                            if (pair[0] <= fp) {
                                break;
                            }
                            fp = pair[0];
                        }
                        LOG_CRITICAL(Debug, "{}", bt);
                        // [XFORM-CONTACT] Frame 3 is eboot+0x348350 (the spawn-at-contact
                        // function, called from eboot+0x2d1663): its rdi is a contact record
                        // {+0x08 object, +0x20 position, +0x30 normal} and the position is what
                        // arrives as 0x7fffffff. rdi lives in r12 there, pushed 4th in the
                        // prologue, so it sits at that frame's rbp-0x20. Name the record's
                        // address (which page, which buffer) and dump it.
                        if (frames >= 4) {
                            u64 contact = 0;
                            SIZE_T got_c = 0;
                            ReadProcessMemory(GetCurrentProcess(),
                                              reinterpret_cast<LPCVOID>(frame_rbp[3] - 0x20),
                                              &contact, sizeof(contact), &got_c);
                            u64 frame4_regs[5]{};
                            SIZE_T got_r = 0;
                            ReadProcessMemory(GetCurrentProcess(),
                                              reinterpret_cast<LPCVOID>(frame_rbp[3] - 0x28),
                                              frame4_regs, sizeof(frame4_regs), &got_r);
                            LOG_CRITICAL(Debug,
                                         "XFORM-CONTACT: record(r12)={:#x} page={:#x} | frame3 "
                                         "rbx={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
                                         contact, contact & ~u64{0x3fff}, frame4_regs[0],
                                         frame4_regs[1], frame4_regs[2], frame4_regs[3],
                                         frame4_regs[4]);
                            dump("xform-contact", contact, 128);
                            if (frames >= 5) {
                                dump("xform-frame4-locals", frame_rbp[4] - 0x80, 128);
                            }
                        }
                    }
                }
                // [PAIRS] eboot+0x5fa3d5 (`cmp r12, [rbp]`): the crashing worker walks a list of
                // 16-byte entries at r10 (rbx = current entry, stride 0x10), each holding a pointer
                // to an object with a u16 at +0xa, and compares against the key object r12. In the
                // 05.09 postmortem rbp held 0x3f6607a9be743e01 - two floats (~0.898, ~-0.238) where
                // a pointer belongs, i.e. the list was overwritten with vertex-like data after the
                // pier cutscene. Dump the list around the bad entry so the extent of the overwrite
                // and the data that landed there are named, not guessed.
                if (MemoryPatcher::g_eboot_address != 0 &&
                    c.Rip == MemoryPatcher::g_eboot_address + 0x5fa3d5) {
                    VideoCore::g_crash_addr_hint.store(c.Rbx, std::memory_order_relaxed);
                    const float* fp = reinterpret_cast<const float*>(&c.Rbp);
                    LOG_CRITICAL(
                        Debug,
                        "PAIRS: entry(rbx)={:#x} base(r10)={:#x} bad(rbp)={:#x} as_floats=({}, {}) "
                        "key(r12)={:#x} obj(rdi)={:#x} prev(rcx)={:#x} idx(rdx)={} r15={:#x}",
                        c.Rbx, c.R10, c.Rbp, fp[0], fp[1], c.R12, c.Rdi, c.Rcx, c.Rdx, c.R15);
                    dump("pairs-rbx-40", c.Rbx >= 0x40 ? c.Rbx - 0x40 : 0, 160);
                    dump("pairs-rbx+a0", c.Rbx + 0xa0, 160);
                    dump("pairs-r10", c.R10, 160);
                    dump("pairs-obj-rdi", c.Rdi, 96);
                    dump("pairs-key-r12", c.R12, 64);
                    // Frame-pointer chain of the worker: the caller names the job (see XFORM-BT).
                    const u64 eb = MemoryPatcher::g_eboot_address;
                    char bt[640]{};
                    int bn = std::snprintf(bt, sizeof(bt), "PAIRS-BT: (rsp chain)");
                    // rbp is clobbered here, so scan the stack for return addresses into eboot.
                    u64 words[48]{};
                    SIZE_T got_w = 0;
                    if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(c.Rsp),
                                          words, sizeof(words), &got_w)) {
                        for (size_t wi = 0; wi < got_w / 8 && bn < 600; ++wi) {
                            if (words[wi] >= eb && words[wi] < eb + 0x1400000) {
                                bn += std::snprintf(
                                    bt + bn, sizeof(bt) - bn, " [+%zx]eboot+%#llx", wi * 8,
                                    static_cast<unsigned long long>(words[wi] - eb));
                            }
                        }
                    }
                    LOG_CRITICAL(Debug, "{}", bt);
                }
                // [postmortem] The verdict line plus the registered partial-window dumps.
                // Strictly BEFORE SymInitialize/StackWalk64: dbghelp can wedge and eat
                // the numbers. Loads only, no exchanges, no locks, no allocations.
                AppendCrashStat(pExp);
                {
                    const s64 pm_now = Postmortem::NowMs();
                    const auto pm_age = [pm_now](s64 v) { return v ? pm_now - v : s64{-1}; };
                    const s64 pm_inflight =
                        Postmortem::g_tw_inflight_since_ms.load(std::memory_order_relaxed);
                    LOG_CRITICAL(
                        Debug,
                        "POSTMORTEM: crash_tid={} tw_tid={} tick_write_age_ms={} "
                        "tick_zero_age_ms={} eop_signal_age_ms={} "
                        "eop_dispatch_age_ms={} eop_lock_max_us={} eop_lock_sum_us={} "
                        "eop_gap_max_ms={} "
                        "tw_av n={} sum_us={} max_us={} inflight_ms={}",
                        GetCurrentThreadId(),
                        Postmortem::g_tick_writer_tid.load(std::memory_order_relaxed),
                        pm_age(Postmortem::g_tick_last_write_ms.load(std::memory_order_relaxed)),
                        pm_age(Postmortem::g_tick_last_zero_ms.load(std::memory_order_relaxed)),
                        pm_age(Postmortem::g_eop_signal_ms.load(std::memory_order_relaxed)),
                        pm_age(Postmortem::g_eop_dispatch_ms.load(std::memory_order_relaxed)),
                        Postmortem::g_eop_lock_wait_max_us.load(std::memory_order_relaxed),
                        Postmortem::g_eop_lock_wait_us.load(std::memory_order_relaxed),
                        Postmortem::g_eop_gap_max_ms.load(std::memory_order_relaxed),
                        Postmortem::g_tw_av_count.load(std::memory_order_relaxed),
                        Postmortem::g_tw_av_us.load(std::memory_order_relaxed),
                        Postmortem::g_tw_av_max_us.load(std::memory_order_relaxed),
                        pm_inflight ? pm_now - pm_inflight : 0);
                    // [SLOWWAIT p.3] GFX parser state at death + last 8 packets + ASC liveness.
                    // Loads only; ring/asc strings built in fixed buffers (no allocations).
                    {
                        const u32 st = Postmortem::g_gfx_state.load(std::memory_order_relaxed);
                        const char* st_name =
                            st == 1 ? "waiting" : (st == 2 ? "starved" : "running");
                        char ring_buf[320];
                        int rb = 0;
                        const u64 seq_now =
                            Postmortem::g_opring_seq.load(std::memory_order_relaxed);
                        for (u64 k = 0; k < 8; ++k) {
                            const u64 idx = seq_now >= 7 ? seq_now - 7 + k : k + 1;
                            if (idx == 0 || idx > seq_now) {
                                continue;
                            }
                            const u64 packed =
                                Postmortem::g_opring[idx & 7].load(std::memory_order_relaxed);
                            const u32 r_op = static_cast<u32>(packed >> 56);
                            const u64 r_seq = (packed >> 24) & 0xFFFFFFFFull;
                            const u64 r_ms = packed & 0xFFFFFFull;
                            const u64 age = (static_cast<u64>(pm_now) - r_ms) & 0xFFFFFFull;
                            if (rb < static_cast<int>(sizeof(ring_buf)) - 40) {
                                rb += std::snprintf(ring_buf + rb, sizeof(ring_buf) - rb,
                                                    " %#x#%llu@%llums", r_op,
                                                    static_cast<unsigned long long>(r_seq),
                                                    static_cast<unsigned long long>(age));
                            }
                        }
                        ring_buf[rb < 0 ? 0 : rb] = '\0';
                        char asc_buf[224];
                        int ab = 0;
                        for (u32 i = 0; i < 8; ++i) {
                            const u64 pkts =
                                Postmortem::g_asc_pkts[i].load(std::memory_order_relaxed);
                            if (pkts == 0) {
                                continue;
                            }
                            const s64 last =
                                Postmortem::g_asc_last_ms[i].load(std::memory_order_relaxed);
                            if (ab < static_cast<int>(sizeof(asc_buf)) - 48) {
                                ab += std::snprintf(
                                    asc_buf + ab, sizeof(asc_buf) - ab, " %u:%llu@%lldms", i,
                                    static_cast<unsigned long long>(pkts),
                                    static_cast<long long>(last ? pm_now - last : -1));
                            }
                        }
                        asc_buf[ab < 0 ? 0 : ab] = '\0';
                        const s64 wrm_since =
                            Postmortem::g_wrm_since_ms.load(std::memory_order_relaxed);
                        LOG_CRITICAL(
                            Debug,
                            "POSTMORTEM-GFX: state={} wrm addr={:#x} func={} ref={:#x} "
                            "mask={:#x} cur={:#x} waiting_ms={} ring=[{} ] asc=[{} ]",
                            st_name, Postmortem::g_wrm_addr.load(std::memory_order_relaxed),
                            Postmortem::g_wrm_func.load(std::memory_order_relaxed),
                            Postmortem::g_wrm_ref.load(std::memory_order_relaxed),
                            Postmortem::g_wrm_mask.load(std::memory_order_relaxed),
                            Postmortem::g_wrm_cur.load(std::memory_order_relaxed),
                            st == 1 && wrm_since ? pm_now - wrm_since : 0, ring_buf, asc_buf);
                    }
                    Postmortem::RunPostmortemDumps();
                }
                // The assert reporter at the known trap site takes four string args via
                // RIP-relative leas; their targets sit at fixed deltas from the trap.
                dump("str-rsi", c.Rip + 0x340452, 96);
                dump("str-rdi", c.Rip + 0x3405AC, 96);
                dump("str-rdx", c.Rip + 0x3405E8, 96);
                dump("str-rcx", c.Rip + 0x351CC2, 96);
            }
        }
        // A bare address says nothing about which emulator routine faulted, so walk
        // the frames and resolve them while the process is still alive.
        // SHADPS4_SYMFRAMES=1 opt-in: dbghelp wedged twice into an unkillable
        // zombie that dragged the whole system down (kernel-stuck process, ssh
        // and Sunshine dead until a hard reboot). POSTMORTEM and raw CRASHDUMPs
        // above carry the payload; symbolization is a luxury we can't afford.
        static const bool sym_frames = [] {
            const char* e = std::getenv("SHADPS4_SYMFRAMES");
            return e && e[0] == '1';
        }();
        if (sym_frames) {
            const HANDLE process = GetCurrentProcess();
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
            SymInitialize(process, nullptr, TRUE);
            alignas(8) char sym_buf[sizeof(SYMBOL_INFO) + 512]{};
            auto* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            sym->MaxNameLen = 511;
            CONTEXT ctx = *pExp->ContextRecord;
            // A jump to a dead address leaves no usable frame: the only surviving clue is
            // the return address the CALL pushed, which names the caller. The job system
            // sometimes lands on zero and sometimes on junk, so treat both the same - the
            // giveaway is that the faulting address IS the instruction pointer.
            const bool jumped_into_nowhere =
                ctx.Rip == 0 || (code == EXCEPTION_ACCESS_VIOLATION &&
                                 pExp->ExceptionRecord->NumberParameters >= 2 &&
                                 static_cast<u64>(pExp->ExceptionRecord->ExceptionInformation[1]) ==
                                     static_cast<u64>(ctx.Rip));
            if (jumped_into_nowhere) {
                LOG_CRITICAL(Debug, "CRASHNULL: rip={:#x} rsp={:#x} rbp={:#x}", ctx.Rip, ctx.Rsp,
                             ctx.Rbp);
                LOG_CRITICAL(Debug,
                             "CRASHREGS: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} "
                             "rdi={:#x} r11={:#x} r12={:#x} r13={:#x} r15={:#x}",
                             ctx.Rax, ctx.Rbx, ctx.Rcx, ctx.Rdx, ctx.Rsi, ctx.Rdi, ctx.R11, ctx.R12,
                             ctx.R13, ctx.R15);
                // The faulting loop walks a table of {func, data} pairs at rbx with a
                // 16-byte stride and a count at [r11+0x24]; dump it to see which slot is
                // empty and what its neighbours are.
                {
                    u32 count = 0;
                    SIZE_T got = 0;
                    if (ReadProcessMemory(process, reinterpret_cast<LPCVOID>(ctx.R11 + 0x24),
                                          &count, sizeof(count), &got) &&
                        got == sizeof(count) && count <= 256) {
                        std::string dump;
                        for (u32 e = 0; e < count; ++e) {
                            DWORD64 pair[2]{};
                            if (!ReadProcessMemory(
                                    process, reinterpret_cast<LPCVOID>(ctx.Rbx + DWORD64{e} * 16),
                                    pair, sizeof(pair), &got) ||
                                got != sizeof(pair)) {
                                break;
                            }
                            fmt::format_to(std::back_inserter(dump), "[{}]={:#x}/{:#x} ", e,
                                           pair[0], pair[1]);
                        }
                        LOG_CRITICAL(Debug, "CRASHTABLE: count={} idx={} {}", count,
                                     static_cast<u32>(ctx.R12), dump);
                        // Is the whole page zero (something re-mapped or wiped it) or only
                        // the table (someone cleared just these entries)? Sample a wide
                        // window around the table and report how much of it survived.
                        {
                            const DWORD64 page = ctx.Rbx & ~u64{0xFFF};
                            u32 nonzero = 0, sampled = 0;
                            for (u32 off = 0; off < 0x1000; off += 8) {
                                DWORD64 word = 0;
                                SIZE_T rd = 0;
                                if (ReadProcessMemory(process,
                                                      reinterpret_cast<LPCVOID>(page + off), &word,
                                                      sizeof(word), &rd) &&
                                    rd == sizeof(word)) {
                                    ++sampled;
                                    if (word != 0) {
                                        ++nonzero;
                                    }
                                }
                            }
                            LOG_CRITICAL(Debug, "CRASHPAGE: page={:#x} nonzero={}/{} words", page,
                                         nonzero, sampled);
                            // Show the neighbourhood: a zeroed hole inside live data points
                            // at a cleared/never-published batch rather than lost memory.
                            std::string around;
                            for (s32 w = -8; w < 24; ++w) {
                                DWORD64 word = 0;
                                SIZE_T rd = 0;
                                const DWORD64 addr = ctx.Rbx + static_cast<s64>(w) * 8;
                                if (ReadProcessMemory(process, reinterpret_cast<LPCVOID>(addr),
                                                      &word, sizeof(word), &rd) &&
                                    rd == sizeof(word)) {
                                    fmt::format_to(std::back_inserter(around), "{:+d}:{:#x} ",
                                                   w * 8, word);
                                }
                            }
                            LOG_CRITICAL(Debug, "CRASHAROUND: rbx={:#x} {}", ctx.Rbx, around);
                            // The loop picks a handler by a 16-bit type id stored in each
                            // 48-byte record at [r11 + i*48 + 0x28]; dump those ids and the
                            // handler each one resolves to. A wild id means the record was
                            // never filled with anything sane.
                            {
                                std::string ids;
                                for (u32 i = 0; i < count && i < 32; ++i) {
                                    u16 type_id = 0;
                                    SIZE_T rd = 0;
                                    const DWORD64 rec = ctx.R11 + DWORD64{i} * 48 + 0x28;
                                    if (!ReadProcessMemory(process, reinterpret_cast<LPCVOID>(rec),
                                                           &type_id, sizeof(type_id), &rd) ||
                                        rd != sizeof(type_id)) {
                                        break;
                                    }
                                    DWORD64 handler = 0;
                                    SIZE_T hrd = 0;
                                    ReadProcessMemory(
                                        process,
                                        reinterpret_cast<LPCVOID>(ctx.Rbx + DWORD64{type_id} * 16),
                                        &handler, sizeof(handler), &hrd);
                                    fmt::format_to(std::back_inserter(ids), "[{}]id={} h={:#x} ", i,
                                                   type_id, handler);
                                }
                                LOG_CRITICAL(Debug, "CRASHIDS: r11={:#x} {}", ctx.R11, ids);
                                // Which job types have a handler at all? The table is static
                                // data filled during start-up, so empty slots mean the
                                // registration for those types never ran.
                                {
                                    std::string table;
                                    for (u32 t = 0; t < 64; ++t) {
                                        DWORD64 h = 0;
                                        SIZE_T rd = 0;
                                        if (!ReadProcessMemory(process,
                                                               reinterpret_cast<LPCVOID>(
                                                                   ctx.Rbx + DWORD64{t} * 16),
                                                               &h, sizeof(h), &rd) ||
                                            rd != sizeof(h)) {
                                            break;
                                        }
                                        if (h != 0) {
                                            fmt::format_to(std::back_inserter(table), "{} ", t);
                                        }
                                    }
                                    LOG_CRITICAL(Debug,
                                                 "CRASHTYPES: registered handlers for ids: {}",
                                                 table);
                                }
                                // Read the same records again a moment later: if they change
                                // while this thread is stopped, somebody is still writing them
                                // and the consumer was racing a live producer.
                                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                                std::string ids2;
                                for (u32 i = 0; i < count && i < 32; ++i) {
                                    u16 type_id = 0;
                                    SIZE_T rd = 0;
                                    const DWORD64 rec = ctx.R11 + DWORD64{i} * 48 + 0x28;
                                    if (!ReadProcessMemory(process, reinterpret_cast<LPCVOID>(rec),
                                                           &type_id, sizeof(type_id), &rd) ||
                                        rd != sizeof(type_id)) {
                                        break;
                                    }
                                    fmt::format_to(std::back_inserter(ids2), "[{}]id={} ", i,
                                                   type_id);
                                }
                                LOG_CRITICAL(Debug, "CRASHIDS2 (+3ms): {}", ids2);
                                // Header of the object plus the failing record in full: an
                                // all-zero record means it was never filled, a partly filled
                                // one means the write was torn.
                                {
                                    std::string hdr, rec;
                                    u8 buf[64]{};
                                    SIZE_T rd = 0;
                                    if (ReadProcessMemory(process,
                                                          reinterpret_cast<LPCVOID>(ctx.R11), buf,
                                                          sizeof(buf), &rd) &&
                                        rd == sizeof(buf)) {
                                        for (u32 b = 0; b < 64; ++b) {
                                            fmt::format_to(std::back_inserter(hdr), "{:02x}",
                                                           buf[b]);
                                        }
                                    }
                                    const DWORD64 rec_addr = ctx.R11 + DWORD64{ctx.R12} * 48 + 0x28;
                                    if (ReadProcessMemory(process,
                                                          reinterpret_cast<LPCVOID>(rec_addr), buf,
                                                          48, &rd) &&
                                        rd == 48) {
                                        for (u32 b = 0; b < 48; ++b) {
                                            fmt::format_to(std::back_inserter(rec), "{:02x}",
                                                           buf[b]);
                                        }
                                    }
                                    LOG_CRITICAL(Debug,
                                                 "CRASHREC: idx={} rec_addr={:#x}\n  header={}\n"
                                                 "  record={}",
                                                 static_cast<u32>(ctx.R12), rec_addr, hdr, rec);
                                }
                            }
                        }
                    }
                }
                for (int i = 0; i < 8; ++i) {
                    const auto slot = ctx.Rsp + static_cast<DWORD64>(i) * 8;
                    DWORD64 value = 0;
                    SIZE_T read = 0;
                    if (!ReadProcessMemory(process, reinterpret_cast<LPCVOID>(slot), &value,
                                           sizeof(value), &read) ||
                        read != sizeof(value) || value == 0) {
                        continue;
                    }
                    DWORD64 disp = 0;
                    if (SymFromAddr(process, value, &disp, sym)) {
                        LOG_CRITICAL(Debug, "CRASHRET [rsp+{:#x}] = {:#x} {}+{:#x}", i * 8, value,
                                     sym->Name, disp);
                    } else {
                        LOG_CRITICAL(Debug, "CRASHRET [rsp+{:#x}] = {:#x}", i * 8, value);
                    }
                }
                // Dump the bytes just before the return address: that is the code that
                // performed the null call, and disassembling it names the callee slot.
                {
                    DWORD64 ret0 = 0;
                    SIZE_T got = 0;
                    if (ctx.Rsp != 0 &&
                        ReadProcessMemory(process, reinterpret_cast<LPCVOID>(ctx.Rsp), &ret0,
                                          sizeof(ret0), &got) &&
                        got == sizeof(ret0) && ret0 > 0x40) {
                        u8 code_bytes[160]{};
                        SIZE_T code_got = 0;
                        const DWORD64 from = ret0 - 144;
                        if (ReadProcessMemory(process, reinterpret_cast<LPCVOID>(from), code_bytes,
                                              sizeof(code_bytes), &code_got) &&
                            code_got > 0) {
                            std::string hex;
                            hex.reserve(code_got * 2);
                            for (SIZE_T b = 0; b < code_got; ++b) {
                                fmt::format_to(std::back_inserter(hex), "{:02x}", code_bytes[b]);
                            }
                            LOG_CRITICAL(Debug, "CRASHCODE at {:#x}: {}", from, hex);
                        }
                    }
                }
                DWORD64 ret_addr = 0;
                SIZE_T ret_read = 0;
                if (ctx.Rsp != 0 &&
                    ReadProcessMemory(process, reinterpret_cast<LPCVOID>(ctx.Rsp), &ret_addr,
                                      sizeof(ret_addr), &ret_read) &&
                    ret_read == sizeof(ret_addr)) {
                    ctx.Rip = ret_addr;
                }
            }
            STACKFRAME64 frame{};
            frame.AddrPC.Offset = ctx.Rip;
            frame.AddrPC.Mode = AddrModeFlat;
            frame.AddrFrame.Offset = ctx.Rbp;
            frame.AddrFrame.Mode = AddrModeFlat;
            frame.AddrStack.Offset = ctx.Rsp;
            frame.AddrStack.Mode = AddrModeFlat;
            for (int i = 0; i < 24; ++i) {
                if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, GetCurrentThread(), &frame,
                                 &ctx, nullptr, SymFunctionTableAccess64, SymGetModuleBase64,
                                 nullptr) ||
                    frame.AddrPC.Offset == 0) {
                    break;
                }
                DWORD64 disp = 0;
                if (SymFromAddr(process, frame.AddrPC.Offset, &disp, sym)) {
                    LOG_CRITICAL(Debug, "CRASHFRAME #{}: {:#x} {}+{:#x}", i, frame.AddrPC.Offset,
                                 sym->Name, disp);
                } else {
                    LOG_CRITICAL(Debug, "CRASHFRAME #{}: {:#x} <no symbol>", i,
                                 frame.AddrPC.Offset);
                }
            }
        }
        Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

#else

// Debug registers are only armed on Windows.
void ArmWriteWatch(u64 address) {}

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

static s32 NativeSiCodeToGuest(s32 sig, s32 code) {
    using namespace Libraries::Kernel;
    switch (sig) {
    case SIGUSR1:
        return POSIX_SI_LWP;
    case SIGSEGV:
        switch (code) {
        case SEGV_MAPERR:
            return POSIX_SEGV_MAPERR;
        case SEGV_ACCERR:
            return POSIX_SEGV_ACCERR;
        }
    case SIGBUS:
        switch (code) {
        case BUS_ADRALN:
            return POSIX_BUS_ADRALN;
        case BUS_ADRERR:
            return POSIX_BUS_ADRERR;
        case BUS_OBJERR:
            return POSIX_BUS_OBJERR;
        }
    case SIGILL:
        switch (code) {
        case ILL_ILLOPC:
            return POSIX_ILL_ILLOPC;
        case ILL_ILLOPN:
            return POSIX_ILL_ILLOPN;
        case ILL_ILLADR:
            return POSIX_ILL_ILLADR;
        case ILL_ILLTRP:
            return POSIX_ILL_ILLTRP;
        case ILL_PRVOPC:
            return POSIX_ILL_PRVOPC;
        case ILL_PRVREG:
            return POSIX_ILL_PRVREG;
        case ILL_COPROC:
            return POSIX_ILL_COPROC;
        case ILL_BADSTK:
            return POSIX_ILL_BADSTK;
        }
    case SIGFPE:
        switch (code) {
        case FPE_INTOVF:
            return POSIX_FPE_INTOVF;
        case FPE_INTDIV:
            return POSIX_FPE_INTDIV;
        case FPE_FLTDIV:
            return POSIX_FPE_FLTDIV;
        case FPE_FLTOVF:
            return POSIX_FPE_FLTOVF;
        case FPE_FLTUND:
            return POSIX_FPE_FLTUND;
        case FPE_FLTRES:
            return POSIX_FPE_FLTRES;
        case FPE_FLTINV:
            return POSIX_FPE_FLTINV;
        case FPE_FLTSUB:
            return POSIX_FPE_FLTSUB;
        }
    case SIGTRAP:
        switch (code) {
        case TRAP_BRKPT:
            return POSIX_TRAP_BRKPT;
        case TRAP_TRACE:
            return POSIX_TRAP_TRACE;
#ifdef __FreeBSD__
        case TRAP_DTRACE:
            return POSIX_TRAP_DTRACE;
#endif
        }

    default:
        return POSIX_SI_NOINFO;
    }
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    using namespace Libraries::Kernel;
    auto* thread = g_curthread;
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    Ucontext context{info, reinterpret_cast<ucontext_t*>(raw_context)};
    Siginfo guest_info{};
    if (info) {
        guest_info = *reinterpret_cast<Siginfo*>(info);
        guest_info._si_signo = sig == SIGUSR1 ? 0 : NativeToOrbisSignal(info->si_signo);
        guest_info._si_errno = NativeToPosixErrno(info->si_errno);
        guest_info._si_code = NativeSiCodeToGuest(sig, info->si_code);
        guest_info._si_addr = (void*)context.uc_mcontext.mc_rip;
    }
    Siginfo* info_p = info ? &guest_info : nullptr;
    Ucontext* context_p = raw_context ? &context : nullptr;

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            if (thread && thread->DispatchSignal(NativeToOrbisSignal(sig), info_p, context_p)) {
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (signals->DispatchIllegalInstruction(raw_context)) {
            return;
        }
    case SIGFPE:
    case SIGTRAP:
    case SIGSYS: {
        if (thread && thread->DispatchSignal(NativeToOrbisSignal(sig), info_p, context_p)) {
            return;
        }

        UNREACHABLE_MSG("Unhandled signal {} at code address {}", sig, fmt::ptr(code_address));
    }
    case SIGSLEEP: {
        // Sleep thread until signal is received again
        sigset_t sigset;
        sigemptyset(&sigset);
        sigaddset(&sigset, SIGSLEEP);
        sigwait(&sigset, &sig);
        break;
    }
    case SIGUSR1:
        if (thread) {
            thread->DispatchPendingSignals(info_p, context_p);
        }
        break;
    default:
        UNREACHABLE_MSG("Unhandled signal {} at code address {}", sig, fmt::ptr(code_address));
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(
        sigaction(SIGSEGV, &action, nullptr) == 0 && sigaction(SIGBUS, &action, nullptr) == 0 &&
            sigaction(SIGILL, &action, nullptr) == 0 && sigaction(SIGFPE, &action, nullptr) == 0 &&
            sigaction(SIGTRAP, &action, nullptr) == 0 && sigaction(SIGSYS, &action, nullptr) == 0 &&
            sigaction(SIGUSR1, &action, nullptr) == 0 && sigaction(SIGSLEEP, &action, nullptr) == 0,
        "Failed to register signal handlers.");
#endif
}

void SignalDispatch::RemoveHandlers() {
    // asserting here would get into an infinite loop until too
    // many nested exceptions makes the OS kill the process
#if defined(_WIN32)
    if (!(RemoveVectoredExceptionHandler(handle))) {
        LOG_CRITICAL(Core, "Failed to remove exception handler.");
        std::quick_exit(1);
    }
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    if (!(sigaction(SIGSEGV, &action, nullptr) == 0 && sigaction(SIGBUS, &action, nullptr) == 0 &&
          sigaction(SIGILL, &action, nullptr) == 0 && sigaction(SIGFPE, &action, nullptr) == 0 &&
          sigaction(SIGTRAP, &action, nullptr) == 0 && sigaction(SIGSYS, &action, nullptr) == 0 &&
          sigaction(SIGUSR1, &action, nullptr) == 0 &&
          sigaction(SIGSLEEP, &action, nullptr) == 0)) {
        LOG_CRITICAL(Core, "Failed to remove signal handlers.");
        std::quick_exit(1);
    }
#endif
}

SignalDispatch::~SignalDispatch() {
    RemoveHandlers();
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
