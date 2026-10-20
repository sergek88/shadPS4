// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "core/memory.h"
#include "core/signals.h"
#include <atomic>
#include <algorithm>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#ifdef ENABLE_USERFAULTFD
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include "common/error.h"
#endif
#else
#include <windows.h>
#include "common/spin_lock.h"
#endif

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

constexpr size_t PM_PAGE_SIZE = 4_KB;
constexpr size_t PM_PAGE_BITS = 12;

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 7;
        // At the moment only buffer cache can request read watchers.
        // And buffers cannot overlap, thus only 1 can exist per page.
        u8 num_read_watchers : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;
#ifdef ENABLE_USERFAULTFD
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        ASSERT_MSG(uffd != -1, "{}", Common::GetLastErrorMsg());

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        ASSERT(ret == 0 && api.api == UFFD_API);

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    void OnMap(VAddr address, size_t size) {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void OnUnmap(VAddr address, size_t size) {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            rasterizer->InvalidateMemory(addr, 1);
        }
    }

    std::jthread ufd_thread;
    int uffd;
#else
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void OnMap(VAddr address, size_t size) {
        // No-op
    }

    void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    // Was this address executable before the tracker touched it? Asking the OS on every
    // protection change is far too expensive - the counter shows up to 380 000 changes per five
    // seconds - so the answer is cached per OS region. Regions of the guest image never move,
    // so after a short warm-up the query disappears entirely.
    static bool IsExecutableRegion(VAddr addr) {
#ifdef _WIN32
        struct Region {
            VAddr begin;
            VAddr end;
            bool exec;
        };
        static std::vector<Region> cache;
        static std::mutex cache_mutex;
        {
            std::scoped_lock lk{cache_mutex};
            const auto it = std::lower_bound(cache.begin(), cache.end(), addr,
                                             [](const Region& r, VAddr a) { return r.end <= a; });
            if (it != cache.end() && addr >= it->begin && addr < it->end) {
                return it->exec;
            }
        }
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi)) != sizeof(mbi)) {
            return false;
        }
        constexpr DWORD ExecMask =
            PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        const Region r{reinterpret_cast<VAddr>(mbi.BaseAddress),
                       reinterpret_cast<VAddr>(mbi.BaseAddress) + mbi.RegionSize,
                       (mbi.Protect & ExecMask) != 0};
        {
            std::scoped_lock lk{cache_mutex};
            const auto pos = std::lower_bound(cache.begin(), cache.end(), r.end,
                                              [](const Region& x, VAddr e) { return x.end < e; });
            cache.insert(pos, r);
        }
        return r.exec;
#else
        return false;
#endif
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        RENDERER_TRACE;
        // [debug] стоимость отслеживания памяти: сколько системных вызовов защиты
        // и сколько байт мы переустанавливаем в секунду.
        {
            static std::atomic<u64> s_calls{0};
            static std::atomic<u64> s_bytes{0};
            static std::atomic<s64> s_next_ms{0};
            const u64 calls = s_calls.fetch_add(1, std::memory_order_relaxed) + 1;
            s_bytes.fetch_add(size, std::memory_order_relaxed);
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            s64 next = s_next_ms.load(std::memory_order_relaxed);
            if (now_ms >= next && s_next_ms.compare_exchange_strong(next, now_ms + 5000)) {
                if (next != 0) {
                    LOG_CRITICAL(Render_Vulkan, "PROTECT: {} calls, {} MB in 5s",
                                 calls, s_bytes.load() >> 20);
                }
                s_calls.store(0, std::memory_order_relaxed);
                s_bytes.store(0, std::memory_order_relaxed);
            }
        }
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        // [fix] SHADPS4_EXECKEEP=1: never take execute permission away.
        // PageState::Perms() only ever returns Read|Write, so every Protect made by the tracker
        // rewrites a page of the guest executable from PAGE_EXECUTE_* down to PAGE_READONLY /
        // PAGE_READWRITE / PAGE_NOACCESS. The next instruction fetch from that page raises a DEP
        // fault, whose ExceptionInformation[0] is 8 - and IsWriteError only tests for 1, so the
        // handler mistakes it for a read, calls ReadMemory, answers "handled", and the very same
        // instruction is retried forever. Measured: 311 000 faults per second on one address
        // (rip == fault address == 0x5d992b0, page 0x5d99000 sitting at PAGE_READONLY).
        // Adding Execute keeps read and write tracking exactly as it was - only instruction
        // fetch stops faulting.
        static const bool keep_exec = [] {
            const char* e = std::getenv("SHADPS4_EXECKEEP");
            return e && e[0] == '1';
        }();
        // Execute is only added where read is already allowed: an execute-without-read page is
        // not a combination the address space knows how to express (it fails VirtualProtect with
        // "the parameter is incorrect"), and a page closed for reading is a GPU-modified one -
        // no code is fetched from those. The broken case is exactly the readable ones, which is
        // where the measured livelock lived (page at PAGE_READONLY, instruction fetch faulting).
        const bool readable = static_cast<bool>(perms & Core::MemoryPermission::Read);
        const bool was_executable = keep_exec && readable && IsExecutableRegion(address);
        // [diag] Guest thread stacks live in this band (every crash-dump rsp of the
        // campaign: 0x12002xxxxx..0x12012xxxxx). A protection landing on a stack
        // page is lethal in a way no handler can report: the fault's own dispatch
        // writes to that same stack, faults again at the VEH prologue, and the
        // process dies mute (WerSvc TerminateProcess, no VEH log, no dump).
        {
            constexpr VAddr StackBandLo = 0x1200000000ull;
            constexpr VAddr StackBandHi = 0x1210000000ull;
            if (address < StackBandHi && address + size > StackBandLo) {
                static std::atomic<u32> reported{0};
                const u32 n = reported.fetch_add(1, std::memory_order_relaxed);
                if (n < 16 || (n % 256) == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "STACKPROT: protect {:#x}+{:#x} perms={:#x} hits guest stack "
                                 "band ({} total)",
                                 address, size, static_cast<u32>(perms), n + 1);
                }
                // [fix] SHADPS4_STACKSHIELD=1: never take Write (or Read) off a
                // guest stack page. A fault whose dispatch frame lands on a
                // protected stack page cannot be delivered by the kernel at all:
                // the process dies mute (WerSvc TerminateProcess; caught twice by
                // SilentProcessExit dumps, thread frozen at the VEH prologue).
                // Tracking loss is covered by sticky CPU authority over the band.
                static const bool stack_shield = [] {
                    const char* e = std::getenv("SHADPS4_STACKSHIELD");
                    return e && e[0] == '1';
                }();
                if (stack_shield &&
                    perms != (Core::MemoryPermission::Read | Core::MemoryPermission::Write)) {
                    return;
                }
            }
        }
        impl.Protect(address, size,
                     was_executable ? (perms | Core::MemoryPermission::Execute) : perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        // [debug] SHADPS4_FAULTINFO=1: the one line that separates the two live theories.
        // ExceptionInformation[0] is 0 for a read, 1 for a write and 8 for an instruction fetch
        // blocked by DEP. IsWriteError only tests for 1, so a DEP fault is mistaken for a read,
        // handed to ReadMemory, reported as handled, and the instruction retries forever.
        // Printing the real faulting address matters too: the READBACK histogram prints a
        // 16 MiB bucket label, not an address, and pointing traps at it wasted a whole round.
#ifdef _WIN32
        {
            static const bool want = [] {
                const char* e = std::getenv("SHADPS4_FAULTINFO");
                return e && e[0] == '1';
            }();
            if (want) {
                static std::atomic<u32> shown{0};
                static std::atomic<u64> total{0};
                static std::atomic<s64> next_ms{0};
                const u64 n = total.fetch_add(1, std::memory_order_relaxed);
                // First 32 for the start-up picture, then one sample every 5 s so the storm
                // phase - which begins minutes later - is described too.
                bool take = shown.load(std::memory_order_relaxed) < 32;
                if (take) {
                    ++shown;
                } else {
                    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count();
                    // Rate watch: sampling on a timer misses a burst that lasts seconds. Measure
                    // the fault rate over one second and, when it explodes, dump a burst of
                    // consecutive samples - that is the only way to describe the storm itself
                    // rather than the calm around it.
                    static std::atomic<s64> window_ms{0};
                    static std::atomic<u64> window_start_n{0};
                    static std::atomic<u32> burst_left{0};
                    s64 wexp = window_ms.load(std::memory_order_relaxed);
                    if (now >= wexp && window_ms.compare_exchange_strong(wexp, now + 1000)) {
                        const u64 prev = window_start_n.exchange(n, std::memory_order_relaxed);
                        const u64 rate = n - prev;
                        if (wexp != 0 && rate > 20000) {
                            burst_left.store(16, std::memory_order_relaxed);
                            LOG_CRITICAL(Render_Vulkan, "FAULTSTORM: {} промахов за секунду", rate);
                        }
                    }
                    if (burst_left.load(std::memory_order_relaxed) > 0) {
                        --burst_left;
                        take = true;
                    } else {
                        s64 expected = next_ms.load(std::memory_order_relaxed);
                        take = now >= expected &&
                               next_ms.compare_exchange_strong(expected, now + 5000);
                    }
                }
                if (take) {
                    auto* ep = static_cast<EXCEPTION_POINTERS*>(context);
                    const auto kind = ep->ExceptionRecord->ExceptionInformation[0];
                    const auto data_addr = ep->ExceptionRecord->ExceptionInformation[1];
                    const auto rip = reinterpret_cast<u64>(ep->ExceptionRecord->ExceptionAddress);
                    MEMORY_BASIC_INFORMATION mbi{};
                    const bool q = VirtualQuery(reinterpret_cast<LPCVOID>(data_addr), &mbi,
                                                sizeof(mbi)) == sizeof(mbi);
                    LOG_CRITICAL(Render_Vulkan,
                                 "FAULTINFO #{}: kind={} ({}) data={:#x} rip={:#x} same={} "
                                 "protect={:#x} state={:#x} base={:#x}",
                                 n, kind,
                                 kind == 0   ? "read"
                                 : kind == 1 ? "write"
                                 : kind == 8 ? "EXECUTE (DEP)"
                                             : "?",
                                 u64(data_addr), rip, u32(data_addr == rip),
                                 q ? u32(mbi.Protect) : 0u, q ? u32(mbi.State) : 0u,
                                 q ? reinterpret_cast<u64>(mbi.BaseAddress) : 0ULL);
                }
            }
        }
#endif
        if (Common::IsWriteError(context)) {
            return rasterizer->InvalidateMemory(addr, 8);
        } else {
            return rasterizer->ReadMemory(addr, 8);
        }
        return false;
    }
#endif

    template <bool track, bool is_read>
    void UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;

        // [debug] SHADPS4_PAGEWATCH=<hex guest address>: name whoever puts this page under
        // watch. A page of the game's own executable image being watched is what paralyses the
        // guest - it faults on its own code and the fault can never be resolved.
        {
            static const VAddr watch = [] {
                const char* e = std::getenv("SHADPS4_PAGEWATCH");
                return e ? static_cast<VAddr>(std::strtoull(e, nullptr, 16)) : VAddr{0};
            }();
            if (watch != 0 && watch >= addr && watch < addr + size) {
                static u32 reported = 0;
                if (reported < 16) {
                    ++reported;
                    LOG_CRITICAL(Render_Vulkan,
                                 "PAGEWATCH: {} {} range {:#x}..{:#x} (size {:#x}) covers the "
                                 "watched address",
                                 track ? "track" : "untrack", is_read ? "read" : "write", addr,
                                 addr + size, size);
                }
            }
        }

        size_t page = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        // Acquire locks for the range of pages
        const auto lock_start = locks.begin() + (page / PAGES_PER_LOCK);
        const auto lock_end = locks.begin() + Common::DivCeil(page_end, PAGES_PER_LOCK);
        Common::RangeLockGuard lk(lock_start, lock_end);

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                Protect(range_begin << PM_PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            // Apply the change to the page state
            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // Only start a new range if the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current range up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    template <bool track, bool is_read>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second) {
            // if all pages are contiguous, use the regular UpdatePageWatchers
            const VAddr start_addr = base_addr + (start_range.first << PM_PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PM_PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PM_PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        std::scoped_lock lk(locks[base_page / PAGES_PER_LOCK]);
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                Protect((range_begin << PM_PAGE_BITS), range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate pages
        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            // Apply the change to the page state
            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // If the page is not being updated, skip it
            if (!update) {
                continue;
            }

            // If the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current rango up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_)
    : impl{std::make_unique<Impl>(rasterizer_)} {}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                     RegionBits& mask) const;

} // namespace VideoCore
