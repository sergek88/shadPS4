// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <filesystem>
#include <array>
#include <atomic>
#include <bit>
#include <mutex>
#include <string>
#include <thread>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include "common/alignment.h"
#include "common/debug.h"
#include "common/scope_exit.h"
#include "common/memory_patcher.h"
#include "common/postmortem.h"
#include "common/thread.h"
#include "core/emulator_settings.h" // [SKINDIRECT-STORES] readbacks mode in the census line
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/backing_import.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

// [POLLMAP] vmaGetAllocationMemoryProperties: whether the poll bitmap sits in HOST_CACHED
// memory decides how much of it one sweep slice may read.
#include <vk_mem_alloc.h>

namespace Vulkan {
// [GPUSTALL] Defined in vk_rasterizer.cpp: contents of the submits still executing.
void DumpStalledSubmits(u64 gpu_tick, u64 waiting_tick, u64 slow_ms);
extern std::atomic<u32> g_dbg_frame_no;
extern std::atomic<u64> g_dbg_upload_bytes;
}

namespace VideoCore {

static int DualHomeMode(); // [DUALHOME] defined with the other dual-home helpers below

namespace PageTrace {
struct Range {
    VAddr base;
    u64 size;
    u32 first;
};
static std::vector<Range> g_ranges;
static std::unique_ptr<std::atomic<u32>[]> g_counters; // pages * NUM_FIELDS, per window
static std::unique_ptr<u32[]> g_total;                  // pages * NUM_FIELDS, cumulative
static u32 g_pages = 0;
static bool g_enabled = false;
static std::unique_ptr<u8[]> g_seen_up;      // the first UP of a page is the cpu.Fill() initial state
static std::unique_ptr<u8[]> g_classes_seen; // bit per class ever assigned (P F R M S I)
static std::atomic<u32> g_stalls{0};

static void Init() {
    const char* p = std::getenv("SHADPS4_PAGETRACE");
    if (!p || !p[0]) {
        return;
    }
    const char* cursor = p;
    u32 first = 0;
    while (*cursor != '\0') {
        u64 b = 0, sz = 0;
        int consumed = 0;
        if (std::sscanf(cursor, "%llx:%llx%n", &b, &sz, &consumed) == 2 && b != 0 && sz != 0) {
            const VAddr base = Common::AlignDown(b, BufferCache::CACHING_PAGESIZE);
            const VAddr end = Common::AlignUp(b + sz, BufferCache::CACHING_PAGESIZE);
            const u32 n = static_cast<u32>((end - base) >> BufferCache::CACHING_PAGEBITS);
            g_ranges.push_back({base, end - base, first});
            first += n;
            cursor += consumed;
        } else {
            break;
        }
        if (*cursor == ',') {
            ++cursor;
        }
    }
    if (g_ranges.empty()) {
        return;
    }
    g_pages = first;
    g_counters.reset(new std::atomic<u32>[static_cast<size_t>(g_pages) * NUM_FIELDS]());
    g_total.reset(new u32[static_cast<size_t>(g_pages) * NUM_FIELDS]());
    g_seen_up.reset(new u8[g_pages]());
    g_classes_seen.reset(new u8[g_pages]());
    g_enabled = true;
    LOG_CRITICAL(Render_Vulkan, "PAGETRACE: armed, {} ranges, {} pages", g_ranges.size(), g_pages);
}

bool Enabled() {
    static const bool on = [] {
        Init();
        return g_enabled;
    }();
    return on;
}

static u32 Slot(VAddr addr) {
    for (const auto& r : g_ranges) {
        if (addr >= r.base && addr < r.base + r.size) {
            return r.first + static_cast<u32>((addr - r.base) >> BufferCache::CACHING_PAGEBITS);
        }
    }
    return UINT32_MAX;
}

static VAddr PageAddr(u32 slot) {
    for (const auto& r : g_ranges) {
        const u32 n = static_cast<u32>(r.size >> BufferCache::CACHING_PAGEBITS);
        if (slot >= r.first && slot < r.first + n) {
            return r.base + (static_cast<u64>(slot - r.first) << BufferCache::CACHING_PAGEBITS);
        }
    }
    return 0;
}

void Note(VAddr addr, Field field) {
    if (!Enabled()) {
        return;
    }
    const u32 slot = Slot(addr);
    if (slot == UINT32_MAX) {
        return;
    }
    if (field == UP && g_seen_up[slot] == 0) {
        // The first SynchronizeBuffer over a fresh region reports every page CPU-dirty
        // (region_manager cpu.Fill()): initial state, not a CPU write.
        g_seen_up[slot] = 1;
        return;
    }
    g_counters[static_cast<size_t>(slot) * NUM_FIELDS + field].fetch_add(1, std::memory_order_relaxed);
}

void NoteStall() {
    if (Enabled()) {
        g_stalls.fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteBitmap(const u32* words, size_t num_words, Field field) {
    if (!Enabled()) {
        return;
    }
    for (const auto& r : g_ranges) {
        const u64 first_page = r.base >> BufferCache::CACHING_PAGEBITS;
        const u64 n = r.size >> BufferCache::CACHING_PAGEBITS;
        for (u64 i = 0; i < n; ++i) {
            const u64 page = first_page + i;
            const size_t w = page >> 5;
            if (w >= num_words) {
                break;
            }
            if (words[w] & (1u << (page & 31))) {
                g_counters[static_cast<size_t>(r.first + i) * NUM_FIELDS + field].fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
    }
}

// [POLLJ] the fault manager sizes and copies the presence-window slices by these ranges
std::vector<std::pair<VAddr, u64>> Ranges() {
    std::vector<std::pair<VAddr, u64>> out;
    if (Enabled()) {
        for (const auto& r : g_ranges) {
            out.emplace_back(r.base, r.size);
        }
    }
    return out;
}
} // namespace PageTrace

// [POLLMAP] Sticky set of the pages the GPU polls (BufferCache::POLLMAP_*): one bit per 16 KiB
// page of the 40-bit guest space, filled by BufferCache::PollMapStep on the GPU thread and read
// lock-free from HomePolicy::WantsPt - the register writer asks for every page of every buffer
// it registers, the fault parser and the priority thread ask for journaled pages, so this path
// takes no mutex. A page never leaves the set within a session (the redirect is sticky, like
// the static SHADPS4_HOMEPOLICY_PT list it extends).
// [POLLMAP v3] "the CPU has written this page at least once" - the persistent discriminator
// the curated 04.09 list had implicitly (job-queue flags/counts the CPU writes and the GPU
// polls). Set from the CPU write fault path (InvalidateMemory); read by the POLLMAP fold. Pages
// the GPU polls but the CPU never writes (GPU-produced data) stay out of the policy - importing
// or redirecting those split the GPU's own stores from its loads (0904r..r4).
namespace CpuWrittenEver {
constexpr u64 kPages = u64{1} << 26; // CACHING_NUMPAGES
static std::atomic<u64>* bits = nullptr;
inline void Init() {
    if (bits == nullptr) {
        bits = new std::atomic<u64>[kPages / 64]();
    }
}
inline void Note(VAddr addr, u64 size) {
    if (bits == nullptr) {
        return;
    }
    const u64 first = addr >> 14, last = (addr + size - 1) >> 14;
    for (u64 p = first; p <= last && p < kPages; ++p) {
        bits[p >> 6].fetch_or(u64{1} << (p & 63), std::memory_order_relaxed);
    }
}
inline bool Has(u64 page) {
    return bits != nullptr && page < kPages &&
           (bits[page >> 6].load(std::memory_order_relaxed) >> (page & 63)) & 1;
}
} // namespace CpuWrittenEver
namespace PollMap {
static constexpr u64 kWords = BufferCache::CACHING_NUMPAGES / 64;
static std::unique_ptr<std::atomic<u64>[]> g_owner;
static std::atomic<std::atomic<u64>*> g_words{nullptr};

static void Init() {
    if (!BufferCache::PollMapEnabled() || g_owner) {
        return;
    }
    g_owner.reset(new std::atomic<u64>[kWords]());
    g_words.store(g_owner.get(), std::memory_order_release);
}

// True when the page was not in the set before. GPU thread (PollMapStep).
static bool Insert(u64 page) {
    std::atomic<u64>* const words = g_words.load(std::memory_order_acquire);
    if (words == nullptr || page >= BufferCache::CACHING_NUMPAGES) {
        return false;
    }
    const u64 bit = u64{1} << (page & 63);
    return (words[page >> 6].fetch_or(bit, std::memory_order_relaxed) & bit) == 0;
}

// Any page of [addr, addr + size) in the set. Any thread; one relaxed load per page.
static bool Contains(VAddr addr, u64 size) {
    std::atomic<u64>* const words = g_words.load(std::memory_order_acquire);
    if (words == nullptr) {
        return false;
    }
    const u64 first = addr >> BufferCache::CACHING_PAGEBITS;
    if (first >= BufferCache::CACHING_NUMPAGES) {
        return false;
    }
    const u64 last_wanted = (addr + std::max<u64>(size, 1) - 1) >> BufferCache::CACHING_PAGEBITS;
    const u64 last = std::min<u64>(last_wanted, BufferCache::CACHING_NUMPAGES - 1);
    for (u64 page = first; page <= last; ++page) {
        if (words[page >> 6].load(std::memory_order_relaxed) & (u64{1} << (page & 63))) {
            return true;
        }
    }
    return false;
}
} // namespace PollMap

namespace HomePolicy {
struct Range {
    VAddr base;
    u64 size;
};
static std::vector<Range> g_import; // sorted, merged, page aligned
// [HOMEPOLICY-PT] SHADPS4_HOMEPOLICY_PT=<hex base>:<hex size>[,...]: pages whose BDA page-table
// entries point at guest RAM (the backing, page by page) even while a VRAM buffer owns them for
// descriptor binds. For CPU-written, GPU-read data whose backing is physically split (the
// per-frame list arena): live loads then read what the CPU wrote, the bind keeps working.
static std::vector<Range> g_pt;
static std::mutex g_mutex;
static bool g_enabled = false;

static void ParseRanges(const char* p, std::vector<Range>& out) {
    if (!p || !p[0]) {
        return;
    }
    const char* cursor = p;
    while (*cursor != '\0') {
        u64 b = 0, sz = 0;
        int consumed = 0;
        if (std::sscanf(cursor, "%llx:%llx%n", &b, &sz, &consumed) == 2 && b != 0 && sz != 0) {
            const VAddr base = Common::AlignDown(b, BufferCache::CACHING_PAGESIZE);
            const VAddr end = Common::AlignUp(b + sz, BufferCache::CACHING_PAGESIZE);
            out.push_back({base, end - base});
            cursor += consumed;
        } else {
            break;
        }
        if (*cursor == ',') {
            ++cursor;
        }
    }
    if (out.empty()) {
        return;
    }
    std::sort(out.begin(), out.end(),
              [](const Range& a, const Range& b) { return a.base < b.base; });
    std::vector<Range> merged;
    for (const auto& r : out) {
        if (!merged.empty() && merged.back().base + merged.back().size >= r.base) {
            merged.back().size = std::max(merged.back().base + merged.back().size, r.base + r.size) -
                                 merged.back().base;
        } else {
            merged.push_back(r);
        }
    }
    out.swap(merged);
}

static void Init() {
    ParseRanges(std::getenv("SHADPS4_HOMEPOLICY_IMPORT"), g_import);
    ParseRanges(std::getenv("SHADPS4_HOMEPOLICY_PT"), g_pt);
    if (g_import.empty() && g_pt.empty()) {
        return;
    }
    g_enabled = true;
    if (!g_import.empty()) {
        u64 total = 0;
        for (const auto& r : g_import) {
            total += r.size;
        }
        LOG_CRITICAL(Render_Vulkan,
                     "HOMEPOLICY: armed, {} import ranges, {} KB, first {:#x} last {:#x}",
                     g_import.size(), total >> 10, g_import.front().base,
                     g_import.back().base + g_import.back().size);
    }
    if (!g_pt.empty()) {
        u64 total = 0;
        for (const auto& r : g_pt) {
            total += r.size;
        }
        LOG_CRITICAL(Render_Vulkan,
                     "HOMEPOLICY-PT: armed, {} page-table ranges, {} KB, first {:#x} last {:#x}",
                     g_pt.size(), total >> 10, g_pt.front().base,
                     g_pt.back().base + g_pt.back().size);
    }
}

bool Enabled() {
    static const bool on = [] {
        Init();
        return g_enabled;
    }();
    return on;
}

// [POLLMAP v2] SHADPS4_POLLMAP_PT=1 keeps the v1 page-table redirect; by default the polled set
// feeds the IMPORT policy instead: the owner buffer of a polled page is (re)created with guest
// RAM as its single home, so descriptor stores, live loads and CPU writes all meet in one
// place. v1 redirected the page table only (0904r..r3): the ND workers store their results
// through unbounded descriptors (is_written is suppressed for them, so no write history),
// those stores landed in the VRAM copy while live loads read guest RAM, and the deck hung
// within seconds three times. The deck was cured on 04.09 by exactly this import policy
// (HOMEPOLICY_IMPORT for the polled pages); the page-table redirect was only ever for the
// list arena that cannot be imported.
// [POLLMAP-OBSERVE] SHADPS4_POLLMAP_OBSERVE=1: collect the set, apply no policy - separates the
// cost of the shader-side marking from the effect of the redirect/import (0904r..r4 all hung).
bool PollMapObserve() {
    static const bool ob = [] {
        const char* e = std::getenv("SHADPS4_POLLMAP_OBSERVE");
        return e && e[0] == '1';
    }();
    return ob;
}

bool PollMapPtMode() {
    static const bool pt = [] {
        const char* e = std::getenv("SHADPS4_POLLMAP_PT");
        return e && e[0] == '1';
    }();
    return pt;
}

bool WantsImport(VAddr addr, u64 size) {
    if (!PollMapObserve() && PollMap::Contains(addr, size)) {
        return true;
    }
    if (!Enabled()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    // first range whose end lies past addr
    const auto it = std::upper_bound(g_import.begin(), g_import.end(), addr,
                                     [](VAddr a, const Range& r) { return a < r.base + r.size; });
    return it != g_import.end() && it->base < addr + size;
}

// [POLLMAP v3.2] static list only (no dynamic set): "is this page the arena's business".
bool StaticWantsPt(VAddr addr, u64 size) {
    if (!Enabled() || g_pt.empty()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    const auto it = std::upper_bound(g_pt.begin(), g_pt.end(), addr,
                                     [](VAddr a, const Range& r) { return a < r.base + r.size; });
    return it != g_pt.end() && it->base < addr + size;
}

bool WantsPt(VAddr addr, u64 size) {
    // [POLLMAP] the dynamic set answers first and without the mutex: this runs for every page
    // of every registered buffer.
    if (!PollMapObserve() && PollMapPtMode() && PollMap::Contains(addr, size)) {
        return true;
    }
    if (!Enabled() || g_pt.empty()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    const auto it = std::upper_bound(g_pt.begin(), g_pt.end(), addr,
                                     [](VAddr a, const Range& r) { return a < r.base + r.size; });
    return it != g_pt.end() && it->base < addr + size;
}

u64 CoveredBytes(VAddr addr, u64 size) {
    if (!Enabled()) {
        return 0;
    }
    std::scoped_lock lk{g_mutex};
    u64 covered = 0;
    for (const auto& r : g_import) {
        const VAddr lo = std::max<VAddr>(r.base, addr);
        const VAddr hi = std::min<VAddr>(r.base + r.size, addr + size);
        if (lo < hi) {
            covered += hi - lo;
        }
    }
    return covered;
}

std::vector<std::pair<VAddr, u64>> Ranges() {
    std::vector<std::pair<VAddr, u64>> out;
    if (!Enabled()) {
        return out;
    }
    std::scoped_lock lk{g_mutex};
    out.reserve(g_import.size());
    for (const auto& r : g_import) {
        out.emplace_back(r.base, r.size);
    }
    return out;
}

void Subtract(VAddr addr, u64 size) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    std::vector<Range> out;
    bool changed = false;
    for (const auto& r : g_import) {
        const VAddr lo = std::max<VAddr>(r.base, addr);
        const VAddr hi = std::min<VAddr>(r.base + r.size, addr + size);
        if (lo >= hi) {
            out.push_back(r);
            continue;
        }
        changed = true;
        if (r.base < lo) {
            out.push_back({r.base, lo - r.base});
        }
        if (hi < r.base + r.size) {
            out.push_back({hi, r.base + r.size - hi});
        }
    }
    if (changed) {
        LOG_CRITICAL(Render_Vulkan, "HOMEPOLICY: unmap {:#x}:{:#x} trims import ranges {} -> {}", addr,
                     size, g_import.size(), out.size());
        g_import.swap(out);
    }
}
} // namespace HomePolicy

thread_local u64 g_bind_pgm_hash = 0;

// [RBQUARTER] readback delivery per 4 KiB tracker page (SHADPS4_RB_QUARTER=1)
static bool RbQuarterEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_RB_QUARTER");
        const bool v = e && e[0] == '1';
        if (v) {
            LOG_CRITICAL(Render_Vulkan, "RBQUARTER: armed - readback delivery per 4 KiB quarter");
        }
        return v;
    }();
    return on;
}
static std::atomic<u64> rb_quarter_delivered{0};
static std::atomic<u64> rb_quarter_refused{0};

// [HOMEPOLICY] counters (GPU thread writes, summary reads on the same thread)
static std::atomic<u64> g_policy_owners{0};
static std::atomic<u64> g_policy_owner_bytes{0};
static std::atomic<u64> g_policy_unions{0};
static std::atomic<u64> g_policy_refused{0};
static std::atomic<u64> g_policy_prearmed{0};
static std::atomic<u64> g_policy_prearmed_bytes{0};
static bool DualKeepMarks();

namespace {
// [SNAPRO] Off by default: SHADPS4_SNAPRO=1 turns it on, SHADPS4_SNAPRO_MAX_MB caps a single
// range (default 32 MB - a bigger one is a poll target or a whole heap, not vertex data).
bool SnaproEnabled() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_SNAPRO");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}
u64 SnaproMinBytes() {
    static const u64 v = [] {
        const char* e = std::getenv("SHADPS4_SNAPRO_MIN_KB");
        const u64 kb = e != nullptr ? std::strtoull(e, nullptr, 10) : 256ull;
        return kb << 10;
    }();
    return v;
}
u32 SnaproMinBinds() {
    static const u32 v = [] {
        const char* e = std::getenv("SHADPS4_SNAPRO_MIN_BINDS");
        const u32 n = e != nullptr ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 8u;
        return n != 0 ? n : 8u;
    }();
    return v;
}
u64 SnaproBudgetBytes() {
    static const u64 v = [] {
        const char* e = std::getenv("SHADPS4_SNAPRO_BUDGET_MB");
        const u64 mb = e != nullptr ? std::strtoull(e, nullptr, 10) : 64ull;
        return (mb != 0 ? mb : 64ull) << 20;
    }();
    return v;
}
// [SNAPRO_STALE] Ceiling experiment: copy each range ONCE (first sight) and never refresh it.
// The picture is expected to go wrong (animated data freezes) - the point is the FPS ceiling of
// "every bounded read-only range is read from VRAM instead of guest memory over PCIe".
// =1: stale from the start (breaks logo videos and menus - the GPU reads CPU-written frames).
// =2: armed - becomes stale once the file C:\Users\user\ctl\stale.on appears (the stand creates
// it on the deck), so the ceiling is measured in gameplay and not in a frozen menu.
bool SnaproStale() {
    static const int mode = [] {
        const char* e = std::getenv("SHADPS4_SNAPRO_STALE");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    if (mode == 1) {
        return true;
    }
    if (mode != 2) {
        return false;
    }
    static bool armed_on = false;
    static u64 last_check = 0;
    const u64 frame = Vulkan::g_dbg_frame_no.load(std::memory_order_relaxed);
    if (!armed_on && frame >= last_check + 60) {
        last_check = frame;
        if (std::filesystem::exists("C:\\Users\\user\\ctl\\stale.on")) {
            armed_on = true;
            LOG_CRITICAL(Render, "SNAPRO_STALE: armed -> ON at frame {}", frame);
        }
    }
    return armed_on;
}
u64 SnaproMaxBytes() {
    static const u64 v = [] {
        const char* e = std::getenv("SHADPS4_SNAPRO_MAX_MB");
        const u64 mb = e != nullptr ? std::strtoull(e, nullptr, 10) : 32ull;
        return (mb != 0 ? mb : 32ull) << 20;
    }();
    return v;
}
// [SKINVRAM] SHADPS4_SKINVRAM=<hex hash>[,<hex hash>...], parsed once (SKIPCS grammar). Empty
// when the env is unset - the hot-path check in SkinVramWanted is then one empty() test.
const std::vector<u64>& SkinVramHashes() {
    static const std::vector<u64> wanted = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_SKINVRAM");
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
    return wanted;
}
// [SKINDIRECT] SHADPS4_SKINDIRECT=<hex hash>[,...], same grammar as SKINVRAM/DESCDUMP. Parsed
// once; empty while the env is unset, and then every entry point is one empty() test and the
// emitted SPIR-V is byte-identical to today.
const std::vector<u64>& SkinDirectHashList() {
    static const std::vector<u64> wanted = [] {
        std::vector<u64> out;
        const char* e = std::getenv("SHADPS4_SKINDIRECT");
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
    return wanted;
}
// [SKINDIRECT] What the recompiler converted, published for the bind-time census. Written once
// per (shader, sharp) at compile time (shader compilation may run on worker threads, hence the
// mutex) and read lock-free afterwards: entries below `count` never change again.
struct SkinDirectEntry {
    u64 hash;
    u32 sharp_idx;
    u32 word;
    bool is_store; // [SKINDIRECT-STORES] output V#: bound WRITTEN binding instead of write_const_live
};
constexpr u32 kSkinDirectMax = 64;
std::array<SkinDirectEntry, kSkinDirectMax> g_skind_reg{};
std::atomic<u32> g_skind_reg_n{0};
std::mutex g_skind_reg_mutex;
std::atomic<u64> g_skind_converted{0}; // reads whose BDA rewrite was declined
std::atomic<u64> g_skind_declined{0};  // matched the word but the V# was not bindable
// [SKINDIRECT-STORES] compile-time census (atomics: shader compiles may run on worker threads)
std::atomic<u64> g_skind_converted_stores{0}; // store IR insts kept on a written binding
std::atomic<u64> g_skind_lowered_stores{0};   // ...that LowerBufferFormatToRaw turned into raw OpStores
std::atomic<u64> g_skind_fmt0_stores{0};      // ...dropped by the FMT0 guard (zero V# permutation)
std::atomic<u64> g_skind_store_shape_odd{0};  // compile-time V# was not the 28-byte record shape
std::atomic<u64> g_skind_meta_skips{0};       // dispatch of a listed shader eaten by IsComputeMetaClear
std::atomic<u32> g_skind_store_entries{0};    // registry entries with is_store

// [LOSTUPD] SHADPS4_LOSTUPD=1. The symptom under investigation is the character keeping its
// previous pose: a limb stays drawn where it was a frame ago. One mechanism produces exactly
// that and no existing counter can see it.
//
// The tracker works in 4 KiB pages, but the game's data does not respect page borders. If the
// CPU writes anything into a page that ALSO holds vertices the GPU has just computed, the whole
// page is marked CPU-dirty, and the next SynchronizeBuffer uploads the guest copy of that page
// over the GPU result. The guest copy never held the new pose - so the old one comes back.
// SynchronizeBuffer already refuses this upload for import-backed and dual-home owners; an
// ordinary cache-owned buffer has no such guard.
//
// Measured only on the READ path (is_written == false). On the written path ForEachUploadRange
// marks the range GPU-modified itself right after the upload, so a check there would count its
// own bookkeeping. The read path is also the interesting one: it is the draw's vertex bind.
//
// The check cannot run inside the upload callback: ForEachUploadRange holds the region lock
// while calling it and IsRegionGpuModified takes the same non-recursive lock - that is a hang,
// not a slowdown. Ranges are therefore collected first and tested after the lock is gone.
std::atomic<u64> g_lostupd_runs{0};    // read-path upload runs landing on GPU-written pages
std::atomic<u64> g_lostupd_bytes{0};   // ...their total size
std::atomic<u64> g_lostupd_total{0};   // all read-path upload runs, for the share
bool LostUpdateProbeFlag() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_LOSTUPD");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

// [VTXBAR] SHADPS4_VTXBAR=1. Second candidate for the character keeping its previous pose.
//
// The skinning worker writes the character's vertices with a compute dispatch; the draw that
// follows reads the same memory as vertex input. That needs a compute->vertex barrier, and
// BindVertexBuffers asks for one only when the range is marked GPU-modified.
//
// A buffer that lives in guest memory (import-backed) never gets that mark: SynchronizeBuffer
// returns before the tracker is touched. The header says so in as many words next to
// SkinDirectOutputIntersects - which exists precisely to paper over this for the experimental
// store conversion. With the conversion off, and it is off in the build that is played, nothing
// covers it: the draw can read the vertices while the dispatch is still writing them, and what
// gets drawn is whatever was there a frame ago - the previous pose.
//
// The flag turns the missing request on and counts it. Asking is nearly free: GetBarrier emits
// only when the buffer's recorded access actually flips, so the read->read draws that follow
// the first one cost one compare.
std::atomic<u64> g_vtxbar_asked{0};     // vertex binds where the import-backed owner was asked
std::atomic<u64> g_vtxbar_emitted{0};   // ...and a barrier really came out
bool VertexBarrierImportFlag() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_VTXBAR");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

// [CONSTCACHE][VERIFY] SHADPS4_CONSTCACHE_VERIFY=1. The constant cache hands back last frame's
// copy whenever IsRegionCpuModified says the guest has not written the range. That is an
// assumption about how the game updates its constants, and an assumption is not a proof: if the
// range is refreshed any other way, the shader is handed stale bytes. For the skinning matrices
// stale bytes are the previous pose - exactly the symptom under investigation.
//
// Verify mode stores a checksum of what was copied and re-checks it on every hit. A mismatch is
// a hit that served content the guest had already changed. Deliberately expensive: it hashes the
// range on every hit, so it is a diagnostic run, never a setting to play with.
std::atomic<u64> g_cc_verify_hits{0};
std::atomic<u64> g_cc_verify_stale{0};
bool ConstCacheVerifyFlag() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_CONSTCACHE_VERIFY");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}
u64 ConstCacheSum(const u8* p, u32 size) {
    // FNV-1a over the range: cheap, no dependencies, and only ever compared with itself.
    u64 h = 1469598103934665603ull;
    for (u32 i = 0; i < size; ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

// [SKINDIRECT-STORES] SHADPS4_SKINDIRECT_STORES=1: the listed shaders' OUTPUT stores (block
// dwords in SHADPS4_SKINDIRECT_STORE_WORDS, default 0,4,8 = the skinning worker's position /
// normal / tangent) keep an ordinary WRITTEN SSBO binding instead of write_const_live. Parsed
// once here so the recompiler (resource_tracking_pass.cpp) and the 5 s report see the same
// values; unset -> every entry point is one bool test and the SPIR-V is byte-identical.
bool SkinDirectStoresFlag() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_SKINDIRECT_STORES");
        if (!(e && e[0] == '1')) {
            return false;
        }
        // [PERMLAYOUT] Hard gate against the disk shader cache (PCACHEDISK). GetProgram
        // (vk_pipeline_cache.cpp) binds EVERY permutation through program->info.buffers - the
        // FIRST compile's list - while a SPECMISS permutation's SPIR-V numbers its bindings from
        // its own Info. With the feature the worker's Info carries three more (written)
        // descriptors than without it, so a program preloaded from a cache written with the
        // other setting shifts every binding after the first output store of each recompiled
        // permutation: stores land in the flatbuf / page table / fault buffer. The profile has no
        // field for the flag (vk_pipeline_serialization.cpp is outside this patch), so the only
        // safe answer while the disk cache is on is to keep the feature off - loudly, so the
        // A/B shows converted_stores=0 with the reason next to it instead of a device fault.
        // Same condition as PipelineCache's serialization init: toml switch, or SHADPS4_PCACHE=1
        // in directory mode.
        const char* pc = std::getenv("SHADPS4_PCACHE");
        const bool pcache_env = pc && pc[0] == '1';
        const bool disk_cache = EmulatorSettings.IsPipelineCacheEnabled() ||
                                (pcache_env && !EmulatorSettings.IsPipelineCacheArchived());
        if (disk_cache) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINDIRECT-STORES: REFUSED - the disk pipeline cache is on (toml={} "
                         "SHADPS4_PCACHE={}); a preloaded program binds permutations with a "
                         "different descriptor layout (PERMLAYOUT). Run with SHADPS4_PCACHE=0 and "
                         "the toml cache off; the census will print converted_stores=0 for this run",
                         EmulatorSettings.IsPipelineCacheEnabled() ? 1 : 0, pcache_env ? 1 : 0);
            return false;
        }
        return true;
    }();
    return v;
}
const std::vector<u32>& SkinDirectStoreWordList() {
    static const std::vector<u32> words = [] {
        const char* e = std::getenv("SHADPS4_SKINDIRECT_STORE_WORDS");
        if (e == nullptr) {
            return std::vector<u32>{0u, 4u, 8u}; // position / normal / tangent of 0x8f067540
        }
        std::vector<u32> out;
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
// SHADPS4_SKINDIRECT_STORE_IMGINV=0: skip texture_cache.InvalidateMemoryFromGPU for the
// converted outputs (the vertex ring aliases no image). Default on - measure the bind loop first.
bool SkinDirectStoreImgInvFlag() {
    static const bool v = [] {
        const char* e = std::getenv("SHADPS4_SKINDIRECT_STORE_IMGINV");
        return e == nullptr || e[0] != '0';
    }();
    return v;
}
// SHADPS4_SKINDIRECT_STORE_VERIFY: run-time detectors (ring of the converted outputs, guest-RAM
// upload landing inside one, vertex-buffer consumers). ON whenever the store conversion is on
// (=0 turns them off): the A/B gates read uploads_in_output / consumer_stream from the census,
// and the detectors cost two compares per upload range or vertex bind. Never on with STORES off,
// so a baseline run keeps today's paths untouched.
bool SkinDirectStoreVerifyFlag() {
    static const bool v = [] {
        if (!SkinDirectStoresFlag()) {
            return false;
        }
        const char* e = std::getenv("SHADPS4_SKINDIRECT_STORE_VERIFY");
        return !(e && e[0] == '0');
    }();
    return v;
}
} // namespace


static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

// [postmortem] Fault-service counters, hoisted from function-local statics in
// WaitForBacking / TryServeFaultFast so the fatal PARTIAL window can be printed
// from the crash handler (the periodic 5s reporters keep zeroing them; the
// postmortem dump does loads only and never disturbs the live windows).
namespace rbstat {
// WaitForBacking (NOBK) counters.
std::atomic<u64> c_wait{0}, c_ok{0}, c_timeout{0}, c_wait_us{0};
std::atomic<VAddr> last_addr{0};
std::atomic<s64> next_ms{0};
// TryServeFaultFast (RBSTAT) counters.
std::atomic<u64> c_fast{0}, c_no_buffer{0}, c_unsubmitted{0}, c_empty{0}, c_changed{0},
    c_lat_us{0}, c_import_heal{0}, c_heal_nobk{0};
std::atomic<s64> next_report_ms{0};
// Maxima of the fatal tail (CAS-max, never reset by the periodic reporters).
std::atomic<u64> rb_lat_max_us{0}, nb_wait_max_us{0};
// Per-thread breakdown, accumulated only when the faulting thread IS the guest
// tick-writer ("GPU Interrupts Handler GfxEop", identified via RINGWATCH).
std::atomic<u64> tw_lock_us{0}, tw_slow_us{0}, tw_fence_us{0}, tw_poll_us{0}, tw_nobk_us{0},
    tw_nobk_cnt{0};
std::atomic<s64> tw_nobk_inflight_ms{0};

inline bool IsTickWriter() {
#ifdef _WIN32
    const u32 tw = Postmortem::g_tick_writer_tid.load(std::memory_order_relaxed);
    return tw != 0 && tw == GetCurrentThreadId();
#else
    return false;
#endif
}

// [DLRING] Last downloads delivered into guest RAM (both readback paths, all four
// TryWriteBacking sites). The 05.09 pier crash (eboot+0x5fa3d5) found two floats where a
// pointer belonged in a list the game had just rebuilt after the cutscene: the shape of a
// stale VRAM copy delivered over reused memory. The ring names which download touched the
// crashing page and how long before the crash; the handler passes the page as a hint.
struct DlRec {
    u64 addr;
    u32 size;
    u8 kind; // 0 fast/quarter 1 fast/full 2 slow/quarter 3 slow/full
    s64 t_ms;
};
constexpr size_t kDlRing = 256;
DlRec dl_ring[kDlRing]{};
std::atomic<u64> dl_seq{0};
inline void NoteDownload(VAddr addr, u64 size, u8 kind) {
    const u64 i = dl_seq.fetch_add(1, std::memory_order_relaxed) % kDlRing;
    dl_ring[i] = DlRec{addr, static_cast<u32>(size), kind, Postmortem::NowMs()};
}
} // namespace rbstat
// Crash-site page passed by the signal handler; rbstat is nested in VideoCore.
std::atomic<u64> g_crash_addr_hint{0};
namespace rbstat {
// Runs inside the exception handler: loads only, no exchange (the counters must
// survive for a second reader), no locks, no allocations.
void DumpReadbackPostmortem() {
    const s64 now = Postmortem::NowMs();
    const s64 wnd = next_report_ms.load(std::memory_order_relaxed);
    LOG_CRITICAL(Render_Vulkan,
                 "POSTMORTEM-RB: window_age_ms={} fast={} no_buf={} unsub={} empty={} changed={} "
                 "heal={} nobk={} lat_sum_us={} lat_max_us={}",
                 wnd ? now - (wnd - 5000) : -1, c_fast.load(std::memory_order_relaxed),
                 c_no_buffer.load(std::memory_order_relaxed),
                 c_unsubmitted.load(std::memory_order_relaxed),
                 c_empty.load(std::memory_order_relaxed), c_changed.load(std::memory_order_relaxed),
                 c_import_heal.load(std::memory_order_relaxed),
                 c_heal_nobk.load(std::memory_order_relaxed),
                 c_lat_us.load(std::memory_order_relaxed),
                 rb_lat_max_us.load(std::memory_order_relaxed));
    LOG_CRITICAL(Render_Vulkan,
                 "POSTMORTEM-NOBK: wait={} ok={} timeout={} sum_us={} max_us={} last={:#x}",
                 c_wait.load(std::memory_order_relaxed), c_ok.load(std::memory_order_relaxed),
                 c_timeout.load(std::memory_order_relaxed),
                 c_wait_us.load(std::memory_order_relaxed),
                 nb_wait_max_us.load(std::memory_order_relaxed),
                 last_addr.load(std::memory_order_relaxed));
    const s64 nb_inflight = tw_nobk_inflight_ms.load(std::memory_order_relaxed);
    LOG_CRITICAL(Render_Vulkan,
                 "POSTMORTEM-TW: lock_us={} slow_us={} fence_us={} poll_us={} nobk_us={} "
                 "nobk_cnt={} nobk_inflight_ms={}",
                 tw_lock_us.load(std::memory_order_relaxed),
                 tw_slow_us.load(std::memory_order_relaxed),
                 tw_fence_us.load(std::memory_order_relaxed),
                 tw_poll_us.load(std::memory_order_relaxed),
                 tw_nobk_us.load(std::memory_order_relaxed),
                 tw_nobk_cnt.load(std::memory_order_relaxed),
                 nb_inflight ? now - nb_inflight : 0);
    {
        const u64 seq = dl_seq.load(std::memory_order_relaxed);
        const u64 hint = VideoCore::g_crash_addr_hint.load(std::memory_order_relaxed);
        constexpr u64 kRingPage = u64{1} << 14; // BufferCache::CACHING_PAGEBITS, not visible here
        const u64 hint_page = hint & ~(kRingPage - 1);
        char line[1400]{};
        int n = std::snprintf(line, sizeof(line), "POSTMORTEM-DLRING: total=%llu hint=%#llx last=",
                              static_cast<unsigned long long>(seq),
                              static_cast<unsigned long long>(hint));
        const u64 shown = std::min<u64>(seq, 24);
        for (u64 k = 0; k < shown && n < 1300; ++k) {
            const DlRec& r = dl_ring[(seq - 1 - k) % kDlRing];
            n += std::snprintf(line + n, sizeof(line) - n, " %llx+%x/%u@-%lldms",
                               static_cast<unsigned long long>(r.addr), r.size, r.kind,
                               static_cast<long long>(now - r.t_ms));
        }
        LOG_CRITICAL(Render_Vulkan, "{}", line);
        if (hint != 0) {
            char hl[1000]{};
            int hn = std::snprintf(hl, sizeof(hl), "POSTMORTEM-DLHINT: page=%#llx hits=",
                                   static_cast<unsigned long long>(hint_page));
            u64 hits = 0;
            const u64 span = std::min<u64>(seq, kDlRing);
            for (u64 k = 0; k < span && hn < 900; ++k) {
                const DlRec& r = dl_ring[(seq - 1 - k) % kDlRing];
                if (r.addr < hint_page + kRingPage && r.addr + r.size > hint_page) {
                    ++hits;
                    hn += std::snprintf(hl + hn, sizeof(hl) - hn, " %llx+%x/%u@-%lldms",
                                        static_cast<unsigned long long>(r.addr), r.size, r.kind,
                                        static_cast<long long>(now - r.t_ms));
                }
            }
            LOG_CRITICAL(Render_Vulkan, "{} (n={})", hl, hits);
        }
    }
}
} // namespace rbstat

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      // [CSBATCH] the stream ring carries a device address: the batch kernel reads the
      // record-time snapshots of the worker's small inputs through raw pointers. Only this
      // ring (not staging/download/device): one dedicated allocation instead of a
      // sub-allocation, same size.
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize,
                    vk::BufferUsageFlagBits::eShaderDeviceAddress},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler,
                           [] {
                               // SHADPS4_PTHOST=1: device-local host-visible (ReBAR) page
                               // table. Residency repair becomes a plain CPU store that a
                               // spinning shader observes immediately - the repair no longer
                               // queues BEHIND the very dispatch that spins on the missing
                               // entry (that ordering was a GPU hang -> driver reset).
                               const char* e = std::getenv("SHADPS4_PTHOST");
                               return e && e[0] == '1' ? MemoryUsage::Stream
                                                       : MemoryUsage::DeviceLocal;
                           }(),
                           0, AllFlags,
                           DualHomeEnabled() ? BDA_PAGETABLE_SIZE + DUAL_TABLE_BYTES
                                             : BDA_PAGETABLE_SIZE} {
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // The page table is device local and starts out holding whatever the driver left in video
    // memory, while every shader treats a zero entry as "this page is not resident". A stray
    // guest pointer - the job system reads plenty of those before its structures are seeded -
    // then lands on a junk entry, passes the null check and is dereferenced as a real device
    // address. The device dies with a ReadInvalid fault at an address that belongs to nothing.
    if (!bda_pagetable_buffer.mapped_data.empty()) {
        std::memset(bda_pagetable_buffer.mapped_data.data(), 0, BDA_PAGETABLE_SIZE);
    } else {
        bda_pagetable_buffer.Fill(0, static_cast<u32>(BDA_PAGETABLE_SIZE), 0);
    }
    // [DUALHOME] window entries are never zero: unmatched pages point at a sink page, so the
    // second store of a raw write can be issued without a branch.
    if (DualHomeEnabled()) {
        if (const char* p = std::getenv("SHADPS4_DUALHOME")) {
            const char* cursor = p;
            while (*cursor != '\0') {
                u64 d_base = 0, d_size = 0;
                int consumed = 0;
                if (std::sscanf(cursor, "%llx:%llx%n", &d_base, &d_size, &consumed) == 2 &&
                    d_base != 0 && d_size != 0) {
                    const VAddr b = Common::AlignDown(d_base, CACHING_PAGESIZE);
                    const VAddr e = Common::AlignUp(d_base + d_size, CACHING_PAGESIZE);
                    if (b >= DUAL_WINDOW_BASE &&
                        e <= DUAL_WINDOW_BASE + (DUAL_WINDOW_PAGES << CACHING_PAGEBITS)) {
                        dual_ranges.emplace_back(b, e - b);
                    } else {
                        LOG_CRITICAL(Render_Vulkan, "DUALHOME: range {:#x}:{:#x} outside the window, ignored",
                                     d_base, d_size);
                    }
                    cursor += consumed;
                } else {
                    break;
                }
                if (*cursor == ',') {
                    ++cursor;
                }
            }
        }
        dual_sink = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::DeviceLocal, 0,
                                             AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                             DUAL_SINK_BYTES);
        Vulkan::SetObjectName(instance.GetDevice(), dual_sink->Handle(), "Dual Home Sink");
        const vk::DeviceAddress sink = dual_sink->BufferDeviceAddress();
        std::vector<vk::DeviceAddress> window(DUAL_WINDOW_PAGES + 1);
        // bit 63 marks a sink entry (not armed): the writeback kernel and poll reads skip it,
        // the dual store masks it off.
        constexpr u64 kSinkBit = u64{1} << 63;
        for (u64 i = 0; i < DUAL_WINDOW_PAGES; ++i) {
            window[i] = (sink + ((i & 63) << CACHING_PAGEBITS)) | kSinkBit;
        }
        window[DUAL_WINDOW_PAGES] = sink | kSinkBit;
        if (!bda_pagetable_buffer.mapped_data.empty()) {
            std::memcpy(bda_pagetable_buffer.mapped_data.data() + BDA_PAGETABLE_SIZE, window.data(),
                        DUAL_TABLE_BYTES);
        } else {
            WriteDataBuffer(bda_pagetable_buffer, BDA_PAGETABLE_SIZE, window.data(),
                            static_cast<u32>(DUAL_TABLE_BYTES));
        }
        LOG_CRITICAL(Render_Vulkan, "DUALHOME: enabled ranges={} window=+{} KB sink={:#x}",
                     dual_ranges.size(), DUAL_TABLE_BYTES >> 10, static_cast<u64>(sink));
    }

    // [SKINVRAM] SHADPS4_SKINVRAM=<hex hash>[,...] (+ SHADPS4_SKINVRAM_MB ring size, _SLOT_KB
    // slot size, _BASE synthetic VA in hex, _LOG=1 verbose, _VERIFY=1 staleness self-check,
    // _GPUMOD=allow): configuration, geometry checks and the ring allocation (here, where the
    // VRAM budget is free). The page-table entries and the runtime state are still set up by
    // the first redirected dispatch on the GPU thread (SkinVramInit), which is where the
    // window can first be checked against real guest mappings. Unset = nothing below runs.
    {
        skin.hashes = SkinVramHashes();
        if (!skin.hashes.empty()) {
            if (const char* v = std::getenv("SHADPS4_SKINVRAM_MB")) {
                const u64 mb = std::strtoull(v, nullptr, 10);
                if (mb != 0) {
                    skin.ring_bytes = mb << 20;
                }
            }
            if (const char* v = std::getenv("SHADPS4_SKINVRAM_SLOT_KB")) {
                const u64 kb = std::strtoull(v, nullptr, 10);
                if (kb != 0) {
                    skin.slot_bytes = kb << 10;
                }
            }
            if (const char* v = std::getenv("SHADPS4_SKINVRAM_BASE")) {
                const u64 b = std::strtoull(v, nullptr, 16);
                if (b != 0) {
                    skin.synth_base = b;
                }
            }
            if (const char* v = std::getenv("SHADPS4_SKINVRAM_LOG")) {
                skin.verbose = v[0] != '\0' && v[0] != '0';
            }
            if (const char* v = std::getenv("SHADPS4_SKINVRAM_VERIFY")) {
                skin.verify = v[0] != '\0' && v[0] != '0';
            }
            if (const char* v = std::getenv("SHADPS4_SKINVRAM_GPUMOD")) {
                // "allow": count the GPU-owned ranges but redirect them anyway (measurement
                // mode). Default: they are a skip, because guest RAM is then not the authority.
                skin.gpumod_allow = std::string_view{v} == "allow";
            }
            const char* refuse = nullptr;
            const char* tag = std::getenv("SHADPS4_BDATAG");
            const char* noderef = std::getenv("SHADPS4_BDA_NODEREF");
            const u64 window = skin.ring_bytes + CACHING_PAGESIZE;
            if (tag != nullptr && std::strtoul(tag, nullptr, 10) != 0) {
                // ChangeRegister stamps the page index into bits 48-63 of every entry and the
                // shader rejects a mismatch; synthetic entries carry no stamp.
                refuse = "SHADPS4_BDATAG is set (synthetic entries carry no page stamp)";
            } else if (noderef != nullptr && noderef[0] == '1') {
                refuse = "SHADPS4_BDA_NODEREF=1 (nothing is dereferenced)";
            } else if (skin.slot_bytes == 0 || (skin.slot_bytes % CACHING_PAGESIZE) != 0) {
                refuse = "slot size must be a multiple of 16 KiB";
            } else if (skin.ring_bytes < 2 * skin.slot_bytes ||
                       (skin.ring_bytes % skin.slot_bytes) != 0 ||
                       skin.ring_bytes / skin.slot_bytes > (u64{1} << 20)) {
                refuse = "ring size must be a multiple of the slot size (2 .. 2^20 slots)";
            } else if ((skin.synth_base & (CACHING_PAGESIZE - 1)) != 0) {
                refuse = "base must be 16 KiB aligned";
            } else if (skin.synth_base >= (u64{1} << 40) ||
                       window > (u64{1} << 40) - skin.synth_base) {
                refuse = "window must fit below 2^40 (page-table coverage)";
            } else if (skin.ring_bytes > (u64{1} << 30)) {
                // VMA is asked to stay WITHIN_BUDGET and a refusal is a hard ASSERT (int3), so
                // an absurd SHADPS4_SKINVRAM_MB must be a clean refusal here, not a crash later.
                refuse = "ring larger than 1 GiB";
            }
            if (refuse != nullptr) {
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM: refused - {} (hashes={} base={:#x} ring={} MB slot={} KB)",
                             refuse, skin.hashes.size(), skin.synth_base, skin.ring_bytes >> 20,
                             skin.slot_bytes >> 10);
            } else {
                skin.num_slots = static_cast<u32>(skin.ring_bytes / skin.slot_bytes);
                skin.config_ok = true;
                // The ring is allocated HERE, not at the first redirected dispatch: the VRAM
                // budget is still untouched (SHADPS4_TEXGC_TRIGGER_MB=9000 plus the imports
                // leave little room mid-frame) and an over-budget vmaCreateBuffer is a hard
                // ASSERT, which must not be able to fire in the middle of a frame.
                SkinVramCreateRing(); // may clear config_ok if the ring is unusable
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM: configured hashes={} first={:#x} base={:#x} ring={} MB "
                             "slot={} KB slots={} log={} verify={} gpumod={} host_ring={} ok={}",
                             skin.hashes.size(), skin.hashes.front(), skin.synth_base,
                             skin.ring_bytes >> 20, skin.slot_bytes >> 10, skin.num_slots,
                             skin.verbose, skin.verify, skin.gpumod_allow ? "allow" : "skip",
                             skin.host_ring, skin.config_ok);
            }
        }
    }

    // [POLLMAP] SHADPS4_POLLMAP=1: the poll bitmap the shaders mark (host-visible|host-coherent,
    // with a device address) and its address slot in the fault buffer (FaultManager sized the
    // buffer for it and zeroed it). Before the early returns below, like BACKIMP. Any failure
    // leaves the slot zero, which the shader treats as "no bitmap" - never fatal.
    if (PollMapEnabled()) {
        PollMap::Init();
    CpuWrittenEver::Init();
        pollmap_buffer = std::make_unique<Buffer>(
            instance, scheduler, MemoryUsage::Download, 0,
            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, POLLMAP_BYTES);
        Vulkan::SetObjectName(instance.GetDevice(), pollmap_buffer->Handle(), "POLLMAP Bitmap");
        Buffer* const fb = fault_manager.GetFaultBuffer();
        const u64 slot = PollMapSlotOffset();
        VkMemoryPropertyFlags mem_flags{};
        vmaGetAllocationMemoryProperties(instance.GetAllocator(), pollmap_buffer->buffer.allocation,
                                         &mem_flags);
        pollmap_cached = (mem_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
        pollmap_pt_host = !bda_pagetable_buffer.mapped_data.empty();
        if (pollmap_buffer->mapped_data.size() < POLLMAP_BYTES) {
            LOG_CRITICAL(Render_Vulkan,
                         "POLLMAP: bitmap buffer is not host-mapped - marking stays disabled");
            pollmap_buffer.reset();
        } else if (!pollmap_buffer->is_coherent) {
            // Download only PREFERS a coherent type; the fold reads the mapping with no
            // invalidate, so a non-coherent type would show it stale lines forever.
            LOG_CRITICAL(Render_Vulkan,
                         "POLLMAP: bitmap memory is not host-coherent (flags {:#x}) - marking "
                         "stays disabled",
                         static_cast<u32>(mem_flags));
            pollmap_buffer.reset();
        } else if (slot + POLLMAP_SLOT_BYTES != fb->SizeBytes()) {
            // The slot is the tail of the fault buffer by construction (FaultManager appends
            // POLLMAP_SLOT_BYTES behind the presence window); anything else means the two
            // layouts drifted apart and the shader would read the address from the wrong place.
            LOG_CRITICAL(Render_Vulkan,
                         "POLLMAP: address slot {:#x}+{} is not the tail of the fault buffer "
                         "({:#x}) - layout drifted, marking stays disabled",
                         slot, POLLMAP_SLOT_BYTES, static_cast<u64>(fb->SizeBytes()));
            pollmap_buffer.reset();
        } else {
            pollmap_host = pollmap_buffer->mapped_data.data();
            std::memset(pollmap_host, 0, POLLMAP_BYTES);
            _mm_sfence();
            const vk::DeviceAddress bda = pollmap_buffer->BufferDeviceAddress();
            // Two 32-bit fills: the slot is 4-byte aligned by construction, and the shader reads
            // it as {lo, hi} anyway. Same command stream as the zeroing, so it lands before any
            // shader runs and no later clear of the fault buffer touches the tail.
            fb->Fill(slot, 4u, static_cast<u32>(bda & 0xFFFFFFFFu));
            fb->Fill(slot + 4, 4u, static_cast<u32>(bda >> 32));
            // Sweep slice: a circle is POLLMAP_REGIONS regions of 32 words. Cached memory reads
            // at cache speed (a 128 KiB slice is tens of microseconds); an uncached type costs a
            // bus round trip per word, so the slice shrinks eightfold there.
            pollmap_sweep_regions_per_pass =
                pollmap_cached ? POLLMAP_REGIONS / 64 : POLLMAP_REGIONS / 512;
            const auto now = std::chrono::steady_clock::now();
            pollmap_sweep_next = now + std::chrono::seconds(2);
            pollmap_report_next = now + std::chrono::seconds(5);
            LOG_CRITICAL(Render_Vulkan,
                         "POLLMAP: armed, bitmap {} MB + summary {} KB at bda {:#x} (memory flags "
                         "{:#x}, cached={}), address slot at fault buffer +{:#x}, {} pages per "
                         "summary bit, sweep slice {} regions, pthost={}",
                         POLLMAP_FINE_BYTES >> 20, POLLMAP_SUMMARY_BYTES >> 10,
                         static_cast<u64>(bda), static_cast<u32>(mem_flags), pollmap_cached, slot,
                         POLLMAP_REGION_PAGES, pollmap_sweep_regions_per_pass, pollmap_pt_host);
            if (!pollmap_pt_host) {
                // Without a host-mapped table WritePageTableRange is a staging copy in the
                // command stream, and that copy executes only after the batch that is spinning:
                // a redirect can never reach the wave it was made for, only the next run of the
                // worker. The stand must run with SHADPS4_PTHOST=1.
                LOG_CRITICAL(Render_Vulkan,
                             "POLLMAP: page table is not host-mapped (SHADPS4_PTHOST=1 needed) - "
                             "redirects reach a spinning wave only after its batch retires");
            }
        }
    }

    {
        const char* uma_env = std::getenv("SHADPS4_UMA");
        uma_enabled = uma_env && uma_env[0] == '1' && instance.IsExternalMemoryHostSupported();
        if (uma_env && uma_env[0] == '1' && !instance.IsExternalMemoryHostSupported()) {
            LOG_WARNING(Render_Vulkan, "SHADPS4_UMA=1 but VK_EXT_external_memory_host missing");
        }
        if (uma_enabled) {
            LOG_INFO(Render_Vulkan, "UMA host imports enabled, min alignment {:#x}",
                     instance.GetMinImportedHostPointerAlignment());
        }
    }

    // STACKSHIELD companion: with stack pages never protectable, CPU writes
    // there are tracker-invisible - sticky CPU authority keeps readback
    // delivery from ever clobbering a live guest stack (same shield the
    // GfxEop tick pages rely on).
    {
        const char* e = std::getenv("SHADPS4_STACKSHIELD");
        if (e && e[0] == '1') {
            std::scoped_lock gm{gpu_modified_mutex};
            cpu_authority_ranges.Add(0x1200000000ull, 0x10000000ull);
            LOG_CRITICAL(Render_Vulkan,
                         "STACKSHIELD: guest stack band under sticky CPU authority");
        }
    }

    // LABELUMA gate. A half-enabled combination with full UMA is fatal by
    // experiment, so it is refused outright.
    {
        const char* e = std::getenv("SHADPS4_LABELUMA");
        label_uma_enabled = e && e[0] == '1' && instance.IsExternalMemoryHostSupported();
        if (e && e[0] == '1' && !instance.IsExternalMemoryHostSupported()) {
            LOG_WARNING(Render_Vulkan, "SHADPS4_LABELUMA=1 but VK_EXT_external_memory_host "
                                       "missing - falling back to readback engine");
        }
        if (label_uma_enabled && uma_enabled) {
            LOG_CRITICAL(Render_Vulkan,
                         "LABELUMA: SHADPS4_UMA=1 set simultaneously - refusing LABELUMA "
                         "(half-enabled UMA combinations are fatal)");
            label_uma_enabled = false;
        }
        if (label_uma_enabled) {
            if (const char* r = std::getenv("SHADPS4_LABELUMA_RANGE")) {
                // Comma-separated hexbase:hexsize list - the callback reads
                // several arenas, one seed rarely covers them all.
                const u64 align = std::max<u64>(
                    instance.GetMinImportedHostPointerAlignment(), CACHING_PAGESIZE);
                const char* cursor = r;
                while (*cursor != '\0') {
                    u64 base = 0, size = 0;
                    int consumed = 0;
                    if (std::sscanf(cursor, "%llx:%llx%n", &base, &size, &consumed) == 2 &&
                        base != 0 && size != 0) {
                        const VAddr abase = Common::AlignDown(base, align);
                        label_seeds.emplace_back(abase,
                                                 Common::AlignUp(base + size, align) - abase);
                        cursor += consumed;
                    } else {
                        break;
                    }
                    if (*cursor == ',') {
                        ++cursor;
                    }
                }
            }
            if (const char* m = std::getenv("SHADPS4_LABELUMA_MAX")) {
                label_budget = std::strtoull(m, nullptr, 10) * 1_MB;
            }
            for (const auto& [sb, ss] : label_seeds) {
                LOG_CRITICAL(Render_Vulkan, "LABELUMA: enabled, seed {:#x}:{:#x}, budget {} MB",
                             sb, ss, label_budget >> 20);
            }
        }
    }

    // [postmortem] Register the readback partial-window dump for the crash handler.
    // Before any early return below - the hook must exist on every configuration.
    Postmortem::RegisterPostmortemDump(&rbstat::DumpReadbackPostmortem);

    // [TDR SPINWATCH/SPINSNAP] Build the rescue contour and start its watchdog BEFORE any early
    // return below (otherwise it would be silently dead on a !CanReportMemoryUsage / fence-fail
    // path). Default-off: SpinRescueInit() no-ops unless SHADPS4_SPINRESCUE is set.
    SpinRescueInit();

    // [BACKIMP] Import the backing SECTION rather than guest virtual addresses. By the time the
    // buffer cache exists the memory manager (and with it the backing) is fully built, the Vulkan
    // device is up, and the guest has not started hammering memory yet - which matters because
    // the driver caches page locks, so this must happen exactly once. Before the early returns
    // below so no configuration can silently skip it.
    {
        Postmortem::StartupT0();
        const char* e = std::getenv("SHADPS4_BACKIMP");
        if (e && e[0] == '1') {
            u32 delay_s = 0;
            if (const char* d = std::getenv("SHADPS4_BACKIMP_DELAY_S")) {
                delay_s = static_cast<u32>(std::strtoul(d, nullptr, 10));
            }
            auto& address_space = memory->GetAddressSpace();
            void* const backing_base = address_space.BackingBase();
            const u64 backing_size = address_space.GetBackingSize();
            const Vulkan::Instance* inst = &instance;
            auto run = [inst, backing_base, backing_size] {
                auto& backimp = Vulkan::BackingImport::Get();
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = backimp.Init(*inst, backing_base, backing_size);
                Postmortem::g_startup_backimp_ms.store(
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - t0)
                                         .count()),
                    std::memory_order_relaxed);
                if (ok) {
                    backimp.SelfTest();
                    // Everything still only planned is pinned off the critical path.
                    backimp.WarmInBackground();
                }
            };
            if (delay_s == 0) {
                run();
            } else {
                LOG_CRITICAL(Render_Vulkan, "BACKIMP: deferred by {} s", delay_s);
                backimp_thread = std::jthread([run, delay_s](std::stop_token stop) {
                    for (u32 i = 0; i < delay_s * 10 && !stop.stop_requested(); ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    if (!stop.stop_requested()) {
                        run();
                    }
                });
            }
            // [BACKIMP] Prefetch ranges: same base:size syntax. The walk itself runs on the GPU
            // thread (BackimpPrefetchStep from OnSubmit) - by then the guest has mapped memory,
            // and repeated passes pick up whatever it maps later.
            if (const char* p = std::getenv("SHADPS4_BACKIMP_PREFETCH")) {
                const char* cursor = p;
                while (*cursor != '\0') {
                    u64 pf_base = 0, pf_size = 0;
                    int consumed = 0;
                    if (std::sscanf(cursor, "%llx:%llx%n", &pf_base, &pf_size, &consumed) == 2 &&
                        pf_base != 0 && pf_size != 0) {
                        backimp_prefetch_ranges.emplace_back(pf_base, pf_size);
                        cursor += consumed;
                    } else {
                        break;
                    }
                    if (*cursor == ',') {
                        ++cursor;
                    }
                }
                LOG_CRITICAL(Render_Vulkan, "BACKIMP: prefetch armed, {} ranges",
                             backimp_prefetch_ranges.size());
            }
        }
    }

    // [VRAMPREFETCH] SHADPS4_VRAM_PREFETCH=<hex base>:<hex size>[,...] - see VramPrefetchStep.
    if (const char* p = std::getenv("SHADPS4_VRAM_PREFETCH")) {
        const char* cursor = p;
        while (*cursor != '\0') {
            u64 pf_base = 0, pf_size = 0;
            int consumed = 0;
            if (std::sscanf(cursor, "%llx:%llx%n", &pf_base, &pf_size, &consumed) == 2 &&
                pf_base != 0 && pf_size != 0) {
                vram_prefetch_ranges.emplace_back(pf_base, pf_size);
                cursor += consumed;
            } else {
                break;
            }
            if (*cursor == ',') {
                ++cursor;
            }
        }
        LOG_CRITICAL(Render_Vulkan, "VRAMPREFETCH: armed, {} ranges", vram_prefetch_ranges.size());
    }

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));

    // Readback engine setup (gated at use by SHADPS4_RBQUEUE): a command pool,
    // two fences and two small host-visible staging buffers for servicing CPU
    // readback faults on the dedicated queue. Built here on the init thread so
    // fault paths never allocate.
    if (instance.HasReadbackQueue()) {
        const vk::CommandPoolCreateInfo pool_info = {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
        };
        auto [pool_result, pool] = instance.GetDevice().createCommandPoolUnique(pool_info);
        if (pool_result == vk::Result::eSuccess) {
            rb_pool = std::move(pool);
            const vk::CommandBufferAllocateInfo alloc_info = {
                .commandPool = *rb_pool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 2,
            };
            auto [cb_result, cbs] =
                instance.GetDevice().allocateCommandBuffersUnique(alloc_info);
            if (cb_result == vk::Result::eSuccess) {
                for (u32 i = 0; i < 2; ++i) {
                    rb_slots[i].cmdbuf = std::move(cbs[i]);
                    auto [f_result, fence] =
                        instance.GetDevice().createFenceUnique(vk::FenceCreateInfo{});
                    if (f_result != vk::Result::eSuccess) {
                        return;
                    }
                    rb_slots[i].fence = std::move(fence);
                    rb_slots[i].staging = std::make_unique<Buffer>(
                        instance, scheduler, MemoryUsage::Download, 0,
                        vk::BufferUsageFlagBits::eTransferDst, 512_KB);
                }
                rb_ready = true;
                LOG_CRITICAL(Render_Vulkan, "RBQUEUE: readback engine ready");
            }
        }
    }
}

BufferCache::~BufferCache() {
    // [TDR] Stop and join the watchdog FIRST - before any member it touches (page_table, fault
    // buffer, rb machinery) is torn down. jthread is the last member so its own destruction would
    // already do this, but being explicit guarantees the ordering against future member churn.
    spin_rescue_thread.request_stop();
    if (spin_rescue_thread.joinable()) {
        spin_rescue_thread.join();
    }
    // [BACKIMP] Same reasoning for the deferred import thread, and the imported chunks have to go
    // back while the device is still alive (the instance outlives the buffer cache).
    backimp_thread.request_stop();
    if (backimp_thread.joinable()) {
        backimp_thread.join();
    }
    Vulkan::BackingImport::Get().Shutdown();
}

// ============================ [TDR SPINWATCH/SPINSNAP] ============================
// Diagnostic spin-rescue contour (SHADPS4_SPINRESCUE = detect | snoop). Behaviour-neutral:
// a 10ms watchdog thread that only logs, plus (snoop) a cross-queue fault-bitmap snapshot and a
// probe compute dispatch on the dedicated rescue queue. Never takes serve_mutex, never uses
// rb_pool, never calls Register/CreateBuffer/FindBuffer/texture_cache/SendCommand/Refresh.

namespace {
// Minimal probe shader: prove the rescue COMPUTE queue makes forward progress during a stall
// (a transfer copy alone would not prove compute progress). Compiled at init via glslang.
constexpr std::string_view kSpinProbeComp = R"(#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer ProbeBuf { uint counter; };
void main() { atomicAdd(counter, 1u); }
)";
} // namespace

void BufferCache::SpinRescueInit() {
    const char* mode = std::getenv("SHADPS4_SPINRESCUE");
    if (mode == nullptr) {
        return; // default-off, bit-identical
    }
    const bool detect = std::strcmp(mode, "detect") == 0;
    const bool import_mode = std::strcmp(mode, "import") == 0 || std::strcmp(mode, "1") == 0;
    const bool full = std::strcmp(mode, "1") == 0;
    const bool snoop = std::strcmp(mode, "snoop") == 0 || import_mode;
    if (!detect && !snoop) {
        return; // unknown value -> off
    }
    spin_enabled = true;
    spin_snoop = snoop;
    spin_import_mode = import_mode; // import+ builds host imports (needs snoop resources)
    spin_full = full;              // full publishes via pt_patch
    if (const char* ms = std::getenv("SHADPS4_SPINRESCUE_MS")) {
        const u32 v = static_cast<u32>(std::strtoul(ms, nullptr, 10));
        if (v != 0) {
            spin_stall_ms = v;
        }
    }
    if (const char* mb = std::getenv("SHADPS4_SPINRESCUE_BUDGET_MB")) {
        const u64 v = std::strtoull(mb, nullptr, 10);
        if (v != 0) {
            spin_budget_bytes = v << 20;
        }
    }
    if (const char* kb = std::getenv("SHADPS4_SPINRESCUE_MAXRUN_KB")) {
        const u64 v = std::strtoull(kb, nullptr, 10);
        if (v != 0) {
            spin_maxrun_bytes = v << 10;
        }
    }
    // Import needs an importable host-memory path; without it we cannot build rescue buffers.
    if (spin_import_mode && !instance.IsExternalMemoryHostSupported()) {
        LOG_CRITICAL(Render_Vulkan,
                     "SPINRESCUE: VK_EXT_external_memory_host missing, import disabled (snoop)");
        spin_import_mode = false;
        spin_full = false;
    }

    const auto device = instance.GetDevice();
    if (spin_snoop) {
        // Snoop needs the dedicated rescue queue (index 3). Never degrade onto queue 2.
        if (!instance.HasRescueQueue()) {
            LOG_CRITICAL(Render_Vulkan,
                         "SPINRESCUE: no dedicated rescue queue, snoop disabled (detect only)");
            spin_snoop = false;
        }
    }
    if (spin_snoop) {
        // Own command pool / cmdbuf / fence - never rb_pool (externally synchronized).
        const vk::CommandPoolCreateInfo pool_info = {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
        };
        auto [pr, pool] = device.createCommandPoolUnique(pool_info);
        auto [fr, fence] = device.createFenceUnique(vk::FenceCreateInfo{});
        bool ok = pr == vk::Result::eSuccess && fr == vk::Result::eSuccess;
        if (ok) {
            spin_cmd_pool = std::move(pool);
            spin_fence = std::move(fence);
            const vk::CommandBufferAllocateInfo cb_info = {
                .commandPool = *spin_cmd_pool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1,
            };
            auto [cr, cbs] = device.allocateCommandBuffersUnique(cb_info);
            if (cr == vk::Result::eSuccess) {
                spin_cmdbuf = std::move(cbs[0]);
            } else {
                ok = false;
            }
        }
        if (ok) {
            const u64 fb_size = fault_manager.GetFaultBufferSize();
            spin_staging = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Download, 0,
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                    fb_size + 4_KB);
            spin_probe_buf = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Download, 0,
                                                      vk::BufferUsageFlagBits::eStorageBuffer, 256);
        }
        // Probe compute pipeline (push-descriptor, one SSBO), mirrors FaultManager setup.
        if (ok) {
            const vk::DescriptorSetLayoutBinding binding = {
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            };
            const vk::DescriptorSetLayoutCreateInfo dl_ci = {
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = 1,
                .pBindings = &binding,
            };
            auto [dlr, dl] = device.createDescriptorSetLayoutUnique(dl_ci);
            vk::ShaderModule module{};
            if (dlr == vk::Result::eSuccess) {
                spin_probe_layout = std::move(dl);
                module = Vulkan::Compile(kSpinProbeComp, vk::ShaderStageFlagBits::eCompute, device,
                                         {});
            }
            if (module) {
                const vk::PipelineShaderStageCreateInfo stage_ci = {
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = module,
                    .pName = "main",
                };
                const vk::PipelineLayoutCreateInfo pl_ci = {
                    .setLayoutCount = 1U,
                    .pSetLayouts = &(*spin_probe_layout),
                };
                auto [plr, pl] = device.createPipelineLayoutUnique(pl_ci);
                if (plr == vk::Result::eSuccess) {
                    spin_probe_pl_layout = std::move(pl);
                    const vk::ComputePipelineCreateInfo cp_ci = {
                        .stage = stage_ci,
                        .layout = *spin_probe_pl_layout,
                    };
                    auto [cpr, cp] = device.createComputePipelineUnique({}, cp_ci);
                    if (cpr == vk::Result::eSuccess) {
                        spin_probe_pipeline = std::move(cp);
                    }
                }
                device.destroyShaderModule(module);
            }
            if (!spin_probe_pipeline) {
                LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: probe pipeline build failed");
                ok = false;
            }
        }
        if (!ok) {
            LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: snoop resource init failed, detect only");
            spin_snoop = false;
        }
    }
    // Import/full need the snoop resources (rescue queue + staging). If snoop degraded, so do they.
    if (!spin_snoop) {
        spin_import_mode = false;
        spin_full = false;
    }

    // [TDR ETAP5] Full publication needs 64-bit buffer atomics for the atomicStore into the table.
    if (spin_full) {
        const auto f12 =
            instance.GetPhysicalDevice()
                .getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
        const bool int64_atomics =
            f12.get<vk::PhysicalDeviceVulkan12Features>().shaderBufferInt64Atomics == VK_TRUE;
        if (!int64_atomics) {
            LOG_CRITICAL(Render_Vulkan,
                         "SPINRESCUE: shaderBufferInt64Atomics unsupported, staying in import mode");
            spin_full = false;
        }
    }
    if (spin_full) {
        // pt_patch pipeline: params-SSBO (binding 0) + page table SSBO (binding 1, uint64_t[]).
        static constexpr std::string_view kPtPatch = R"(#version 450
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Entry { uint page_idx; uint entry_lo; uint entry_hi; };
layout(std430, binding = 0) readonly buffer Params { uint count; uint pad0; uint pad1; uint pad2;
                                                     Entry entries[]; };
layout(std430, binding = 1) buffer Table { uint64_t table[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    uint64_t e = packUint2x32(uvec2(entries[i].entry_lo, entries[i].entry_hi));
    atomicExchange(table[entries[i].page_idx], e);
}
)";
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        const vk::DescriptorSetLayoutCreateInfo dl_ci = {
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = 2,
            .pBindings = bindings.data(),
        };
        bool ok = true;
        auto [dlr, dl] = device.createDescriptorSetLayoutUnique(dl_ci);
        vk::ShaderModule module{};
        if (dlr == vk::Result::eSuccess) {
            spin_patch_layout = std::move(dl);
            module = Vulkan::Compile(kPtPatch, vk::ShaderStageFlagBits::eCompute, device, {});
        }
        if (module) {
            const vk::PipelineShaderStageCreateInfo stage_ci = {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = module,
                .pName = "main",
            };
            const vk::PipelineLayoutCreateInfo pl_ci = {
                .setLayoutCount = 1U,
                .pSetLayouts = &(*spin_patch_layout),
            };
            auto [plr, pl] = device.createPipelineLayoutUnique(pl_ci);
            if (plr == vk::Result::eSuccess) {
                spin_patch_pl_layout = std::move(pl);
                const vk::ComputePipelineCreateInfo cp_ci = {
                    .stage = stage_ci,
                    .layout = *spin_patch_pl_layout,
                };
                auto [cpr, cp] = device.createComputePipelineUnique({}, cp_ci);
                if (cpr == vk::Result::eSuccess) {
                    spin_patch_pipeline = std::move(cp);
                }
            }
            device.destroyShaderModule(module);
        }
        // params: 16-byte header (count + 3 pad) + 256 * 12-byte entries.
        spin_params_buf = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                                   vk::BufferUsageFlagBits::eStorageBuffer,
                                                   16 + 256 * 12);
        if (!ok || !spin_patch_pipeline) {
            LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: pt_patch pipeline build failed, import only");
            spin_full = false;
        }
    }

    LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: armed mode={} stall_ms={} snoop={} import={} full={}",
                 mode, spin_stall_ms, spin_snoop, spin_import_mode, spin_full);
    spin_rescue_thread = std::jthread([this](std::stop_token st) { SpinRescueLoop(st); });
}

bool BufferCache::SpinSubmitWait(u64 timeout_ns) {
    // Submits the already-recorded spin_cmdbuf to the rescue queue and waits. Returns false on
    // timeout or device loss (logged by caller / here). Never asserts - the log is the artifact.
    const auto device = instance.GetDevice();
    (void)device.resetFences(1, &*spin_fence);
    const vk::CommandBuffer cb = *spin_cmdbuf;
    const vk::SubmitInfo si = {
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    const vk::Result sr = instance.GetRescueQueue().submit(si, *spin_fence);
    if (sr != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "SPINSNAP: submit failed: {}", vk::to_string(sr));
        return false;
    }
    const vk::Result wr = device.waitForFences(1, &*spin_fence, VK_TRUE, timeout_ns);
    if (wr != vk::Result::eSuccess) {
        return false; // eTimeout / eErrorDeviceLost - caller logs the specific verdict
    }
    return true;
}

void BufferCache::SpinRescueSnapshot(u64 gpu_tick, u64 stalled_ms) {
    const auto device = instance.GetDevice();
    const u64 fb_size = fault_manager.GetFaultBufferSize();
    const vk::Buffer fault_buf = fault_manager.GetFaultBuffer()->Handle();
    constexpr u64 kTimeout = 100'000'000ULL; // 100 ms

    const auto record_copy = [&] {
        (void)spin_cmdbuf->reset();
        (void)spin_cmdbuf->begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // Same queue family, bits are monotonic 0->1 and only cleared by the (stalled) parser
        // dispatch - no barriers/semaphores needed for a best-effort diagnostic read.
        spin_cmdbuf->copyBuffer(fault_buf, spin_staging->Handle(),
                                vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = fb_size});
        (void)spin_cmdbuf->end();
    };

    // --- Snapshot A (one retry after 50ms on a hard stall before giving up) ---
    record_copy();
    if (!SpinSubmitWait(kTimeout)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        record_copy();
        if (!SpinSubmitWait(kTimeout)) {
            LOG_CRITICAL(Render_Vulkan, "SPINSNAP: copy timeout (gpu_tick={} stalled {} ms)",
                         gpu_tick, stalled_ms);
            return; // rescue queue serialized behind the stall - verdict on its own
        }
    }
    const u32 words = static_cast<u32>(fb_size / sizeof(u32));
    std::vector<u32> snap_a(words);
    std::memcpy(snap_a.data(), spin_staging->mapped_data.data(), fb_size);
    PageTrace::NoteBitmap(snap_a.data(), snap_a.size(), PageTrace::SPA);

    // --- Probe compute dispatch (forward-progress linchpin) ---
    {
        (void)spin_cmdbuf->reset();
        (void)spin_cmdbuf->begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        const vk::DescriptorBufferInfo pb_info = {
            .buffer = spin_probe_buf->Handle(),
            .offset = 0,
            .range = 256,
        };
        const vk::WriteDescriptorSet write = {
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &pb_info,
        };
        spin_cmdbuf->bindPipeline(vk::PipelineBindPoint::eCompute, *spin_probe_pipeline);
        spin_cmdbuf->pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *spin_probe_pl_layout, 0,
                                          write);
        spin_cmdbuf->dispatch(1, 1, 1);
        (void)spin_cmdbuf->end();
        const auto t0 = std::chrono::steady_clock::now();
        if (SpinSubmitWait(kTimeout)) {
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
            LOG_CRITICAL(Render_Vulkan, "SPINSNAP: probe compute ok, latency {} us", us);
        } else {
            LOG_CRITICAL(Render_Vulkan,
                         "SPINSNAP: probe compute timeout - atomic publish path not viable");
        }
    }

    // --- Diff: clear the bitmap, let the spin re-set actively-faulting pages, snapshot B ---
    {
        (void)spin_cmdbuf->reset();
        (void)spin_cmdbuf->begin(
            vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        spin_cmdbuf->fillBuffer(fault_buf, 0, fb_size, 0);
        (void)spin_cmdbuf->end();
        if (!SpinSubmitWait(kTimeout)) {
            LOG_CRITICAL(Render_Vulkan, "SPINSNAP: bitmap clear timeout");
            return;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    record_copy();
    if (!SpinSubmitWait(kTimeout)) {
        LOG_CRITICAL(Render_Vulkan, "SPINSNAP: diff copy timeout");
        return;
    }
    std::vector<u32> snap_b(words);
    std::memcpy(snap_b.data(), spin_staging->mapped_data.data(), fb_size);
    PageTrace::NoteBitmap(snap_b.data(), snap_b.size(), PageTrace::SPB);

    // --- CPU decode: VA = (word*32 + findLSB) << 14. Actively-faulting (B) pages first. ---
    boost::container::small_vector<u64, 16> top;
    u64 total_pages = 0;
    const auto collect = [&](const std::vector<u32>& snap) {
        for (u32 w = 0; w < words && top.size() < 16; ++w) {
            u32 bits = snap[w];
            while (bits != 0 && top.size() < 16) {
                const u32 lsb = std::countr_zero(bits);
                bits &= bits - 1;
                top.push_back((u64{w} * 32 + lsb) << 14);
            }
        }
    };
    for (u32 w = 0; w < words; ++w) {
        total_pages += std::popcount(snap_b[w]);
    }
    collect(snap_b);
    if (top.size() < 16) {
        collect(snap_a); // fill remaining slots from the full snapshot
    }
    std::string list;
    for (const u64 a : top) {
        list += fmt::format(" {:#x}", a);
    }
    LOG_CRITICAL(Render_Vulkan, "SPINSNAP: {} pages, top:{}", total_pages, list);

    // [TDR ETAP4/5] Feed candidates from the FULL A-snapshot (the wave WALKS the pages, touching
    // each once, so the B-diff is empty on the next round). B-diff is used only to prioritize
    // (actively-faulting pages first). The stage-4 filters (IsFaultRegionValid rejects
    // 0x0/0x3f800000 junk, device-entry==0, gpu_modified-without-authority skip, ratchet, budget)
    // do the real screening downstream - they are untouched.
    if (spin_import_mode) {
        constexpr u32 kFeedCap = 64;
        boost::container::small_vector<u64, 64> feed;
        std::unordered_set<u64> seen;
        const auto feed_from = [&](const std::vector<u32>& snap) {
            for (u32 w = 0; w < words && feed.size() < kFeedCap; ++w) {
                u32 bits = snap[w];
                while (bits != 0 && feed.size() < kFeedCap) {
                    const u32 lsb = std::countr_zero(bits);
                    bits &= bits - 1;
                    const u64 va = (u64{w} * 32 + lsb) << 14;
                    if (seen.insert(va).second) {
                        feed.push_back(va);
                    }
                }
            }
        };
        feed_from(snap_b); // actively-faulting first (priority)
        feed_from(snap_a); // then the full snapshot up to the cap
        if (!feed.empty()) {
            SpinImportAndPublish(feed, gpu_tick);
        }
    }
}

// [TDR ETAP4] Copy one 8-byte page-table entry (DEVICE truth, not the CPU page_table) via the
// rescue queue into the 4KB scratch tail of spin_staging. Returns false on timeout/device loss.
bool BufferCache::SpinReadDeviceEntry(u64 page, vk::DeviceAddress& out) {
    const u64 fb_size = fault_manager.GetFaultBufferSize();
    const vk::Buffer table = bda_pagetable_buffer.Handle();
    (void)spin_cmdbuf->reset();
    (void)spin_cmdbuf->begin(
        vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    spin_cmdbuf->copyBuffer(table, spin_staging->Handle(),
                            vk::BufferCopy{.srcOffset = page * sizeof(vk::DeviceAddress),
                                           .dstOffset = fb_size, .size = sizeof(vk::DeviceAddress)});
    (void)spin_cmdbuf->end();
    if (!SpinSubmitWait(100'000'000ULL)) {
        return false;
    }
    std::memcpy(&out, spin_staging->mapped_data.data() + fb_size, sizeof(out));
    return true;
}

// [TDR ETAP5] Publish {page_idx, entry64} pairs into the device page table with a 64-bit
// atomicStore on the rescue queue. TOCTOU + submit handled by the caller; returns false on
// timeout/device loss.
bool BufferCache::SpinPatchEntries(
    const boost::container::small_vector<std::pair<u64, vk::DeviceAddress>, 32>& entries) {
    if (entries.empty() || !spin_full) {
        return false;
    }
    u8* p = spin_params_buf->mapped_data.data();
    const u32 count = static_cast<u32>(std::min<size_t>(entries.size(), 256));
    std::memcpy(p, &count, sizeof(count));
    for (u32 i = 0; i < count; ++i) {
        const u32 page_idx = static_cast<u32>(entries[i].first);
        const u32 lo = static_cast<u32>(entries[i].second & 0xFFFFFFFFu);
        const u32 hi = static_cast<u32>(entries[i].second >> 32);
        u8* e = p + 16 + i * 12;
        std::memcpy(e + 0, &page_idx, 4);
        std::memcpy(e + 4, &lo, 4);
        std::memcpy(e + 8, &hi, 4);
    }
    (void)spin_cmdbuf->reset();
    (void)spin_cmdbuf->begin(
        vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    const vk::DescriptorBufferInfo params_info = {
        .buffer = spin_params_buf->Handle(), .offset = 0, .range = 16 + 256 * 12};
    const vk::DescriptorBufferInfo table_info = {
        .buffer = bda_pagetable_buffer.Handle(), .offset = 0, .range = BDA_PAGETABLE_SIZE};
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {.dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &params_info},
        {.dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &table_info},
    }};
    spin_cmdbuf->bindPipeline(vk::PipelineBindPoint::eCompute, *spin_patch_pipeline);
    spin_cmdbuf->pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *spin_patch_pl_layout, 0,
                                      writes);
    spin_cmdbuf->dispatch((count + 63) / 64, 1, 1);
    (void)spin_cmdbuf->end();
    return SpinSubmitWait(100'000'000ULL);
}

vk::DeviceAddress BufferCache::RescueBdaFor(VAddr addr) {
    std::scoped_lock lk{spin_import_mutex};
    if (spin_imports.empty()) {
        return 0;
    }
    auto it = spin_imports.upper_bound(addr);
    if (it == spin_imports.begin()) {
        return 0;
    }
    --it;
    const SpinImport& imp = it->second;
    if (addr >= imp.base && addr < imp.base + imp.size) {
        return imp.bda + (addr - imp.base);
    }
    return 0;
}

bool BufferCache::LiveWriteLandsInGuest(VAddr addr) {
    if ((addr >> 40) != 0) {
        return false; // outside the 40-bit guest space: not ours to judge, replay decides
    }
    std::scoped_lock lk{buffers_mutex};
    const u64 page = addr >> CACHING_PAGEBITS;
    const BufferId owner = page_table[page].buffer_id;
    if (owner) {
        const Buffer& buffer = slot_buffers[owner];
        return !buffer.is_deleted && buffer.is_import_backed;
    }
    if (HostBdaFor(addr) != 0) {
        return true;
    }
    return RescueBdaFor(addr) != 0;
}

void BufferCache::SpinReleaseImports(VAddr addr, u64 size) {
    if (!spin_enabled || size == 0) {
        return;
    }
    boost::container::small_vector<std::pair<u64, vk::DeviceAddress>, 32> zeros;
    {
        std::scoped_lock lk{spin_import_mutex};
        for (auto cur = spin_imports.begin(); cur != spin_imports.end();) {
            const SpinImport& imp = cur->second;
            if (imp.base < addr + size && addr < imp.base + imp.size) {
                for (u64 p = imp.base; p < imp.base + imp.size; p += CACHING_PAGESIZE) {
                    if (zeros.size() < 256) {
                        zeros.push_back({p >> CACHING_PAGEBITS, vk::DeviceAddress{0}});
                    }
                }
                if (!imp.backing) {
                    spin_imported_bytes -= std::min(spin_imported_bytes, imp.size);
                }
                cur = spin_imports.erase(cur);
            } else {
                ++cur;
            }
        }
    }
    // Zero the device table entries (works during a stall too); the imported Buffers stay alive
    // for the session (bounded by budget) so their memory is never freed under an in-flight deref.
    if (!zeros.empty() && spin_full) {
        SpinPatchEntries(zeros);
    }
}

void BufferCache::SpinImportAndPublish(std::span<const u64> pages, u64 gpu_tick) {
    // [TDR] No re-confirm of stagnation for the IMPORT itself: with SPINGUARD the wave frees
    // itself and stalls are now short (~300ms), so the tick has usually moved by the time we
    // decode the snapshot - re-checking here cancelled every import despite live candidates.
    // Candidates came from a stall snapshot; importing a valid owner-less page is harmless and
    // cheap, and the downstream filters (IsFaultRegionValid / device-entry==0 / gpu_modified skip
    // / ratchet / budget) do the screening. Stagnation is re-checked only in the publish TOCTOU.
    spin_incident_imports = 0;
    const size_t candidate_count = pages.size();

    // Filter candidates page by page, collect valid ones (already page-aligned VAs).
    boost::container::small_vector<VAddr, 64> valid;
    for (const VAddr page_addr : pages) {
        const u64 page = page_addr >> CACHING_PAGEBITS;
        // (1) mapped guest memory only - garbage pointers are legally answered 0 by the guard.
        if (!IsFaultRegionValid(page_addr, CACHING_PAGESIZE)) {
            continue;
        }
        // (2) DEVICE entry must be 0 (the fatal-flaw criterion: NOT the CPU page_table). A live
        // device entry means the page is already published - healing it would be wrong.
        vk::DeviceAddress dev_entry = 0;
        if (!SpinReadDeviceEntry(page, dev_entry)) {
            return; // rescue queue serialized behind the stall - abandon this incident
        }
        if (dev_entry != 0) {
            continue;
        }
        // (3) truth in VRAM -> skip (v1). gpu_modified without cpu_authority and not CPU-dirty.
        {
            std::scoped_lock gm{gpu_modified_mutex};
            const bool gpu_mod =
                gpu_modified_ranges.Intersects(page_addr, CACHING_PAGESIZE);
            const bool cpu_auth = cpu_authority_ranges.Intersects(page_addr, CACHING_PAGESIZE);
            if (gpu_mod && !cpu_auth) {
                continue;
            }
        }
        // (4) ratchet: a page healed >=2 times without tick progress goes to a stop-list.
        if (spin_ratchet[page] >= 2) {
            continue;
        }
        // already covered by a live import? then it is already handled - skip building a new one.
        if (RescueBdaFor(page_addr) != 0) {
            continue;
        }
        valid.push_back(page_addr);
    }
    const size_t valid_count = valid.size();
    // NOTE: do not early-return on an empty `valid` - SPINFEED must still print below so we can
    // see where candidates were lost (N candidates -> V valid -> M imported).
    std::ranges::sort(valid);

    // Coalesce adjacent pages into runs (<= maxrun), import each run, publish its entries.
    const u64 align = std::max<u64>(instance.GetMinImportedHostPointerAlignment(), CACHING_PAGESIZE);
    boost::container::small_vector<std::pair<u64, vk::DeviceAddress>, 32> to_publish;
    boost::container::small_vector<VAddr, 32> corrections;
    // [BACKIMP] With the backing import alive, rescue buffers bind over already-pinned memory:
    // no pinning budget is consumed, so the per-incident cap can be far looser (bounded only to
    // keep the net itself from stalling). spin_imported_bytes counts PRIVATE-path bytes only -
    // the byte budget keeps guarding the fallback that does pin.
    const bool backing_avail = Vulkan::BackingImport::Get().IsAvailable();
    const u32 incident_cap = backing_avail ? 64 : 16;
    size_t i = 0;
    while (i < valid.size()) {
        if (spin_incident_imports >= incident_cap || spin_imported_bytes >= spin_budget_bytes) {
            LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: budget/incident cap reached ({} bytes, {})",
                         spin_imported_bytes, spin_incident_imports);
            break;
        }
        VAddr run_lo = valid[i];
        VAddr run_hi = valid[i] + CACHING_PAGESIZE;
        size_t j = i + 1;
        while (j < valid.size() && valid[j] == run_hi &&
               (run_hi + CACHING_PAGESIZE - run_lo) <= spin_maxrun_bytes) {
            run_hi += CACHING_PAGESIZE;
            ++j;
        }
        const VAddr base = Common::AlignDown(run_lo, align);
        const u64 isize = Common::AlignUp(run_hi, align) - base;
        auto imp = Buffer::CreateHostImport(instance, scheduler, base, isize, "rescue");
        if (!imp) {
            LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: host import {:#x}+{:#x} failed", base, isize);
            i = j;
            continue;
        }
        const vk::DeviceAddress base_bda = imp->BufferDeviceAddress();
        const bool backed = imp->is_backing_backed;
        {
            std::scoped_lock lk{spin_import_mutex};
            spin_import_bufs.push_back(std::make_unique<Buffer>(std::move(*imp)));
            spin_imports[base] =
                SpinImport{.base = base, .size = isize, .bda = base_bda, .backing = backed};
            if (!backed) {
                spin_imported_bytes += isize;
            }
            ++spin_incident_imports;
        }
        LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: import #{} {:#x}+{:#x} -> bda {:#x} ({})",
                     spin_incident_imports, base, isize, u64(base_bda),
                     backed ? "backing" : "private");
        for (VAddr pa = run_lo; pa < run_hi; pa += CACHING_PAGESIZE) {
            const u64 page = pa >> CACHING_PAGEBITS;
            ++spin_ratchet[page];
            to_publish.push_back({page, base_bda + (pa - base)});
            corrections.push_back(pa);
        }
        i = j;
    }
    // [TDR ETAP4] Feed accounting, printed UNCONDITIONALLY after every feed pass (not throttled):
    // shows where candidates are lost - N snapshot candidates -> V surviving filters -> M imported.
    LOG_CRITICAL(Render_Vulkan, "SPINFEED: {} candidates -> {} valid -> {} imported",
                 candidate_count, valid_count, spin_incident_imports);
    if (to_publish.empty() || !spin_full) {
        return; // import-only mode stops here (table untouched, TDRs remain by design)
    }

    // [TDR ETAP5] TOCTOU: under buffers_mutex revalidate ONLY that the pages are still owner-less
    // (buffer_id==0), then publish. We no longer require an ongoing stall - publishing a rescue
    // bda for a still-owner-less page is safe and the honest Register hook supersedes it later.
    {
        std::scoped_lock lk{buffers_mutex};
        boost::container::small_vector<std::pair<u64, vk::DeviceAddress>, 32> revalidated;
        for (const auto& [page, entry] : to_publish) {
            if (!page_table[page].buffer_id) { // still no CPU owner -> safe to publish rescue bda
                revalidated.push_back({page, entry});
            }
        }
        if (!revalidated.empty()) {
            if (SpinPatchEntries(revalidated)) {
                LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: published {} entries (gpu_tick={})",
                             revalidated.size(), gpu_tick);
            } else {
                LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: pt_patch submit/wait failed");
            }
        }
    }
}

#ifdef _WIN32
// K32* live in kernel32.dll on every supported Windows, so no psapi.h and no extra link input.
extern "C" __declspec(dllimport) int __stdcall K32EnumProcessModules(void*, void**, unsigned long,
                                                                    unsigned long*);
extern "C" __declspec(dllimport) unsigned long __stdcall K32GetModuleBaseNameA(void*, void*, char*,
                                                                               unsigned long);
struct HostModInfo {
    void* base;
    unsigned long size;
    void* entry;
};
extern "C" __declspec(dllimport) int __stdcall K32GetModuleInformation(void*, void*, HostModInfo*,
                                                                      unsigned long);
#endif

namespace {
// [GUESTSTALL] Snapshot of every registered GUEST thread while the EOP stream is silent.
// Deliberately dbghelp-free: OpenThread + SuspendThread + GetThreadContext + ResumeThread and
// plain arithmetic against the module table. Contexts are collected FIRST and only logged
// afterwards - formatting while a guest thread is suspended can deadlock on the logger's own
// lock if that thread happened to hold it.
void GuestStallSnapshot(u64 silent_ms) {
#ifdef _WIN32
    struct Shot {
        u32 tid;
        const char* name;
        CONTEXT ctx;
        bool ok;
        const char* wait_what;
        u64 wait_a0;
        u64 wait_a1;
        s64 wait_since;
    };
    boost::container::small_vector<Shot, 32> shots;
    const DWORD self_tid = GetCurrentThreadId();
    for (u32 i = 0; i < Postmortem::kMaxGuestThreads; ++i) {
        const u32 tid = Postmortem::g_guest_threads[i].tid.load(std::memory_order_acquire);
        if (tid == 0 || tid == 0xFFFFFFFFu || tid == self_tid) {
            continue;
        }
        const HANDLE th =
            OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, static_cast<DWORD>(tid));
        if (th == nullptr) {
            continue;
        }
        Shot shot{tid, Postmortem::g_guest_threads[i].name, {}, false, nullptr, 0, 0, 0};
        shot.wait_what = Postmortem::g_guest_threads[i].wait_what.load(std::memory_order_acquire);
        shot.wait_a0 = Postmortem::g_guest_threads[i].wait_a0.load(std::memory_order_relaxed);
        shot.wait_a1 = Postmortem::g_guest_threads[i].wait_a1.load(std::memory_order_relaxed);
        shot.wait_since =
            Postmortem::g_guest_threads[i].wait_since_ms.load(std::memory_order_relaxed);
        shot.ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        if (SuspendThread(th) != static_cast<DWORD>(-1)) {
            shot.ok = GetThreadContext(th, &shot.ctx) != 0;
            ResumeThread(th);
        }
        CloseHandle(th);
        if (shot.ok) {
            shots.push_back(shot);
        }
        if (shots.size() >= 32) {
            break;
        }
    }

    // rip -> "<module>+0x..." using the MODBASE table; no symbolisation. Returns true when the
    // RIP is inside a guest module (i.e. the thread is running GUEST code).
    const auto where = [](u64 rip, char* out, size_t out_size) {
        for (u32 i = 0; i < Postmortem::kMaxGuestModules; ++i) {
            const u64 base = Postmortem::g_guest_modules[i].base.load(std::memory_order_acquire);
            if (base == 0 || base == ~u64{0}) {
                continue;
            }
            const u64 size = Postmortem::g_guest_modules[i].size.load(std::memory_order_relaxed);
            if (rip >= base && rip < base + size) {
                std::snprintf(out, out_size, "%s+%#llx", Postmortem::g_guest_modules[i].name,
                              static_cast<unsigned long long>(rip - base));
                return true;
            }
        }
        std::snprintf(out, out_size, "%s", rip >= 0x700000000000ull ? "host" : "guest");
        return false;
    };

    // Same ring the crash dump reads, but the stalled slot is unknown here (no RBX), so report
    // the newest tick across the 32 slots. ReadProcessMemory so an unmapped slot fails quietly.
    u64 tick_best = 0;
    u32 tick_slot = 0;
    if (MemoryPatcher::g_eboot_address != 0) {
        const HANDLE self = GetCurrentProcess();
        const u64 ring = MemoryPatcher::g_eboot_address + 0x10CDB40;
        for (u32 s = 0; s < 32; ++s) {
            u64 v = 0;
            SIZE_T got = 0;
            if (ReadProcessMemory(self, reinterpret_cast<LPCVOID>(ring + u64{s} * 0xA50 + 0xA08),
                                  &v, sizeof(v), &got) &&
                got == sizeof(v) && v > tick_best) {
                tick_best = v;
                tick_slot = s;
            }
        }
    }
    const s64 now_ms = Postmortem::NowMs();
    const s64 tick_write = Postmortem::g_tick_last_write_ms.load(std::memory_order_relaxed);
    LOG_CRITICAL(Render_Vulkan,
                 "GUESTSTALL: eop_silent={}ms threads={} tick_max={:#x}@slot{} "
                 "tick_write_age_ms={} tw_tid={}",
                 silent_ms, shots.size(), tick_best, tick_slot,
                 tick_write ? now_ms - tick_write : -1,
                 Postmortem::g_tick_writer_tid.load(std::memory_order_relaxed));

    // Same-RIP detection across consecutive incidents (watchdog thread only, no locking).
    static std::unordered_map<u32, u64> last_rip;
    const u32 tw_tid = Postmortem::g_tick_writer_tid.load(std::memory_order_relaxed);
    for (const auto& s : shots) {
        char loc[96];
        const bool in_guest = where(s.ctx.Rip, loc, sizeof(loc));
        auto it = last_rip.find(s.tid);
        // Only GUEST code can spin. A thread parked in ntdll sits at the same RIP by
        // definition, and flagging that as spinning sent the last investigation the wrong way.
        const bool same = in_guest && it != last_rip.end() && it->second == s.ctx.Rip;
        last_rip[s.tid] = s.ctx.Rip;
        char waiting[224];
        if (s.wait_what != nullptr) {
            const s64 waited = s.wait_since ? now_ms - s.wait_since : -1;
            // [CONDTRACE] For a condvar wait, the decisive fact is whether anyone signalled it
            // and WHEN relative to the wait starting.
            const bool is_cond = std::strncmp(s.wait_what, "pthread_cond_", 13) == 0;
            const Postmortem::CondStatSlot* cs =
                is_cond ? Postmortem::CondSlot(s.wait_a0, false) : nullptr;
            if (cs != nullptr) {
                const s64 last = cs->last_ms.load(std::memory_order_relaxed);
                const s64 sig_age = last ? now_ms - last : -1;
                const char* verdict = sig_age < 0                 ? "NEVER_SIGNALLED"
                                      : (sig_age > waited)        ? "ALL_SIGNALS_PREDATE_WAIT"
                                                                  : "SIGNALLED_DURING_WAIT";
                std::snprintf(waiting, sizeof(waiting),
                              "<%s a0=%#llx a1=%#llx for %lldms last_signal_age=%lldms "
                              "signals_total=%llu %s>",
                              s.wait_what, static_cast<unsigned long long>(s.wait_a0),
                              static_cast<unsigned long long>(s.wait_a1),
                              static_cast<long long>(waited), static_cast<long long>(sig_age),
                              static_cast<unsigned long long>(
                                  cs->count.load(std::memory_order_relaxed)),
                              verdict);
            } else if (is_cond) {
                std::snprintf(waiting, sizeof(waiting),
                              "<%s a0=%#llx a1=%#llx for %lldms signals_total=0 NEVER_SIGNALLED>",
                              s.wait_what, static_cast<unsigned long long>(s.wait_a0),
                              static_cast<unsigned long long>(s.wait_a1),
                              static_cast<long long>(waited));
            } else {
                std::snprintf(waiting, sizeof(waiting), "<%s a0=%#llx a1=%#llx for %lldms>",
                              s.wait_what, static_cast<unsigned long long>(s.wait_a0),
                              static_cast<unsigned long long>(s.wait_a1),
                              static_cast<long long>(waited));
            }
        } else {
            std::snprintf(waiting, sizeof(waiting), "%s", in_guest ? "<none>" : "<none/host>");
        }
        LOG_CRITICAL(Render_Vulkan,
                     "GUESTSTALL: tid={}{} name={} rip={:#x} rsp={:#x} module={} waiting_on={} "
                     "rax={:#x} rcx={:#x} rdx={:#x} rbx={:#x} rsi={:#x} rdi={:#x}{}",
                     s.tid, s.tid == tw_tid ? "*TICKWRITER" : "", s.name, s.ctx.Rip, s.ctx.Rsp,
                     loc, waiting, s.ctx.Rax, s.ctx.Rcx, s.ctx.Rdx, s.ctx.Rbx, s.ctx.Rsi,
                     s.ctx.Rdi, same ? " SPINNING_AT_SAME_RIP" : "");
        // Unlabelled host wait: the top of the stack usually holds the return address into
        // whichever wrapper called into ntdll, which names the primitive even when no marker
        // covers it. ReadProcessMemory so an unreadable stack fails quietly; still no dbghelp.
        if (s.wait_what == nullptr && !in_guest) {
            const HANDLE self = GetCurrentProcess();
            char line[224];
            int n = std::snprintf(line, sizeof(line), "GUESTSTALL-STACK: tid=%u", s.tid);
            for (u32 k = 0; k < 3; ++k) {
                u64 v = 0;
                SIZE_T got = 0;
                if (!ReadProcessMemory(self, reinterpret_cast<LPCVOID>(s.ctx.Rsp + k * 8), &v,
                                       sizeof(v), &got) ||
                    got != sizeof(v)) {
                    break;
                }
                char vloc[96];
                where(v, vloc, sizeof(vloc));
                if (n < static_cast<int>(sizeof(line)) - 64) {
                    n += std::snprintf(line + n, sizeof(line) - n, " [rsp+%u]=%#llx(%s)", k * 8,
                                       static_cast<unsigned long long>(v), vloc);
                }
            }
            LOG_CRITICAL(Render_Vulkan, "{}", line);
        }
    }

    // --- [CONDTRACE] Last 8 condvar signals in the process, newest last ---
    {
        char line[320];
        int n = std::snprintf(line, sizeof(line), "GUESTSTALL-SIGNALS:");
        const u64 seq_now = Postmortem::g_cond_ring_seq.load(std::memory_order_relaxed);
        const u64 start = seq_now >= 8 ? seq_now - 8 : 0;
        for (u64 k = start; k < seq_now; ++k) {
            const Postmortem::CondSignalRec& r = Postmortem::g_cond_ring[k % Postmortem::kCondRing];
            const s64 ms = r.ms.load(std::memory_order_acquire);
            if (n < static_cast<int>(sizeof(line)) - 56) {
                n += std::snprintf(line + n, sizeof(line) - n, " %#llx%s@%lldms(tid%u)",
                                   static_cast<unsigned long long>(
                                       r.cond.load(std::memory_order_relaxed)),
                                   r.bcast.load(std::memory_order_relaxed) ? "B" : "S",
                                   static_cast<long long>(ms ? now_ms - ms : -1),
                                   r.tid.load(std::memory_order_relaxed));
            }
        }
        if (seq_now == 0) {
            std::snprintf(line + n, sizeof(line) - n, " <none - no cond signal in the whole run>");
        }
        LOG_CRITICAL(Render_Vulkan, "{}", line);
    }

    // --- Timeline of the frame loop, plus the oldest link ---
    const auto age = [now_ms](s64 stamp) { return stamp ? now_ms - stamp : s64{-1}; };
    const s64 a_submit = age(Postmortem::g_last_submit_ms.load(std::memory_order_relaxed));
    const s64 a_sdone = age(Postmortem::g_last_submit_done_ms.load(std::memory_order_relaxed));
    const s64 a_flipreq = age(Postmortem::g_last_flip_req_ms.load(std::memory_order_relaxed));
    const s64 a_flip = age(Postmortem::g_last_flip_ms.load(std::memory_order_relaxed));
    const s64 a_vblank = age(Postmortem::g_last_vblank_ms.load(std::memory_order_relaxed));
    const s64 a_vbtick = age(Postmortem::g_vblank_tick_ms.load(std::memory_order_relaxed));
    const s64 a_eopsig = age(Postmortem::g_eop_signal_ms.load(std::memory_order_relaxed));
    const s64 a_eopdisp = age(Postmortem::g_eop_dispatch_ms.load(std::memory_order_relaxed));
    const s64 a_tick = age(tick_write);
    LOG_CRITICAL(Render_Vulkan,
                 "GUESTSTALL-TIME: submit={}ms submitdone={}ms flipreq={}ms flip={}ms "
                 "vblank_wait={}ms vblank_tick={}ms eop_sig={}ms eop_disp={}ms tick={}ms | "
                 "submits_total={} flipreqs_total={} flips_total={} vblank_ticks={}",
                 a_submit, a_sdone, a_flipreq, a_flip, a_vblank, a_vbtick, a_eopsig, a_eopdisp,
                 a_tick, Postmortem::g_submits_total.load(std::memory_order_relaxed),
                 Postmortem::g_flip_reqs_total.load(std::memory_order_relaxed),
                 Postmortem::g_flips_total.load(std::memory_order_relaxed),
                 Postmortem::g_vblank_ticks.load(std::memory_order_relaxed));

    // Oldest link = the one that stopped first. Links never seen at all (-1) are skipped:
    // "never happened" is a different verdict from "happened long ago".
    const std::pair<const char*, s64> links[] = {
        {"submit", a_submit},        {"submitdone", a_sdone},   {"flipreq", a_flipreq},
        {"flip", a_flip},            {"vblank_wait", a_vblank}, {"vblank_tick", a_vbtick},
        {"eop_signal", a_eopsig},    {"eop_dispatch", a_eopdisp}, {"tick_write", a_tick},
    };
    const char* first_name = "none";
    s64 first_age = -1;
    for (const auto& [n, v] : links) {
        if (v > first_age) {
            first_age = v;
            first_name = n;
        }
    }
    LOG_CRITICAL(Render_Vulkan, "GUESTSTALL-FIRST: {} stopped first ({}ms ago)", first_name,
                 first_age);

    // --- Submission accounting: names the difference between "the parser has work it cannot
    // finish" and "there is nothing left that could ever satisfy the waiter". ---
    {
        const s64 hb_age = age(Postmortem::g_heartbeat_ms.load(std::memory_order_relaxed));
        const u32 pstate = Postmortem::g_gfx_state.load(std::memory_order_relaxed);
        LOG_CRITICAL(Render_Vulkan,
                     "GUESTSTALL-GPU: submission_lock={} submits_in={} submits_done={} "
                     "gfx_inflight={} asc_inflight={} flips_req={} flips_mat={} "
                     "heartbeat={} heartbeat_age={}ms last_retire={}ms parser_state={}",
                     Postmortem::g_submission_lock.load(std::memory_order_relaxed),
                     Postmortem::g_submits_in.load(std::memory_order_relaxed),
                     Postmortem::g_submits_done.load(std::memory_order_relaxed),
                     Postmortem::g_num_gfx_submits.load(std::memory_order_relaxed),
                     Postmortem::g_num_asc_submits.load(std::memory_order_relaxed),
                     Postmortem::g_flips_requested.load(std::memory_order_relaxed),
                     Postmortem::g_flips_materialized.load(std::memory_order_relaxed),
                     Postmortem::g_parse_heartbeat.load(std::memory_order_relaxed), hb_age,
                     age(Postmortem::g_last_retire_ms.load(std::memory_order_relaxed)),
                     pstate == 1 ? "waiting" : (pstate == 2 ? "starved" : "running"));
        char sl[192];
        int n = std::snprintf(sl, sizeof(sl), "GUESTSTALL-SUBMIT:");
        static const char* const site_names[] = {"submitDone_raise", "gpuidle_clear",
                                                 "gpuidle_FORCED", "site3"};
        for (u32 i = 0; i < Postmortem::kSubSites; ++i) {
            const u64 inc = Postmortem::g_sub_inc[i].load(std::memory_order_relaxed);
            const u64 dec = Postmortem::g_sub_dec[i].load(std::memory_order_relaxed);
            if (inc == 0 && dec == 0) {
                continue;
            }
            if (n < static_cast<int>(sizeof(sl)) - 48) {
                n += std::snprintf(sl + n, sizeof(sl) - n, " %s inc=%llu dec=%llu |",
                                   site_names[i], static_cast<unsigned long long>(inc),
                                   static_cast<unsigned long long>(dec));
            }
        }
        LOG_CRITICAL(Render_Vulkan, "{}", sl);
    }

    // --- Flip pipeline: is the presenter starved, or are we the ones not delivering? ---
    LOG_CRITICAL(Render_Vulkan,
                 "GUESTSTALL-FLIP: pending={} gc_queue={} present_queued={} "
                 "flipreq_vs_flip={}ms (positive = we are behind the guest)",
                 Postmortem::g_flip_pending.load(std::memory_order_relaxed),
                 Postmortem::g_flip_gc_queue.load(std::memory_order_relaxed),
                 Postmortem::g_present_queued.load(std::memory_order_relaxed),
                 (a_flip >= 0 && a_flipreq >= 0) ? a_flip - a_flipreq : s64{-1});

    // --- SUBFIX liveness: say WHY it did not fire, so silence is never ambiguous ---
    {
        const u64 gfx_inflight = Postmortem::g_num_gfx_submits.load(std::memory_order_relaxed);
        if (gfx_inflight != 0) {
            LOG_CRITICAL(Lib_GnmDriver, "SUBFIX: not-eligible gfx_inflight={}", gfx_inflight);
        }
    }

    // ================= HOSTSTALL: OUR OWN threads =================
    // The parser is a host thread, so it was invisible to every snapshot so far - which is
    // exactly where the wedge turned out to be. Same Suspend/GetContext/Resume discipline:
    // collect first, format afterwards, never hold a thread suspended across a log call.
    {
        // What the parser is chewing on right now.
        const s64 pkt_us = Parserprof::g_cur_pkt_us.load(std::memory_order_acquire);
        const s64 now_us = Parserprof::NowUs();
        const s64 ph_us = Parserprof::g_cur_phase_us.load(std::memory_order_relaxed);
        LOG_CRITICAL(Render_Vulkan,
                     "HOSTSTALL-PKT: op={:#x} seq={} in_packet={}ms phase={} phase_us={} "
                     "heartbeat={}",
                     Parserprof::g_cur_op.load(std::memory_order_relaxed),
                     Parserprof::g_cur_seq.load(std::memory_order_relaxed),
                     pkt_us ? (now_us - pkt_us) / 1000 : -1,
                     Parserprof::PhaseName(Parserprof::g_cur_phase.load(std::memory_order_relaxed)),
                     ph_us ? now_us - ph_us : -1,
                     Postmortem::g_parse_heartbeat.load(std::memory_order_relaxed));

        // Loaded host modules, so a RIP resolves to nvoglv64 / vulkan-1 / ntdll / shadps4.exe
        // instead of a bare number. Enumerated fresh per incident (cheap, bounded).
        struct HostMod {
            u64 base;
            u64 size;
            char name[40];
        };
        boost::container::small_vector<HostMod, 64> mods;
        {
            const HANDLE self = GetCurrentProcess();
            void* handles[192];
            unsigned long needed = 0;
            if (K32EnumProcessModules(self, handles, sizeof(handles), &needed)) {
                const u32 count = std::min<u32>(needed / sizeof(void*), 192);
                for (u32 i = 0; i < count; ++i) {
                    HostModInfo mi{};
                    if (!K32GetModuleInformation(self, handles[i], &mi, sizeof(mi))) {
                        continue;
                    }
                    HostMod m{reinterpret_cast<u64>(mi.base), mi.size, {}};
                    K32GetModuleBaseNameA(self, handles[i], m.name, sizeof(m.name));
                    mods.push_back(m);
                }
            }
        }
        const auto host_where = [&mods](u64 rip, char* out, size_t out_size) {
            for (const auto& m : mods) {
                if (rip >= m.base && rip < m.base + m.size) {
                    std::snprintf(out, out_size, "%s+%#llx", m.name,
                                  static_cast<unsigned long long>(rip - m.base));
                    return;
                }
            }
            std::snprintf(out, out_size, "%s", "?");
        };

        boost::container::small_vector<Shot, 24> host_shots;
        for (u32 i = 0; i < Postmortem::kMaxHostThreads; ++i) {
            const u32 tid = Postmortem::g_host_threads[i].tid.load(std::memory_order_acquire);
            if (tid == 0 || tid == 0xFFFFFFFFu || tid == self_tid) {
                continue;
            }
            // Guest threads are named through the same helper; they already got a full report.
            bool is_guest = false;
            for (u32 j = 0; j < Postmortem::kMaxGuestThreads && !is_guest; ++j) {
                is_guest = Postmortem::g_guest_threads[j].tid.load(std::memory_order_acquire) == tid;
            }
            if (is_guest) {
                continue;
            }
            const HANDLE th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE,
                                         static_cast<DWORD>(tid));
            if (th == nullptr) {
                continue;
            }
            Shot shot{tid, Postmortem::g_host_threads[i].name, {}, false, nullptr, 0, 0, 0};
            shot.wait_what = Postmortem::g_host_threads[i].wait_what.load(std::memory_order_acquire);
            shot.wait_a0 = Postmortem::g_host_threads[i].wait_a0.load(std::memory_order_relaxed);
            shot.wait_a1 = Postmortem::g_host_threads[i].wait_a1.load(std::memory_order_relaxed);
            shot.wait_since =
                Postmortem::g_host_threads[i].wait_since_ms.load(std::memory_order_relaxed);
            shot.ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (SuspendThread(th) != static_cast<DWORD>(-1)) {
                shot.ok = GetThreadContext(th, &shot.ctx) != 0;
                ResumeThread(th);
            }
            CloseHandle(th);
            if (shot.ok) {
                host_shots.push_back(shot);
            }
        }
        const HANDLE self = GetCurrentProcess();
        for (const auto& s : host_shots) {
            char loc[96];
            host_where(s.ctx.Rip, loc, sizeof(loc));
            char waiting[160];
            if (s.wait_what != nullptr) {
                std::snprintf(waiting, sizeof(waiting), "<%s a0=%#llx a1=%#llx for %lldms>",
                              s.wait_what, static_cast<unsigned long long>(s.wait_a0),
                              static_cast<unsigned long long>(s.wait_a1),
                              static_cast<long long>(s.wait_since ? now_ms - s.wait_since : -1));
            } else {
                std::snprintf(waiting, sizeof(waiting), "<none>");
            }
            LOG_CRITICAL(Render_Vulkan,
                         "HOSTSTALL: name={} tid={} rip={:#x} rsp={:#x} module={} waiting_on={} "
                         "rax={:#x} rcx={:#x} rdx={:#x} rbx={:#x} rsi={:#x} rdi={:#x}",
                         s.name, s.tid, s.ctx.Rip, s.ctx.Rsp, loc, waiting, s.ctx.Rax, s.ctx.Rcx,
                         s.ctx.Rdx, s.ctx.Rbx, s.ctx.Rsi, s.ctx.Rdi);
            char line[256];
            int n = std::snprintf(line, sizeof(line), "HOSTSTALL-STACK: name=%s", s.name);
            for (u32 k = 0; k < 3; ++k) {
                u64 v = 0;
                SIZE_T got = 0;
                if (!ReadProcessMemory(self, reinterpret_cast<LPCVOID>(s.ctx.Rsp + k * 8), &v,
                                       sizeof(v), &got) ||
                    got != sizeof(v)) {
                    break;
                }
                char vloc[96];
                host_where(v, vloc, sizeof(vloc));
                if (n < static_cast<int>(sizeof(line)) - 72) {
                    n += std::snprintf(line + n, sizeof(line) - n, " [rsp+%u]=%#llx(%s)", k * 8,
                                       static_cast<unsigned long long>(v), vloc);
                }
            }
            LOG_CRITICAL(Render_Vulkan, "{}", line);
        }
    }
#endif
}
} // Anonymous namespace

void BufferCache::SpinRescueLoop(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:SpinRescue");
    const auto device = instance.GetDevice();
    const vk::Semaphore sem = scheduler.GetMasterSemaphore()->Handle();
    u64 last_counter = ~u64{0};
    auto last_change = std::chrono::steady_clock::now();
    u64 cycles = 0;
    bool acted = false; // one-shot per stall episode (until the counter moves again)
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++cycles;
        // NOT Refresh(): that ASSERTs on device lost and would kill the process before the log.
        auto [r, counter] = device.getSemaphoreCounterValue(sem);
        if (r != vk::Result::eSuccess) {
            LOG_CRITICAL(Render_Vulkan, "SPINRESCUE: device lost after {} cycles ({})", cycles,
                         vk::to_string(r));
            return; // stop the contour
        }
        const auto now = std::chrono::steady_clock::now();
        if (counter != last_counter) {
            last_counter = counter;
            last_change = now;
            acted = false;
            continue;
        }
        const u64 current = scheduler.CurrentTick();
        const bool behind = counter + 1 < current; // == !AllSubmittedRetired()
        const u64 held_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_change).count());
        // [GUESTSTALL] Independent trigger. The GPU-side condition below cannot see this class
        // of stall at all: when the guest stops submitting, the GPU drains and `behind` goes
        // FALSE, yet the EOP stream is dead and the engine fuse (~2s) is already counting. So
        // watch EOP silence directly, one snapshot per incident, 8 incidents per run.
        if (Parserprof::Enabled()) {
            static u32 gs_incidents = 0;
            static bool gs_armed = true;
            const s64 last_eop = Postmortem::g_eop_signal_ms.load(std::memory_order_relaxed);
            const s64 now_ms = Postmortem::NowMs();
            const s64 silent = last_eop != 0 ? now_ms - last_eop : 0;
            if (silent < 200) {
                gs_armed = true; // EOP alive again - re-arm for the next incident
            } else if (silent >= 800 && gs_armed && gs_incidents < 8) {
                gs_armed = false;
                ++gs_incidents;
                GuestStallSnapshot(static_cast<u64>(silent));
            }
        }
        if (behind && held_ms >= spin_stall_ms && !acted) {
            PageTrace::NoteStall();
            LOG_CRITICAL(Render_Vulkan, "SPINWATCH: gpu_tick={} stalled {} ms submitted={}", counter,
                         held_ms, current);
            // [GPUSTALL] Name the submits that are actually still executing, and any compute in
            // them that has been in flight long enough to be a non-terminating shader rather
            // than heavy work. Diagnostic only - nothing is killed.
            static const u64 dispkill_ms = [] {
                const char* e = std::getenv("SHADPS4_DISPKILL_MS");
                return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
            }();
            const u64 slow_ms = dispkill_ms != 0 ? dispkill_ms : 200ull;
            Vulkan::DumpStalledSubmits(counter, current, slow_ms);
            // Is our own backing/prefetch machinery implicated? Print the live tally so a
            // rebind storm during execution would be visible.
            Vulkan::BackingImport::Get().LogSummaryNow("gpustall");
            if (spin_snoop) {
                SpinRescueSnapshot(counter, held_ms);
            }
            acted = true; // hold off until the counter advances (avoids per-10ms spam)
        }
    }
}


// [debug] SHADPS4_BUFWATCH=<hex guest address>: report every buffer that covers this address.
// The read-fault histogram names a 16 MiB range; this names the buffer living in it.
static bool BufWatched(VAddr begin, VAddr end) {
    static const VAddr watch = [] {
        const char* e = std::getenv("SHADPS4_BUFWATCH");
        return e ? static_cast<VAddr>(std::strtoull(e, nullptr, 16)) : VAddr{0};
    }();
    return watch != 0 && watch >= begin && watch < end;
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size) {
    PageTrace::Note(device_addr, PageTrace::CW);
    // [POLLMAP v3.1] only fault-sized invalidations count as "the CPU wrote this page": the
    // same entry point is used for whole-region map/unmap invalidations (gigabytes), which
    // flagged every page (179: notcpu=0 of 796) and made the discriminator meaningless.
    if (size <= 4 * CACHING_PAGESIZE) {
        CpuWrittenEver::Note(device_addr, size);
    }
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
}

void BufferCache::OnGuestMap(VAddr addr, u64 size) {
    // [SKINVRAM] A guest mapping over the synthetic window: from now on the fault path could
    // give those pages a buffer owner whose Register/Unregister rewrites our entries, so the
    // redirect is switched off for good (guest thread: only the atomic and const config here).
    if (skin.config_ok && addr < skin.synth_base + skin.ring_bytes + CACHING_PAGESIZE &&
        addr + size > skin.synth_base && !skin.clobbered.exchange(true)) {
        LOG_CRITICAL(Render_Vulkan,
                     "SKINVRAM: guest mapped {:#x}+{:#x} over the synthetic window - redirect "
                     "disabled",
                     addr, size);
    }
    if (!label_uma_enabled || label_seeds.empty()) {
        return;
    }
    // Only ranges the guest actually mapped for GPU use are promotable; clamp
    // each seed by the fresh mapping.
    for (const auto& [sbase, ssize] : label_seeds) {
        const VAddr lo = std::max<VAddr>(addr, sbase);
        const VAddr hi = std::min<VAddr>(addr + size, sbase + ssize);
        if (lo >= hi) {
            continue;
        }
        std::scoped_lock gm{gpu_modified_mutex};
        pending_label_promotions.emplace_back(lo, hi - lo);
        LOG_CRITICAL(Render_Vulkan, "LABELUMA: armed {:#x}:{:#x} by guest mapping", lo, hi - lo);
    }
}

void BufferCache::RequestLabelPromotion(VAddr addr, u64 size) {
    if (!label_uma_enabled) {
        return;
    }
    std::scoped_lock gm{gpu_modified_mutex};
    pending_label_promotions.emplace_back(addr, size);
}

bool BufferCache::IntersectsLabelImport(VAddr addr, u64 size) {
    if (!label_uma_enabled) {
        return false;
    }
    std::scoped_lock gm{gpu_modified_mutex};
    return label_import_ranges.Intersects(addr, size);
}

void BufferCache::ReleaseLabelRegion(VAddr addr, u64 size) {
    // [TDR ETAP5] Guest unmap: drop any rescue imports over this range and zero their table
    // entries first (runs regardless of LABELUMA - this is the unmap hook for the rescue contour).
    SpinReleaseImports(addr, size);
    // [HOMEPOLICY] the list is sticky: the guest releases the whole 0x1100000000 region
    // (2.9 GB) during loading before the job heaps are mapped, and trimming here emptied the
    // policy for the rest of the run (liveloop73). Unmapped pages are protected by the
    // IsFaultRegionValid check in PolicyPrefetchStep; a role change after re-map is accepted.
    if (HomePolicy::Enabled()) {
        static u32 logged = 0;
        if (logged < 8 && HomePolicy::WantsImport(addr, size)) {
            ++logged;
            LOG_CRITICAL(Render_Vulkan, "HOMEPOLICY: unmap {:#x}:{:#x} touches import ranges (kept)", addr,
                         size);
        }
    }
    if (!label_uma_enabled || !IntersectsLabelImport(addr, size)) {
        return;
    }
    // The guest is unmapping pages an import-backed buffer lives on: the buffer
    // must die and every in-flight submit that saw it must retire BEFORE the
    // guest reuses the host pages. Runs via the GPU thread for cache safety.
    liverpool->SendCommand<true>([this, addr, size] {
        boost::container::small_vector<BufferId, 4> victims;
        ForEachBufferInRange(addr, size, [&](BufferId id, Buffer& b) {
            if (b.is_import_backed) {
                victims.push_back(id);
            }
        });
        for (const BufferId id : victims) {
            DeleteBuffer(id);
        }
    });
    scheduler.Finish();
    for (int spin = 0; spin < 1000; ++spin) {
        {
            std::scoped_lock lk{buffers_mutex};
            bool pinned = false;
            for (const auto& [idx, pins] : buffer_pins) {
                if (pins != 0) {
                    pinned = true;
                    break;
                }
            }
            if (!pinned) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void BufferCache::PromoteLabelRegions() {
    if (!label_uma_enabled) {
        return;
    }
    std::vector<std::pair<VAddr, u64>> pending;
    {
        std::scoped_lock gm{gpu_modified_mutex};
        ++label_promote_passes;
        pending.swap(pending_label_promotions);
        // Due OOM retries re-enter the queue with attempt accounting.
        for (auto it = label_retries.begin(); it != label_retries.end();) {
            if (label_promote_passes >= it->not_before_pass) {
                if (++it->attempts <= 3) {
                    pending.emplace_back(it->base, it->size);
                    it->not_before_pass = label_promote_passes + 90;
                    ++it;
                } else {
                    label_failed_ranges.Add(it->base, it->size);
                    LOG_CRITICAL(Render_Vulkan,
                                 "LABELUMA: import for {:#x} kept failing, staying on readbacks",
                                 it->base);
                    it = label_retries.erase(it);
                }
            } else {
                ++it;
            }
        }
    }
    for (const auto& [base, size] : pending) {
        // Already owned by an import-backed buffer covering the range - done.
        const BufferId owner = page_table[base >> CACHING_PAGEBITS].buffer_id;
        if (owner && slot_buffers[owner].is_import_backed &&
            slot_buffers[owner].IsInBounds(base, size)) {
            continue;
        }
        // An image over the range would lose its image->buffer sync silently.
        if (texture_cache.FindImageFromRange(base, size)) {
            LOG_CRITICAL(Render_Vulkan,
                         "LABELUMA: range {:#x}:{:#x} owned by an image, promotion skipped",
                         base, size);
            continue;
        }
        {
            std::scoped_lock gm{gpu_modified_mutex};
            if (label_failed_ranges.Intersects(base, size)) {
                continue;
            }
        }
        label_promote_intent = true;
        const BufferId buffer_id = CreateBuffer(base, static_cast<u32>(size));
        label_promote_intent = false;
        Buffer& buffer = slot_buffers[buffer_id];
        if (!buffer.is_import_backed) {
            continue;
        }
        {
            std::scoped_lock gm{gpu_modified_mutex};
            label_import_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes());
            std::erase_if(label_retries, [&](const LabelRetry& r) {
                return r.base >= buffer.CpuAddr() &&
                       r.base < buffer.CpuAddr() + buffer.SizeBytes();
            });
        }
        LOG_CRITICAL(Render_Vulkan, "LABELUMA: promoted {:#x}:{:#x} bda={:#x}",
                     buffer.CpuAddr(), buffer.SizeBytes(), buffer.BufferDeviceAddress());
        // Pre-promotion tracker marks inside the span become orphans: their tick
        // entries get served or migrated away, but a mark with no tick entry can
        // only refault - the fast path sees "empty", the slow path refuses import
        // buffers, and page-granular healing cannot lift a protection unit whose
        // neighbour pages are also marked (seen live: a 2500/5s refault storm on
        // level load that starved the GfxEop callback into the frame-ring
        // assert). Cleanse the whole span once the migration copies retire -
        // any earlier and the CPU reads stale bytes mid-copy (the v1 wound).
        scheduler.DeferOperation([this, span_base = buffer.CpuAddr(),
                                  span_size = buffer.SizeBytes()] {
            {
                std::scoped_lock gm{gpu_modified_mutex};
                gpu_modified_ranges.Subtract(span_base, span_size);
            }
            memory_tracker->UnmarkRegionAsGpuModified(span_base, span_size);
            LOG_CRITICAL(Render_Vulkan, "LABELUMA: cleansed {:#x}:{:#x} at retire",
                         span_base, span_size);
        });
    }
}

void BufferCache::MarkGpuWrittenRegion(VAddr device_addr, u64 size) {
    // [DUALHOME] the CPU reads guest memory, where every raw store already landed: no marks,
    // no readback protection, no CPU-waits-for-GPU.
    if (DualHomeEnabled() && !DualKeepMarks() && IsDualRange(device_addr, size)) {
        return;
    }
    // The tick map entry must exist BEFORE the tracker arms protection: a fault
    // serviced between the two would otherwise find no tick and skip waiting.
    {
        std::scoped_lock lk{gpu_modified_mutex};
        if (label_uma_enabled && label_import_ranges.Intersects(device_addr, size)) {
            // Import-backed region: GPU stores land in guest RAM directly, a
            // mark would only resurrect faults and readbacks for it.
            return;
        }
        gpu_modified_ranges.Add(device_addr, size, scheduler.CurrentTick());
    }
    memory_tracker->MarkRegionAsGpuModified(device_addr, size);
}



// [fix] NOBK: MEM_RESERVE inside the PRT "pretend mapped" aperture is a hole in
// host backing, not tracker protection - unmark cannot cure it (protection bits
// are already clear, mprotect does not conjure memory), only the guest's own
// commit can. Distinguishing the two costs one VirtualQuery (~1us).
static bool IsBackingCommitted(VAddr addr);

bool BufferCache::WaitForBacking(VAddr device_addr) {
    const bool pm_tw = rbstat::IsTickWriter();
    if (pm_tw) {
        rbstat::tw_nobk_inflight_ms.store(Postmortem::NowMs(), std::memory_order_relaxed);
    }
    rbstat::c_wait.fetch_add(1, std::memory_order_relaxed);
    rbstat::last_addr.store(device_addr, std::memory_order_relaxed);
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(5);
    u32 sleep_us = 0;
    u32 spins = 0;
    bool committed = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (IsBackingCommitted(device_addr)) {
            committed = true;
            break;
        }
        if (spins++ < 4) {
            std::this_thread::yield();
            continue;
        }
        sleep_us = sleep_us == 0 ? 50 : std::min(sleep_us * 2, 500u);
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
    (committed ? rbstat::c_ok : rbstat::c_timeout).fetch_add(1, std::memory_order_relaxed);
    const u64 nb_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    rbstat::c_wait_us.fetch_add(nb_us, std::memory_order_relaxed);
    Postmortem::AtomicMax(rbstat::nb_wait_max_us, nb_us);
    if (pm_tw) {
        rbstat::tw_nobk_us.fetch_add(nb_us, std::memory_order_relaxed);
        rbstat::tw_nobk_cnt.fetch_add(1, std::memory_order_relaxed);
        rbstat::tw_nobk_inflight_ms.store(0, std::memory_order_relaxed);
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    s64 expected = rbstat::next_ms.load(std::memory_order_relaxed);
    if (now_ms >= expected && rbstat::next_ms.compare_exchange_strong(expected, now_ms + 5000)) {
        if (expected != 0) {
            const u64 waits = rbstat::c_wait.exchange(0);
            LOG_CRITICAL(Render_Vulkan, "NOBK: wait={} ok={} timeout={} avg_us={} last={:#x}",
                         waits, rbstat::c_ok.exchange(0), rbstat::c_timeout.exchange(0),
                         waits ? rbstat::c_wait_us.exchange(0) / waits : 0,
                         rbstat::last_addr.load(std::memory_order_relaxed));
        }
    }
    // Always "handled": the guest is alive here and will get its commit (the
    // HONESTFAULT experiment showed a refusal collapses survival 90s -> 11s).
    // A timeout is fine - the next fault lands back here, and a cycle now
    // costs one VirtualQuery with no mutex held.
    return true;
}

static bool IsBackingCommitted(VAddr addr) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi)) != sizeof(mbi)) {
        return true; // query failed - keep prior behaviour
    }
    return mbi.State == MEM_COMMIT;
#else
    return true;
#endif
}

u32 BufferCache::PageHomeKind(VAddr addr) {
    u32 kind = 0;
    {
        std::scoped_lock lk{buffers_mutex};
        const BufferId buffer_id = page_table[addr >> CACHING_PAGEBITS].buffer_id;
        if (buffer_id) {
            const Buffer& buffer = slot_buffers[buffer_id];
            if (!buffer.is_deleted) {
                kind = buffer.is_backing_backed ? 3 : (buffer.is_import_backed ? 2 : 1);
            }
        }
    }
    // [HOMEPOLICY-PT] 4 = the page-table entry points at guest RAM regardless of the cache owner.
    if ((kind == 0 || kind == 1) && HomePolicy::WantsPt(addr, 4) && PolicyPtBda(addr) != 0) {
        return 4;
    }
    return kind;
}

u32 BufferCache::PeekGuestDword(VAddr addr, u32* out_value, u32* out_home) {
    *out_home = PageHomeKind(addr);
    *out_value = 0;
#ifdef _WIN32
    SIZE_T got = 0;
    if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(addr), out_value,
                          sizeof(u32), &got) &&
        got == sizeof(u32)) {
        return 1;
    }
    return 0;
#else
    std::memcpy(out_value, reinterpret_cast<const void*>(addr), sizeof(u32));
    return 1;
#endif
}

// ---------------------------------------------------------------------------------------------
// [SKINVRAM] SHADPS4_SKINVRAM: the ND skinning worker (0x8f067540) reads a 128-byte descriptor
// block (ud_0|ud_1<<32), a vertex count dword (ud_2|ud_3<<32) and, through V#s stored in block
// dwords 16/17, 20/21, 24/25, the per-vertex influence headers (stride 8), the influence dword
// list and the bone matrix palette - all as raw pointers, so the recompiler lowers every read to
// a page-table walk. Those pages live in guest memory (DESCDUMP: headers/list home=3, palette
// home=4), and 217 dispatches per frame turn into ~26-37 ms of dependent PCIe loads. Here every
// such dispatch gets a private slot in a device-local ring: the five ranges are written into the
// slot, the block COPY is patched so its three V#s name the slot copies, and ud_0..3 in the push
// constants are rewritten to the slot copies of the block and of the count. The ring is
// host-visible DEVICE_LOCAL (ReBAR) whenever the driver offers such memory, so filling a slot is
// one CPU store with no transfer and no barrier; without ReBAR it falls back to a staging copy
// whose barrier is handed to the caller for the batch it already issues. A fixed synthetic
// guest-VA window maps 1:1 onto the ring through page-table
// entries written once (entry i = ring_bda + i * 16 KiB), so the unchanged shader walks the
// table into VRAM. Guest memory, the block page, its owner and the fault path are never touched.
// Why not patch the VRAM copy of the block: ObtainBuffer(blk+64, 24, false) returns a stream
// copy the walk never reaches, is_written=true would read the synthetic bases back into guest
// memory, and the block page has a VRAM owner in only ~86% of the samples.
// ---------------------------------------------------------------------------------------------
namespace {
// Defined in the BDA journal block further down this file (same unnamed namespace); lets the
// ring show up by name in a VK_EXT_device_fault dump under SHADPS4_BDADBG=1.
void BdaLogAdd(u64 begin, u64 end, u64 guest, u32 id, u64 tick);

// Bulk guest read that never takes the fault path (same reason as PeekGuestDword): a readback-
// protected or unmapped page answers "not readable" instead of trapping the GPU thread.
bool ReadGuestBytes(VAddr addr, void* dst, size_t n) {
#ifdef _WIN32
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(addr), dst, n, &got) &&
           got == n;
#else
    std::memcpy(dst, reinterpret_cast<const void*>(addr), n);
    return true;
#endif
}
} // namespace

bool BufferCache::PeekGuestBytes(VAddr addr, void* dst, size_t n) {
    return ReadGuestBytes(addr, dst, n);
}

bool BufferCache::PtRedirectedInside(VAddr base, u64 size) {
    if (!HomePolicy::WantsPt(base, size)) {
        return false;
    }
    const VAddr end = base + size;
    for (VAddr p = Common::AlignDown(base, CACHING_PAGESIZE); p < end; p += CACHING_PAGESIZE) {
        if (HomePolicy::WantsPt(p, CACHING_PAGESIZE) && PolicyPtBda(p) != 0) {
            return true;
        }
    }
    return false;
}

bool BufferCache::SkinVramWanted(u64 pgm_hash) {
    const auto& wanted = SkinVramHashes();
    if (wanted.empty()) {
        return false;
    }
    for (const u64 h : wanted) {
        if (h == pgm_hash) {
            return true;
        }
    }
    return false;
}

bool BufferCache::SkinVramPageOwned(VAddr addr) {
    // find() never allocates a level (operator[] would); GPU thread is the only writer.
    const PageData* pd = page_table.find(addr >> CACHING_PAGEBITS);
    return pd != nullptr && static_cast<bool>(pd->buffer_id);
}

bool BufferCache::PublishSkinEntries(bool allow_wait) {
    // One entry per 16 KiB synthetic page, guard page included: plain device address of the
    // page, bit 63 clear, no tag (the feature refuses to start under BDATAG / BDA_NODEREF).
    const u64 n = skin.ring_bytes / CACHING_PAGESIZE + 1;
    const u32 bytes = static_cast<u32>(n * sizeof(vk::DeviceAddress));
    const u64 offset = (skin.synth_base >> CACHING_PAGEBITS) * sizeof(vk::DeviceAddress);
    if (!bda_pagetable_buffer.mapped_data.empty()) {
        // SHADPS4_PTHOST=1: a plain host store. The values never change, so a store racing an
        // in-flight reader of the previous lap is harmless.
        auto* dst =
            reinterpret_cast<vk::DeviceAddress*>(bda_pagetable_buffer.mapped_data.data() + offset);
        for (u64 i = 0; i < n; ++i) {
            dst[i] = skin.ring_bda + (i << CACHING_PAGEBITS);
        }
        _mm_sfence();
        return true;
    }
    // Device-local table: staging copy in the current command buffer (WriteDataBuffer's recipe),
    // hence ordered before the dispatch recorded right after it. The wrap-time re-publish maps
    // the staging ring without waiting so it can never stall the parser or split the command
    // buffer between the slot copy and its dispatch; a refusal is retried on the next redirect.
    const auto [staging, src_off] = staging_buffer.Map(bytes, 8, allow_wait);
    if (staging == nullptr) {
        return false;
    }
    auto* e = reinterpret_cast<vk::DeviceAddress*>(staging);
    for (u64 i = 0; i < n; ++i) {
        e[i] = skin.ring_bda + (i << CACHING_PAGEBITS);
    }
    staging_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = bda_pagetable_buffer.Handle(),
        .offset = offset,
        .size = bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = bda_pagetable_buffer.Handle(),
        .offset = offset,
        .size = bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(staging_buffer.Handle(), bda_pagetable_buffer.Handle(),
                      vk::BufferCopy{.srcOffset = src_off, .dstOffset = offset, .size = bytes});
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
    return true;
}

void BufferCache::SkinVramCreateRing() {
    // Called from the constructor, where the VRAM budget is still free: VMA is asked to stay
    // WITHIN_BUDGET and turns a refusal into a hard ASSERT, which must not be reachable from
    // the middle of a frame.
    const u64 window = skin.ring_bytes + CACHING_PAGESIZE;
    const auto flags = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    // First choice: DEVICE_LOCAL *and* host visible (ReBAR) - MemoryUsage::Stream is the class
    // the fork already uses for the 512 MiB page table under SHADPS4_PTHOST. A slot is then
    // filled with one CPU store: no staging buffer, no vkCmdCopyBuffer and, above all, no
    // transfer->compute barrier in a loop that runs ~217 times per frame (the very drains
    // CSBARRIER exists to delete). Host writes issued before vkQueueSubmit are visible to the
    // device by definition, and the slot's previous reader is already retired (IsFree).
    skin.ring = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Stream, 0, flags, window);
    VkMemoryPropertyFlags props{};
    vmaGetAllocationMemoryProperties(instance.GetAllocator(), skin.ring->buffer.allocation, &props);
    const bool device_local = (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    const bool coherent = (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    skin.host_ring = device_local && coherent && !skin.ring->mapped_data.empty();
    if (!skin.host_ring) {
        // No ReBAR: a Stream allocation lands in system RAM, which would reinstate over PCIe
        // exactly the reads this feature exists to remove. Fall back to plain VRAM plus the
        // staging copy (whose barrier is handed to the caller, never recorded standalone).
        LOG_CRITICAL(Render_Vulkan,
                     "SKINVRAM: no host-visible device-local memory (props={:#x} mapped={}) - "
                     "falling back to DEVICE_LOCAL + staging copy",
                     static_cast<u32>(props), !skin.ring->mapped_data.empty());
        skin.ring.reset();
        skin.ring = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::DeviceLocal, 0,
                                             flags, window);
    }
    Vulkan::SetObjectName(instance.GetDevice(), skin.ring->Handle(), "SkinVram Ring");
    skin.ring_bda = skin.ring->BufferDeviceAddress();
    if (skin.ring_bda == 0 || (skin.ring_bda & 0xF) != 0) {
        // read_const_live_x4 stays on its 16-byte vector path only if the copied bytes keep
        // their guest alignment; the layout preserves the low 8 address bits, the base must
        // not spoil it. Never used by the GPU yet, so it can go right away.
        LOG_CRITICAL(Render_Vulkan,
                     "SKINVRAM: refused, ring device address {:#x} is zero or not 16-byte aligned",
                     static_cast<u64>(skin.ring_bda));
        skin.ring.reset();
        skin.ring_bda = 0;
        skin.host_ring = false;
        skin.config_ok = false;
        return;
    }
    BdaLogAdd(skin.ring_bda, skin.ring_bda + window, skin.synth_base, 0x5EED0000u, 0);
}

void BufferCache::SkinVramInit() {
    skin.init_tried = true;
    if (!skin.config_ok || skin.ring == nullptr) {
        return;
    }
    const u64 window = skin.ring_bytes + CACHING_PAGESIZE;
    // The window must carry no guest state: not mapped for the GPU (IsFaultRegionValid ==
    // Rasterizer::IsMapped, so the fault path can never cache it) and owned by no buffer (so
    // no Register/Unregister ever rewrites its entries). One-time, ~8k pages.
    for (VAddr p = skin.synth_base; p < skin.synth_base + window; p += CACHING_PAGESIZE) {
        const bool mapped = IsFaultRegionValid(p, CACHING_PAGESIZE);
        const bool owned = SkinVramPageOwned(p);
        if (mapped || owned) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: refused, page {:#x} is {} - set SHADPS4_SKINVRAM_BASE="
                         "fe00000000",
                         p, mapped ? "guest-mapped" : "buffer-owned");
            // No entries were published, so the window is not ours: drop the ring and stop
            // claiming its pages in the fault parser.
            skin.ring.reset();
            skin.ring_bda = 0;
            skin.host_ring = false;
            return;
        }
    }
    // Entries are written once, here: the slot <-> VA mapping never changes and nothing can
    // rewrite it (the init scan proved the window is unowned, the per-dispatch owner guard and
    // OnGuestMap switch the feature off if that ever stops being true). A failure must be
    // fatal to the feature - with a zero window read_const falls back to the flatbuf while
    // read_const_dynamic/_live_x4 return zero, i.e. silently wrong skinning at full speed.
    // The publish may wait on the staging ring: nothing of this dispatch has been recorded
    // yet, so a flush here cannot separate a slot copy from its dispatch.
    if (!PublishSkinEntries(true)) {
        LOG_CRITICAL(Render_Vulkan, "SKINVRAM: refused, page-table publish failed");
        skin.ring.reset();
        skin.ring_bda = 0;
        skin.host_ring = false;
        return;
    }
    skin.slot_tick.assign(skin.num_slots, 0);
    skin.scratch.reserve(skin.slot_bytes);
    if (skin.verify) {
        skin.verify_image.reserve(skin.slot_bytes);
        skin.verify_tmp.reserve(skin.slot_bytes);
    }
    skin.next_slot = 0;
    skin.pending_slot = -1;
    skin.report_next = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    skin.ok = true;
    LOG_CRITICAL(Render_Vulkan,
                 "SKINVRAM: on hashes={} base={:#x} ring={} MB slot={} KB slots={} bda={:#x} "
                 "guard={:#x} host_ring={}",
                 skin.hashes.size(), skin.synth_base, skin.ring_bytes >> 20, skin.slot_bytes >> 10,
                 skin.num_slots, static_cast<u64>(skin.ring_bda),
                 skin.synth_base + skin.ring_bytes, skin.host_ring);
}

bool BufferCache::SkinVramNoteFaultPage(VAddr page) {
    if (!skin.config_ok || skin.ring_bda == 0) {
        return false;
    }
    const u64 window = skin.ring_bytes + CACHING_PAGESIZE;
    if (page < skin.synth_base || page >= skin.synth_base + window) {
        return false;
    }
    skin.fault_pages.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void BufferCache::SkinVramVerifyPrev() {
    // [SKINVRAM_VERIFY] The three inputs were snapshotted on the parser thread while the
    // dispatch was being recorded; the original shader read them when the dispatch executed.
    // Re-reading them now and comparing against the snapshot says whether the bytes are still
    // what we copied. A non-zero verify_mismatch means the record-time snapshot is not safe -
    // some writer (CPU or a GPU job whose result the BDATRACK replay put back) touches those
    // ranges between record and execute.
    if (!skin.verify_armed) {
        return;
    }
    skin.verify_armed = false;
    for (const auto& r : skin.verify_ranges) {
        if (r.size == 0) {
            continue;
        }
        skin.verify_tmp.resize(r.size);
        if (!ReadGuestBytes(r.guest, skin.verify_tmp.data(), r.size)) {
            continue;
        }
        ++skin.verify_checked;
        if (std::memcmp(skin.verify_tmp.data(), skin.verify_image.data() + r.off, r.size) != 0) {
            const u64 n = ++skin.verify_mismatch;
            if (n <= 8 || (n % 4096) == 0) {
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM-VERIFY: {:#x}+{:#x} changed after the snapshot ({} of {} "
                             "checks) - the record-time copy can be stale",
                             r.guest, r.size, n, skin.verify_checked);
            }
        }
    }
}

// [SKINDIRECT] ---------------------------------------------------------------------------------
const std::vector<u64>& BufferCache::SkinDirectHashes() {
    return SkinDirectHashList();
}

bool BufferCache::SkinDirectWanted(u64 pgm_hash) {
    const auto& wanted = SkinDirectHashList();
    if (wanted.empty()) {
        return false;
    }
    return std::find(wanted.begin(), wanted.end(), pgm_hash) != wanted.end();
}

bool BufferCache::SkinDirectRegister(u64 pgm_hash, u32 sharp_idx, u32 word) {
    g_skind_converted.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lk{g_skind_reg_mutex};
    const u32 n = g_skind_reg_n.load(std::memory_order_relaxed);
    for (u32 i = 0; i < n; ++i) {
        // The registry is shared with the store side: a sharp that is both read (WORDS) and
        // written (STORE_WORDS) needs one entry per kind, or the second kind never registers.
        if (g_skind_reg[i].hash == pgm_hash && g_skind_reg[i].sharp_idx == sharp_idx &&
            !g_skind_reg[i].is_store) {
            return false;
        }
    }
    if (n >= kSkinDirectMax) {
        return false;
    }
    g_skind_reg[n] = SkinDirectEntry{pgm_hash, sharp_idx, word, false};
    // Publish after the entry is written: the bind-time reader takes the count with acquire and
    // then walks entries that are immutable from here on.
    g_skind_reg_n.store(n + 1, std::memory_order_release);
    return true;
}

void BufferCache::SkinDirectNoteDecline() {
    g_skind_declined.fetch_add(1, std::memory_order_relaxed);
}

// [SKINDIRECT-STORES] ---------------------------------------------------------------------------
bool BufferCache::SkinDirectStoresEnabled() {
    return SkinDirectStoresFlag();
}

const std::vector<u32>& BufferCache::SkinDirectStoreWords() {
    return SkinDirectStoreWordList();
}

bool BufferCache::SkinDirectStoreImgInv() {
    return SkinDirectStoreImgInvFlag();
}

bool BufferCache::SkinDirectStoreVerify() {
    return SkinDirectStoreVerifyFlag();
}

bool BufferCache::SkinDirectRegisterStore(u64 pgm_hash, u32 sharp_idx, u32 word, bool shape_odd,
                                          bool needs_lowering) {
    g_skind_converted_stores.fetch_add(1, std::memory_order_relaxed);
    if (shape_odd) {
        g_skind_store_shape_odd.fetch_add(1, std::memory_order_relaxed);
    }
    if (!needs_lowering) {
        // raw StoreBufferU32/F32xN: nothing left to lower, the OpStores are already there
        g_skind_lowered_stores.fetch_add(1, std::memory_order_relaxed);
    }
    std::scoped_lock lk{g_skind_reg_mutex};
    const u32 n = g_skind_reg_n.load(std::memory_order_relaxed);
    for (u32 i = 0; i < n; ++i) {
        // Dedup per kind (see SkinDirectRegister): a read entry for the same sharp must not
        // swallow the store entry, or store_entries stays 0 and the `lowered` gate fails.
        if (g_skind_reg[i].hash == pgm_hash && g_skind_reg[i].sharp_idx == sharp_idx &&
            g_skind_reg[i].is_store) {
            return false;
        }
    }
    if (n >= kSkinDirectMax) {
        return false;
    }
    g_skind_reg[n] = SkinDirectEntry{pgm_hash, sharp_idx, word, true};
    g_skind_store_entries.fetch_add(1, std::memory_order_relaxed);
    g_skind_reg_n.store(n + 1, std::memory_order_release);
    return true;
}

void BufferCache::SkinDirectNoteStoreLowered(u64 pgm_hash, u32 sharp_idx, bool emitted) {
    const u32 n = g_skind_reg_n.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (g_skind_reg[i].hash == pgm_hash && g_skind_reg[i].sharp_idx == sharp_idx &&
            g_skind_reg[i].is_store) {
            (emitted ? g_skind_lowered_stores : g_skind_fmt0_stores)
                .fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}

void BufferCache::SkinDirectNoteMetaSkip(u64 pgm_hash, VAddr base) {
    const u64 n = g_skind_meta_skips.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 8) {
        LOG_CRITICAL(Render_Vulkan,
                     "[SKINDIRECT-STORE] dispatch of {:#x} SKIPPED as metadata clear (written "
                     "descriptor at {:#x}) - the worker did not run ({} so far)",
                     pgm_hash, base, n);
    }
}

void BufferCache::SkinDirectNoteImgInvSkipped() {
    ++skind.store_imginv_skipped;
}

u8 BufferCache::SkinDirectNote(u64 pgm_hash, u32 sharp_idx, VAddr base, u64 size,
                               bool is_written) {
    const u32 n = g_skind_reg_n.load(std::memory_order_acquire);
    u32 word = 0;
    bool found = false;
    for (u32 i = 0; i < n; ++i) {
        // The entry of the descriptor's own kind: a written descriptor (Descriptors::Add ORs
        // is_written on dedup) is judged as the store it carries, a read-only one as the read.
        if (g_skind_reg[i].hash == pgm_hash && g_skind_reg[i].sharp_idx == sharp_idx &&
            g_skind_reg[i].is_store == is_written) {
            word = g_skind_reg[i].word;
            found = true;
            break;
        }
    }
    if (!found) {
        if (n == 0) {
            // Listed shader, nothing converted: the report still has to appear, otherwise a run
            // where the recompiler matched nothing is silent and looks like a run with no data.
            SkinVramReport();
        }
        return 0; // some other descriptor of the same shader; not ours to judge
    }
    if (is_written) {
        // [SKINDIRECT-STORES] the bind loop answers base 0 / size 0 with BufferId{} ->
        // VK_NULL_HANDLE (vk_rasterizer.cpp) -> nullDescriptor: every store of this dispatch
        // into this V# is discarded. Expected for the zero-V# permutation (an empty job slot:
        // its typed store was already dropped at compile time by the FMT0 guard, nothing is
        // lost); for a real V# it would be lost work, hence one loud line either way. The rest
        // of the store census needs the owner and runs after ObtainBuffer
        // (SkinDirectNoteStoreBind). NO gpumod probe for stores: a converted output IS
        // GPU-modified after its first dispatch by design; probing would spam and pollute the
        // read census.
        if (base == 0 || size == 0) {
            ++skind.store_null;
            if (!skind.logged_store_null) {
                skind.logged_store_null = true;
                LOG_CRITICAL(Render_Vulkan,
                             "[SKINDIRECT-STORE] pgm={:#x} block dword {} (sharp_idx={}) resolved "
                             "to base={:#x} size={:#x} - null descriptor: the dispatch's stores "
                             "into this V# are discarded (expected for the zero-V#/FMT0 "
                             "permutation, whose store was dropped at compile time; lost work "
                             "if fmt0 stays 0 in the census)",
                             pgm_hash, word, sharp_idx, base, size);
            }
        }
        SkinVramReport();
        return 2;
    }
    if (!skind.probe_read) {
        skind.probe_read = true;
        if (const char* e = std::getenv("SHADPS4_SKINDIRECT_GPUMOD")) {
            skind.gpumod_probe = e[0] != '0';
        }
    }
    if (base == 0 || size == 0) {
        // The bind path answers this with BufferId{} -> a null descriptor -> robust zero reads,
        // which is byte-for-byte the miss value read_const_dynamic returns. Behaviour is safe;
        // the count is what says how often the descriptor was not there at all.
        ++skind.null_desc;
        if (!skind.logged_null) {
            skind.logged_null = true;
            LOG_CRITICAL(Render_Vulkan,
                         "[SKINDIRECT] pgm={:#x} block dword {} (sharp_idx={}) resolved to "
                         "base={:#x} size={:#x} - null descriptor, reads return zero",
                         pgm_hash, word, sharp_idx, base, size);
        }
        return 1;
    }
    ++skind.bound;
    // A bounded input of this worker is tens of KiB (headers 0x4268, list 0x2134, palette 0x30 in
    // the DESCDUMP samples). Anything past 64 MiB is not the descriptor we matched at compile
    // time, so it is worth a loud line even though the binding path handles it.
    if (size > (u64{64} << 20)) {
        ++skind.oversize;
        if (!skind.logged_oversize) {
            skind.logged_oversize = true;
            LOG_CRITICAL(Render_Vulkan,
                         "[SKINDIRECT] pgm={:#x} block dword {} (sharp_idx={}) declares {:#x} "
                         "bytes at {:#x} - far larger than the bounded input it was matched on",
                         pgm_hash, word, sharp_idx, size, base);
        }
    }
    // The authority test SkinVramRedirect applies to the same ranges: a range whose fresh
    // contents live in VRAM is stale in guest RAM, and a buffer binding reads the cache's copy.
    // Counting only - this is the number that would explain a visual regression.
    // [SKINDIRECT] the probe is off by default and never runs on an oversize range: an unbounded
    // V# would hand IsRegionGpuModified a multi-gigabyte scan inside the per-dispatch bind loop,
    // which both hurts and inflates the very number the experiment measures (reviewer, 07.09).
    if (skind.gpumod_probe && size <= (16ull << 20) && IsRegionGpuModified(base, size)) {
        ++skind.gpumod;
        if (!skind.logged_gpumod) {
            skind.logged_gpumod = true;
            LOG_CRITICAL(Render_Vulkan,
                         "[SKINDIRECT] pgm={:#x} block dword {} (sharp_idx={}) at {:#x}+{:#x} is "
                         "GPU-modified - guest RAM is not the authority for this range",
                         pgm_hash, word, sharp_idx, base, size);
        }
    }
    SkinVramReport();
    return 1;
}

void BufferCache::SkinDirectNoteStoreBind(VAddr base, u64 size, const Buffer& owner,
                                          bool unbounded, bool barrier_emitted) {
    ++skind.store_bound;
    // Home by the owner's flags, AFTER ObtainBuffer: on the first touch there is no owner before
    // it, and PageHomeKind would cost buffers_mutex + an address-space lookup per bind.
    const u8 home = owner.is_backing_backed ? 3 : (owner.is_import_backed ? 2 : 1);
    ++skind.store_home[home];
    if (barrier_emitted) {
        ++skind.store_barriers;
    }
    if (unbounded) {
        ++skind.store_unbounded;
        if (!skind.logged_store_unbounded) {
            skind.logged_store_unbounded = true;
            LOG_CRITICAL(Render_Vulkan,
                         "[SKINDIRECT-STORE] output V# at {:#x} is UNBOUNDED ({:#x} bytes): "
                         "is_written suppressed by the bind loop, no GPU mark, no barrier",
                         base, size);
        }
    }
    if (size > (u64{64} << 20)) {
        ++skind.store_oversize;
        if (!skind.logged_store_oversize) {
            skind.logged_store_oversize = true;
            LOG_CRITICAL(Render_Vulkan, "[SKINDIRECT-STORE] output V# at {:#x} declares {:#x} bytes",
                         base, size);
        }
    }
    // [R1] home 4: the page-table entry of a listed page points at guest RAM while the bound
    // store lands in the VRAM owner - live readers and the CPU keep reading the old bytes, and
    // the PT heal (NoteGpuWrittenPage: another live store on the same 16 KiB page flags it
    // CPU-modified) would upload that stale guest RAM over the output. Only a VRAM owner can
    // diverge (WritePageTableRange and ChangeRegister skip the redirect for import owners), and
    // only a page that actually has a backing address is redirected (PageHomeKind's rule) -
    // WantsPt alone over-reports. The fork's remedy for an arena that must keep guest RAM as
    // its home before the GPU's first touch is SHADPS4_BACKIMP_PREFETCH (the skinning ring
    // 0x1100000000+0x1000000 is listed there today), so the line names it.
    if (home == 1 && HomePolicy::WantsPt(base, size)) {
        bool redirected = false;
        const VAddr end = base + size;
        for (VAddr p = Common::AlignDown(base, CACHING_PAGESIZE); p < end && !redirected;
             p += CACHING_PAGESIZE) {
            redirected = HomePolicy::WantsPt(p, CACHING_PAGESIZE) && PolicyPtBda(p) != 0;
        }
        if (redirected) {
            ++skind.store_pt;
            if (!skind.logged_store_pt) {
                skind.logged_store_pt = true;
                LOG_CRITICAL(Render_Vulkan,
                             "[SKINDIRECT-STORE] output {:#x}+{:#x} has a PT-redirected page under "
                             "a VRAM owner: bound store -> VRAM, PT readers/CPU -> guest RAM, and "
                             "the PT heal uploads stale guest RAM over it (home-4 divergence). Give "
                             "the arena an import owner (SHADPS4_BACKIMP_PREFETCH) or drop it from "
                             "SHADPS4_HOMEPOLICY_PT; pt_hits counts every such bind",
                             base, size);
            }
        }
    }
    // [R2] home 1 with readbacks Disabled: guest RAM never receives this output (no fault, no
    // download), so a CPU reader sees pre-dispatch bytes, and a CPU write into a 4 KiB tracker
    // page shared with the output re-arms CPU-dirty -> the next SynchronizeBuffer uploads guest
    // RAM (which never had the output) over the VRAM copy. Relaxed/Precise download first
    // (InvalidateRegion). The old path replayed dwords into guest RAM (lossily); this one does
    // not. Counted per run by uploads_in_output (VERIFY, on by default).
    if (home == 1 && !skind.logged_store_home1_noreadback &&
        EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Disabled) {
        skind.logged_store_home1_noreadback = true;
        LOG_CRITICAL(Render_Vulkan,
                     "[SKINDIRECT-STORE] output {:#x}+{:#x} bound to a VRAM owner with readbacks "
                     "Disabled: the CPU and BDA readers keep pre-dispatch bytes for this range; "
                     "uploads_in_output > 0 in the census means a stale guest-RAM upload clobbered "
                     "it (lost update)",
                     base, size);
    }
    // The consumer guard (SkinDirectOutputIntersects): bounded outputs only - an unbounded or
    // oversize V# would drag gigabytes into the set and force every small bind in them off the
    // stream path.
    if (!unbounded && size <= (u64{64} << 20)) {
        skind.out_ranges.Add(base, size);
        skind.out_armed = true;
    }
    if (SkinDirectStoreVerifyFlag()) {
        skind.ring[skind.ring_n++ % skind.ring.size()] =
            SkinDirect::Out{base, size, scheduler.CurrentTick(), home};
        skind.ring_lo = std::min<VAddr>(skind.ring_lo, base);
        skind.ring_hi = std::max<VAddr>(skind.ring_hi, base + size);
    }
}

void BufferCache::SkinDirectVerifyUpload(VAddr addr, u64 size) {
    if (skind.ring_n == 0 || addr + size <= skind.ring_lo || addr >= skind.ring_hi) {
        return;
    }
    const u32 m = std::min<u32>(skind.ring_n, static_cast<u32>(skind.ring.size()));
    for (u32 i = 0; i < m; ++i) {
        const auto& o = skind.ring[i];
        if (addr < o.base + o.size && o.base < addr + size) {
            ++skind.uploads_in_output;
            if (skind.logged_uploads < 8) {
                ++skind.logged_uploads;
                // readbacks Disabled: this upload overwrote GPU results (lost update).
                // Relaxed/Precise: InvalidateRegion downloaded first, the upload is legitimate.
                LOG_CRITICAL(Render_Vulkan,
                             "[SKINDIRECT-STORE] guest-RAM upload {:#x}+{:#x} lands inside converted "
                             "output {:#x}+{:#x} (home {}, bound tick {}, now {}, readbacks={})",
                             addr, size, o.base, o.size, static_cast<u32>(o.home), o.tick,
                             scheduler.CurrentTick(),
                             static_cast<int>(EmulatorSettings.GetReadbacksMode()));
            }
            return;
        }
    }
}

void BufferCache::SkinDirectVerifyConsumer(VAddr addr, u64 size, bool stream, bool barrier) {
    if (skind.ring_n == 0 || addr + size <= skind.ring_lo || addr >= skind.ring_hi) {
        return;
    }
    const u32 m = std::min<u32>(skind.ring_n, static_cast<u32>(skind.ring.size()));
    for (u32 i = 0; i < m; ++i) {
        const auto& o = skind.ring[i];
        if (addr < o.base + o.size && o.base < addr + size) {
            ++skind.consumer_hits;
            if (stream) {
                ++skind.consumer_stream; // record-time snapshot: predates the dispatch. Must be 0.
            }
            if (!barrier) {
                ++skind.consumer_nobarrier; // no compute->vertex barrier requested. Must be 0.
            }
            return;
        }
    }
}

void BufferCache::SkinVramReport() {
    const auto now = std::chrono::steady_clock::now();
    if (now < skin.report_next) {
        return;
    }
    skin.report_next = now + std::chrono::seconds(5);
    // [LOSTUPD] its own gate, so the line exists in a run that names no shader at all.
    if (LostUpdateProbeFlag()) {
        LOG_CRITICAL(Render_Vulkan, "LOSTUPD: runs_over_gpu={} bytes_over_gpu={} read_runs={}",
                     g_lostupd_runs.load(std::memory_order_relaxed),
                     g_lostupd_bytes.load(std::memory_order_relaxed),
                     g_lostupd_total.load(std::memory_order_relaxed));
    }
    if (VertexBarrierImportFlag()) {
        LOG_CRITICAL(Render_Vulkan, "VTXBAR: asked={} emitted={}",
                     g_vtxbar_asked.load(std::memory_order_relaxed),
                     g_vtxbar_emitted.load(std::memory_order_relaxed));
    }
    if (ConstCacheVerifyFlag()) {
        LOG_CRITICAL(Render_Vulkan, "CONSTCACHE-VERIFY: hits={} stale={}",
                     g_cc_verify_hits.load(std::memory_order_relaxed),
                     g_cc_verify_stale.load(std::memory_order_relaxed));
    }
    // [SKINDIRECT] shares this timer so the feature never adds a second one. Printed only when
    // the env named at least one shader, so a run without it sees exactly today's output.
    if (!SkinDirectHashList().empty()) {
        LOG_CRITICAL(Render_Vulkan,
                     "SKINDIRECT: converted_reads={} compile_declines={} bound={} null={} "
                     "gpumod={} oversize={} shaders_x_words={} gpumod_probe={}",
                     g_skind_converted.load(std::memory_order_relaxed),
                     g_skind_declined.load(std::memory_order_relaxed), skind.bound,
                     skind.null_desc, skind.gpumod, skind.oversize,
                     g_skind_reg_n.load(std::memory_order_acquire),
                     skind.gpumod_probe ? 1 : 0);
        // [SKINDIRECT-STORES] the proof line of the store conversion: converted_stores counts
        // recompiler actions (3 IR stores x compiles for the worker), bound counts dispatches
        // whose output descriptor reached a cache-owned buffer. A flag being on is not proof.
        if (SkinDirectStoresFlag()) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINDIRECT-STORES: converted_stores={} lowered={} fmt0={} shape_odd={} "
                         "store_entries={} | bound={} null={} oversize={} unbounded={} "
                         "home[vram/import/backing]={}/{}/{} pt_hits={} barriers={} "
                         "imginv_skipped={} metaclear_skips={} readbacks={} | out_guard: "
                         "armed={} stream_skips={} snapro_skips={} vtx_barriers={} | verify={}: "
                         "uploads_in_output={} consumer_hits={} consumer_stream={} "
                         "consumer_nobarrier={}",
                         g_skind_converted_stores.load(std::memory_order_relaxed),
                         g_skind_lowered_stores.load(std::memory_order_relaxed),
                         g_skind_fmt0_stores.load(std::memory_order_relaxed),
                         g_skind_store_shape_odd.load(std::memory_order_relaxed),
                         g_skind_store_entries.load(std::memory_order_relaxed), skind.store_bound,
                         skind.store_null, skind.store_oversize, skind.store_unbounded,
                         skind.store_home[1], skind.store_home[2], skind.store_home[3],
                         skind.store_pt, skind.store_barriers, skind.store_imginv_skipped,
                         g_skind_meta_skips.load(std::memory_order_relaxed),
                         static_cast<int>(EmulatorSettings.GetReadbacksMode()),
                         skind.out_armed ? 1 : 0, skind.out_stream_skips, skind.out_snapro_skips,
                         skind.out_vtx_barriers, SkinDirectStoreVerifyFlag() ? 1 : 0,
                         skind.uploads_in_output, skind.consumer_hits, skind.consumer_stream,
                         skind.consumer_nobarrier);
        }
    }
    if (!skin.config_ok) {
        return;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "SKINVRAM: redirected={} bytes={} KB skipped udmask={} unread={} decode={} "
                 "too_big={} ring_busy={} staging={} clobbered={} gpu_auth={} short_hdr={} "
                 "wraps={} slots={} slot_kb={} host_ring={} off={} pt_faults={} verify={}/{}",
                 skin.redirected, skin.bytes >> 10, skin.skipped[SkinVram::UdMask],
                 skin.skipped[SkinVram::Unread], skin.skipped[SkinVram::Decode],
                 skin.skipped[SkinVram::TooBig], skin.skipped[SkinVram::RingBusy],
                 skin.skipped[SkinVram::Staging], skin.skipped[SkinVram::Clobbered],
                 skin.skipped[SkinVram::GpuAuth], skin.skipped[SkinVram::ShortHdr], skin.wraps,
                 skin.num_slots, skin.slot_bytes >> 10, skin.host_ring,
                 skin.clobbered.load(std::memory_order_relaxed) ? 1 : 0,
                 skin.fault_pages.load(std::memory_order_relaxed), skin.verify_mismatch,
                 skin.verify_checked);
}

void BufferCache::SkinVramRedirect(
    u64 pgm_hash, u32 ud_mask, u32* ud_regs,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    if (!skin.config_ok) {
        return;
    }
    const bool off = skin.clobbered.load(std::memory_order_relaxed);
    if (!skin.init_tried && !off) {
        SkinVramInit();
    }
    if (!skin.ok) {
        return;
    }
    // Reported before the `off` return, so a run that disabled itself keeps saying so (off=1)
    // instead of looking like a run where the feature never came up.
    SkinVramReport();
    if (off) {
        return;
    }
    if (skin.verify) {
        SkinVramVerifyPrev();
    }
    // A skipped dispatch keeps the old path (ud_regs untouched): count it, and let the first
    // occurrence of each reason be logged with its details.
    const auto skip = [&](SkinVram::Skip why) -> bool {
        ++skin.skipped[why];
        if (skin.skip_logged[why]) {
            return false;
        }
        skin.skip_logged[why] = true;
        return true;
    };

    // 1. PushUd packs user_data[k] for the set bits of ud_mask in ascending order and the
    //    SPIR-V reads ud_regs[0..3]; only with bits 0..3 all set is ud_regs[i] == user_data[i],
    //    the mapping the block decode relies on.
    if ((ud_mask & 0xFu) != 0xFu) {
        if (skip(SkinVram::UdMask)) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: pgm={:#x} ud_mask={:#x} does not cover user data 0..3 - not "
                         "redirected",
                         pgm_hash, ud_mask);
        }
        return;
    }
    // 2. pointers
    constexpr u64 kVaLimit = u64{1} << 40; // page-table coverage
    const u64 blk = u64{ud_regs[0]} | (u64{ud_regs[1]} << 32);
    const u64 cnt = u64{ud_regs[2]} | (u64{ud_regs[3]} << 32);
    // [SKINVRAM] bound the END of every range, not its start: a range that starts below 2^40
    // and crosses it would index the memory tracker one element past its top tier.
    if (blk == 0 || cnt == 0 || blk > kVaLimit - 128 || cnt > kVaLimit - 4 || (blk & 3) != 0 ||
        (cnt & 3) != 0) {
        if (skip(SkinVram::Decode)) {
            LOG_CRITICAL(Render_Vulkan, "SKINVRAM: pgm={:#x} blk={:#x} cnt={:#x} rejected",
                         pgm_hash, blk, cnt);
        }
        return;
    }
    // 2b. authority. The copy is taken from guest RAM, so guest RAM must BE the data. A page
    //     whose fresh contents live in a VRAM buffer (home=1 in 57/293 DESCDUMP samples for the
    //     header V#) or that the GPU wrote through a raw address (PAGEJOURNAL marks those the
    //     same way) is stale in RAM and the original path would have read the VRAM copy through
    //     the page table. ReadProcessMemory cannot see any of that: read protection is only
    //     installed in GpuReadbacksMode::Precise and the stand runs the default Disabled, so
    //     the "unread" skip can never fire for it. This is the check that can.
    const auto gpu_owned = [&](VAddr a, u64 n, const char* what) -> bool {
        if (!IsRegionGpuModified(a, n)) {
            return false;
        }
        if (skip(SkinVram::GpuAuth)) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: pgm={:#x} {} at {:#x}+{:#x} is GPU-modified - guest RAM is "
                         "not the authority{}",
                         pgm_hash, what, a, n,
                         skin.gpumod_allow ? " (SHADPS4_SKINVRAM_GPUMOD=allow: redirected anyway)"
                                           : "");
        }
        return !skin.gpumod_allow;
    };
    if (gpu_owned(blk, 128, "block") || gpu_owned(cnt, 4, "count")) {
        return;
    }
    // 3. block + count
    u32 d[32]{};
    u32 cnt_v = 0;
    if (!ReadGuestBytes(blk, d, sizeof(d)) || !ReadGuestBytes(cnt, &cnt_v, sizeof(cnt_v))) {
        if (skip(SkinVram::Unread)) {
            LOG_CRITICAL(Render_Vulkan, "SKINVRAM: pgm={:#x} blk={:#x} / cnt={:#x} not readable",
                         pgm_hash, blk, cnt);
        }
        return;
    }
    // 4. decode the three V#s (DESCDUMP's decode): base = dword0 | (dword1 & 0xfff) << 32,
    //    stride = dword1 bits 16..29, records = dword2; size = records * max(stride, 1).
    struct Range {
        VAddr guest;
        u64 size;
        u64 off;
    };
    // slot layout order: count dword, block, headers, list, palette
    Range ranges[5] = {{cnt, 4, 0}, {blk, sizeof(d), 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    static constexpr u32 kWords[3] = {16u, 20u, 24u};
    static constexpr u32 kStride[3] = {8u, 0u, 0u};
    static constexpr u32 kMinRecords[3] = {1u, 4u, 48u};
    for (u32 k = 0; k < 3; ++k) {
        const u32 w = kWords[k];
        const u64 base = u64{d[w]} | (u64{d[w + 1] & 0xfffu} << 32);
        const u32 stride = (d[w + 1] >> 16) & 0x3fffu;
        const u32 records = d[w + 2];
        const u64 size = u64{records} * (stride != 0 ? stride : 1u);
        if (stride != kStride[k] || records < kMinRecords[k] || base == 0 || base >= kVaLimit ||
            size > kVaLimit - base) { // [SKINVRAM] end of range inside the table, see above
            if (skip(SkinVram::Decode)) {
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM: pgm={:#x} blk={:#x} v#@{} rejected: {:08x} {:08x} {:08x} "
                             "{:08x} (base={:#x} stride={} records={} size={:#x})",
                             pgm_hash, blk, w, d[w], d[w + 1], d[w + 2], d[w + 3], base, stride,
                             records, size);
            }
            return;
        }
        // A well-formed range that simply does not fit is a sizing problem, not a malformed
        // descriptor: it must land on the counter whose hint names the knob that fixes it.
        if (size > skin.slot_bytes) {
            if (skip(SkinVram::TooBig)) {
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM: pgm={:#x} v#@{} needs {:#x} bytes, slot is {} - raise "
                             "SHADPS4_SKINVRAM_SLOT_KB",
                             pgm_hash, w, size, skin.slot_bytes);
            }
            return;
        }
        if (gpu_owned(base, size, k == 0 ? "headers" : k == 1 ? "list" : "palette")) {
            return;
        }
        ranges[2 + k] = Range{base, size, 0};
    }
    // 4b. The shader bounds the header index by the vertex count (ud_2|ud_3), never by the V#'s
    //     num_records - it never reads that dword at all. In guest memory an under-declared
    //     num_records is harmless (the bytes after the array are the rest of a larger one); in
    //     a slot the same over-read hits padding and then THIS dispatch's own influence-list
    //     copy, i.e. wrong bone indices for the tail of the mesh with every counter at zero.
    //     DESCDUMP found cnt == header records in 293/293 samples; enforce it instead of
    //     printing it.
    // +8: the last header record can be fetched by a 16-byte vector load, which in a slot would
    // reach into this dispatch's own influence-list copy rather than the guest's neighbour bytes.
    if (u64{cnt_v} * 8 + 8 > ranges[2].size) {
        if (skip(SkinVram::ShortHdr)) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: pgm={:#x} blk={:#x} vertex count {} needs {:#x} header bytes, "
                         "the V# declares {:#x} - not redirected",
                         pgm_hash, blk, cnt_v, u64{cnt_v} * 8, ranges[2].size);
        }
        return;
    }
    // 5. slot layout: every range starts at a 256-byte boundary plus the low 8 bits of its
    //    guest address, so 16-byte alignment (the read_const_live_x4 vector path) is identical
    //    to guest memory, and ends padded to 256 so no 16-byte read crosses into the next
    //    range; the guard page absorbs an over-read at the end of the last slot.
    u64 cursor = 0;
    for (Range& x : ranges) {
        x.off = Common::AlignUp(cursor, 256) + (x.guest & 0xFFu);
        cursor = Common::AlignUp(x.off + x.size, 256);
    }
    if (cursor > skin.slot_bytes) {
        if (skip(SkinVram::TooBig)) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: pgm={:#x} needs {} bytes, slot is {} (hdr={:#x} list={:#x} "
                         "pal={:#x}) - raise SHADPS4_SKINVRAM_SLOT_KB",
                         pgm_hash, cursor, skin.slot_bytes, ranges[2].size, ranges[3].size,
                         ranges[4].size);
        }
        return;
    }
    // 6. slot: recycled only once the cmdbuf that read it retired; never waited on (the next
    //    dispatch retries the same slot). Owner guard: a buffer registered over the slot means
    //    its entries are no longer ours - permanent stop.
    const u32 slot = skin.next_slot;
    if (skin.slot_tick[slot] != 0 && !scheduler.IsFree(skin.slot_tick[slot])) {
        if (skip(SkinVram::RingBusy)) {
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: ring busy (slot {} tick {} not retired, current {}) - raise "
                         "SHADPS4_SKINVRAM_MB",
                         slot, skin.slot_tick[slot], scheduler.CurrentTick());
        }
        return;
    }
    const VAddr slot_va = skin.synth_base + u64{slot} * skin.slot_bytes;
    for (VAddr p = slot_va; p < slot_va + skin.slot_bytes; p += CACHING_PAGESIZE) {
        if (SkinVramPageOwned(p)) {
            skin.clobbered.store(true, std::memory_order_relaxed);
            skip(SkinVram::Clobbered);
            LOG_CRITICAL(Render_Vulkan,
                         "SKINVRAM: synthetic page {:#x} acquired a buffer owner - redirect "
                         "disabled",
                         p);
            return;
        }
    }
    // 7. all guest reads before any GPU-side resource is touched, so a failure leaves nothing
    //    half-recorded. The block and the count were read above.
    //    Destination: with a ReBAR ring the bytes go straight into the mapped slot (no staging,
    //    no transfer, no barrier at all); otherwise into scratch, which is then staged. The
    //    verify self-check needs a plain-RAM image to compare against, so it forces scratch.
    const u64 slot_off = u64{slot} * skin.slot_bytes;
    const bool build_in_scratch = !skin.host_ring || skin.verify;
    u8* dst = nullptr;
    if (build_in_scratch) {
        skin.scratch.resize(cursor); // reserved to slot_bytes at init: no reallocation
        dst = skin.scratch.data();
    } else {
        dst = skin.ring->mapped_data.data() + slot_off;
    }
    for (u32 k = 2; k < 5; ++k) {
        if (!ReadGuestBytes(ranges[k].guest, dst + ranges[k].off, ranges[k].size)) {
            if (skip(SkinVram::Unread)) {
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM: pgm={:#x} range {} at {:#x}+{:#x} not readable", pgm_hash,
                             k - 2, ranges[k].guest, ranges[k].size);
            }
            return;
        }
    }
    std::memcpy(dst + ranges[0].off, &cnt_v, sizeof(cnt_v));
    // 8. patch the block COPY: the three V# bases now name the slot copies. Stride/flag bits of
    //    dword 1 are kept; dwords 0..9 (the write bases the shader stores through) and
    //    everything else keep their guest values.
    for (u32 k = 0; k < 3; ++k) {
        const u32 w = kWords[k];
        const u64 s = slot_va + ranges[2 + k].off;
        d[w] = static_cast<u32>(s);
        d[w + 1] = (d[w + 1] & ~0xfffu) | static_cast<u32>((s >> 32) & 0xfffu);
    }
    std::memcpy(dst + ranges[1].off, d, sizeof(d));
    // 8b. [SKINVRAM_VERIFY] keep this dispatch's image and its three guest ranges; the next
    //     redirect re-reads them and reports any change (see SkinVramVerifyPrev).
    if (skin.verify) {
        skin.verify_image.assign(skin.scratch.begin(), skin.scratch.begin() + cursor);
        for (u32 k = 0; k < 3; ++k) {
            skin.verify_ranges[k] = {ranges[2 + k].guest, ranges[2 + k].off, ranges[2 + k].size};
        }
        skin.verify_armed = true;
    }
    // [SKINVRAM] the slot lives in write-combined device memory on the ReBAR path: drain the
    //     store buffer before the dispatch is recorded, as every other host->device store in
    //     this file does. Cheap (one fence per redirected dispatch, no pipeline effect).
    if (skin.host_ring) {
        _mm_sfence();
    }
    // 9/10. publish the slot to the device.
    //     ReBAR ring: nothing to record. The store lands in device memory now and a host write
    //     issued before vkQueueSubmit is visible to the device by definition - no staging
    //     buffer, no vkCmdCopyBuffer and, above all, no transfer->compute barrier ~217 times
    //     per frame. The slot's previous reader is retired (IsFree above), so no reader can be
    //     racing this store.
    //     Fallback ring (no ReBAR): staging copy in the current command buffer, and the
    //     barrier is APPENDED to the caller's list so Pipeline::BindResources folds it into
    //     the single pipelineBarrier2 it issues anyway - never a standalone global barrier.
    //     Commit() watches CurrentTick, so the staged bytes live until this tick retires; the
    //     map never waits on a ring wrap - a refusal is a skip.
    if (!skin.host_ring) {
        const auto [staging, src_off] = staging_buffer.Map(cursor, 256, /*allow_wait=*/false);
        if (staging == nullptr) {
            if (skip(SkinVram::Staging)) {
                LOG_CRITICAL(Render_Vulkan,
                             "SKINVRAM: staging ring full ({} bytes wanted) - skipped", cursor);
            }
            return;
        }
        std::memcpy(staging, skin.scratch.data(), cursor);
        staging_buffer.Commit();
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.copyBuffer(
            staging_buffer.Handle(), skin.ring->Handle(),
            vk::BufferCopy{.srcOffset = src_off, .dstOffset = slot_off, .size = cursor});
        // Ranged, and its src stage is Transfer, so the CSBARRIER compute->compute filter
        // (which runs before this hook) can never match it even if it saw it.
        barriers.emplace_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = skin.ring->Handle(),
            .offset = slot_off,
            .size = cursor,
        });
    } else if (skin.verify) {
        std::memcpy(skin.ring->mapped_data.data() + slot_off, skin.scratch.data(), cursor);
    }
    // 11. push constants: ud_0/ud_1 -> slot copy of the block, ud_2/ud_3 -> slot copy of the
    //     count (full 64-bit pointers; the V# hi dwords patched above carry bits 32-43).
    const u64 blk_va = slot_va + ranges[1].off;
    const u64 cnt_va = slot_va + ranges[0].off;
    ud_regs[0] = static_cast<u32>(blk_va);
    ud_regs[1] = static_cast<u32>(blk_va >> 32);
    ud_regs[2] = static_cast<u32>(cnt_va);
    ud_regs[3] = static_cast<u32>(cnt_va >> 32);
    // 12. bookkeeping; SkinVramStamp re-stamps the slot right after the dispatch is recorded.
    skin.slot_tick[slot] = scheduler.CurrentTick();
    skin.pending_slot = static_cast<s32>(slot);
    skin.next_slot = (slot + 1) % skin.num_slots;
    if (skin.next_slot == 0) {
        // No re-publish here: the entries are a pure function of ring_bda, nothing in the tree
        // can rewrite them (the init scan proved the window unowned; OnGuestMap and the
        // per-dispatch owner guard switch the feature off if that changes), and the barriers
        // plus the 64 KiB copy it used to pay bought byte-identical values in the hot loop.
        ++skin.wraps;
    }
    ++skin.redirected;
    skin.bytes += cursor;
    // 13. verbose sample
    if (skin.verbose && (skin.redirected <= 16 || skin.redirected % 2000 == 0)) {
        LOG_CRITICAL(Render_Vulkan,
                     "SKINVRAM-LOG #{} blk={:#x} cnt={:#x}={} hdr={:#x}+{:#x} list={:#x}+{:#x} "
                     "pal={:#x}+{:#x} -> slot={} synth={:#x} used={} tick={} host={}",
                     skin.redirected, blk, cnt, cnt_v, ranges[2].guest, ranges[2].size,
                     ranges[3].guest, ranges[3].size, ranges[4].guest, ranges[4].size, slot,
                     slot_va, cursor, skin.slot_tick[slot], skin.host_ring);
    }
}

void BufferCache::SkinVramStamp(u64 tick) {
    if (skin.pending_slot >= 0) {
        skin.slot_tick[static_cast<u32>(skin.pending_slot)] = tick;
        skin.pending_slot = -1;
    }
}

bool BufferCache::TryServeFaultFast(VAddr device_addr, u64 size, bool is_write) {
    static const bool enabled = [] {
        const char* e = std::getenv("SHADPS4_RBQUEUE");
        return e && e[0] == '1';
    }();
    // Service counters (hoisted to rbstat for the postmortem partial-window dump),
    // reported every 5 s from whichever fault thread comes by.
    using namespace rbstat;
    const auto report = [&] {
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        s64 expected = next_report_ms.load(std::memory_order_relaxed);
        if (now_ms >= expected && next_report_ms.compare_exchange_strong(expected, now_ms + 5000)) {
            const u64 fast = c_fast.exchange(0);
            if (expected != 0) {
                LOG_CRITICAL(Render_Vulkan,
                             "RBSTAT: fast={} no_buf={} unsub={} empty={} changed={} heal={} "
                             "nobk={} avg_us={}",
                             fast, c_no_buffer.exchange(0), c_unsubmitted.exchange(0),
                             c_empty.exchange(0), c_changed.exchange(0),
                             c_import_heal.exchange(0), c_heal_nobk.exchange(0),
                             fast ? c_lat_us.exchange(0) / fast : 0);
            }
        }
    };
    if (!enabled || !rb_ready) {
        return false;
    }
    report();
    const auto t0 = std::chrono::steady_clock::now();

    // [fix] NOBK: a genuine AV on a hole in the PRT backing. No mutex helps:
    // copies and unmarks are powerless, only the guest's own commit cures it.
    // Wait for that commit OUTSIDE every lock - a "handled" answer with an
    // instant retry spins the OS at ~700 faults/s per thread and serialises
    // the guest on serve_mutex until the frame callback starves.
    static const bool nobk_wait = [] {
        const char* e = std::getenv("SHADPS4_NOBKWAIT");
        return !e || e[0] != '0';
    }();
    if (nobk_wait && label_uma_enabled && !IsBackingCommitted(device_addr)) {
        return WaitForBacking(device_addr);
    }

    // Fault service is serialised: fast path here, slow path and GC take the
    // same lock, so a window is never half-served by two paths at once.
    // [postmortem] unique_lock instead of scoped_lock purely to time the acquire
    // when this thread is the guest tick-writer; locking semantics unchanged.
    const bool pm_tw = IsTickWriter();
    const auto pm_lk0 = std::chrono::steady_clock::now();
    std::unique_lock serve_lk{serve_mutex};
    if (pm_tw) {
        tw_lock_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - pm_lk0)
                                 .count(),
                             std::memory_order_relaxed);
    }

    // Existing-buffer lookup with a POD snapshot and a pin. No CreateBuffer on
    // this thread, ever - buffer creation records into the GPU thread's
    // command buffer.
    vk::Buffer vk_buffer{};
    VAddr buf_start = 0;
    u64 buf_size = 0;
    u64 content_tick = 0;
    bool import_backed = false;
    bool backing_backed = false;
    u32 pinned_index = 0;
    {
        std::scoped_lock lk{buffers_mutex};
        const BufferId buffer_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!buffer_id) {
            c_no_buffer.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.is_deleted || !buffer.IsInBounds(device_addr, size)) {
            c_no_buffer.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        vk_buffer = buffer.Handle();
        buf_start = buffer.CpuAddr();
        buf_size = buffer.SizeBytes();
        content_tick = buffer.content_tick;
        import_backed = buffer.is_import_backed;
        backing_backed = buffer.is_backing_backed;
        pinned_index = buffer_id.index;
        ++buffer_pins[pinned_index];
    }
    const auto unpin = [&] {
        std::scoped_lock lk{buffers_mutex};
        auto it = buffer_pins.find(pinned_index);
        if (it != buffer_pins.end() && it->second != 0) {
            --it->second;
        }
    };

    // Same widening as the slow path: dirty data comes as scattered islands.
    constexpr u64 WindowSize = 512_KB;
    const VAddr win_start = std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), buf_start);
    const VAddr win_end =
        std::min<VAddr>(std::max<VAddr>(win_start + WindowSize, device_addr + size),
                        buf_start + buf_size);

    // One critical section: collect ranges, find the newest writing tick,
    // reject unsubmitted work, and only then subtract - a gap between collect
    // and subtract would lose marks forever.
    struct CopyRange {
        VAddr start;
        VAddr end;
        u64 staging_offset;
    };
    boost::container::small_vector<CopyRange, 16> ranges;
    u64 staging_used = 0;
    u64 needed_tick = content_tick;
    for (int attempt = 0;; ++attempt) {
        bool wait_for_submit = false;
        {
            std::scoped_lock gm{gpu_modified_mutex};
            ranges.clear();
            staging_used = 0;
            needed_tick = content_tick;
            bool overflow = false;
            gpu_modified_ranges.ForEachInRange(
                win_start, win_end - win_start, [&](VAddr start, VAddr end, const u64& tick) {
                    const u64 len = end - start;
                    const u64 aligned = (len + 63) & ~u64{63};
                    if (staging_used + aligned > 512_KB) {
                        overflow = true;
                        return;
                    }
                    ranges.push_back({start, end, staging_used});
                    staging_used += aligned;
                    needed_tick = std::max(needed_tick, tick);
                });
            if (overflow) {
                c_changed.fetch_add(1, std::memory_order_relaxed);
                unpin();
                return false;
            }
            if (ranges.empty()) {
                // An import-backed page with no tick entry refaults forever if we
                // punt here: the slow path refuses import-owned buffers, and
                // delivery would be identity anyway (the import IS guest RAM).
                // Orphaned marks showed up as a 1000+/5s refault storm on level
                // load that starved the GfxEop callback into the frame-ring
                // assert. Unprotecting IS the whole service.
                const VAddr heal_start = Common::AlignDown(device_addr, CACHING_PAGESIZE);
                const VAddr heal_end =
                    Common::AlignUp(device_addr + size, CACHING_PAGESIZE);
                // [BACKIMP] A backing buffer is import-backed too, and in `all` mode it is
                // reached through the ORDINARY CreateBuffer path, so it never enters
                // label_import_ranges - without this the heal above can never fire for it and
                // the page refaults forever (exactly the storm the comment describes, only now
                // on every guest range instead of the seeds). The buffer's own flag is the
                // authority; the range set is only LABELUMA's bookkeeping.
                if (backing_backed ||
                    (label_uma_enabled &&
                     label_import_ranges.Contains(heal_start, heal_end - heal_start))) {
                    if (!IsBackingCommitted(device_addr)) {
                        // Decommitted between the early check and here: unmark is
                        // a physical no-op and mprotect does not conjure memory.
                        // Hand the retry back - the next fault takes the NOBK
                        // wait path before any mutex.
                        c_heal_nobk.fetch_add(1, std::memory_order_relaxed);
                        unpin();
                        return true;
                    }
                    memory_tracker->UnmarkRegionAsGpuModified(heal_start,
                                                              heal_end - heal_start);
                    if (is_write) {
                        cpu_authority_ranges.Add(device_addr, size);
                        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                    }
                    c_import_heal.fetch_add(1, std::memory_order_relaxed);
                    unpin();
                    return true;
                }
                // Nothing GPU-dirty in the window: do NOT unmark anything (a
                // fresh mark may land right after); let the slow path decide.
                c_empty.fetch_add(1, std::memory_order_relaxed);
                unpin();
                return false;
            }
            if (needed_tick < scheduler.CurrentTick()) {
                for (const auto& r : ranges) {
                    gpu_modified_ranges.Subtract(r.start, r.end - r.start);
                }
                break;
            }
            // The writing commands are still being recorded; ask the GPU
            // thread to flush (throttled to one outstanding request).
            if (!rb_flush_pending.exchange(true, std::memory_order_acq_rel)) {
                liverpool->SendCommand<false>([this] {
                    scheduler.Flush();
                    rb_flush_pending.store(false, std::memory_order_release);
                });
            }
            c_unsubmitted.fetch_add(1, std::memory_order_relaxed);
            if (attempt != 0) {
                unpin();
                return false;
            }
            wait_for_submit = true;
        }
        // Off every lock: wait only for SUBMISSION of the writing tick (an
        // atomic read) - execution is covered by the GPU-side timeline wait in
        // our own submit. Submission after a flush request is millisecond-scale;
        // a stuck parser makes us fall back fast instead of freezing game
        // threads (a 2-second inline wait tried here earlier froze the game).
        if (wait_for_submit) {
            const auto pm_poll0 = std::chrono::steady_clock::now();
            const auto deadline = pm_poll0 + std::chrono::milliseconds(30);
            bool submitted = false;
            while (std::chrono::steady_clock::now() < deadline) {
                if (scheduler.CurrentTick() > needed_tick) {
                    submitted = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            if (pm_tw) {
                tw_poll_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - pm_poll0)
                                         .count(),
                                     std::memory_order_relaxed);
            }
            if (!submitted) {
                unpin();
                return false;
            }
        }
    }

    // The buffer may have been swapped or deleted between snapshot and now;
    // roll the marks back and fall back if so.
    {
        std::scoped_lock lk{buffers_mutex};
        const BufferId check_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        const bool same = check_id && check_id.index == pinned_index &&
                          !slot_buffers[check_id].is_deleted;
        if (!same) {
            std::scoped_lock gm{gpu_modified_mutex};
            for (const auto& r : ranges) {
                gpu_modified_ranges.Add(r.start, r.end - r.start, needed_tick);
            }
            c_changed.fetch_add(1, std::memory_order_relaxed);
            auto it = buffer_pins.find(pinned_index);
            if (it != buffer_pins.end() && it->second != 0) {
                --it->second;
            }
            return false;
        }
    }

    // Record and submit the download on the dedicated queue with a GPU-side
    // wait for the newest writing tick: the copy starts only after the writes
    // executed, and no CPU ever waits on the parser for this.
    auto& slot = rb_slots[rb_slot_index];
    rb_slot_index = (rb_slot_index + 1) % rb_slots.size();
    const vk::CommandBuffer cmdbuf = *slot.cmdbuf;
    (void)cmdbuf.reset();
    (void)cmdbuf.begin(
        vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    boost::container::small_vector<vk::BufferCopy, 16> copies;
    for (const auto& r : ranges) {
        copies.push_back(vk::BufferCopy{
            .srcOffset = r.start - buf_start,
            .dstOffset = r.staging_offset,
            .size = r.end - r.start,
        });
    }
    cmdbuf.copyBuffer(vk_buffer, slot.staging->Handle(), copies);
    (void)cmdbuf.end();

    const u64 wait_value = needed_tick;
    const vk::Semaphore timeline = scheduler.GetMasterSemaphore()->Handle();
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eAllCommands;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_value,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &timeline,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdbuf,
    };
    {
        std::scoped_lock sub_lk{rb_submit_mutex};
        (void)instance.GetDevice().resetFences(1, &*slot.fence);
        const auto submit_result = instance.GetReadbackQueue().submit(submit_info, *slot.fence);
        if (submit_result != vk::Result::eSuccess) {
            std::scoped_lock gm{gpu_modified_mutex};
            for (const auto& r : ranges) {
                gpu_modified_ranges.Add(r.start, r.end - r.start, needed_tick);
            }
            c_changed.fetch_add(1, std::memory_order_relaxed);
            unpin();
            return false;
        }
    }
    const auto pm_f0 = std::chrono::steady_clock::now();
    // [R6 D5] FENCEWAIT: under SHADPS4_RINGSENSE poll the readback fence in 1s slices and log the
    // stall (needed_tick vs KnownGpuTick) - names fence-starvation of the readback engine under a
    // sagging main-queue retire. Flag off => original single infinite wait, bit-identical.
    static const bool pm_ringsense = [] {
        const char* e = std::getenv("SHADPS4_RINGSENSE");
        return e && e[0] == '1';
    }();
    if (pm_ringsense) {
        u64 pm_waited_ms = 0;
        while (instance.GetDevice().waitForFences(1, &*slot.fence, VK_TRUE, 1'000'000'000ULL) ==
               vk::Result::eTimeout) {
            pm_waited_ms += 1000;
            static std::atomic<u32> pm_fw{0};
            if (pm_fw.fetch_add(1, std::memory_order_relaxed) < 16) {
                const u64 known = scheduler.GetMasterSemaphore()->KnownGpuTick();
                LOG_CRITICAL(Render_Vulkan,
                             "FENCEWAIT: waited_ms={} needed_tick={} known_tick={} delta={}",
                             pm_waited_ms, needed_tick, known, needed_tick - known);
            }
        }
    } else {
        (void)instance.GetDevice().waitForFences(1, &*slot.fence, VK_TRUE,
                                                 std::numeric_limits<u64>::max());
    }
    if (pm_tw) {
        tw_fence_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - pm_f0)
                                  .count(),
                              std::memory_order_relaxed);
    }

    // Deliver the bytes, then unmark - but never a page that has been
    // re-marked with newer GPU work while we copied (rounded out to caching
    // pages: the tracker flips whole pages, and unmarking a partially-fresh
    // page would silently lose the new data).
    auto* memory = Core::Memory::Instance();
    for (const auto& r : ranges) {
        // Import-backed source IS the destination RAM: delivery is identity at
        // best and a stale rollback at worst. Unmark below is all that's needed.
        if (import_backed) {
            continue;
        }
        // CPU-written data is newer than the GPU copy by protocol: the guest's
        // GfxEop callback stores frame ticks with plain CPU writes into pages
        // that also carry a GPU-modified mark, and a stale download delivered
        // over them is exactly the zero tick the game asserts on. Never clobber.
        // [RBQUARTER] SHADPS4_RB_QUARTER=1: refuse per tracker page (4 KiB) instead of the
        // whole 16 KiB delivery. A job page where the CPU owns one quarter and the GPU
        // writes another had every delivery refused (PAGETRACE: 0x1101d98000 cra=128 of
        // crs=129), so the CPU never saw those results at all.
        if (RbQuarterEnabled()) {
            constexpr u64 kQuarter = 4096;
            for (VAddr q = r.start; q < r.end; q += kQuarter) {
                const VAddr qe = std::min<VAddr>(r.end, q + kQuarter);
                bool owned = false;
                {
                    std::scoped_lock gm{gpu_modified_mutex};
                    owned = cpu_authority_ranges.Intersects(q, qe - q);
                }
                if (owned || memory_tracker->IsRegionCpuModified(q, qe - q)) {
                    PageTrace::Note(q, PageTrace::CRA);
                    rb_quarter_refused.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                memory->TryWriteBacking(std::bit_cast<u8*>(q),
                                        slot.staging->mapped_data.data() + r.staging_offset +
                                            (q - r.start),
                                        static_cast<u32>(qe - q));
                rbstat::NoteDownload(q, qe - q, 0);
                rb_quarter_delivered.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        {
            std::scoped_lock gm{gpu_modified_mutex};
            if (cpu_authority_ranges.Intersects(r.start, r.end - r.start)) {
                PageTrace::Note(r.start, PageTrace::CRA);
                continue;
            }
        }
        if (memory_tracker->IsRegionCpuModified(r.start, r.end - r.start)) {
            PageTrace::Note(r.start, PageTrace::CRA);
            continue;
        }
        memory->TryWriteBacking(std::bit_cast<u8*>(r.start),
                                slot.staging->mapped_data.data() + r.staging_offset,
                                static_cast<u32>(r.end - r.start));
        rbstat::NoteDownload(r.start, r.end - r.start, 1);
    }
    {
        std::scoped_lock gm{gpu_modified_mutex};
        for (const auto& r : ranges) {
            boost::container::small_vector<std::pair<VAddr, VAddr>, 8> fresh;
            gpu_modified_ranges.ForEachInRange(r.start, r.end - r.start,
                                               [&](VAddr fs, VAddr fe, const u64&) {
                                                   fresh.push_back({
                                                       Common::AlignDown(fs, CACHING_PAGESIZE),
                                                       Common::AlignUp(fe, CACHING_PAGESIZE),
                                                   });
                                               });
            VAddr cur = r.start;
            const auto unmark_upto = [&](VAddr upto) {
                if (upto > cur) {
                    memory_tracker->UnmarkRegionAsGpuModified(cur, upto - cur);
                }
            };
            for (const auto& [fs, fe] : fresh) {
                unmark_upto(std::min(fs, r.end));
                cur = std::max(cur, std::min(fe, r.end));
            }
            unmark_upto(r.end);
        }
    }
    if (is_write) {
        {
            std::scoped_lock gm{gpu_modified_mutex};
            cpu_authority_ranges.Add(device_addr, size);
        }
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
    unpin();
    c_fast.fetch_add(1, std::memory_order_relaxed);
    PageTrace::Note(device_addr, PageTrace::CRS);
    const u64 pm_lat_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
    c_lat_us.fetch_add(pm_lat_us, std::memory_order_relaxed);
    Postmortem::AtomicMax(rb_lat_max_us, pm_lat_us);
    return true;
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    if (!is_write) {
        PageTrace::Note(device_addr, PageTrace::CR);
    }
    // [probe] SHADPS4_RINGWATCH=1: hardware write watch on the frame-ring tick of
    // slot 0 (eboot+0x10CDB40+0xA08). Names the thread and guest RIP that publishes
    // frame ticks - the whole readback-engine plan rests on it being the guest's
    // GfxEop callback, so verify before building. Armed here because by the first
    // readback fault the game's worker threads all exist.
    {
        static const bool ring_watch = [] {
            const char* e = std::getenv("SHADPS4_RINGWATCH");
            return e && e[0] == '1';
        }();
        static std::atomic<bool> rw_armed{false};
        if (ring_watch && MemoryPatcher::g_eboot_address != 0 &&
            !rw_armed.exchange(true, std::memory_order_relaxed)) {
            const VAddr tick0 = MemoryPatcher::g_eboot_address + 0x10CDB40 + 0xA08;
            Core::ArmWriteWatch(tick0);
            LOG_CRITICAL(Render_Vulkan, "RINGWATCH armed at {:#x} (slot 0 tick)", tick0);
        }
    }
    {
        // Does the CPU ever fault on GPU-written memory, i.e. is the writeback path armed?
        static std::atomic<u64> reads{0};
        static std::atomic<s64> next_ms{0};
        const u64 n = reads.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
        // A bare count says a storm is happening but not where. Bucket the faulting address by
        // 16 MiB so the report can name the range that keeps being marked GPU-modified and never
        // cleared - that range is the one stalling the guest.
        constexpr u32 BucketBits = 24;
        constexpr u32 BucketCount = 1u << (40 - BucketBits);
        static std::array<std::atomic<u32>, BucketCount> buckets{};
        const u64 bucket = (device_addr >> BucketBits);
        if (bucket < BucketCount) {
            buckets[bucket].fetch_add(1, std::memory_order_relaxed);
        }
        // A fault on a page no buffer covers cannot be resolved: the handler reports success,
        // nothing unprotects the page, the guest retries the same instruction and faults again.
        // Counting these separates "busy readback" from "livelock".
        static std::atomic<u64> unowned{0};
        if (!IsRegionRegistered(device_addr, size)) {
            unowned.fetch_add(1, std::memory_order_relaxed);
        }
        // Fine per-MB histogram for the mixed 0x113x arena: the 16MB buckets
        // are too coarse to carve a label seed out of it.
        constexpr VAddr FineBase = 0x1130000000ull;
        constexpr u32 FineCount = 256; // 256 MB span, 1 MB buckets
        static std::array<std::atomic<u32>, FineCount> fine{};
        if (device_addr >= FineBase && device_addr < FineBase + (u64{FineCount} << 20)) {
            fine[(device_addr - FineBase) >> 20].fetch_add(1, std::memory_order_relaxed);
        }
        s64 expected = next_ms.load(std::memory_order_relaxed);
        if (now >= expected && next_ms.compare_exchange_strong(expected, now + 5000)) {
            if (expected != 0) {
                std::array<std::pair<u32, u32>, 3> top{};
                for (u32 i = 0; i < BucketCount; ++i) {
                    const u32 v = buckets[i].load(std::memory_order_relaxed);
                    if (v > top[2].first) {
                        top[2] = {v, i};
                        std::sort(top.begin(), top.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                    }
                }
                LOG_CRITICAL(Render_Vulkan,
                             "READBACK: {} faults in 5s, {} on pages NO BUFFER owns | worst "
                             "{:#x}:{} {:#x}:{} {:#x}:{}",
                             n, unowned.exchange(0, std::memory_order_relaxed),
                             u64{top[0].second} << BucketBits, top[0].first,
                             u64{top[1].second} << BucketBits, top[1].first,
                             u64{top[2].second} << BucketBits, top[2].first);
                for (auto& b : buckets) {
                    b.store(0, std::memory_order_relaxed);
                }
                std::array<std::pair<u32, u32>, 4> ftop{};
                for (u32 i = 0; i < FineCount; ++i) {
                    const u32 v = fine[i].load(std::memory_order_relaxed);
                    if (v > ftop[3].first) {
                        ftop[3] = {v, i};
                        std::sort(ftop.begin(), ftop.end(),
                                  [](const auto& a2, const auto& b2) {
                                      return a2.first > b2.first;
                                  });
                    }
                    fine[i].store(0, std::memory_order_relaxed);
                }
                if (ftop[0].first != 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "READFINE: {:#x}:{} {:#x}:{} {:#x}:{} {:#x}:{}",
                                 FineBase + (u64{ftop[0].second} << 20), ftop[0].first,
                                 FineBase + (u64{ftop[1].second} << 20), ftop[1].first,
                                 FineBase + (u64{ftop[2].second} << 20), ftop[2].first,
                                 FineBase + (u64{ftop[3].second} << 20), ftop[3].first);
                }
            }
            reads.store(0, std::memory_order_relaxed);
        }
    }
    if (TryServeFaultFast(device_addr, size, is_write)) {
        return;
    }
    // [postmortem] Time the synchronous slow path when it runs on the tick-writer.
    const bool pm_slow_tw = rbstat::IsTickWriter();
    const auto pm_slow0 = std::chrono::steady_clock::now();
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        // Shares serve_mutex with the fast path: a window must never be
        // half-served by both at once (subtracted marks vs whole-window unmark).
        std::scoped_lock serve_lk{serve_mutex};
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        // GPU-modified ranges come as many small scattered islands, so the download
        // is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr buf_start = buffer.CpuAddr();
        const VAddr buf_end = buf_start + buffer.SizeBytes();
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), buf_start);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), buf_end);
        DownloadBufferMemory<false>(buffer, window_start, window_end - window_start);
        if (is_write) {
            {
                std::scoped_lock gm{gpu_modified_mutex};
                cpu_authority_ranges.Add(device_addr, size);
            }
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    });
    if (pm_slow_tw) {
        rbstat::tw_slow_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - pm_slow0)
                                         .count(),
                                     std::memory_order_relaxed);
    }
}

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    // Lock order: gpu_modified_mutex -> RegionManager::lock. The callback below
    // runs under the region lock, so the map lock must be taken out here; map
    // accesses inside run lock-free because we already hold it.
    std::scoped_lock gm_lk{gpu_modified_mutex};
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(
                device_addr_out, range_size,
                [&](VAddr start, VAddr end, const u64&) { add_download(start, end); });
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        for (const auto& copy : copies) {
            if (buffer.is_import_backed) {
                break;
            }
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            // Same protocol rule as the fast path: CPU-written bytes are newer
            // than the GPU copy - never clobber them with a stale download.
            if (RbQuarterEnabled()) {
                // [RBQUARTER] per 4 KiB tracker page, see TryServeFaultFast
                constexpr u64 kQuarter = 4096;
                const VAddr copy_end = copy_device_addr + copy.size;
                for (VAddr q = copy_device_addr; q < copy_end; q += kQuarter) {
                    const VAddr qe = std::min<VAddr>(copy_end, q + kQuarter);
                    if (cpu_authority_ranges.Intersects(q, qe - q) ||
                        memory_tracker->IsRegionCpuModified(q, qe - q)) {
                        PageTrace::Note(q, PageTrace::CRA);
                        rb_quarter_refused.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    memory->TryWriteBacking(std::bit_cast<u8*>(q),
                                            download + dst_offset + (q - copy_device_addr),
                                            static_cast<u32>(qe - q));
                    rbstat::NoteDownload(q, qe - q, 2);
                    rb_quarter_delivered.fetch_add(1, std::memory_order_relaxed);
                }
                continue;
            }
            if (cpu_authority_ranges.Intersects(copy_device_addr, copy.size) ||
                memory_tracker->IsRegionCpuModified(copy_device_addr, copy.size)) {
                PageTrace::Note(copy_device_addr, PageTrace::CRA);
                continue;
            }
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size);
            rbstat::NoteDownload(copy_device_addr, copy.size, 3);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
    };
    if constexpr (async) {
        scheduler.DeferOperation(write_data);
    } else {
        scheduler.Finish();
        write_data();
    }
}

void BufferCache::BindVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        const bool gpu_mod = IsRegionGpuModified(range.base_address, size);
        // [SKINDIRECT-STORES] the skinning worker's converted output read as vertex data: an
        // import-backed owner has no tracker mark, so gpu_mod alone would skip the compute->
        // vertex barrier. GetBarrier itself only emits on an access flip, so asking is free
        // for the read->read draws that follow the first one.
        const bool sd_out = !gpu_mod && SkinDirectOutputIntersects(range.base_address, size);
        if (sd_out) {
            ++skind.out_vtx_barriers;
        }
        // [VTXBAR] see the flag's declaration: an import-backed owner carries no tracker mark,
        // so gpu_mod is always false for it and the compute->vertex barrier is never requested.
        const bool import_vtx =
            VertexBarrierImportFlag() && !gpu_mod && !sd_out && buffer->is_import_backed;
        if (gpu_mod || sd_out || import_vtx) {
            if (import_vtx) {
                g_vtxbar_asked.fetch_add(1, std::memory_order_relaxed);
            }
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                if (import_vtx) {
                    g_vtxbar_emitted.fetch_add(1, std::memory_order_relaxed);
                }
                barriers.emplace_back(*barrier);
            }
        }
        // [SKINDIRECT-STORES][VERIFY] who reads the converted outputs as vertex data, and whether
        // it got a record-time snapshot (stream) or a barrier-protected cache buffer.
        if (SkinDirectStoreVerifyFlag()) {
            SkinDirectVerifyConsumer(range.base_address, size, buffer == &stream_buffer,
                                     gpu_mod || sd_out);
        }
    }

    // Bind vertex buffers
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->vk_buffer);
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void BufferCache::BindIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(vk_buffer->Handle(), offset, index_type);
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes) &&
            !IntersectsLabelImport(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes) &&
        !IntersectsLabelImport(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !IntersectsLabelImport(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        if (BufWatched(dst, dst + num_bytes)) {
            static u32 reported = 0;
            if (reported < 12) {
                ++reported;
                LOG_CRITICAL(Render_Vulkan, "GPUMODWATCH: CopyBuffer marks {:#x}..{:#x}", dst,
                             dst + num_bytes);
            }
        }
        if (!buffer.is_import_backed) {
            std::scoped_lock lk{gpu_modified_mutex};
            gpu_modified_ranges.Add(dst, num_bytes, scheduler.CurrentTick());
        }
        return buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

void BufferCache::DebugCompareWatched(VAddr device_addr) {
    // Counters of the job object: two of them agree, the third runs ahead by one and is the
    // one the consumer trusts. Watching them over time shows when they diverge.
    {
        const u32* hdr = std::bit_cast<const u32*>(device_addr);
        static u32 last_a = ~0u, last_b = ~0u, last_c = ~0u;
        const u32 a = hdr[0x1C / 4];
        const u32 b = hdr[0x20 / 4];
        const u32 c = hdr[0x24 / 4];
        if (a != last_a || b != last_b || c != last_c) {
            last_a = a;
            last_b = b;
            last_c = c;
            LOG_CRITICAL(Render_Vulkan, "JOBHDR: addr={:#x} +1C={} +20={} +24={}{}", device_addr, a,
                         b, c, c > b ? "  <-- consumer will overrun" : "");
        }
        return;
    }
    // Is the GPU copy of this range fresher than guest memory? If yes, the data exists and
    // only the trip back is missing; if both are stale, nothing ever wrote it.
    const BufferId id = FindBuffer(device_addr, 64);
    Buffer& buffer = slot_buffers[id];
    const u64 offset_in_buffer = device_addr - buffer.CpuAddr();
    const auto [download, offset] = download_buffer.Map(64);
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(),
                      vk::BufferCopy{.srcOffset = offset_in_buffer, .dstOffset = offset, .size = 64});
    scheduler.Finish();
    std::string gpu_hex, guest_hex;
    const u8* guest = std::bit_cast<const u8*>(device_addr);
    for (u32 i = 0; i < 32; ++i) {
        fmt::format_to(std::back_inserter(gpu_hex), "{:02x}", download[i]);
        fmt::format_to(std::back_inserter(guest_hex), "{:02x}", guest[i]);
    }
    LOG_CRITICAL(Render_Vulkan, "WATCHCMP addr={:#x}\n  gpu  ={}\n  guest={}", device_addr,
                 gpu_hex, guest_hex);
}

// [CONSTCACHE] SHADPS4_CONSTCACHE=1 (default 0). The same constant ranges are snapshotted into
// the stream ring hundreds of thousands of times a second; if the guest has not touched a range
// and the ring has not reused its memory, last frame's offset is still valid.
//
// Correctness, spelled out because the previous cache in this file was wrong:
//  (a) the key is an EXACT (address, size) pair - a hash lookup, never a range containment test,
//      so there is no subtraction that can wrap. This is what the lookup cache got wrong.
//  (b) an entry is only used when BOTH the frame number AND the stream ring's wrap generation
//      still match, and only after IsRegionCpuModified says the guest has not written since.
//  (c) the whole map is cleared on a frame boundary or a wrap, so a stale key cannot exist -
//      there is nothing to keep in sync, only to discard.
u64 BufferCache::ConstCacheCopy(VAddr device_addr, u32 size) {
    // Default OFF: the reference build (shadps4_OPT_1716) enables this explicitly, so the
    // default stays identical to what has actually been played. SHADPS4_CONSTCACHE=1.
    static const bool enabled = [] {
        const char* e = std::getenv("SHADPS4_CONSTCACHE");
        return e != nullptr && e[0] == '1';
    }();
    if (!enabled) {
        return stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
    }
    // EXACTLY the 17:16 logic. Two later "improvements" were reverted:
    //  - dropping the per-frame clear hung the loader;
    //  - clearing on a wrap detected during Copy made texture flicker WORSE, which contradicts
    //    the staleness theory that motivated it. Until measurements explain that, this function
    //    does not change. The counters below exist to produce those measurements.
    const u64 frame = Vulkan::g_dbg_frame_no.load(std::memory_order_relaxed);
    const u64 wrap = stream_buffer.WrapGeneration();
    if (frame != const_cache_frame || wrap != const_cache_wrap) {
        Postmortem::g_constcache_clears.fetch_add(1, std::memory_order_relaxed);
        if (frame != const_cache_frame) {
            Postmortem::g_constcache_clear_frame.fetch_add(1, std::memory_order_relaxed);
        }
        if (wrap != const_cache_wrap) {
            Postmortem::g_constcache_clear_wrap.fetch_add(1, std::memory_order_relaxed);
        }
        const_cache.clear();
        const_cache_frame = frame;
        const_cache_wrap = wrap;
    }
    const bool verify = ConstCacheVerifyFlag();
    const auto key = std::make_pair(device_addr, size);
    if (const auto it = const_cache.find(key); it != const_cache.end()) {
        if (!memory_tracker->IsRegionCpuModified(device_addr, size)) {
            Postmortem::g_constcache_hits.fetch_add(1, std::memory_order_relaxed);
            if (verify) {
                g_cc_verify_hits.fetch_add(1, std::memory_order_relaxed);
                const u64 now_sum =
                    ConstCacheSum(std::bit_cast<const u8*>(device_addr), size);
                if (now_sum != it->second.sum) {
                    g_cc_verify_stale.fetch_add(1, std::memory_order_relaxed);
                }
            }
            return it->second.offset;
        }
        Postmortem::g_constcache_cpumod.fetch_add(1, std::memory_order_relaxed);
        const_cache.erase(it);
    }
    Postmortem::g_constcache_misses.fetch_add(1, std::memory_order_relaxed);
    const u64 offset = stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
    const u64 wrap_after = stream_buffer.WrapGeneration();
    if (wrap_after != wrap) {
        // MEASURE ONLY - no clear here. Whether this case needs one is the open question.
        Postmortem::g_constcache_wrap_in_copy.fetch_add(1, std::memory_order_relaxed);
    }
    const_cache_wrap = wrap_after;
    const_cache[key] = ConstEntry{
        offset, verify ? ConstCacheSum(std::bit_cast<const u8*>(device_addr), size) : 0};
    return offset;
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id,
                                                  u32 sync_limit) {
    // [SLOWPKT] buf phase (lookup/CreateBuffer/synchronize).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhBuf};
    // [PROBECOST 10.09] This timer used to be an unconditional UsTimer: two clock reads plus an
    // atomic on every one of ~1.77M calls per 5 s window, i.e. ~240 us per frame of pure
    // measurement, paid even when nobody looks at the number. UsTimerOpt is the same sink under
    // the existing SHADPS4_BUFPROF_LEVEL gate, so the detailed breakdown is still available on
    // demand and costs nothing when it is off. The call counter stays: it is one relaxed atomic
    // and the census needs it.
    Postmortem::UsTimerOpt buf_timer{Postmortem::g_buf_obtain_us};
    Postmortem::g_buf_obtains.fetch_add(1, std::memory_order_relaxed);
    Postmortem::g_buf_probes.fetch_add(2, std::memory_order_relaxed);
    // [SLOWOBTAIN] SHADPS4_SLOWOBTAIN=<us>: name every bind that takes longer than this, with
    // the path it took (stream copy vs cached buffer) and where inside it the time went.
    // SLOWPKT only says "buf=8 ms, binds=1"; this says which buffer and which step.
    static const u64 slow_obtain_us = [] {
        const char* e = std::getenv("SHADPS4_SLOWOBTAIN");
        return e ? std::strtoull(e, nullptr, 10) : u64{0};
    }();
    using SoClock = std::chrono::steady_clock;
    SoClock::time_point so_t0{}, so_t_find{}, so_t_sync{};
    if (slow_obtain_us) {
        so_t0 = SoClock::now();
    }
    const auto so_us = [](SoClock::time_point a, SoClock::time_point b) {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
    };
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= CACHING_PAGESIZE) {
        bool gpu_mod;
        {
            Postmortem::UsTimerOpt t{Postmortem::g_buf_gpumod_us};
            gpu_mod = IsRegionGpuModified(device_addr, size);
        }
        bool label_int = false;
        if (!gpu_mod) {
            Postmortem::UsTimerOpt t{Postmortem::g_buf_labelint_us};
            label_int = IntersectsLabelImport(device_addr, size);
        }
        // [SKINDIRECT-STORES] a converted output of the skinning worker: its import-backed
        // owner carries no tracker mark, so gpu_mod is false and a record-time copy here would
        // predate the dispatch that fills it (the same reason LABELUMA ranges skip this path).
        // One bool test until the first output is bound.
        bool sd_out = false;
        if (!gpu_mod && !label_int && SkinDirectOutputIntersects(device_addr, size)) {
            sd_out = true;
            ++skind.out_stream_skips;
        }
        if (!gpu_mod && !label_int && !sd_out) {
            // Import-backed ranges must not be snapshotted at record time: in-flight
            // GPU stores would be missing from the copy.
            Postmortem::UsTimerOpt t{Postmortem::g_buf_streamcopy_us};
            Postmortem::g_buf_streamcopies.fetch_add(1, std::memory_order_relaxed);
            Postmortem::g_stream_hist[size < 256          ? 0
                                      : size < 1024       ? 1
                                      : size < CACHING_PAGESIZE ? 2
                                                          : 3]
                .fetch_add(1, std::memory_order_relaxed);
            // [TOUCHSTAT] This path never reaches the backing import: it copies into the
            // device-local stream buffer, so these bytes are already where we would want them.
            Parserprof::g_ts_stream_bytes.fetch_add(size, std::memory_order_relaxed);
            Parserprof::g_ts_stream_calls.fetch_add(1, std::memory_order_relaxed);
            const u64 offset = ConstCacheCopy(device_addr, size);
            if (slow_obtain_us) {
                const u64 total = so_us(so_t0, SoClock::now());
                if (total >= slow_obtain_us) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "SLOWOBTAIN: stream addr={:#x} size={:#x} took {}us "
                                 "(stream buffer wrap/wait?)",
                                 device_addr, size, total);
                }
            }
            return {&stream_buffer, offset};
        }
    }
    if (IsBufferInvalid(buffer_id)) {
        Postmortem::UsTimerOpt t{Postmortem::g_buf_find_us};
        buffer_id = FindBuffer(device_addr, size);
    }
    if (slow_obtain_us) {
        so_t_find = SoClock::now();
    }
    Buffer& buffer = slot_buffers[buffer_id];
    // [TOUCHSTAT] Per-frame census of what the GPU reaches for. `bound` counts every request with
    // repeats; `unique` counts each buffer once per frame, on its first touch - that is exactly
    // what a "copy into VRAM on first bind" scheme would have to move per frame.
    {
        if (is_written) {
            Parserprof::g_ts_bound_rw.fetch_add(size, std::memory_order_relaxed);
        } else {
            Parserprof::g_ts_bound_ro.fetch_add(size, std::memory_order_relaxed);
        }
        const u64 ts_frame = Vulkan::g_dbg_frame_no.load(std::memory_order_relaxed) + 1;
        if (buffer.touch_stamp != ts_frame) {
            buffer.touch_stamp = ts_frame;
            const u64 whole = buffer.SizeBytes();
            Parserprof::g_ts_unique_bufs.fetch_add(1, std::memory_order_relaxed);
            if (is_written) {
                Parserprof::g_ts_unique_rw.fetch_add(whole, std::memory_order_relaxed);
            } else {
                Parserprof::g_ts_unique_ro.fetch_add(whole, std::memory_order_relaxed);
            }
        }
    }
    // [SNAPRO] Read-only guest range -> serve it from a VRAM copy refreshed once per frame.
    // Excluded on purpose: anything the GPU writes (it must stay where the guest sees it) and
    // every unbounded descriptor (sync_limit != 0) - those are ND's job queues, which compute
    // shaders POLL in a loop. A poll shader reading a snapshot never leaves the loop, and that
    // is exactly how seeds/small/bywrite used to hang.
    // [SKINDIRECT-STORES] and never a converted skinning output: the once-per-frame copy would
    // serve the previous dispatch's vertices (the worker writes the import owner directly).
    // Evaluated only for a SNAPRO candidate, so a run without SNAPRO pays nothing here.
    const bool snapro_candidate = SnaproEnabled() && !is_written && sync_limit == 0 &&
                                  buffer.is_import_backed &&
                                  buffer.SizeBytes() <= SnaproMaxBytes() &&
                                  buffer.SizeBytes() >= SnaproMinBytes() && buffer.CpuAddr() != 0;
    if (snapro_candidate && SkinDirectOutputIntersects(device_addr, size)) {
        ++skind.out_snapro_skips;
    } else if (snapro_candidate) {
        const u64 snap_frame = Vulkan::g_dbg_frame_no.load(std::memory_order_relaxed) + 1;
        auto [it, inserted] = snap_shadows.try_emplace(buffer.CpuAddr());
        SnapShadow& sh = it->second;
        if (inserted || sh.size != buffer.SizeBytes()) {
            // First sight of this range (or it changed size): make the VRAM copy now, but keep
            // reading the guest memory this frame - the copy is only filled at the next frame
            // boundary, before drawing starts.
            sh.buf = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::DeviceLocal,
                                              buffer.CpuAddr(), AllFlags, buffer.SizeBytes());
            sh.size = buffer.SizeBytes();
            sh.fresh_frame = 0;
        }
        if (sh.used_frame != snap_frame) {
            sh.binds_prev = sh.binds_now;
            sh.binds_now = 0;
        }
        sh.used_frame = snap_frame;
        ++sh.binds_now;
        // Not "refreshed in exactly this frame": the frame counter ticks between the refresh
        // (OnSubmit) and the binds that follow, so a strict match almost never held and the
        // copies were made for nothing. One frame of slack is what the scheme promises anyway.
        if (sh.buf && sh.fresh_frame != 0 && (SnaproStale() || snap_frame - sh.fresh_frame <= 1)) {
            ++snap_hits;
            return {sh.buf.get(), static_cast<u32>(buffer.Offset(device_addr))};
        }
        ++snap_misses;
    }
    // sync_limit: для unbounded-дескрипторов («вся память», ND) синхронизируем
    // только окно у базы — полный диапазон в гигабайты означает пере-заливку
    // всего, что гость записал куда угодно, на каждом draw (наблюдали 12 c).
    // Плюс не чаще раза за флип: сотни draw за кадр биндят один и тот же
    // дескриптор, а окно между флипами гость не меняет.
    if (sync_limit) {
        const u64 flip_no = Vulkan::g_dbg_frame_no.load();
        if (unbounded_sync_stamp != flip_no) {
            unbounded_sync_stamp = flip_no;
            Postmortem::UsTimerOpt t{Postmortem::g_buf_sync_us};
            SynchronizeBuffer(buffer, device_addr, std::min(size, sync_limit), is_written,
                              is_texel_buffer);
        }
    } else {
        {
            Postmortem::UsTimerOpt t{Postmortem::g_buf_sync_us};
            SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
        }
        // A bounded "writeback window" with is_written=true was tried here to arm the
        // download path for unbounded descriptors. It changed nothing measurable and was
        // followed by an alignment assertion in BindBuffers, so it is gone.
    }
    if (slow_obtain_us) {
        so_t_sync = SoClock::now();
    }
    if (is_written) {
        Postmortem::UsTimerOpt wt{Postmortem::g_buf_written_us};
        // [WRITESTAT] Census by write behaviour. O(1) per call, no iteration, works in any mode.
        {
            const u64 frame = Vulkan::g_dbg_frame_no.load(std::memory_order_relaxed);
            if (!buffer.counted_written) {
                buffer.counted_written = true;
                buffer.write_first_frame = frame;
                // [BYWRITE] Teach the placement heuristic that this address range is written,
                // which fixes FUTURE buffers in the zone. No-op unless the mode is on. Runs once
                // per buffer, never per draw.
                auto& bw_imp = Vulkan::BackingImport::Get();
                bw_imp.NoteGuestWrite(buffer.CpuAddr(), buffer.SizeBytes());
                // And fix THIS buffer: it lives in VRAM, the GPU is about to write it, so the
                // guest would keep reading the pre-write bytes out of its own memory. Queue it
                // for demotion at the next submit boundary - never here, in the middle of a
                // draw, where its device address is already published in the BDA table and
                // recorded commands still reference it.
                if (bw_imp.GetMode() == Vulkan::BackingImport::Mode::ByWrite &&
                    !buffer.is_import_backed && !buffer.bywrite_queued &&
                    bywrite_requeue.size() < 4096) {
                    buffer.bywrite_queued = true;
                    bywrite_requeue.emplace_back(buffer.CpuAddr(), buffer.SizeBytes());
                    ++bywrite_seen;
                    if (bywrite_seen <= 64 || (bywrite_seen & 63) == 0) {
                        const u64 bw_addr = buffer.CpuAddr();
                        const u64 bw_kb = buffer.SizeBytes() >> 10;
                        const u64 bw_total = bywrite_seen;
                        LOG_CRITICAL(Render_Vulkan,
                                     "BYWRITE: late-write buf={:#x} size={} KB -> "
                                     "action=requeue (total={})",
                                     bw_addr, bw_kb, bw_total);
                    }
                }
                Parserprof::g_ws_written_bufs.fetch_add(1, std::memory_order_relaxed);
                Parserprof::g_ws_written_bytes.fetch_add(buffer.SizeBytes(),
                                                         std::memory_order_relaxed);
            }
            if (!buffer.counted_late && frame > Parserprof::WriteStatEarlyFrames()) {
                buffer.counted_late = true;
                Parserprof::g_ws_late_bufs.fetch_add(1, std::memory_order_relaxed);
                Parserprof::g_ws_late_bytes.fetch_add(buffer.SizeBytes(),
                                                      std::memory_order_relaxed);
            }
        }
        // An unbounded descriptor covers gigabytes; marking all of it GPU-modified
        // poisons the tracker for the whole heap. Mark only the window around the
        // base, which is where the shader actually writes.
        static constexpr u64 UnboundedMarkWindow = 16_MB;
        const u64 marked = size >= 1_GB ? std::min<u64>(size, UnboundedMarkWindow) : size;
        if (BufWatched(device_addr, device_addr + marked)) {
            static u32 reported = 0;
            if (reported < 12) {
                ++reported;
                LOG_CRITICAL(Render_Vulkan,
                             "GPUMODWATCH: ObtainBuffer marks {:#x}..{:#x} (asked {:#x}) - a "
                             "descriptor base landing here paints guest code as GPU-modified",
                             device_addr, device_addr + marked, size);
            }
        }
        if (!buffer.is_import_backed &&
            !(DualHomeEnabled() && !DualKeepMarks() && IsDualRange(device_addr, marked))) {
            // Lock acquisition timed separately: 830k contended acquisitions per window would
            // look exactly like the unaccounted time we are hunting.
            {
                Postmortem::UsTimerOpt lw{Postmortem::g_buf_lockwait_us};
                std::scoped_lock lk{gpu_modified_mutex};
                gpu_modified_ranges.Add(device_addr, marked, scheduler.CurrentTick());
            }
        }
        // Was a getenv on every written access - a linear scan of the environment block in the
        // hot path. Behaviourally identical, just read once.
        static const bool dbglog = [] {
            const char* e = std::getenv("SHADPS4_DBGLOG");
            return e != nullptr && e[0] == '1';
        }();
        if (dbglog) {
            LOG_WARNING(Render_Vulkan, "GPUMOD: guest={:#x} size={:#x}", device_addr, marked);
        }
    }
    {
        Postmortem::UsTimerOpt rt{Postmortem::g_buf_ret_us};
        // The job-record array the guest consumer crashes on always sits at the same
        // offset inside its allocation, so it can be spotted by its low bits.
        static const u64 watch_low = [] {
            const char* e = std::getenv("SHADPS4_WATCHLOW");
            return e ? std::strtoull(e, nullptr, 0) : u64{0};
        }();
        // The record lives at a fixed offset inside a 1 MiB block, so look for a buffer
        // that CONTAINS such an address rather than starting at one.
        VAddr watched = 0;
        if (watch_low != 0) {
            VAddr candidate = (device_addr & ~u64{0xFFFFF}) | watch_low;
            if (candidate < device_addr) {
                candidate += 1_MB;
            }
            // Only the guest heap where the job structures live, otherwise every buffer
            // in the address space matches the low-bit pattern.
            if (candidate < device_addr + size && candidate >= 0x1010000000ULL &&
                candidate < 0x1030000000ULL) {
                watched = candidate;
            }
        }
        if (watched != 0 && debug_watched_addr == 0) {
            // Many allocations have data at this offset, so confirm the object by its
            // signature: the first job record carries type id 54 at +0x28 and the counters
            // at +0x1C/+0x20/+0x24 are small.
            const u32* hdr = std::bit_cast<const u32*>(watched);
            if ((hdr[0x28 / 4] & 0xFFFF) == 54 && hdr[0x24 / 4] < 64 && hdr[0x20 / 4] < 64) {
                debug_watched_addr = watched;
                LOG_CRITICAL(Render_Vulkan, "JOBOBJ found at {:#x}", watched);
                // Watch the counter the consumer trusts: whoever bumps it to a value the
                // records do not back is the producer we are looking for.
                Core::ArmWriteWatch(watched + 0x24);
            }
        }
    }
    if (slow_obtain_us) {
        const auto so_end = SoClock::now();
        const u64 total = so_us(so_t0, so_end);
        if (total >= slow_obtain_us) {
            LOG_CRITICAL(Render_Vulkan,
                         "SLOWOBTAIN: cached addr={:#x} size={:#x} written={} texel={} "
                         "sync_limit={:#x} buf=[cpu={:#x} size={:#x} imp={} del={}] find={}us "
                         "sync={}us rest={}us total={}us",
                         device_addr, size, is_written, is_texel_buffer, sync_limit,
                         buffer.CpuAddr(), buffer.SizeBytes(), buffer.is_import_backed,
                         buffer.is_deleted, so_us(so_t0, so_t_find), so_us(so_t_find, so_t_sync),
                         so_us(so_t_sync, so_end), total);
        }
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    // [SLOWPKT] buf phase (image-side obtain; usually nests under the tex probe and is
    // attributed here, not there - exclusive accounting).
    Parserprof::PhaseProbe slow_probe{Parserprof::PhBuf};
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size) || IntersectsLabelImport(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::HasBufferAt(VAddr addr) {
    return static_cast<bool>(page_table[addr >> CACHING_PAGEBITS].buffer_id);
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Read from fault paths on guest threads; buffer_ranges mutates on the GPU
    // thread under the same lock.
    std::scoped_lock lk{buffers_mutex};
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    // NOTE: a one-entry lookup cache was tried here and REMOVED. It was both worthless and
    // wrong. Worthless because this function is already O(1) - one page-table index plus a
    // bounds check - so there was nothing to save. Wrong because the hit test
    //     device_addr >= begin && size <= end - device_addr
    // underflows for any address ABOVE the cached buffer: `end - device_addr` wraps to a huge
    // unsigned value, the size test passes, and the cache returns a buffer that does not cover
    // the address at all. Draws then got another buffer's device address, and shaders sat on
    // garbage pointers for 30 seconds. Do not reintroduce it without a range test that cannot
    // wrap - and even then, measure first: there is no cost here to recover.
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    // [HOMEPOLICY] walls: an import-backed policy owner that lies OUTSIDE the request is never
    // absorbed into a neighbour's union - the union would either be refused ('split' backing)
    // and eat the page into VRAM, or grow the import over the whole zone (liveloop75: 4.4 MB
    // listed -> 44 MB of import owners). The union stops at the owner's edge instead.
    const VAddr req_begin = device_addr;
    const VAddr req_end = end;
    const bool policy_walls = HomePolicy::Enabled();
    static u32 wall_logged = 0;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        if (policy_walls && overlap.is_import_backed) {
            const VAddr o_begin = overlap.CpuAddr();
            const VAddr o_end = o_begin + overlap.SizeBytes();
            const bool inside_request = o_end > req_begin && o_begin < req_end;
            if (!inside_request && HomePolicy::WantsImport(o_begin, overlap.SizeBytes())) {
                // Reached only by rightward growth (the left stream leap is disabled below), so
                // every picked buffer lies left of this owner: clipping `end` here is safe.
                if (o_begin >= req_end) {
                    end = std::min(end, o_begin);
                    if (wall_logged < 16) {
                        ++wall_logged;
                        LOG_CRITICAL(Render_Vulkan,
                                     "HOMEPOLICY: wall at {:#x} clips union {:#x}+{:#x} (request {:#x}+{:#x})",
                                     o_begin, begin, end - begin, req_begin, req_end - req_begin);
                    }
                    break;
                }
                continue; // owner left of the request (should not happen without a left leap)
            }
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            // [HOMEPOLICY] no leftward leap with walls on: a re-walk from the left could pick
            // buffers beyond a policy owner and the wall clip would then be unsafe.
            if (expands_left && !policy_walls) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    // COURT-MANDATED GUARD: an import-backed destination IS guest RAM. The old
    // VRAM snapshot must never be blitted over it wholesale - the snapshot
    // predates the freshest CPU writes (frame ticks live there), and clobbering
    // them is exactly the guest assert this whole design exists to fix. Migrate
    // only the islands the GPU actually owns: gpu-modified minus CPU-authority
    // minus CPU-dirty (the same rule the readback delivery uses).
    if (new_buffer.is_import_backed) {
        boost::container::small_vector<vk::BufferCopy, 8> copies;
        {
            std::scoped_lock gm{gpu_modified_mutex};
            gpu_modified_ranges.ForEachInRange(
                overlap.CpuAddr(), overlap.SizeBytes(),
                [&](VAddr start, VAddr end, const u64&) {
                    if (cpu_authority_ranges.Intersects(start, end - start)) {
                        return;
                    }
                    if (memory_tracker->IsRegionCpuModified(start, end - start)) {
                        return;
                    }
                    copies.push_back(vk::BufferCopy{
                        .srcOffset = start - overlap.CpuAddr(),
                        .dstOffset = start - new_buffer.CpuAddr(),
                        .size = end - start,
                    });
                });
        }
        if (!copies.empty()) {
            scheduler.EndRendering();
            const auto cmdbuf2 = scheduler.CommandBuffer();
            boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre2{};
            if (auto b1 = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                             vk::PipelineStageFlagBits2::eTransfer)) {
                pre2.push_back(*b1);
            }
            if (auto b2 = new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                                vk::PipelineStageFlagBits2::eTransfer,
                                                static_cast<u32>(dst_base_offset))) {
                pre2.push_back(*b2);
            }
            if (!pre2.empty()) {
                cmdbuf2.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = static_cast<u32>(pre2.size()),
                    .pBufferMemoryBarriers = pre2.data(),
                });
            }
            cmdbuf2.copyBuffer(overlap.Handle(), new_buffer.Handle(), copies);
        }
        DeleteBuffer(overlap_id);
        return;
    }
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    Postmortem::UsTimer create_timer{Postmortem::g_buf_create_us};
    Postmortem::g_buf_creates.fetch_add(1, std::memory_order_relaxed);
    VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    // [BACKIMP] Widen backing-eligible ranges to a coarse granule. The count of live buffers is
    // the real cost here (lookup structures, handles, GC pressure), and for a backing buffer a
    // wider window is free: the memory is shared with the guest either way, and import-backed
    // buffers neither upload nor download, so a coarser span carries no sync hazard. Widening
    // BEFORE ResolveOverlaps lets the existing overlap machinery fold neighbours into one
    // buffer instead of minting a new one per small range.
    // Default 0 = OFF: widening every range unconditionally was a regression (see below), so it
    // stays opt-in until measured on its own.
    static const u64 granule = [] {
        const char* e = std::getenv("SHADPS4_BACKIMP_GRANULE_KB");
        return e != nullptr ? std::strtoull(e, nullptr, 10) * 1024ull : 0ull;
    }();
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    OverlapResult overlap = [&] {
        Postmortem::UsTimer t{Postmortem::g_buf_overlap_us};
        return ResolveOverlaps(device_addr, wanted_size);
    }();
    // [BACKIMP] Widening is attempted ONLY on a true miss - when the narrow range already
    // overlaps nothing. Widening before the first resolve (the 15:31 build) meant a 16x wider
    // span caught neighbouring buffers, and every catch costs a JoinOverlap: a GPU copy plus
    // DeleteBuffer of the victim, plus a Register over 16x more pages, each page a BDA write
    // recorded into the frame's command buffer. That cascade is why frames collapsed from the
    // first seconds. Restricted to a miss, no join can be created that did not already exist.
    if (granule > CACHING_PAGESIZE && overlap.ids.empty() && !label_promote_intent) {
        auto& backimp = Vulkan::BackingImport::Get();
        if (backimp.IsAvailable() && backimp.ModeAllows(device_addr, wanted_size)) {
            const VAddr wide_begin = Common::AlignDown(device_addr, granule);
            const VAddr wide_end = Common::AlignUp(device_addr_end, granule);
            if (Core::Memory::Instance()
                    ->GetAddressSpace()
                    .GetBackingOffset(wide_begin, wide_end - wide_begin)
                    .has_value()) {
                Postmortem::UsTimer t{Postmortem::g_buf_overlap_us};
                const auto wide = ResolveOverlaps(wide_begin, static_cast<u32>(wide_end -
                                                                              wide_begin));
                // Still a miss after widening: nothing to join, so the wider window is free.
                if (wide.ids.empty()) {
                    device_addr = wide_begin;
                    device_addr_end = wide_end;
                    wanted_size = static_cast<u32>(wide_end - wide_begin);
                    overlap = wide;
                }
            }
        }
    }
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    // LABELUMA promotion intent: back the new owner with imported guest RAM.
    // Any failure falls through to the ordinary VMA path (and is remembered,
    // so a hopeless range is not retried every frame).
    BufferId new_buffer_id{};
    if (label_promote_intent && static_cast<u64>(size) <= label_budget &&
        !label_failed_ranges.Intersects(overlap.begin, size)) {
        if (auto imported =
                Buffer::CreateHostImport(instance, scheduler, overlap.begin, size, "label")) {
            new_buffer_id = slot_buffers.insert(std::move(*imported));
        } else {
            std::scoped_lock gm{gpu_modified_mutex};
            // The freed import of a just-demoted buffer lands only at tick
            // retire; give it time and retry before writing the range off.
            bool retried = false;
            for (auto& r : label_retries) {
                if (r.base == overlap.begin) {
                    retried = true;
                    break;
                }
            }
            if (!retried) {
                label_retries.push_back({overlap.begin, u64{size},
                                         label_promote_passes + 90, 0});
                LOG_CRITICAL(Render_Vulkan,
                             "LABELUMA: import failed for {:#x}:{:#x}, retry scheduled",
                             overlap.begin, size);
            }
        }
    }
    // [BACKIMP] Ordinary buffers: if this range is a plain window into the backing section and
    // the mode covers it, bind over the existing import instead of allocating VRAM. Skipped
    // when the LABELUMA branch above already tried (CreateHostImport prefers backing itself),
    // so a range is never attempted - or rejected in the log - twice per call.
    // [DUALHOME] a dual range always gets a VRAM owner; its second home is armed first so the
    // window entries precede the primary entries in command order.
    const bool dual_range = DualHomeEnabled() && DualHomeMode() < 3 && IsDualRange(overlap.begin, size);
    if (dual_range) {
        ArmDualRangesFor(overlap.begin, size);
    }
    // [HOMEPOLICY] a range touching a policy page gets guest memory as its single home: the
    // GPU's stores land where the CPU reads them, no readback and no journal replay in between.
    // A union with unlisted pages is correct but slower (counted); a refusal falls through to
    // the ordinary path (logged).
    if (!new_buffer_id && !label_promote_intent && !dual_range && HomePolicy::Enabled() &&
        HomePolicy::WantsImport(overlap.begin, size)) {
        const char* reason = "unavailable";
        static u32 refused_logged = 0;
        static u32 union_logged = 0;
        if (texture_cache.FindImageFromRange(overlap.begin, size, false)) {
            reason = "needs-staging";
        } else if (auto backed = Buffer::CreateBackingBacked(instance, scheduler, overlap.begin, size,
                                                             &reason, "policy", /*force_mode=*/true)) {
            new_buffer_id = slot_buffers.insert(std::move(*backed));
            ++g_policy_owners;
            g_policy_owner_bytes += size;
            const u64 covered = HomePolicy::CoveredBytes(overlap.begin, size);
            if (covered < size) {
                ++g_policy_unions;
                if (union_logged < 32) {
                    ++union_logged;
                    LOG_CRITICAL(Render_Vulkan,
                                 "HOMEPOLICY: union {:#x}+{:#x} forced import ({} KB listed, {} overlaps, "
                                 "request {:#x}+{:#x}, binder pgm={:#x})",
                                 overlap.begin, size, covered >> 10, overlap.ids.size(), device_addr,
                                 wanted_size, g_bind_pgm_hash);
                }
            }
        }
        // [HOMEPOLICY-DIRECT] The backing section is physically split under the per-frame list
        // arena (0x1018..0x101f), so every window into it is refused ("split") and the range
        // used to fall to a VRAM home - the GPU then read a copy the CPU was not writing
        // (LISTCOUNT: count sane on the host, home=1, TDR on the GPU). A policy page must keep
        // guest memory as its home: import the guest range by host pointer instead, which is
        // contiguous in virtual space. Only a staging conflict (an image over the range) stays
        // refused.
        if (!new_buffer_id && std::strcmp(reason, "needs-staging") != 0) {
            if (auto direct = Buffer::CreateHostImport(instance, scheduler, overlap.begin, size,
                                                       "policy-direct")) {
                new_buffer_id = slot_buffers.insert(std::move(*direct));
                ++g_policy_owners;
                g_policy_owner_bytes += size;
                static u32 direct_logged = 0;
                if (direct_logged < 32) {
                    ++direct_logged;
                    LOG_CRITICAL(Render_Vulkan,
                                 "HOMEPOLICY: direct host import {:#x}+{:#x} after backing refusal "
                                 "({}), binder pgm={:#x}",
                                 overlap.begin, size, reason, g_bind_pgm_hash);
                }
            } else {
                reason = "direct-failed";
            }
        }
        if (!new_buffer_id) {
            ++g_policy_refused;
            if (refused_logged < 32) {
                ++refused_logged;
                LOG_CRITICAL(Render_Vulkan, "HOMEPOLICY: import refused for {:#x}:{:#x} ({}), VRAM home",
                             overlap.begin, size, reason);
            }
        }
    }
    if (!new_buffer_id && !label_promote_intent && !dual_range) {
        const char* reason = "unavailable";
        auto& backimp = Vulkan::BackingImport::Get();
        // An image over the range keeps its own copy in VRAM and syncs it through the buffer;
        // handing the buffer guest RAM instead breaks that sync silently. LABELUMA refuses such
        // ranges for exactly this reason, and `all` mode meets them constantly, so it needs the
        // same guard - staging semantics the backing path cannot provide.
        const bool owned_by_image = backimp.IsAvailable() && backimp.ModeAllows(overlap.begin, size) &&
                                    texture_cache.FindImageFromRange(overlap.begin, size, false);
        if (owned_by_image) {
            backimp.NoteReject(overlap.begin, size, "needs-staging");
        } else if (auto backed = Buffer::CreateBackingBacked(instance, scheduler, overlap.begin,
                                                             size, &reason, "regular")) {
            new_buffer_id = slot_buffers.insert(std::move(*backed));
        } else if (backimp.IsAvailable() && std::strcmp(reason, "mode") != 0) {
            backimp.NoteReject(overlap.begin, size, reason);
        }
    }
    if (!new_buffer_id) {
        new_buffer_id =
            slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                                AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    }
    auto& new_buffer = slot_buffers[new_buffer_id];
    // JoinOverlap records its copies into the current command buffer; nothing in
    // this buffer is trustworthy until that tick retires.
    new_buffer.content_tick = scheduler.CurrentTick();
    {
        Postmortem::UsTimer t{Postmortem::g_buf_overlap_us};
        for (const BufferId overlap_id : overlap.ids) {
            JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
        }
    }
    {
        Postmortem::UsTimer t{Postmortem::g_buf_register_us};
        Register(new_buffer_id);
    }
    Postmortem::g_res_buffers.store(slot_buffers.size(), std::memory_order_relaxed);
    Parserprof::g_ws_total_bufs.fetch_add(1, std::memory_order_relaxed);
    Parserprof::g_ws_total_bytes.fetch_add(new_buffer.SizeBytes(), std::memory_order_relaxed);
    // [BYWRITE] Where did this buffer land? No-op outside bywrite mode.
    Vulkan::BackingImport::Get().NoteBufferPlacement(new_buffer.is_backing_backed,
                                                     new_buffer.SizeBytes());
    return new_buffer_id;
}

void BufferCache::SnaproRefreshStep() {
    if (!SnaproEnabled() || snap_shadows.empty()) {
        return;
    }
    const u64 frame = Vulkan::g_dbg_frame_no.load(std::memory_order_relaxed) + 1;
    // OnSubmit fires on EVERY submit, not once per frame. Copying there means EndRendering in
    // the middle of drawing, several times a frame - the render pass is cut and part of the
    // frame is lost. After a cutscene that showed up as a black screen. Refresh once per frame,
    // on the first submit after the frame counter moves.
    if (snap_last_refresh_frame == frame) {
        return;
    }
    snap_last_refresh_frame = frame;

    u64 bytes = 0;
    u32 copies = 0;
    u32 over_budget = 0;
    const u64 budget = SnaproBudgetBytes();

    // Decide the whole batch BEFORE touching the scheduler. EndRendering breaks the current
    // render pass, and calling it every frame even when there is nothing to copy stalled the
    // pipeline - in the menu that was audible as stuttering sound.
    boost::container::small_vector<std::pair<vk::Buffer, vk::Buffer>, 32> batch;
    boost::container::small_vector<u64, 32> batch_sizes;

    for (auto it = snap_shadows.begin(); it != snap_shadows.end();) {
        SnapShadow& sh = it->second;
        // Drop snapshots nothing has asked for in a while - they only cost VRAM.
        if (frame > sh.used_frame + 120) {
            it = snap_shadows.erase(it);
            continue;
        }
        // A copy only pays off when the range is read many times per frame: copy once, read
        // twenty times. Ranges bound once or twice cost more to copy than to read - that is
        // exactly the menu, where 6 huge buffers were copied (48 MB/frame) for almost no hits
        // and the audio started stuttering.
        // Stale mode: one copy per range for its whole life.
        if (SnaproStale() && sh.fresh_frame != 0) {
            ++it;
            continue;
        }
        if (sh.binds_prev < SnaproMinBinds()) {
            ++it;
            continue;
        }
        // Refresh only what the last couple of frames actually used. Refreshing everything
        // touched in the last two seconds cost 74 MB/frame instead of the ~22 MB measured.
        if (frame > sh.used_frame + 2) {
            ++it;
            continue;
        }
        const BufferId src_id = FindBuffer(it->first, static_cast<u32>(sh.size));
        Buffer& src = slot_buffers[src_id];
        // The range must still be the same guest buffer living in guest RAM. If it was recreated
        // with another size, or promoted out of the backing import, drop the snapshot.
        if (!src.is_import_backed || src.CpuAddr() != it->first || src.SizeBytes() != sh.size) {
            it = snap_shadows.erase(it);
            continue;
        }
        // Hard ceiling per frame. Without it the copy set grew to 180 MB/frame during a
        // cutscene and every thread stalled waiting on the GPU. Anything past the ceiling
        // simply stays unrefreshed and is read straight from guest memory, as before.
        if (bytes + sh.size > budget) {
            ++over_budget;
            ++it;
            continue;
        }
        batch.emplace_back(src.Handle(), sh.buf->Handle());
        batch_sizes.push_back(sh.size);
        sh.fresh_frame = frame;
        bytes += sh.size;
        ++copies;
        ++it;
    }

    if (copies != 0) {
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        for (size_t i = 0; i < batch.size(); ++i) {
            cmdbuf.copyBuffer(
                batch[i].first, batch[i].second,
                vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = batch_sizes[i]});
        }
        // One barrier for the whole batch: the copies must land before any shader reads them.
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
        });
    }
    snap_copied_bytes += bytes;
    snap_copies += copies;

    // Report every ~300 frames: MB per frame is the number the whole idea stands on.
    static u64 last_report = 0;
    if (frame >= last_report + 300) {
        const u64 span = frame - last_report;
        const u64 mb_per_frame = (snap_copied_bytes / span) >> 20;
        const u64 copies_per_frame = snap_copies / span;
        const u64 hits = snap_hits;
        const u64 misses = snap_misses;
        LOG_CRITICAL(Render,
                     "SNAPSTAT: {} MB/frame in {} copies/frame | live_ranges={} | "
                     "served_from_vram={} fell_back={} | over_budget={}",
                     mb_per_frame, copies_per_frame, snap_shadows.size(), hits, misses,
                     over_budget);
        snap_copied_bytes = 0;
        snap_copies = 0;
        snap_hits = 0;
        snap_misses = 0;
        last_report = frame;
    }
}

void BufferCache::BywriteCorrectionStep() {
    if (bywrite_requeue.empty()) {
        return;
    }
    // GPU thread only (Rasterizer::OnSubmit), the same point RunGarbageCollector runs from.
    // Nothing here happens inside a render pass or between a bind and its draw.
    //
    // Why demotion and not "mark the range CPU-dirty and let SynchronizeBuffer re-upload":
    // the trigger is a GPU write, not a CPU write. Uploading guest memory over a buffer the
    // shader has just written is a lost update - the exact hazard SynchronizeBuffer refuses to
    // risk for import-backed buffers, and the one JoinOverlap guards against. So the correction
    // has to run the other way round: pull the GPU-owned bytes down into guest memory, then let
    // the range be re-created as backing, where guest memory IS the buffer and no direction of
    // staleness exists any more.
    static const u64 max_fix_bytes = [] {
        const char* e = std::getenv("SHADPS4_BYWRITE_FIX_MAX_MB");
        const u64 mb = e ? std::strtoull(e, nullptr, 10) : 0;
        return (mb ? mb : 256ull) << 20;
    }();
    static const bool fix_enabled = [] {
        // Escape hatch only: the correction is part of the bywrite mode, which is itself opt-in.
        const char* e = std::getenv("SHADPS4_BYWRITE_FIX");
        return !(e && e[0] == '0');
    }();
    if (!fix_enabled) {
        bywrite_requeue.clear();
        return;
    }
    constexpr u32 kMaxPerPass = 4; // each fix stalls on a synchronous download
    const auto t0 = std::chrono::steady_clock::now();
    u32 done = 0;
    size_t cursor = 0;
    for (; cursor < bywrite_requeue.size() && done < kMaxPerPass; ++cursor) {
        const auto [addr, size] = bywrite_requeue[cursor];
        // Resolve the owner now instead of trusting a stored BufferId: slot indices carry no
        // generation counter, so an id parked across a frame can silently name a different
        // buffer. The page table is always current.
        const BufferId owner = page_table[addr >> CACHING_PAGEBITS].buffer_id;
        if (!owner || slot_buffers[owner].is_deleted) {
            continue;
        }
        Buffer& buffer = slot_buffers[owner];
        buffer.bywrite_queued = false;
        if (buffer.is_import_backed || buffer.CpuAddr() != addr || buffer.SizeBytes() != size) {
            // Already in guest memory, or the range was re-carved meanwhile - nothing to do.
            continue;
        }
        // Bounded ratchet, same shape as the LABELUMA promotion ratchet: if CreateBackingBacked
        // keeps refusing this range for a structural reason (owned by an image, cross-chunk,
        // non-zero delta), demoting it again every frame would be an expensive infinite loop.
        u32& tries = bywrite_fix_counts[addr >> 20];
        if (tries >= 4) {
            ++bywrite_gaveup;
            continue;
        }
        if (size > max_fix_bytes) {
            ++bywrite_skipped_big;
            continue;
        }
        ++tries;
        {
            // Exactly the demotion recipe the garbage collector uses, in its order: the download
            // must be synchronous, because marking a region CPU-modified while its GPU-modified
            // flag is still set asks the page manager for write-only permission, which does not
            // exist and fires an assertion.
            std::scoped_lock serve_lk{serve_mutex};
            DownloadBufferMemory<false>(buffer, buffer.CpuAddr(), buffer.SizeBytes());
            memory_tracker->MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
            DeleteBuffer(owner);
        }
        ++done;
        ++bywrite_fixed;
    }
    bywrite_requeue.erase(bywrite_requeue.begin(),
                          bywrite_requeue.begin() + static_cast<std::ptrdiff_t>(cursor));
    if (done != 0) {
        bywrite_fix_us += static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0)
                .count());
        // Read every counter into a local before the log call.
        const u64 f = bywrite_fixed;
        const u64 q = bywrite_requeue.size();
        const u64 big = bywrite_skipped_big;
        const u64 gave = bywrite_gaveup;
        const u64 us = bywrite_fix_us;
        if (f <= 64 || (f & 63) == 0) {
            LOG_CRITICAL(Render_Vulkan,
                         "BYWRITE: fixed={} queued={} skipped_big={} gaveup={} total_us={}", f, q,
                         big, gave, us);
        }
    }
}

// [VRAMPREFETCH] Give the listed guest ranges VRAM buffers (page-table entries) before the GPU
// touches them. A raw-address write to a page nobody cached yet is dropped by the shader, and
// the job heap (0x1101...) takes ~330k such writes at the cutscene->deck transition (GPUFAULT/
// DROPLOG, 03.09) - the slots keep the game's 0x7FFFFFFF filler and the validator asserts.
// Importing the range instead (BACKIMP_PREFETCH) keeps the writes but puts the worker's hot
// data behind PCIe (disp 16 -> 43 ms/frame). VRAM keeps it fast; readbacks carry results back.
static u64 VramPrefetchAutoBytes() {
    static const u64 v = [] {
        const char* e = std::getenv("SHADPS4_VRAM_PREFETCH_AUTO");
        return e ? static_cast<u64>(std::strtoull(e, nullptr, 10)) << 20 : u64{0};
    }();
    return v;
}

void BufferCache::NoteGpuFaultPage(VAddr page) {
    const u64 block = VramPrefetchAutoBytes();
    if (block == 0) {
        return;
    }
    const VAddr base = page & ~(block - 1);
    std::scoped_lock lk{vram_auto_mutex};
    if (std::find(vram_auto_blocks.begin(), vram_auto_blocks.end(), base) != vram_auto_blocks.end() ||
        std::find(vram_auto_done_blocks.begin(), vram_auto_done_blocks.end(), base) !=
            vram_auto_done_blocks.end()) {
        return;
    }
    if (vram_auto_blocks.size() < 256) {
        vram_auto_blocks.push_back(base);
    }
}

void BufferCache::NoteGpuWrittenPage(VAddr page) {
    {
        std::scoped_lock lk{buffers_mutex};
        const BufferId owner = page_table[page >> CACHING_PAGEBITS].buffer_id;
        if (owner) {
            const Buffer& b = slot_buffers[owner];
            if (!b.is_deleted && b.is_import_backed) {
                return; // the store already landed in guest memory
            }
        }
    }
    // [HOMEPOLICY-PT] the entry pointed at guest RAM: the store landed there and the CPU sees it
    // now. The VRAM owner (if any) holds the stale copy - flag the page CPU-modified so the next
    // bind uploads guest RAM over it. Protecting it as GPU-written would do the opposite: the
    // CPU's next read would pull the stale VRAM copy over the fresh store.
    if (HomePolicy::WantsPt(page, CACHING_PAGESIZE) && PolicyPtBda(page) != 0) {
        // Order matters: a page still flagged GPU-modified plus CPU-modified computes to a
        // write-only protection, which PageManager::Protect asserts on (liveloop141). Clear the
        // GPU flag first, exactly as the fast-path import heal does.
        memory_tracker->UnmarkRegionAsGpuModified(page, CACHING_PAGESIZE);
        memory_tracker->MarkRegionAsCpuModified(page, CACHING_PAGESIZE);
        static std::atomic<u64> pt_gpu_writes{0};
        const u64 n = pt_gpu_writes.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 4 || n % 65536 == 0) {
            LOG_CRITICAL(Render_Vulkan,
                         "HOMEPOLICY-PT: GPU wrote listed page {:#x} in guest RAM, VRAM copy flagged "
                         "for upload ({} so far)",
                         page, n);
        }
        return;
    }
    MarkGpuWrittenRegion(page, CACHING_PAGESIZE);
    static const bool sticky = [] {
        const char* e = std::getenv("SHADPS4_PAGEJOURNAL");
        return e && e[0] == '2';
    }();
    if (sticky) {
        std::scoped_lock lk{pagej_mutex};
        if (pagej_new.size() < 65536) {
            pagej_new.push_back(page);
        }
    }
}

void BufferCache::PageJournalStickyStep() {
    static const bool sticky = [] {
        const char* e = std::getenv("SHADPS4_PAGEJOURNAL");
        return e && e[0] == '2';
    }();
    if (!sticky) {
        return;
    }
    {
        std::scoped_lock lk{pagej_mutex};
        for (const VAddr p : pagej_new) {
            if (std::find(pagej_sticky.begin(), pagej_sticky.end(), p) == pagej_sticky.end() &&
                pagej_sticky.size() < 16384) {
                pagej_sticky.push_back(p);
            }
        }
        pagej_new.clear();
    }
    // GPU thread, submit boundary: arm protection for the whole working set before the GPU
    // touches it again. Import-backed owners are skipped inside (stores land in guest RAM).
    u32 marked = 0;
    for (const VAddr p : pagej_sticky) {
        BufferId owner{};
        {
            std::scoped_lock lk{buffers_mutex};
            owner = page_table[p >> CACHING_PAGEBITS].buffer_id;
            if (owner) {
                const Buffer& b = slot_buffers[owner];
                if (!b.is_deleted && b.is_import_backed) {
                    continue;
                }
            }
        }
        // [HOMEPOLICY-PT] guest RAM is the home: protecting would readback the stale VRAM copy.
        if (HomePolicy::WantsPt(p, CACHING_PAGESIZE) && PolicyPtBda(p) != 0) {
            continue;
        }
        MarkGpuWrittenRegion(p, CACHING_PAGESIZE);
        ++marked;
    }
    pagej_sticky_marks += marked;
    static auto next = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now >= next) {
        next = now + std::chrono::seconds(5);
        LOG_CRITICAL(Render_Vulkan, "PAGEJ-STICKY: {} pages in set, {} marks since last report",
                     pagej_sticky.size(), pagej_sticky_marks);
        pagej_sticky_marks = 0;
    }
}

// ---------------------------------------------------------------------------------------------
// [DUALHOME]
bool BufferCache::DualHomeEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DUALHOME");
        return e && e[0] != '\0';
    }();
    return on;
}

// 1 = shader dual store, 2 = writeback kernel + forced VRAM owners, 3 = writeback kernel only
// (owners stay whatever the ordinary path makes them: VRAM after first touch, rescue imports for
// polled pages); pieces are created in force mode without admitting the range to ModeAllows.
static int DualHomeMode() {
    static const int v = [] {
        const char* e = std::getenv("SHADPS4_DUALHOME_MODE");
        return e ? std::atoi(e) : 2;
    }();
    return v;
}
// SHADPS4_DUALHOME_MARKS=1: keep GPU-modified marks (readback protection) for dual pages too.
// Test of the theory that skipping them lets the CPU read stale results without a fault and
// the game then feeds the GPU garbage (2x dispatch time, hangs) - liveloop63 vs 58/61/62.
static bool DualKeepMarks() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DUALHOME_MARKS");
        return e && e[0] == '1';
    }();
    return on;
}

bool BufferCache::IsDualRange(VAddr addr, u64 size) const {
    for (const auto& [b, s] : dual_ranges) {
        if (addr < b + s && b < addr + size) {
            return true;
        }
    }
    return false;
}

const BufferCache::DualPiece* BufferCache::DualPieceFor(VAddr addr) const {
    for (const auto& p : dual_pieces) {
        if (addr >= p->base && addr < p->base + p->size) {
            return p.get();
        }
    }
    return nullptr;
}

void BufferCache::PublishDualEntries(const DualPiece& piece) {
    const u64 first_page = piece.base >> CACHING_PAGEBITS;
    const u64 num_pages = piece.size >> CACHING_PAGEBITS;
    std::vector<vk::DeviceAddress> entries(num_pages);
    const vk::DeviceAddress bda = piece.buffer->BufferDeviceAddress();
    for (u64 i = 0; i < num_pages; ++i) {
        entries[i] = bda + (i << CACHING_PAGEBITS);
    }
    const u64 offset = BDA_PAGETABLE_SIZE + (first_page - DUAL_WINDOW_BASE_PAGE) * sizeof(vk::DeviceAddress);
    if (!bda_pagetable_buffer.mapped_data.empty()) {
        std::memcpy(bda_pagetable_buffer.mapped_data.data() + offset, entries.data(),
                    num_pages * sizeof(vk::DeviceAddress));
        _mm_sfence();
    } else {
        WriteDataBuffer(bda_pagetable_buffer, offset, entries.data(),
                        static_cast<u32>(num_pages * sizeof(vk::DeviceAddress)));
    }
}

// Arm the intersection of [addr, addr+size) with the dual ranges: one import piece per
// contiguous backing window (largest the backing accepts, halving on 'split'). GPU thread.
void BufferCache::ArmDualRangesFor(VAddr addr, u64 size) {
    auto& backimp = Vulkan::BackingImport::Get();
    if (!backimp.IsAvailable()) {
        return;
    }
    const auto& aspace = Core::Memory::Instance()->GetAddressSpace();
    constexpr u64 kMaxChunk = 8_MB;
    for (const auto& [rb, rs] : dual_ranges) {
        const VAddr lo = std::max<VAddr>(addr, rb);
        const VAddr hi = std::min<VAddr>(addr + size, rb + rs);
        if (lo >= hi) {
            continue;
        }
        if (DualHomeMode() < 3) {
            backimp.AllowRange(rb, rs);
        }
        VAddr pos = Common::AlignDown(lo, CACHING_PAGESIZE);
        const VAddr end = Common::AlignUp(hi, CACHING_PAGESIZE);
        while (pos < end) {
            if (const DualPiece* have = DualPieceFor(pos)) {
                pos = have->base + have->size;
                continue;
            }
            u64 sz = Common::AlignDown(std::min<u64>(kMaxChunk, end - pos), CACHING_PAGESIZE);
            // clip at the next existing piece
            for (const auto& p : dual_pieces) {
                if (p->base > pos && p->base < pos + sz) {
                    sz = p->base - pos;
                }
            }
            std::optional<u64> off;
            bool ineligible = false;
            while (sz >= CACHING_PAGESIZE) {
                const char* why = nullptr;
                off = aspace.GetBackingOffset(pos, sz, &why);
                if (off) {
                    if (backimp.CoversContiguously(*off, sz)) {
                        break;
                    }
                    off.reset();
                } else if (why == nullptr || std::strcmp(why, "split") != 0) {
                    ineligible = true;
                    break;
                }
                sz = Common::AlignDown(sz / 2, CACHING_PAGESIZE);
            }
            if (!off) {
                if (ineligible) {
                    static u32 logged = 0;
                    if (logged < 16) {
                        ++logged;
                        LOG_CRITICAL(Render_Vulkan, "DUALHOME: {:#x} not armable yet (unmapped/ineligible)", pos);
                    }
                }
                break; // retried by DualArmStep
            }
            const char* reason = "";
            auto backed = Buffer::CreateBackingBacked(instance, scheduler, pos, sz, &reason, "dual",
                                                      DualHomeMode() >= 3);
            if (!backed) {
                LOG_CRITICAL(Render_Vulkan, "DUALHOME: piece {:#x}:{:#x} refused ({})", pos, sz, reason);
                break;
            }
            auto piece = std::make_unique<DualPiece>();
            piece->base = pos;
            piece->size = sz;
            piece->buffer = std::make_unique<Buffer>(std::move(*backed));
            PublishDualEntries(*piece);
            dual_armed_bytes += sz;
            LOG_CRITICAL(Render_Vulkan, "DUALHOME: armed {:#x}:{:#x} bda={:#x} pieces={} total={} MB",
                         pos, sz, static_cast<u64>(piece->buffer->BufferDeviceAddress()),
                         dual_pieces.size() + 1, dual_armed_bytes >> 20);
            dual_pieces.push_back(std::move(piece));
            pos += sz;
        }
    }
}

void BufferCache::DualArmStep() {
    if (!DualHomeEnabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (dual_next == std::chrono::steady_clock::time_point{} || now >= dual_next) {
        dual_next = now + std::chrono::seconds(3);
        u64 want = 0;
        for (const auto& [b, s] : dual_ranges) {
            want += s;
            ArmDualRangesFor(b, s);
        }
        if (dual_report_next == std::chrono::steady_clock::time_point{} || now >= dual_report_next) {
            dual_report_next = now + std::chrono::seconds(5);
            LOG_CRITICAL(Render_Vulkan,
                         "DUALSTAT: armed {} of {} MB in {} pieces | piece copies={} staging bytes={}",
                         dual_armed_bytes >> 20, want >> 20, dual_pieces.size(), dual_piece_copies,
                         dual_staging_bytes);
        }
    }
}

void BufferCache::PageTraceStep() {
    if (!PageTrace::Enabled()) {
        return;
    }
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    static u32 window_no = 0;
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds(5);
    ++window_no;
    using namespace PageTrace;
    struct Row {
        u32 slot;
        std::array<u32, NUM_FIELDS> c;
        char cls;
        char home;
    };
    std::vector<Row> rows;
    std::array<u32, 8> cls_count{}; // P M R S I U F (0..6)
    std::array<u64, NUM_FIELDS> sums{};
    const auto cls_index = [](char c) -> u32 {
        switch (c) {
        case 'P': return 0;
        case 'M': return 1;
        case 'R': return 2;
        case 'S': return 3;
        case 'I': return 4;
        case 'F': return 6;
        default: return 5;
        }
    };
    const u32 stalls = g_stalls.exchange(0, std::memory_order_relaxed);
    u32 spb_pages = 0;
    u32 pj_pages = 0; // [POLLJ] pages a poll loop read this window (cached ones included)
    for (u32 slot = 0; slot < g_pages; ++slot) {
        Row row{slot, {}, 'U', '-'};
        bool any = false;
        for (u32 f = 0; f < NUM_FIELDS; ++f) {
            const u32 v = g_counters[static_cast<size_t>(slot) * NUM_FIELDS + f].exchange(0, std::memory_order_relaxed);
            row.c[f] = v;
            g_total[static_cast<size_t>(slot) * NUM_FIELDS + f] += v;
            sums[f] += v;
            any = any || v != 0;
        }
        const auto& c = row.c;
        // Snapshot A is the whole bitmap since the parser's last clear (first-touch evidence);
        // only a page re-faulting within 40 ms of the clear (B) is the one the GPU spins on.
        // [POLLJ] PJ is the shader's own word: the page was read from a poll loop, whatever
        // its home - the class the bitmap snapshots could never give a cached page.
        if (c[SPB] || c[PJ]) {
            row.cls = 'P';
            spb_pages += c[SPB] != 0;
            pj_pages += c[PJ] != 0;
        } else if (c[GW] && (c[CR] + c[CRS]) >= 4 * c[GW]) {
            row.cls = 'F'; // CPU polls a flag the GPU writes
        } else if (c[CR] || c[CRS]) {
            // CPU read fault on a GPU-marked page: the mark may come from a written bind
            // (no page-journal entry, gw=0) - liveloop93: one such page cost ~12 ms per frame.
            row.cls = 'R';
        } else if (c[GW] && (c[UP] || c[CW])) {
            row.cls = 'M';
        } else if (c[GW]) {
            row.cls = 'S';
        } else if (c[UP] || c[CW]) {
            row.cls = 'I';
        }
        ++cls_count[cls_index(row.cls)];
        if (row.cls != 'U') {
            g_classes_seen[slot] |= static_cast<u8>(1u << cls_index(row.cls));
        }
        if (!any) {
            continue;
        }
        const VAddr addr = PageAddr(slot);
        {
            std::scoped_lock lk{buffers_mutex};
            const BufferId owner = page_table[addr >> CACHING_PAGEBITS].buffer_id;
            if (owner) {
                const Buffer& b = slot_buffers[owner];
                row.home = b.is_deleted ? 'd' : (b.is_import_backed ? 'I' : 'V');
            }
        }
        if (row.home == '-' && RescueBdaFor(addr) != 0) {
            row.home = 'R';
        }
        rows.push_back(row);
    }
    LOG_CRITICAL(Render_Vulkan,
                 "PAGETRACE w{}: P={} F={} M={} R={} S={} I={} U={} active={} | ft={} gw={} dl={} cw={} "
                 "cr={} crs={} cra={} up={} spa={} spb={} pj={} stalls={} spb_pages={} pj_pages={}",
                 window_no, cls_count[0], cls_count[6], cls_count[1], cls_count[2], cls_count[3],
                 cls_count[4], cls_count[5], rows.size(), sums[FT], sums[GW], sums[DL], sums[CW],
                 sums[CR], sums[CRS], sums[CRA], sums[UP], sums[SPA], sums[SPB], sums[PJ], stalls,
                 spb_pages, pj_pages);
    if (stalls != 0 && spb_pages == 0 && pj_pages == 0) {
        LOG_CRITICAL(Render_Vulkan,
                     "PAGETRACE: {} stall(s) this window with no re-faulting traced page - the polled "
                     "page is CACHED or outside the traced ranges (POLLJ needed)",
                     stalls);
    }
    const auto top = [&](const char* tag, Field key) {
        std::vector<const Row*> v;
        for (const auto& r : rows) {
            if (r.c[key]) {
                v.push_back(&r);
            }
        }
        std::sort(v.begin(), v.end(), [key](const Row* a, const Row* b) { return a->c[key] > b->c[key]; });
        std::string line;
        for (size_t i = 0; i < v.size() && i < 8; ++i) {
            const Row& r = *v[i];
            fmt::format_to(std::back_inserter(line),
                           "{:#x}[{}{}] gw={} cr={} crs={} cra={} up={} cw={} ft={} dl={} sp={}/{} pj={} | ",
                           PageAddr(r.slot), r.cls, r.home, r.c[GW], r.c[CR], r.c[CRS], r.c[CRA],
                           r.c[UP], r.c[CW], r.c[FT], r.c[DL], r.c[SPA], r.c[SPB], r.c[PJ]);
        }
        if (!line.empty()) {
            LOG_CRITICAL(Render_Vulkan, "PAGETRACE-TOP {}: {}", tag, line);
        }
    };
    top("gw", GW);
    top("cr", CR);
    top("crs", CRS);
    top("cra", CRA);
    top("up", UP);
    top("ft", FT);
    top("sp", SPB);
    top("pj", PJ);
    // one row per active page into a file the stand can pull
    static const std::string path = [] {
        const char* e = std::getenv("SHADPS4_PAGETRACE_FILE");
        return std::string{e && e[0] ? e : "C:\\Users\\user\\ctl\\pagetrace.txt"};
    }();
    std::ofstream out(path, std::ios::app);
    if (out) {
        for (const auto& r : rows) {
            out << window_no << ' ' << fmt::format("{:#x}", PageAddr(r.slot)) << ' ' << r.cls << ' ' << r.home;
            for (u32 f = 0; f < NUM_FIELDS; ++f) {
                out << ' ' << r.c[f];
            }
            out << '\n';
        }
    }
    // Crash-safe cumulative map, rewritten every window: runs die in 0.7-4 min, so an exit-time
    // dump is unreliable. Row: addr classes_seen ft gw dl cw cr crs up spa spb cra pj (cumulative).
    std::ofstream map(path + ".map", std::ios::trunc);
    if (map) {
        static constexpr char kClassChars[] = "PMRSIUF";
        for (u32 slot = 0; slot < g_pages; ++slot) {
            const u32* t = &g_total[static_cast<size_t>(slot) * NUM_FIELDS];
            bool any = false;
            for (u32 f = 0; f < NUM_FIELDS; ++f) {
                any = any || t[f] != 0;
            }
            if (!any) {
                continue;
            }
            std::string seen;
            for (u32 i = 0; i < 7; ++i) {
                if (g_classes_seen[slot] & (1u << i)) {
                    seen.push_back(kClassChars[i]);
                }
            }
            map << fmt::format("{:#x}", PageAddr(slot)) << ' ' << (seen.empty() ? "-" : seen);
            for (u32 f = 0; f < NUM_FIELDS; ++f) {
                map << ' ' << t[f];
            }
            map << '\n';
        }
    }
}

void BufferCache::PolicyPrefetchStep() {
    if (!HomePolicy::Enabled()) {
        return;
    }
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    static auto next_summary = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_summary) {
        next_summary = now + std::chrono::seconds(5);
        LOG_CRITICAL(Render_Vulkan,
                     "HOMEPOLICY: owners={} ({} KB) prearmed={} ({} KB) unions={} refused={}",
                     g_policy_owners.load(), g_policy_owner_bytes.load() >> 10, g_policy_prearmed.load(),
                     g_policy_prearmed_bytes.load() >> 10, g_policy_unions.load(),
                     g_policy_refused.load());
    }
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds(3);
    // GPU thread (Rasterizer::OnSubmit): the sole page_table writer, so the owner checks below
    // need no buffers_mutex; CreateBuffer is called exactly the way ObtainBuffer calls it.
    const auto& aspace = Core::Memory::Instance()->GetAddressSpace();
    auto& backimp = Vulkan::BackingImport::Get();
    constexpr u64 kMaxRun = 256_KB;
    constexpr u32 kMaxPerPass = 32;
    u32 created = 0;
    u64 created_bytes = 0;
    static u32 skipped_logged = 0;
    const auto owned = [&](VAddr page) { return page_table[page >> CACHING_PAGEBITS].buffer_id.operator bool(); };
    for (const auto& [base, range_size] : HomePolicy::Ranges()) {
        VAddr pos = base;
        const VAddr end = base + range_size;
        while (pos < end && created < kMaxPerPass) {
            if (owned(pos) || !IsFaultRegionValid(pos, CACHING_PAGESIZE)) {
                pos += CACHING_PAGESIZE;
                continue;
            }
            VAddr run_end = pos + CACHING_PAGESIZE;
            while (run_end < end && run_end - pos < kMaxRun && !owned(run_end) &&
                   IsFaultRegionValid(run_end, CACHING_PAGESIZE)) {
                run_end += CACHING_PAGESIZE;
            }
            u64 sz = run_end - pos;
            std::optional<u64> off;
            bool ineligible = false;
            while (sz >= CACHING_PAGESIZE) {
                const char* why = nullptr;
                off = aspace.GetBackingOffset(pos, sz, &why);
                if (off) {
                    if (backimp.CoversContiguously(*off, sz)) {
                        break;
                    }
                    off.reset();
                } else if (why == nullptr || std::strcmp(why, "split") != 0) {
                    ineligible = true;
                    break;
                }
                sz = Common::AlignDown(sz / 2, CACHING_PAGESIZE);
            }
            if (!off) {
                if (skipped_logged < 16) {
                    ++skipped_logged;
                    LOG_CRITICAL(Render_Vulkan, "HOMEPOLICY: {:#x}:{:#x} not armable ({})", pos,
                                 run_end - pos, ineligible ? "ineligible" : "no contiguous backing");
                }
                pos = run_end;
                continue;
            }
            const BufferId id = CreateBuffer(pos, static_cast<u32>(sz));
            if (id && slot_buffers[id].is_import_backed) {
                ++created;
                created_bytes += sz;
            }
            pos += sz;
        }
    }
    if (created != 0) {
        g_policy_prearmed += created;
        g_policy_prearmed_bytes += created_bytes;
        LOG_CRITICAL(Render_Vulkan, "HOMEPOLICY: prearmed {} runs ({} KB) this pass, total {} ({} KB)",
                     created, created_bytes >> 10, g_policy_prearmed.load(),
                     g_policy_prearmed_bytes.load() >> 10);
    }
}

// [LIVEREPLAY] SHADPS4_LIVEREPLAY_SKIP=1: page-table entries that resolve into an import of the
// backing mirror (= guest memory) carry bit 63, so write_const_live can skip the raw-address
// write log for them (the store already landed where the CPU reads) and the host replay skips
// them too. get_bda_pointer masks the bit off before dereferencing.
static bool LiveReplaySkipEnabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("SHADPS4_LIVEREPLAY_SKIP");
        return e && e[0] == '1';
    }();
    return enabled;
}
static constexpr vk::DeviceAddress kBdaLandedBit = vk::DeviceAddress{1} << 63;

// [POLLMAP] GPU thread (Rasterizer::OnSubmit -> BackimpPrefetchStep), once per submit. The
// bitmap is host-coherent, so the fold is a plain read of mapped memory - no copy, no batch, no
// priority-thread hop: the 8 KiB summary is scanned every pass, the 32 words behind each set
// summary bit are read, and every page not yet in the sticky set is inserted and queued. The
// drain then rewrites the queued pages' table entries on this very thread - the only one
// allowed to write page_table and the device table (same rule as PolicyPrefetchStep). Doing
// both here (instead of a fold on the priority thread plus a cross-thread queue) costs one 8 KiB
// scan per submit and nothing else, and needs no locking at all.
// Races with the GPU are all benign: the fine bits are never cleared by the host, so a mark
// whose summary bit the host's clear ate (or whose two atomics became visible out of order) is
// picked up by the incremental sweep of the fine bitmap (a slice per pass, a circle per 2 s).
// With PTHOST the table is host-mapped and the priority thread stores into it too (the deferred
// register publish, ChangeRegister): those entries are re-derived at publish time, and the
// re-apply at the end stores every redirected entry again in slices, so a store that slipped
// through cannot undo a redirect for more than a few passes.
void BufferCache::PollMapStep() {
    if (pollmap_host == nullptr) {
        return;
    }
    static const bool pollmap_all = [] {
        const char* e = std::getenv("SHADPS4_POLLMAP_ALL");
        return e && e[0] == '1';
    }();
    static u64 pollmap_gwskipped = 0;
    static u64 pollmap_notcpu = 0;
    static u64 pollmap_staticpt = 0;
    const auto now = std::chrono::steady_clock::now();
    ++pollmap_folds;
    const u32* const fine = reinterpret_cast<const u32*>(pollmap_host);
    u32* const summary = reinterpret_cast<u32*>(pollmap_host + POLLMAP_FINE_BYTES);
    const auto load_mapped = [](const u32* p) { return *static_cast<const volatile u32*>(p); };
    constexpr size_t kReportPages = 48; // newly marked pages named in the 5 s report
    u32 fresh = 0;
    const auto scan_region = [&](u64 region) {
        const u64 first_word = region * POLLMAP_REGION_WORDS;
        for (u64 w = 0; w < POLLMAP_REGION_WORDS; ++w) {
            u32 bits = load_mapped(fine + first_word + w);
            while (bits != 0) {
                const u32 b = static_cast<u32>(std::countr_zero(bits));
                bits &= bits - 1;
                const u64 page = (first_word + w) * 32 + b;
                // [POLLMAP-GW] 06.09 liveloop162: redirecting every polled page (766 on the
                // deck against the 135 of the curated list) put pages the GPU also writes
                // through bounded descriptors onto guest RAM - live loads then read the CPU's
                // bytes while the GPU's results stay in the VRAM copy, the worker follows a
                // garbage pointer and the GPU faults (ReadInvalid 0x4a542b000). A page that is
                // GPU-modified at fold time is left out of the set; the 2 s sweep re-evaluates
                // it, so a page that is only transiently dirty still gets its redirect.
                // v1.2 (liveloop165): the transient GPU-modified mark is cleared by downloads
                // long before the fold sees it (gwskipped stayed 0 and the deck still hung), so
                // the persistent per-buffer history is used: a page whose owner buffer has ever
                // been bound as written by the GPU (WRITESTAT write_first_frame) is skipped.
                if (!pollmap_all && !CpuWrittenEver::Has(page)) {
                    ++pollmap_notcpu;
                    continue; // GPU-polled but never CPU-written: not a flag/queue page
                }
                // [POLLMAP v3.2] pages under the static page-table list (the per-frame list
                // arena, 0x1005..0x1026) are already served by the PT redirect; importing their
                // owners is refused as "split" and falls to a direct host import that the GPU then
                // faults on (185: ReadInvalid at load). Leave them to the static policy.
                if (!pollmap_all && HomePolicy::StaticWantsPt(page << CACHING_PAGEBITS, CACHING_PAGESIZE)) {
                    ++pollmap_staticpt;
                    continue;
                }
                bool gpu_writer = false;
                if (!pollmap_all && HomePolicy::PollMapPtMode()) {
                    const BufferId owner = page_table[page].buffer_id;
                    gpu_writer = (owner && slot_buffers[owner].write_first_frame != 0) ||
                                 memory_tracker->IsRegionGpuModified(page << CACHING_PAGEBITS,
                                                                     CACHING_PAGESIZE);
                }
                if (gpu_writer) {
                    ++pollmap_gwskipped;
                    if (pollmap_gwskipped <= 16) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "POLLMAP: page {:#x} is GPU-written - not redirected (#{})",
                                     page << CACHING_PAGEBITS, pollmap_gwskipped);
                    }
                    continue;
                }
                if (PollMap::Insert(page)) {
                    pollmap_pending.push_back({page, 0});
                    ++fresh;
                    if (pollmap_report_pages.size() < kReportPages) {
                        pollmap_report_pages.push_back(page << CACHING_PAGEBITS);
                    }
                }
            }
        }
    };
    for (u64 sw = 0; sw < POLLMAP_SUMMARY_WORDS; ++sw) {
        const u32 bits_now = load_mapped(summary + sw);
        if (bits_now == 0) {
            continue;
        }
        // Clear BEFORE reading the regions: a mark that lands during the scan sets the bit
        // again and the next pass rereads the region. A clear that races the GPU's atomic can
        // only lose in the safe direction (bit stays set -> one more scan; bit lost -> sweep).
        *static_cast<volatile u32*>(summary + sw) = 0;
        u32 bits = bits_now;
        while (bits != 0) {
            const u32 b = static_cast<u32>(std::countr_zero(bits));
            bits &= bits - 1;
            scan_region(sw * 32 + b);
        }
    }
    // Incremental sweep: a circle starts once the 2 s pause is over and then advances one slice
    // per pass until it wraps. CPU time is accounted per slice - the wall time of a circle spans
    // many submits and says nothing about the cost.
    if (pollmap_sweep_cursor != 0 || now >= pollmap_sweep_next) {
        if (pollmap_sweep_cursor == 0) {
            pollmap_sweep_started = now;
            pollmap_sweep_work_us = 0;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const u64 end =
            std::min<u64>(pollmap_sweep_cursor + pollmap_sweep_regions_per_pass, POLLMAP_REGIONS);
        for (u64 region = pollmap_sweep_cursor; region < end; ++region) {
            scan_region(region);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const u64 slice_us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        pollmap_sweep_work_us += slice_us;
        pollmap_slice_max_us = std::max(pollmap_slice_max_us, slice_us);
        pollmap_sweep_cursor = end;
        if (pollmap_sweep_cursor >= POLLMAP_REGIONS) {
            pollmap_sweep_cursor = 0;
            pollmap_sweep_us = pollmap_sweep_work_us;
            pollmap_sweep_span_ms = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - pollmap_sweep_started)
                    .count());
            ++pollmap_sweeps;
            pollmap_sweep_next = t1 + std::chrono::seconds(2);
        }
    }
    pollmap_marked += fresh;

    // Drain: the entry can only be rewritten once the page's backing chunk is imported and has
    // a whole-chunk buffer (PolicyPtBda -> BackingImport::PageBda imports the chunk on demand,
    // exactly the path the static list takes). A page with no backing address stays queued and
    // is retried next pass; after kMaxAttempts passes it is given up on (a stray pointer the
    // poll read, or a chunk whose import the driver refused) but stays in the sticky set, so a
    // later register of a buffer over it still applies the redirect.
    if (!pollmap_pending.empty()) {
        auto& backimp = Vulkan::BackingImport::Get();
        if (!backimp.IsAvailable()) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_CRITICAL(Render_Vulkan,
                             "POLLMAP: backing import unavailable (SHADPS4_BACKIMP=1 needed) - "
                             "{} polled pages wait for it",
                             pollmap_pending.size());
            }
        } else {
            constexpr u32 kMaxPerPass = 256;
            constexpr u32 kMaxAttempts = 16;
            u32 budget = kMaxPerPass;
            size_t keep = 0;
            const size_t total = pollmap_pending.size();
            boost::container::small_vector<u64, 256> ready; // pages with a backing address
            for (size_t i = 0; i < total; ++i) {
                PollMapPending e = pollmap_pending[i];
                if (budget == 0) {
                    pollmap_pending[keep++] = e;
                    continue;
                }
                --budget;
                const VAddr page_addr = e.page << CACHING_PAGEBITS;
                if (!HomePolicy::PollMapPtMode()) {
                    // [POLLMAP v2] import mode: the page is in the set, WantsImport answers true,
                    // the next buffer (re)created over it gets guest RAM as its home. Nothing to
                    // write into the page table here.
                    const BufferId owner = page_table[e.page].buffer_id;
                    ++pollmap_redirected;
                    if (pollmap_redirected <= 8 || pollmap_redirected % 256 == 0) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "POLLMAP: page {:#x} marked for import (owner {}, {} marked)",
                                     page_addr,
                                     owner ? (slot_buffers[owner].is_import_backed ? "import"
                                                                                    : "vram")
                                           : "none",
                                     pollmap_redirected);
                    }
                    continue;
                }
                const vk::DeviceAddress bda = PolicyPtBda(page_addr);
                if (bda == 0) {
                    if (++e.attempts >= kMaxAttempts) {
                        ++pollmap_dropped;
                        if (pollmap_dropped <= 8) {
                            LOG_CRITICAL(Render_Vulkan,
                                         "POLLMAP: page {:#x} has no backing address after {} "
                                         "passes (unmapped, or the chunk import was refused) - "
                                         "dropped from the queue, kept in the set",
                                         page_addr, e.attempts);
                        }
                        continue;
                    }
                    pollmap_pending[keep++] = e;
                    continue;
                }
                const BufferId owner = page_table[e.page].buffer_id;
                const char* owner_kind =
                    owner ? (slot_buffers[owner].is_import_backed ? "import" : "vram") : "none";
                ready.push_back(e.page);
                // The entry WritePageTableRange puts in for a page under a non-import owner;
                // replayed by the PTHOST re-apply below.
                pollmap_applied.push_back(
                    {e.page, LiveReplaySkipEnabled() ? (bda | kBdaLandedBit) : bda});
                ++pollmap_redirected;
                if (pollmap_redirected <= 8 || pollmap_redirected % 256 == 0) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "POLLMAP: page {:#x} -> backing {:#x} (owner {}, {} redirected, "
                                 "{} pending)",
                                 page_addr, static_cast<u64>(bda), owner_kind, pollmap_redirected,
                                 keep + (total - i - 1));
                }
            }
            pollmap_pending.resize(keep);
            // WantsPt is true for every ready page (they are in the set), so the writer puts the
            // backing entry in unless an import-backed owner already points at guest RAM.
            // Adjacent pages go in one call: without PTHOST each call is a staging copy plus two
            // barriers in the command stream. host_import_mutex: HostBdaFor inside reads
            // host_imports, which the UMA import/release paths change from another thread
            // (FlushPendingHostImports takes the same lock for the same call; the order
            // host_import_mutex -> backimp.mutex is the existing one).
            std::sort(ready.begin(), ready.end());
            std::scoped_lock hlk{host_import_mutex};
            for (size_t i = 0; i < ready.size();) {
                size_t j = i + 1;
                while (j < ready.size() && ready[j] == ready[j - 1] + 1) {
                    ++j;
                }
                WritePageTableRange(ready[i], j - i);
                i = j;
            }
        }
    }

    // PTHOST re-apply: a slice of the redirected entries stored again, plain host stores into
    // the mapped table (what WritePageTableRange does with PTHOST), no lookup and no lock - the
    // entry of a page was fixed when it was redirected. An import-backed owner registered over
    // such a page since then gets the backing address instead of its own: the same guest bytes.
    // Without PTHOST every writer sits in the one command stream in program order, so there is
    // nothing to repair (and each rewrite would be a copy plus barriers).
    if (pollmap_pt_host && !pollmap_applied.empty()) {
        constexpr size_t kReapplyPerPass = 64;
        u8* const table = bda_pagetable_buffer.mapped_data.data();
        const size_t n = std::min(kReapplyPerPass, pollmap_applied.size());
        for (size_t i = 0; i < n; ++i) {
            if (pollmap_reapply_cursor >= pollmap_applied.size()) {
                pollmap_reapply_cursor = 0;
            }
            const PollMapApplied& a = pollmap_applied[pollmap_reapply_cursor++];
            std::memcpy(table + a.page * sizeof(vk::DeviceAddress), &a.entry, sizeof(a.entry));
        }
        _mm_sfence();
        pollmap_reapplied += n;
    }

    if (now >= pollmap_report_next) {
        pollmap_report_next = now + std::chrono::seconds(5);
        // The new pages are named so a session that dies in a TDR before the drain ran still
        // leaves a seed for SHADPS4_HOMEPOLICY_PT in its log.
        std::string names;
        for (const VAddr p : pollmap_report_pages) {
            fmt::format_to(std::back_inserter(names), "{:#x} ", p);
        }
        u32 pollmap_import_owned = 0;
    for (const VAddr ra : pollmap_report_pages) {
        const BufferId o = page_table[ra >> CACHING_PAGEBITS].buffer_id;
        if (o && slot_buffers[o].is_import_backed) {
            ++pollmap_import_owned;
        }
    }
    LOG_CRITICAL(Render_Vulkan,
                     "POLLMAP: marked={} redirected={} gwskipped={} notcpu={} staticpt={} import_owned={}/{} pending={} dropped={} reapplied={} "
                     "folds={} sweeps={} (last circle {} us of work over {} ms, worst slice {} "
                     "us, cursor {}/{}) pthost={} new since last report: {}{}",
                     pollmap_marked, pollmap_redirected, pollmap_gwskipped, pollmap_notcpu, pollmap_staticpt, pollmap_import_owned, pollmap_report_pages.size(), pollmap_pending.size(), pollmap_dropped,
                     pollmap_reapplied, pollmap_folds, pollmap_sweeps, pollmap_sweep_us,
                     pollmap_sweep_span_ms, pollmap_slice_max_us, pollmap_sweep_cursor,
                     POLLMAP_REGIONS, pollmap_pt_host,
                     names.empty() ? std::string{"none"} : names,
                     pollmap_report_pages.size() >= kReportPages ? "..." : "");
        pollmap_report_pages.clear();
    }
}

void BufferCache::VramPrefetchStep() {
    // [VRAMPREFETCH-AUTO] fold the blocks the fault parser noted into the range list; the
    // first touch of a block still loses its writes, but the rest of the block is covered
    // before the burst spreads over it.
    if (VramPrefetchAutoBytes() != 0) {
        std::scoped_lock lk{vram_auto_mutex};
        for (const VAddr base : vram_auto_blocks) {
            vram_prefetch_ranges.emplace_back(base, VramPrefetchAutoBytes());
            vram_auto_done_blocks.push_back(base);
            LOG_CRITICAL(Render_Vulkan, "VRAMPREFETCH: auto block {:#x} ({} MB) queued", base,
                         VramPrefetchAutoBytes() >> 20);
        }
        if (!vram_auto_blocks.empty()) {
            vram_auto_blocks.clear();
            vram_prefetch_next = {}; // run this pass now
        }
    }
    if (vram_prefetch_ranges.empty()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (vram_prefetch_next != std::chrono::steady_clock::time_point{} && now < vram_prefetch_next) {
        return;
    }
    vram_prefetch_next = now + std::chrono::seconds(3);
    constexpr u64 kMaxChunk = 8_MB;
    constexpr u32 kMaxPerPass = 64;
    u32 created = 0;
    u64 created_bytes = 0;
    for (const auto& [range_base, range_size] : vram_prefetch_ranges) {
        VAddr pos = Common::AlignDown(range_base, CACHING_PAGESIZE);
        const VAddr range_end = Common::AlignUp(range_base + range_size, CACHING_PAGESIZE);
        while (pos < range_end && created < kMaxPerPass) {
            if (const BufferId owner = page_table[pos >> CACHING_PAGEBITS].buffer_id) {
                const Buffer& b = slot_buffers[owner];
                VAddr next = pos + CACHING_PAGESIZE;
                if (!b.is_deleted && b.IsInBounds(pos, CACHING_PAGESIZE)) {
                    next = std::max<VAddr>(next, b.CpuAddr() + b.SizeBytes());
                }
                pos = std::min<VAddr>(Common::AlignUp(next, CACHING_PAGESIZE), range_end);
                continue;
            }
            if (!IsFaultRegionValid(pos, CACHING_PAGESIZE)) {
                pos += CACHING_PAGESIZE; // unmapped (yet): a later pass retries
                continue;
            }
            VAddr run_end = pos + CACHING_PAGESIZE;
            while (run_end < range_end && run_end - pos < kMaxChunk &&
                   !page_table[run_end >> CACHING_PAGEBITS].buffer_id &&
                   IsFaultRegionValid(run_end, CACHING_PAGESIZE)) {
                run_end += CACHING_PAGESIZE;
            }
            if (texture_cache.FindImageFromRange(pos, run_end - pos, false)) {
                pos = run_end;
                continue;
            }
            const BufferId id = CreateBuffer(pos, static_cast<u32>(run_end - pos));
            ++created;
            created_bytes += slot_buffers[id].SizeBytes();
            pos = run_end;
        }
    }
    if (created != 0) {
        vram_prefetch_buffers += created;
        vram_prefetch_bytes += created_bytes;
        LOG_CRITICAL(Render_Vulkan, "VRAMPREFETCH: +{} buffers ({} MB) this pass, total {} ({} MB)",
                     created, created_bytes >> 20, vram_prefetch_buffers, vram_prefetch_bytes >> 20);
    }
}

void BufferCache::BackimpPrefetchStep() {
    PageTraceStep();
    PolicyPrefetchStep();
    PollMapStep(); // [POLLMAP] no-op unless SHADPS4_POLLMAP=1
    if (RbQuarterEnabled()) {
        static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        const auto now = std::chrono::steady_clock::now();
        if (now >= next) {
            next = now + std::chrono::seconds(5);
            LOG_CRITICAL(Render_Vulkan, "RBQUARTER: delivered={} refused={} (4 KiB quarters, cumulative)",
                         rb_quarter_delivered.load(), rb_quarter_refused.load());
        }
    }
    DualArmStep();
    VramPrefetchStep();
    PageJournalStickyStep();
    // GPU thread only (Rasterizer::OnSubmit). That thread is the sole page_table/slot_buffers
    // writer, so the reads below need no buffers_mutex; CreateBuffer is called exactly the way
    // ObtainBuffer calls it.
    if (backimp_prefetch_ranges.empty()) {
        return;
    }
    auto& backimp = Vulkan::BackingImport::Get();
    if (!backimp.IsAvailable() || backimp.GetMode() == Vulkan::BackingImport::Mode::Off) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (backimp_prefetch_next != std::chrono::steady_clock::time_point{} &&
        now < backimp_prefetch_next) {
        return;
    }
    backimp_prefetch_next = now + std::chrono::seconds(3);

    const auto& aspace = Core::Memory::Instance()->GetAddressSpace();
    constexpr u64 kMaxChunk = 8_MB;
    // Windows maps views on 64KB boundaries, so probing every 64KB boundary cannot miss the
    // start of a mapped island while keeping a fully-unmapped pass cheap (one query per 64KB).
    constexpr u64 kProbeStep = 0x10000;
    constexpr u32 kMaxPerPass = 64; // bound the GPU-thread work per pass
    u32 created = 0;
    u64 pass_skipped = 0;
    for (const auto& [range_base, range_size] : backimp_prefetch_ranges) {
        VAddr pos = Common::AlignDown(range_base, CACHING_PAGESIZE);
        const VAddr range_end = Common::AlignUp(range_base + range_size, CACHING_PAGESIZE);
        while (pos < range_end && created < kMaxPerPass) {
            // Already owned - hop over the whole owner. Covers both earlier prefetch passes and
            // buffers the game's own draws created meanwhile.
            if (const BufferId owner = page_table[pos >> CACHING_PAGEBITS].buffer_id) {
                const Buffer& b = slot_buffers[owner];
                VAddr next = pos + CACHING_PAGESIZE;
                if (!b.is_deleted && b.IsInBounds(pos, CACHING_PAGESIZE)) {
                    next = std::max<VAddr>(next, b.CpuAddr() + b.SizeBytes());
                }
                pos = std::min<VAddr>(Common::AlignUp(next, CACHING_PAGESIZE), range_end);
                continue;
            }
            // Largest window into the backing starting here. Success is monotonic in size within
            // one view, so shrink by halves - but only a "split" refusal can be cured by
            // shrinking; not-mapped/fd/no-phys describe the region containing pos itself.
            u64 sz = Common::AlignDown(std::min<u64>(kMaxChunk, range_end - pos), CACHING_PAGESIZE);
            std::optional<u64> off;
            bool region_ineligible = false;
            while (sz >= CACHING_PAGESIZE) {
                const char* why = nullptr;
                off = aspace.GetBackingOffset(pos, sz, &why);
                if (off) {
                    if (backimp.CoversContiguously(*off, sz)) {
                        break; // good window (cross-chunk would also be cured by shrinking)
                    }
                    off.reset();
                } else if (why == nullptr || std::strcmp(why, "split") != 0) {
                    region_ineligible = true;
                    break;
                }
                sz = Common::AlignDown(sz / 2, CACHING_PAGESIZE);
            }
            if (!off) {
                // Silent skip: unmapped parts are expected, a later pass retries after the guest
                // maps them.
                const VAddr next = region_ineligible
                                       ? Common::AlignUp(pos + 1, kProbeStep)
                                       : pos + CACHING_PAGESIZE;
                pass_skipped += std::min<VAddr>(next, range_end) - pos;
                pos = std::min<VAddr>(next, range_end);
                continue;
            }
            // Clip at the first page something else owns so CreateBuffer's overlap resolution
            // cannot widen the request beyond the window just validated.
            for (VAddr p = pos; p < pos + sz; p += CACHING_PAGESIZE) {
                if (page_table[p >> CACHING_PAGEBITS].buffer_id) {
                    sz = p - pos;
                    break;
                }
            }
            if (sz == 0) {
                continue; // owner starts exactly here; the head check hops over it next round
            }
            // Same guard LABELUMA promotion has: an image over the range needs staging
            // semantics the backing cannot provide.
            if (texture_cache.FindImageFromRange(pos, sz, false)) {
                if (backimp_prefetch_logged < 64) {
                    ++backimp_prefetch_logged;
                    LOG_CRITICAL(Render_Vulkan,
                                 "BACKIMP: prefetch guest={:#x} size={:#x} skip(image)", pos, sz);
                }
                pass_skipped += sz;
                pos += sz;
                continue;
            }
            const BufferId id = CreateBuffer(pos, static_cast<u32>(sz));
            ++created;
            const Buffer& b = slot_buffers[id];
            const u64 got = b.SizeBytes();
            if (b.is_backing_backed) {
                ++backimp_prefetch_buffers;
                backimp_prefetch_covered += got;
                if (backimp_prefetch_logged < 64) {
                    ++backimp_prefetch_logged;
                    LOG_CRITICAL(Render_Vulkan, "BACKIMP: prefetch guest={:#x} size={:#x} ok", pos,
                                 got);
                }
            } else {
                // Should not happen (all checks re-run inside CreateBackingBacked) - if it does,
                // the log must show it: the device table now points at an unsynced VRAM buffer.
                pass_skipped += got;
                if (backimp_prefetch_logged < 64) {
                    ++backimp_prefetch_logged;
                    LOG_CRITICAL(Render_Vulkan,
                                 "BACKIMP: prefetch guest={:#x} size={:#x} skip(vram-fallback)",
                                 pos, got);
                }
            }
            pos = std::min<VAddr>(Common::AlignUp(b.CpuAddr() + got, CACHING_PAGESIZE), range_end);
        }
        if (created >= kMaxPerPass) {
            break;
        }
    }
    if (created != 0) {
        LOG_CRITICAL(Render_Vulkan,
                     "BACKIMP: prefetch done ranges={} buffers={} covered={} MB skipped={} MB",
                     backimp_prefetch_ranges.size(), backimp_prefetch_buffers,
                     backimp_prefetch_covered >> 20, pass_skipped >> 20);
    }
}

void BufferCache::ProcessFaultBuffer() {
    if (debug_watched_addr != 0) {
        DebugCompareWatched(debug_watched_addr);
    } else if (const char* watch = std::getenv("SHADPS4_WATCHLOW"); watch != nullptr) {
        // Hunt for the job object directly: it always sits at the same offset inside a 1 MiB
        // block of the guest heap, so only a few hundred addresses need checking, and its
        // header carries a recognisable signature. Reads go through ReadProcessMemory so
        // unmapped candidates fail harmlessly.
        static auto next = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now >= next) {
            next = now + std::chrono::seconds(2);
            const u64 low = std::strtoull(watch, nullptr, 0);
            const HANDLE self = GetCurrentProcess();
            for (VAddr base = 0x1010000000ULL; base < 0x1030000000ULL; base += 1_MB) {
                const VAddr candidate = base | low;
                u32 hdr[16]{};
                SIZE_T got = 0;
                if (!ReadProcessMemory(self, reinterpret_cast<LPCVOID>(candidate), hdr,
                                       sizeof(hdr), &got) ||
                    got != sizeof(hdr)) {
                    continue;
                }
                if ((hdr[0x28 / 4] & 0xFFFF) == 54 && hdr[0x24 / 4] > 0 && hdr[0x24 / 4] < 64 &&
                    hdr[0x20 / 4] < 64) {
                    debug_watched_addr = candidate;
                    LOG_CRITICAL(Render_Vulkan, "JOBOBJ found by scan at {:#x} (+20={} +24={})",
                                 candidate, hdr[0x20 / 4], hdr[0x24 / 4]);
                    // Watch the record the consumer trips over (index 8, 48-byte stride,
                    // payload at +0x28) rather than the counters: the counters are written
                    // as running totals by code that is easy to find, the record is not.
                    const u64 record_offset = 0x28 + u64{8} * 48;
                    Core::ArmWriteWatch(candidate + record_offset);
                    LOG_CRITICAL(Render_Vulkan, "JOBREC watch at {:#x} (record 8)",
                                 candidate + record_offset);
                    break;
                }
            }
        }
    }
    FlushPendingHostImports();
    // [BACKIMP] Cheap steady_clock check; prints at most once per five seconds.
    Vulkan::BackingImport::Get().MaybeLogSummary();
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::ImportHostRegion(VAddr addr, u64 size) {
    if (!uma_enabled || size == 0) {
        return;
    }
    const u64 align = instance.GetMinImportedHostPointerAlignment();
    const VAddr base = Common::AlignDown(addr, align);
    const u64 aligned_size = Common::AlignUp(addr + size, align) - base;
    const auto device = instance.GetDevice();

    const auto host_props = device.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        reinterpret_cast<void*>(base));
    if (host_props.result != vk::Result::eSuccess || host_props.value.memoryTypeBits == 0) {
        LOG_WARNING(Render_Vulkan, "UMA: host pointer {:#x} not importable ({})", base,
                    vk::to_string(host_props.result));
        return;
    }
    // Prefer a host-visible coherent type among the allowed bits.
    const auto mem_props = instance.GetPhysicalDevice().getMemoryProperties();
    u32 type_index = std::numeric_limits<u32>::max();
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (!(host_props.value.memoryTypeBits & (1u << i))) {
            continue;
        }
        const auto flags = mem_props.memoryTypes[i].propertyFlags;
        if (flags & vk::MemoryPropertyFlagBits::eHostVisible &&
            flags & vk::MemoryPropertyFlagBits::eHostCoherent) {
            type_index = i;
            break;
        }
        if (type_index == std::numeric_limits<u32>::max()) {
            type_index = i;
        }
    }

    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_ci{
        vk::BufferCreateInfo{
            .size = aligned_size,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        vk::ExternalMemoryBufferCreateInfo{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        },
    };
    auto buffer_res = device.createBuffer(buffer_ci.get());
    if (buffer_res.result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "UMA: buffer creation failed for {:#x}+{:#x}", base,
                    aligned_size);
        return;
    }
    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT,
                             vk::MemoryAllocateFlagsInfo>
        alloc_ci{
            vk::MemoryAllocateInfo{
                .allocationSize = aligned_size,
                .memoryTypeIndex = type_index,
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = reinterpret_cast<void*>(base),
            },
            vk::MemoryAllocateFlagsInfo{
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
            },
        };
    auto memory_res = device.allocateMemory(alloc_ci.get());
    if (memory_res.result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "UMA: import of {:#x}+{:#x} failed ({})", base, aligned_size,
                    vk::to_string(memory_res.result));
        device.destroyBuffer(buffer_res.value);
        return;
    }
    if (device.bindBufferMemory(buffer_res.value, memory_res.value, 0) != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "UMA: bind failed for {:#x}", base);
        device.destroyBuffer(buffer_res.value);
        device.freeMemory(memory_res.value);
        return;
    }
    const vk::DeviceAddress bda = device.getBufferAddress(vk::BufferDeviceAddressInfo{
        .buffer = buffer_res.value,
    });

    {
        std::scoped_lock lk{host_import_mutex};
        host_imports[base] = HostImport{
            .base = base,
            .size = aligned_size,
            .buffer = buffer_res.value,
            .memory = memory_res.value,
            .bda = bda,
        };
        pending_host_fills.push_back({base, aligned_size});
    }
    LOG_INFO(Render_Vulkan, "UMA: imported {:#x}+{:#x} -> bda {:#x}", base, aligned_size, bda);
}

void BufferCache::ReleaseHostRegion(VAddr addr, u64 size) {
    if (!uma_enabled || size == 0) {
        return;
    }
    std::vector<HostImport> released;
    {
        std::scoped_lock lk{host_import_mutex};
        auto it = host_imports.lower_bound(addr + size);
        while (it != host_imports.begin()) {
            --it;
            if (it->second.base + it->second.size <= addr) {
                break;
            }
            released.push_back(it->second);
            it = host_imports.erase(it);
        }
        if (!released.empty()) {
            // Queue a refill so the affected pages drop back to zero entries.
            pending_host_fills.push_back({addr, size});
        }
    }
    for (const auto& imp : released) {
        // The GPU may still be executing work that reads the import; destroy
        // once the current tick is known to have completed.
        scheduler.DeferOperation([device = instance.GetDevice(), imp] {
            device.destroyBuffer(imp.buffer);
            device.freeMemory(imp.memory);
        });
    }
}

vk::DeviceAddress BufferCache::HostBdaFor(VAddr addr) {
    auto it = host_imports.upper_bound(addr);
    if (it == host_imports.begin()) {
        return 0;
    }
    --it;
    const auto& imp = it->second;
    if (addr < imp.base || addr >= imp.base + imp.size) {
        return 0;
    }
    return imp.bda + (addr - imp.base);
}

// [HOMEPOLICY-PT] Guest-RAM device address of one page through the backing, page by page: the
// backing under the list arena is physically split, so no window serves a bind there, but a
// single page is always contiguous. 0 when the page has no backing (unmapped / no chunk).
vk::DeviceAddress BufferCache::PolicyPtBda(VAddr page_addr) {
    const auto& address_space = Core::Memory::Instance()->GetAddressSpace();
    const char* why = nullptr;
    const auto off = address_space.GetBackingOffset(page_addr, CACHING_PAGESIZE, &why);
    if (!off) {
        return 0;
    }
    const auto bda = Vulkan::BackingImport::Get().PageBda(*off);
    return bda ? *bda : 0;
}

void BufferCache::WritePageTableRange(u64 page_begin, u64 size_pages) {
    boost::container::small_vector<vk::DeviceAddress, 256> entries;
    entries.reserve(size_pages);
    static std::atomic<u64> pt_redirects{0};
    static std::atomic<u64> pt_misses{0};
    for (u64 i = 0; i < size_pages; ++i) {
        const u64 page = page_begin + i;
        const VAddr page_addr = page << CACHING_PAGEBITS;
        // [HOMEPOLICY-PT] a listed page points at guest RAM whatever owns it in the cache: the
        // VRAM owner (if any) still serves descriptor binds, live loads read the CPU's bytes.
        if (HomePolicy::WantsPt(page_addr, CACHING_PAGESIZE)) {
            const BufferId owner = page_table[page].buffer_id;
            const bool import_owner = owner && slot_buffers[owner].is_import_backed;
            if (!import_owner) {
                if (vk::DeviceAddress e = PolicyPtBda(page_addr); e != 0) {
                    if (LiveReplaySkipEnabled()) {
                        e |= kBdaLandedBit;
                    }
                    entries.push_back(e);
                    const u64 n = pt_redirects.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (n <= 4 || n % 65536 == 0) {
                        LOG_CRITICAL(Render_Vulkan,
                                     "HOMEPOLICY-PT: page {:#x} -> backing {:#x} (owner {}, redirects {}, "
                                     "misses {})",
                                     page_addr, e & ~kBdaLandedBit, owner ? "vram" : "none", n,
                                     pt_misses.load(std::memory_order_relaxed));
                    }
                    continue;
                }
                const u64 m = pt_misses.fetch_add(1, std::memory_order_relaxed) + 1;
                if (m <= 4 || m % 4096 == 0) {
                    LOG_CRITICAL(Render_Vulkan, "HOMEPOLICY-PT: page {:#x} has no backing address (misses {})",
                                 page_addr, m);
                }
            }
        }
        if (const BufferId owner = page_table[page].buffer_id; owner) {
            // A cached VRAM buffer owns the page: keep the fast entry.
            const Buffer& buffer = slot_buffers[owner];
            vk::DeviceAddress e = buffer.BufferDeviceAddress() + (page_addr - buffer.CpuAddr());
            if (LiveReplaySkipEnabled() && buffer.is_import_backed) {
                e |= kBdaLandedBit;
            }
            entries.push_back(e);
        } else {
            vk::DeviceAddress e = HostBdaFor(page_addr);
            if (LiveReplaySkipEnabled() && e != 0) {
                e |= kBdaLandedBit;
            }
            entries.push_back(e);
        }
    }
    if (!bda_pagetable_buffer.mapped_data.empty()) {
        std::memcpy(bda_pagetable_buffer.mapped_data.data() +
                        page_begin * sizeof(vk::DeviceAddress),
                    entries.data(), entries.size() * sizeof(vk::DeviceAddress));
        _mm_sfence();
        return;
    }
    WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress), entries.data(),
                    size_pages * sizeof(vk::DeviceAddress));
}

void BufferCache::FlushPendingHostImports() {
    std::vector<PendingHostFill> fills;
    {
        std::scoped_lock lk{host_import_mutex};
        fills.swap(pending_host_fills);
    }
    for (const auto& fill : fills) {
        const u64 page_begin = fill.addr >> CACHING_PAGEBITS;
        const u64 page_end = Common::DivCeil(fill.addr + fill.size, CACHING_PAGESIZE);
        std::scoped_lock lk{host_import_mutex};
        WritePageTableRange(page_begin, page_end - page_begin);
    }
}

// [debug] Journal of every host address range ever published into the BDA page table, with the
// tick it was registered, unregistered and finally destroyed. VK_EXT_device_fault reports raw
// host addresses; this turns such an address into the name of the guest buffer that owned it
// and tells whether its memory was still alive at the time of the fault.
namespace {
struct BdaRecord {
    u64 begin{}, end{};
    u64 guest{};
    u32 id{};
    u64 tick_reg{}, tick_unreg{}, tick_destroy{};
};
constexpr size_t BdaLogMax = 8192;
std::mutex g_bda_mutex;
std::vector<BdaRecord> g_bda_log;
size_t g_bda_pos = 0;

// Mirror of what the page table currently publishes, so a fault address can be answered with
// "the table still maps this" (the zeroing on unregister is broken) versus "nothing maps this"
// (the stale reference lives in a descriptor, not in the table).
struct LiveEntry {
    u64 bda{};
    u64 guest{};
    u32 id{};
};
std::unordered_map<u64, LiveEntry> g_bda_live;

// Recently unregistered pages with the address they used to publish. The table must read zero
// here; if it still holds the old address, the zeroing on unregister never reached the device
// and that is exactly the stale pointer the shader dereferences.
// Pages that were unregistered and never taken back. These MUST read zero in the table; a
// non-zero value here is a pointer into memory that no buffer owns any more, which is exactly
// what a shader would dereference into nowhere.
std::unordered_map<u64, u64> g_bda_orphans;

// All of the journalling below is diagnostic only and must stay OFF by default: it takes a
// global lock on every page of every buffer registration, and this game registers thousands of
// buffers a second. Leaving it always-on cut a known 161-second run down to 33 and made the
// device fault "disappear" simply because the game died sooner.
bool BdaDebugEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BDADBG");
        return e && e[0] == '1';
    }();
    return on;
}

void BdaLogAdd(u64 begin, u64 end, u64 guest, u32 id, u64 tick) {
    if (begin == 0 || !BdaDebugEnabled()) {
        return;
    }
    std::scoped_lock lk{g_bda_mutex};
    if (g_bda_log.size() < BdaLogMax) {
        g_bda_log.push_back({begin, end, guest, id, tick, 0, 0});
    } else {
        g_bda_log[g_bda_pos] = {begin, end, guest, id, tick, 0, 0};
        g_bda_pos = (g_bda_pos + 1) % BdaLogMax;
    }
}

void BdaLiveSet(u64 page, u64 bda, u64 guest, u32 id) {
    if (!BdaDebugEnabled()) {
        return;
    }

    std::scoped_lock lk{g_bda_mutex};
    g_bda_live[page] = LiveEntry{bda, guest, id};
    g_bda_orphans.erase(page);
}

void BdaLiveClear(u64 page_begin, u64 page_end) {
    if (!BdaDebugEnabled()) {
        return;
    }

    std::scoped_lock lk{g_bda_mutex};
    for (u64 p = page_begin; p != page_end; ++p) {
        if (const auto it = g_bda_live.find(p); it != g_bda_live.end()) {
            g_bda_orphans[p] = it->second.bda;
            g_bda_live.erase(it);
        }
    }
}

void BdaLogMark(u32 id, u64 tick, bool destroyed) {
    if (!BdaDebugEnabled()) {
        return;
    }

    std::scoped_lock lk{g_bda_mutex};
    for (auto it = g_bda_log.rbegin(); it != g_bda_log.rend(); ++it) {
        if (it->id != id) {
            continue;
        }
        if (destroyed) {
            if (it->tick_destroy == 0) {
                it->tick_destroy = tick;
                return;
            }
        } else if (it->tick_unreg == 0) {
            it->tick_unreg = tick;
            return;
        }
    }
}
} // namespace

std::string BufferCache::DescribeHostAddress(u64 host_addr) {
    std::scoped_lock lk{g_bda_mutex};
    std::string out;
    for (const auto& r : g_bda_log) {
        if (host_addr < r.begin || host_addr >= r.end) {
            continue;
        }
        out += fmt::format(" [id={} host={:#x}..{:#x} guest={:#x} off={:#x} reg@{} unreg@{} "
                           "destroy@{}]",
                           r.id, r.begin, r.end, r.guest, host_addr - r.begin, r.tick_reg,
                           r.tick_unreg, r.tick_destroy);
    }
    u32 live_hits = 0;
    for (const auto& [page, e] : g_bda_live) {
        if (host_addr < e.bda || host_addr >= e.bda + CACHING_PAGESIZE) {
            continue;
        }
        if (live_hits++ < 4) {
            out += fmt::format(" LIVE-ENTRY[page={:#x} guest={:#x} bda={:#x} id={}]", page,
                               e.guest, e.bda, e.id);
        }
    }
    if (live_hits == 0) {
        out += " (no live page table entry maps this address)";
    }
    return out.empty() ? std::string{"<address never published to the page table>"} : out;
}

// [ATTRIB] every live page-table entry whose device address falls inside [base, base+size):
// the guest pages still pointing into an object the driver has already unbound (188/193: the
// blur draw read a dead object 5600 ticks after its unbind).
std::string BufferCache::DescribeHostRange(u64 base, u64 size) {
    std::scoped_lock lk{g_bda_mutex};
    std::string out;
    u32 hits = 0;
    for (const auto& [page, e] : g_bda_live) {
        if (e.bda < base || e.bda >= base + size) {
            continue;
        }
        if (hits++ < 16) {
            out += fmt::format(" [page={:#x} guest={:#x} bda={:#x} id={}]", page, e.guest, e.bda,
                               e.id);
        }
    }
    for (const auto& r : g_bda_log) {
        if (r.end <= base || r.begin >= base + size) {
            continue;
        }
        out += fmt::format(" LOG[id={} host={:#x}..{:#x} guest={:#x} reg@{} unreg@{} destroy@{}]",
                           r.id, r.begin, r.end, r.guest, r.tick_reg, r.tick_unreg,
                           r.tick_destroy);
    }
    return fmt::format("live entries into range: {}{}", hits, out);
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

// Page table updates are recorded straight into the scheduler's command buffer. A Vulkan
// command buffer may only be recorded by one thread at a time, so if the cache is ever entered
// from a guest thread while the GPU thread is recording draws, the recording is corrupted and a
// page table write can be lost - leaving the shader with an address whose buffer is already gone.
static void CheckSingleThreaded(const char* what) {
    static std::atomic<std::size_t> owner{0};
    static std::atomic<u32> reported{0};
    const auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::size_t expected = 0;
    if (owner.compare_exchange_strong(expected, id)) {
        return;
    }
    if (expected != id && reported.fetch_add(1) < 8) {
        LOG_CRITICAL(Render_Vulkan, "THREADMIX: {} called from thread {:#x}, owner is {:#x}", what,
                     id, expected);
    }
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    CheckSingleThreaded(insert ? "Register" : "Unregister");
    // The command-buffer writes below stay GPU-thread-only; the lock makes the
    // page_table/buffer_ranges bookkeeping visible to the fault fast path.
    std::scoped_lock lk{buffers_mutex};
    Buffer& buffer = slot_buffers[buffer_id];
    if (BufWatched(buffer.CpuAddr(), buffer.CpuAddr() + buffer.SizeBytes())) {
        static u32 reported = 0;
        if (reported < 24) {
            ++reported;
            LOG_CRITICAL(Render_Vulkan, "BUFWATCH: {} id={} guest={:#x}..{:#x} size={:#x} tick={}",
                         insert ? "reg" : "unreg", buffer_id.index, buffer.CpuAddr(),
                         buffer.CpuAddr() + buffer.SizeBytes(), buffer.SizeBytes(),
                         scheduler.CurrentTick());
        }
    }
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    // [debug] журнал device-адресов: сопоставляется с DEVFAULT ReadInvalid,
    // чтобы понять, чей это адрес и был ли буфер уже удалён.
    // У части буферов BDA нет — запрашивать его там нельзя (ASSERT).
    if (const char* dbg = std::getenv("SHADPS4_DBGLOG"); dbg && dbg[0] == '1') {
        const vk::DeviceAddress dbg_bda =
            buffer.Handle() ? static_cast<vk::DeviceAddress>(buffer.buffer.bda_addr) : 0;
        LOG_WARNING(Render_Vulkan, "BDAMAP: {} id={} bda={:#x} size={:#x} guest={:#x} tick={}",
                    insert ? "reg" : "unreg", buffer_id.index, dbg_bda, size, device_addr_begin,
                    scheduler.CurrentTick());
    }

    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        // [bisect] SHADPS4_BDATAG=1 stamps the low 16 bits of the page index into the unused
        // top 16 bits of every entry (device addresses here are well under 2^48). The shader
        // checks the stamp before dereferencing, so an entry that belongs to a different page -
        // the only way a correct-looking address can point at nothing - is caught instead of
        // being followed into unmapped memory.
        // 1 = stamp the page index (catches an entry that belongs to another page)
        // 2 = stamp a constant (catches only garbage in the top bits, never a page mismatch)
        static const u32 tag_mode = [] {
            const char* e = std::getenv("SHADPS4_BDATAG");
            return e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
        }();
        const bool tag_entries = tag_mode != 0;
        static std::atomic<u64> pt_reg_redirects{0};
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            if (LiveReplaySkipEnabled() && !tag_entries && buffer.is_import_backed) {
                addr |= kBdaLandedBit;
            }
            // [HOMEPOLICY-PT] a listed page keeps guest RAM as its device address even under a
            // VRAM owner: the entry points into the backing page by page, the buffer still
            // serves descriptor binds. (This is the writer for every registered buffer;
            // WritePageTableRange only covers host-import and unregister paths.)
            if (!buffer.is_import_backed && !tag_entries) {
                const VAddr page_addr = (page_begin + i) << CACHING_PAGEBITS;
                if (HomePolicy::WantsPt(page_addr, CACHING_PAGESIZE)) {
                    if (const vk::DeviceAddress e = PolicyPtBda(page_addr); e != 0) {
                        addr = LiveReplaySkipEnabled() ? (e | kBdaLandedBit) : e;
                        const u64 n = pt_reg_redirects.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (n <= 4 || n % 65536 == 0) {
                            LOG_CRITICAL(Render_Vulkan,
                                         "HOMEPOLICY-PT: register page {:#x} -> backing {:#x} "
                                         "(buffer {:#x}+{:#x}, redirects {})",
                                         page_addr, e, buffer.CpuAddr(), buffer.SizeBytes(), n);
                        }
                    }
                }
            }
            if (tag_entries) {
                if (addr >> 48 != 0) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        LOG_CRITICAL(Render_Vulkan,
                                     "BDATAG: device address {:#x} uses more than 48 bits, "
                                     "tagging is unsafe on this driver",
                                     u64(addr));
                    }
                } else {
                    const u64 stamp = tag_mode == 2 ? 1ULL : ((page_begin + i) & 0xFFFF);
                    addr |= vk::DeviceAddress{stamp} << 48;
                }
            }
            bda_addrs.push_back(addr);
        }
        if (!bda_pagetable_buffer.mapped_data.empty()) {
            // Ordering: the queued write this replaces used to land AFTER the
            // JoinOverlap copies that fill the buffer, because both sat in the
            // same command stream. A host store publishes instantly, so hold it
            // until the buffer's content tick retires - otherwise a shader can
            // read garbage through a fresh entry (observed as silent deaths on
            // chapter load with PTHOST v1).
            const u64 content_tick = buffer.content_tick;
            const bool deferred = !scheduler.GetMasterSemaphore()->IsFree(content_tick);
            // [POLLMAP] a deferred publish carries entries computed now and stores them from
            // the priority thread later; by then PollMapStep may have redirected some of these
            // pages to guest RAM, and the VRAM address would overwrite the redirect for good
            // (the page is already in the sticky set, nothing queues it again). Re-derive the
            // entries of set pages at publish time - guest RAM never holds garbage, so the
            // content-tick hold does not apply to them. The immediate path just computed the
            // same entries and skips the second walk.
            const bool pollmap_recheck =
                deferred && PollMapEnabled() && !buffer.is_import_backed && !tag_entries;
            auto publish = [this, page_begin, pollmap_recheck,
                            addrs = boost::container::small_vector<vk::DeviceAddress, 256>(
                                bda_addrs.begin(), bda_addrs.end())]() mutable {
                if (pollmap_recheck) {
                    for (size_t i = 0; i < addrs.size(); ++i) {
                        const VAddr page_addr = (page_begin + i) << CACHING_PAGEBITS;
                        if (!HomePolicy::WantsPt(page_addr, CACHING_PAGESIZE)) {
                            continue;
                        }
                        if (const vk::DeviceAddress e = PolicyPtBda(page_addr); e != 0) {
                            addrs[i] = LiveReplaySkipEnabled() ? (e | kBdaLandedBit) : e;
                        }
                    }
                }
                std::memcpy(bda_pagetable_buffer.mapped_data.data() +
                                page_begin * sizeof(vk::DeviceAddress),
                            addrs.data(), addrs.size() * sizeof(vk::DeviceAddress));
                _mm_sfence();
            };
            if (!deferred) {
                publish();
            } else {
                scheduler.DeferPriorityOperation(std::move(publish));
            }
        } else {
            WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                            bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        }
        BdaLogAdd(bda_addrs.front(), bda_addrs.front() + (size_pages << CACHING_PAGEBITS),
                  device_addr_begin, buffer_id.index, scheduler.CurrentTick());
        for (u64 i = 0; i < size_pages; ++i) {
            BdaLiveSet(page_begin + i, bda_addrs[i], device_addr_begin, buffer_id.index);
        }
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
        // [TDR ETAP5] Honest register wins over any rescue import on these pages: the owner's real
        // bda is now published above, so drop the rescue coverage (RescueBdaFor -> 0). We keep the
        // imported Buffer alive for the session (bounded by budget) to avoid a mid-flight free.
        if (spin_enabled) {
            std::scoped_lock slk{spin_import_mutex};
            for (auto cur = spin_imports.begin(); cur != spin_imports.end();) {
                const SpinImport& imp = cur->second;
                if (imp.base < device_addr_end && device_addr_begin < imp.base + imp.size) {
                    spin_imported_bytes -= std::min(spin_imported_bytes, imp.size);
                    cur = spin_imports.erase(cur);
                } else {
                    ++cur;
                }
            }
        }
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        BdaLogMark(buffer_id.index, scheduler.CurrentTick(), false);
        BdaLiveClear(page_begin, page_end);
        lru_cache.Free(buffer.LRUId());
        if (uma_enabled) {
            // Freed pages fall back to live host entries where an import
            // covers them (zero otherwise), keeping DMA reads functional.
            std::scoped_lock lk{host_import_mutex};
            WritePageTableRange(page_begin, size_pages);
        } else {
            const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
            if (!bda_pagetable_buffer.mapped_data.empty()) {
                std::memset(bda_pagetable_buffer.mapped_data.data() +
                                page_begin * sizeof(vk::DeviceAddress),
                            0, size_pages * sizeof(vk::DeviceAddress));
                _mm_sfence();
            } else {
                bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
            }
            // [POLLMAP] a redirected page keeps its guest-RAM entry across the unregister. With
            // the entry zeroed, the next poll read faults, the fault manager stretches a fresh
            // VRAM owner over the page and the poll spins on a stale copy until that owner's
            // register writer runs. Covers the static list too, but only with the env on. One
            // range query and one writer call: for the pages outside the set WritePageTableRange
            // produces the very zero just written (no owner left, no host import in this
            // branch), so the outcome equals per-page runs at a single traversal.
            // host_import_mutex for HostBdaFor inside, as the UMA branch above takes it.
            if (PollMapEnabled() && HomePolicy::WantsPt(page_begin << CACHING_PAGEBITS,
                                                        size_pages << CACHING_PAGEBITS)) {
                std::scoped_lock hlk{host_import_mutex};
                WritePageTableRange(page_begin, size_pages);
            }
            // [TDR ETAP5] For pages still covered by a live rescue import, republish its bda
            // instead of the zero just written - otherwise the next access re-faults and re-spins.
            // The main queue is alive at Unregister time (WriteDataBuffer is fine).
            if (spin_enabled) {
                for (u64 page = page_begin; page != page_end; ++page) {
                    const vk::DeviceAddress rbda = RescueBdaFor(page << CACHING_PAGEBITS);
                    if (rbda != 0) {
                        WriteDataBuffer(bda_pagetable_buffer, page * sizeof(vk::DeviceAddress), &rbda,
                                        sizeof(rbda));
                    }
                }
            }
        }
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    if (buffer.is_import_backed) {
        // Coherent guest RAM: CPU writes are already visible to the GPU, an
        // upload would overwrite fresher GPU stores with a stale CPU snapshot
        // (lost update), and clearing CPU-dirty would only re-arm write faults.
        static std::atomic<bool> texel_warned{false};
        if (is_texel_buffer && !texel_warned.exchange(true)) {
            LOG_CRITICAL(Render_Vulkan,
                         "LABELUMA: texel-buffer sync skipped on import-backed buffer");
        }
        return true;
    }
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    // [DUALHOME] a dual owner is refilled by GPU copies from the import piece (guest memory =
    // union of CPU and GPU writes, read at execution time), never from a CPU snapshot that could
    // overwrite a GPU store made after the snapshot.
    const bool dual_owner =
        DualHomeEnabled() && IsDualRange(buffer.CpuAddr(), buffer.SizeBytes());
    struct PieceCopy {
        vk::Buffer src;
        vk::BufferCopy copy;
    };
    boost::container::small_vector<PieceCopy, 8> piece_copies;
    // [LOSTUPD] ranges are collected here and tested after ForEachUploadRange returns: the
    // callback runs under the region lock and the check would take the same lock again.
    const bool lostupd_probe = LostUpdateProbeFlag() && !is_written;
    boost::container::small_vector<std::pair<VAddr, u64>, 8> lostupd_ranges;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            // [SKINDIRECT-STORES][VERIFY] a guest-RAM upload landing inside a converted output:
            // reads the ring only, takes no lock (the lambda already holds the region lock).
            if (SkinDirectStoreVerifyFlag()) {
                SkinDirectVerifyUpload(device_addr_out, range_size);
            }
            if (lostupd_probe) {
                lostupd_ranges.emplace_back(device_addr_out, range_size);
            }
            if (PageTrace::Enabled()) {
                for (VAddr p = Common::AlignDown(device_addr_out, CACHING_PAGESIZE);
                     p < device_addr_out + range_size; p += CACHING_PAGESIZE) {
                    PageTrace::Note(p, PageTrace::UP);
                }
            }
            if (dual_owner) {
                // split the run at piece boundaries; only a run no piece covers falls back
                VAddr cur = device_addr_out;
                const VAddr run_end = device_addr_out + range_size;
                while (cur < run_end) {
                    const DualPiece* piece = DualPieceFor(cur);
                    if (!piece) {
                        break;
                    }
                    const VAddr part_end = std::min<VAddr>(run_end, piece->base + piece->size);
                    piece_copies.push_back({piece->buffer->Handle(),
                                            vk::BufferCopy{cur - piece->base, cur - buffer_start,
                                                           part_end - cur}});
                    ++dual_piece_copies;
                    cur = part_end;
                }
                if (cur >= run_end) {
                    return;
                }
                dual_staging_bytes += run_end - cur;
                copies.emplace_back(total_size_bytes, cur - buffer_start, run_end - cur);
                total_size_bytes += run_end - cur;
                return;
            }
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    // [LOSTUPD] the region lock is gone by now, so the check is safe here. Tallies are
    // thread-local and flushed rarely: two atomic adds per range would cost more than they
    // measure (11.09 lesson).
    if (lostupd_probe && !lostupd_ranges.empty()) {
        static thread_local u64 t_runs = 0, t_bytes = 0, t_total = 0;
        for (const auto& [addr, len] : lostupd_ranges) {
            ++t_total;
            if (memory_tracker->IsRegionGpuModified(addr, len)) {
                ++t_runs;
                t_bytes += len;
            }
        }
        if (t_total >= 4096) {
            g_lostupd_runs.fetch_add(t_runs, std::memory_order_relaxed);
            g_lostupd_bytes.fetch_add(t_bytes, std::memory_order_relaxed);
            g_lostupd_total.fetch_add(t_total, std::memory_order_relaxed);
            t_runs = t_bytes = t_total = 0;
        }
    }

    if (total_size_bytes || !piece_copies.empty()) {
        buffer.last_upload_tick = scheduler.CurrentTick();
        Vulkan::g_dbg_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
        if (total_size_bytes >= 32_MB) {
            LOG_WARNING(Render_Vulkan, "BIGUP: sync {} MB (buffer {} MB, texel={}, written={})",
                        total_size_bytes >> 20, buffer.SizeBytes() >> 20, is_texel_buffer,
                        is_written);
        }
    }
    if (src_buffer || !piece_copies.empty()) {
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        if (src_buffer) {
            cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
        }
        if (!piece_copies.empty()) {
            // [DUALHOME] Only mode 1 has shaders writing into the piece (dual stores) that must be
            // ordered before the transfer read. In modes 2/3 the piece is written by the CPU
            // (host-coherent) and by the writeback kernel behind its own barrier - and a global
            // MemoryWrite->TransferRead barrier here, ~150 times per frame in front of every
            // worker dispatch, flushed L2 and cost the worker 2x (liveloop56/58/61).
            if (DualHomeMode() == 1) {
                const vk::MemoryBarrier2 piece_barrier = {
                    .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .memoryBarrierCount = 1,
                    .pMemoryBarriers = &piece_barrier,
                });
            }
            for (const auto& pc : piece_copies) {
                cmdbuf.copyBuffer(pc.src, buffer.buffer, pc.copy);
            }
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        TouchBuffer(buffer);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false);
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

// [debug] SHADPS4_PTCHECK=N: every N submits, read a sample of the page table back from the
// device and compare it with the CPU mirror. A mismatch proves the table itself is being
// corrupted; a match proves the entries are right and the damage happens elsewhere on the
// dereference path.
void BufferCache::DebugVerifyPageTable() {
    static const u32 every = [] {
        const char* e = std::getenv("SHADPS4_PTCHECK");
        return e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    if (every == 0) {
        return;
    }
    static u64 counter = 0;
    if ((counter++ % every) != 0) {
        return;
    }

    // Half the sample is live pages (the table must match the mirror), half is pages that were
    // unregistered (the table must read zero). The second half is the interesting one.
    boost::container::small_vector<std::pair<u64, u64>, 128> sample;
    size_t orphan_count = 0;
    size_t orphans_sampled = 0;
    {
        std::scoped_lock lk{g_bda_mutex};
        for (const auto& [page, e] : g_bda_live) {
            sample.emplace_back(page, e.bda);
            if (sample.size() >= 32) {
                break;
            }
        }
        size_t taken = 0;
        for (const auto& [page, old_bda] : g_bda_orphans) {
            sample.emplace_back(page, 0ULL);
            if (++taken >= 96) {
                break;
            }
        }
        orphan_count = g_bda_orphans.size();
        orphans_sampled = taken;
    }
    if (sample.empty()) {
        return;
    }

    // Also pull the tag-mismatch counter the shader keeps in the fault buffer.
    sample.emplace_back(0, 0);
    const u64 bytes = sample.size() * sizeof(u64);
    const auto [dst, offset] = download_buffer.Map(bytes);
    if (dst == nullptr) {
        return;
    }
    download_buffer.Commit();

    boost::container::small_vector<vk::BufferCopy, 128> copies;
    for (u32 i = 0; i + 1 < sample.size(); ++i) {
        copies.push_back(vk::BufferCopy{
            .srcOffset = sample[i].first * sizeof(vk::DeviceAddress),
            .dstOffset = offset + i * sizeof(u64),
            .size = sizeof(u64),
        });
    }

    const u64 tag_counter_dst = offset + (sample.size() - 1) * sizeof(u64);
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = bda_pagetable_buffer.Handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre,
    });
    cmdbuf.copyBuffer(bda_pagetable_buffer.Handle(), download_buffer.Handle(), copies);
    {
        auto* fault = fault_manager.GetFaultBuffer();
        const vk::BufferCopy tag_copy{
            .srcOffset = (CACHING_NUMPAGES / 32 + 1) * sizeof(u32),
            .dstOffset = tag_counter_dst,
            .size = sizeof(u32),
        };
        cmdbuf.copyBuffer(fault->Handle(), download_buffer.Handle(), tag_copy);
    }
    scheduler.Finish();
    u32 tag_bad_count = 0;
    std::memcpy(&tag_bad_count, dst + (sample.size() - 1) * sizeof(u64), sizeof(u32));

    u32 bad = 0;
    for (u32 i = 0; i + 1 < sample.size(); ++i) {
        u64 got = 0;
        std::memcpy(&got, dst + i * sizeof(u64), sizeof(u64));
        if (got == sample[i].second) {
            continue;
        }
        if (bad < 6) {
            LOG_CRITICAL(Render_Vulkan, "PTBAD: page={:#x} guest={:#x} table={:#x} expected={:#x}",
                         sample[i].first, sample[i].first << CACHING_PAGEBITS, got,
                         sample[i].second);
        }
        ++bad;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "PTCHECK: {} of {} disagree | TAGBAD={} | live={} orphans={} (sampled {}) "
                 "buffers={} used={}MB vram={}MB",
                 bad, sample.size() - 1, tag_bad_count, sample.size() - 1 - orphans_sampled,
                 orphan_count, orphans_sampled, slot_buffers.size(), total_used_memory >> 20,
                 instance.CanReportMemoryUsage() ? (instance.GetDeviceMemoryUsage() >> 20) : 0);
}

// [BACKIMPGC] SHADPS4_BACKIMP_GC_KEEP: live backing buffers to keep before collecting.
static u64 BackingGcKeep() {
    static const u64 keep = [] {
        const char* e = std::getenv("SHADPS4_BACKIMP_GC_KEEP");
        const u64 v = e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
        return v != 0 ? v : 1024ull;
    }();
    return keep;
}

void BufferCache::RunGarbageCollector() {
    DebugVerifyPageTable();
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    // Every cache buffer asks for a device address, and VMA gives those a DEDICATED
    // VkDeviceMemory each. Collecting only by bytes therefore misses the real limit: this game
    // reaches five thousand live buffers - five thousand allocations - in ninety seconds while
    // using under two gigabytes, so the byte trigger never fires and nothing is ever reclaimed.
    // Gated while its side effects are being measured: the collector stalls on a synchronous
    // download, and a change that destabilises the run masks the very fault being hunted.
    static const bool collect_by_count = [] {
        const char* e = std::getenv("SHADPS4_BUFGC");
        return e && e[0] == '1';
    }();
    const u64 live_buffers = slot_buffers.size();
    Postmortem::g_res_buffers.store(live_buffers, std::memory_order_relaxed);
    Postmortem::g_res_vram_mb.store(total_used_memory >> 20, std::memory_order_relaxed);
    // Non-backing cache buffers each ask for a device address, and VMA gives those a DEDICATED
    // VkDeviceMemory - that is what counts against maxMemoryAllocationCount (4096 on NVIDIA).
    Postmortem::g_live_dedicated.store(
        live_buffers - std::min<u64>(live_buffers, Postmortem::g_res_backing_bufs.load(
                                                       std::memory_order_relaxed)),
        std::memory_order_relaxed);
    if (Postmortem::g_max_mem_allocs.load(std::memory_order_relaxed) == 0) {
        Postmortem::g_max_mem_allocs.store(
            instance.GetPhysicalDevice().getProperties().limits.maxMemoryAllocationCount,
            std::memory_order_relaxed);
    }
    // [BACKIMP] Reclaiming BACKING buffers needs neither the byte trigger nor the synchronous
    // download the flag above was guarding, so it gets its own count trigger. Without this the
    // cache grew monotonically for the whole session in MODE=all: every buffer is import-backed
    // there, so clean_up bailed on all of them and the `buf` phase grew with the cache.
    // Default OFF: the default build must behave exactly like the configuration that shipped
    // working. Every reclaim experiment is opt-in and measured on its own.
    static const bool backing_gc = [] {
        const char* e = std::getenv("SHADPS4_BACKIMP_GC");
        return e != nullptr && e[0] == '1';
    }();
    const bool too_many_backing =
        backing_gc &&
        Postmortem::g_res_backing_bufs.load(std::memory_order_relaxed) > BackingGcKeep();
    const bool too_many_allocations =
        (collect_by_count && live_buffers >= BufferCountSoftLimit) || too_many_backing;
    if (total_used_memory < trigger_gc_memory && !too_many_allocations) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory ||
                            (collect_by_count && live_buffers >= BufferCountHardLimit);
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        // The download must be synchronous here: the asynchronous form clears the GPU-modified
        // flag only later, and marking the same region CPU-modified while the GPU flag is still
        // set asks the page manager for write-only permission, which does not exist
        // (readable = !gpu, writeable = cpu) - it fires an assertion and kills the emulator.
        if (buffer.is_import_backed) {
            // Sysmem-backed: eviction frees no VRAM and would silently resurrect
            // the fault class this buffer exists to remove.
            return;
        }
        std::scoped_lock serve_lk{serve_mutex};
        DownloadBufferMemory<false>(buffer, buffer.CpuAddr(), buffer.SizeBytes());
        memory_tracker->MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        DeleteBuffer(buffer_id);
    };
    // The call the collector was missing: without it clean_up was dead code and the cache grew
    // without bound. Same shape the texture cache already uses. Kept behind the same flag as
    // the count trigger until its effect on stability is measured - reclaiming buffers stalls
    // on a synchronous download, which is not free.
    if (collect_by_count) {
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
        if (aggressive) {
            lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
        }
    }

    // [BACKIMPGC] Backing buffers are collected in TWO PHASES: the LRU walk only COLLECTS ids,
    // and the deletion happens after the walk has finished. Deleting from inside the walk (the
    // first attempt) mutated the very containers the iteration and its callee both touch -
    // DeleteBuffer -> Unregister frees the LRU node, rewrites page_table, buffer_ranges and the
    // device BDA table - and the run fail-fasted (0xC0000409) the moment the threshold was first
    // crossed. Splitting the phases removes that entire class of question rather than arguing
    // about which container tolerated it.
    if (too_many_backing) {
        static u64 last_gc_tick = 0;
        const u64 keep = BackingGcKeep();
        const u64 live_now = Postmortem::g_res_backing_bufs.load(std::memory_order_relaxed);
        const u64 over = live_now > keep ? live_now - keep : 0;
        // Cadence scales with the overshoot. A fixed "64 every 32 ticks" could not keep up with
        // a cache that grows in the thousands - measured kept=7215 against KEEP=1024 - so the
        // further we are over, the more often we run.
        const u64 interval = over > keep ? 1 : (over > keep / 4 ? 4 : 32);
        if (gc_tick - last_gc_tick >= interval) {
            last_gc_tick = gc_tick;
            // Aim at the target instead of a fixed slice, but never spend more than the time
            // budget below inside a frame.
            const u64 want = std::max<u64>(64, over);
            boost::container::small_vector<BufferId, 512> victims;
            {
                std::scoped_lock lk{buffers_mutex};
                u64 live = live_now;
                lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, [&](BufferId id) {
                    if (live <= keep || victims.size() >= want) {
                        return;
                    }
                    const Buffer& b = slot_buffers[id];
                    // (a) not touched recently, (b) nothing of it is still in flight, and it is
                    // neither already deleted nor pinned by a fault service in progress.
                    if (!b.is_backing_backed || b.is_deleted) {
                        return;
                    }
                    if (auto it = buffer_pins.find(id.index);
                        it != buffer_pins.end() && it->second != 0) {
                        return;
                    }
                    if (!scheduler.IsFree(b.content_tick) ||
                        !scheduler.IsFree(b.last_upload_tick)) {
                        return;
                    }
                    victims.push_back(id);
                    --live;
                });
            }
            // (c)+(d): DeleteBuffer -> Unregister clears page_table, buffer_ranges and the
            // device BDA entries BEFORE anything is destroyed, and the VkBuffer itself only
            // dies in DeferErase, after the tick retires and the pin count is zero. Held under
            // serve_mutex exactly like the sibling clean_up path, so a fault service cannot be
            // half-way through the same buffer.
            u64 collected = 0;
            const auto gc_t0 = std::chrono::steady_clock::now();
            static const u64 budget_us = [] {
                const char* e = std::getenv("SHADPS4_BACKIMP_GC_MS");
                const u64 v = e != nullptr ? std::strtoull(e, nullptr, 10) : 0ull;
                return (v != 0 ? v : 3ull) * 1000ull;
            }();
            if (!victims.empty()) {
                std::scoped_lock serve_lk{serve_mutex};
                for (const BufferId id : victims) {
                    if (slot_buffers[id].is_deleted) {
                        continue;
                    }
                    DeleteBuffer(id);
                    ++collected;
                    // Time budget: a frame hitch is worse than one more pass next tick, and
                    // with interval=1 while far over, the remainder is picked up immediately.
                    if ((collected & 31) == 0 &&
                        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - gc_t0)
                                             .count()) > budget_us) {
                        break;
                    }
                }
            }
            Postmortem::g_gc_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - gc_t0)
                                     .count()),
                std::memory_order_relaxed);
            static u64 gc_reported = 0;
            if (collected != 0 && gc_reported++ % 8 == 0) {
                LOG_CRITICAL(Render_Vulkan,
                             "BACKIMPGC: collected={} kept={} want={} over={} took={}us",
                             collected,
                             Postmortem::g_res_backing_bufs.load(std::memory_order_relaxed), want,
                             over,
                             std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - gc_t0)
                                 .count());
            }
        }
    }
    static u64 last_report = 0;
    if (gc_tick - last_report > 600) {
        last_report = gc_tick;
        LOG_CRITICAL(Render_Vulkan, "BUFGC: live={} -> {} (aggressive={})", live_buffers,
                     slot_buffers.size(), aggressive);
    }
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeferErase(BufferId buffer_id, u32 extra_ticks) {
    // [experiment] SHADPS4_NOFREE=1 never hands buffer memory back to the driver. If the device
    // stops dying, the fault comes from a page table entry that still points at a buffer whose
    // memory was already released - and the cure is to recycle buffers instead of destroying
    // them, so every published address stays mapped for the lifetime of the process.
    static const bool never_free = [] {
        const char* e = std::getenv("SHADPS4_NOFREE");
        return e && e[0] == '1';
    }();
    if (never_free) {
        BdaLogMark(buffer_id.index, scheduler.CurrentTick(), true);
        return;
    }
    scheduler.DeferOperation([this, buffer_id, extra_ticks] {
        if (extra_ticks > 0) {
            DeferErase(buffer_id, extra_ticks - 1);
            return;
        }
        {
            std::scoped_lock lk{buffers_mutex};
            // A pinned buffer is mid-download on the readback queue; push the
            // erase out by one more tick instead of pulling memory from under it.
            if (auto it = buffer_pins.find(buffer_id.index);
                it != buffer_pins.end() && it->second != 0) {
                DeferErase(buffer_id, 1);
                return;
            }
            BdaLogMark(buffer_id.index, scheduler.CurrentTick(), true);
            slot_buffers.erase(buffer_id);
            buffer_pins.erase(buffer_id.index);
        }
    });
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.is_import_backed) {
        // Demotion: marks and readbacks come back into force for the range
        // BEFORE any store lands in whatever VRAM buffer replaces this one.
        bool repromote = false;
        {
            std::scoped_lock gm{gpu_modified_mutex};
            label_import_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
            // [BACKIMP] A backing buffer costs no pinning budget and needs no promotion
            // economy: the next CreateBuffer over the range binds one again for free. Skip
            // the demote/ratchet bookkeeping (and its per-delete log) for those - only the
            // range subtraction above is shared.
            if (!buffer.is_backing_backed) {
                LOG_CRITICAL(Render_Vulkan, "LABELUMA: demoted {:#x}:{:#x}", buffer.CpuAddr(),
                             buffer.SizeBytes());
                // Ratchet: re-promote the same range; ResolveOverlaps will union it
                // with whatever VRAM buffer captured the pages, so the next import
                // owns both and the ping-pong converges. Bounded per arena.
                u32& count = label_repromote_counts[buffer.CpuAddr() >> 24];
                if (++count <= 4) {
                    repromote = true;
                } else {
                    label_failed_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes());
                    LOG_CRITICAL(Render_Vulkan,
                                 "LABELUMA: arena {:#x} ping-pongs, staying on readbacks",
                                 buffer.CpuAddr());
                }
            }
        }
        if (repromote) {
            RequestLabelPromotion(buffer.CpuAddr(), buffer.SizeBytes());
        }
    }
    Unregister(buffer_id);
    // [experiment] SHADPS4_DELAYFREE=N holds the buffer for N extra completed ticks.
    // VK_EXT_device_fault caught the GPU reading a buffer whose memory had already been
    // handed back; if extra ticks make the fault go away, the lifetime is the problem.
    static const u32 extra_ticks = [] {
        const char* e = std::getenv("SHADPS4_DELAYFREE");
        return e ? static_cast<u32>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    DeferErase(buffer_id, extra_ticks);
    buffer.is_deleted = true;
}

} // namespace VideoCore
