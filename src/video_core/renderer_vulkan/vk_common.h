// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Include vulkan-hpp header
#define VK_ENABLE_BETA_EXTENSIONS
#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_SETTERS
#define VULKAN_HPP_HAS_SPACESHIP_OPERATOR
#define VULKAN_HPP_NO_EXCEPTIONS
// Define assert-on-result to nothing to instead return the result for our handling.
#define VULKAN_HPP_ASSERT_ON_RESULT

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#include <vulkan/vulkan.hpp>
#pragma clang diagnostic pop

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

// [BINDSKIP] Tracks the pipeline currently bound in the scheduler's command buffer, so a draw
// that would re-bind the same one can skip the call. Measured: 70452 binds per 5s window, ALL
// of them redundant.
//
// Correctness rests on invalidation, and the tracked state is per COMMAND BUFFER, not per pass:
// Vulkan keeps pipeline bindings across vkCmdBeginRendering/EndRendering but resets them when a
// command buffer begins. Twelve other places in the renderer (blits, detiling, FSR/PP passes,
// the fault manager) record their own bindPipeline into the SAME buffer between our draws, so
// each of them must invalidate too - otherwise we would skip a bind that is no longer current
// and draw with someone else's pipeline.
#include <atomic>
#include <cstdlib>
#include <cstdint>
namespace Vulkan {
inline std::atomic<std::uint64_t> g_bound_pipeline{0};
inline std::atomic<std::uint64_t> g_bind_skipped{0};
inline std::atomic<std::uint64_t> g_bind_issued{0};

inline void InvalidateBoundPipeline() {
    g_bound_pipeline.store(0, std::memory_order_relaxed);
}

inline bool BindSkipEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BINDSKIP");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

/// Returns true when the caller should issue vkCmdBindPipeline.
inline bool ShouldBindPipeline(vk::Pipeline pipeline) {
    const auto handle = reinterpret_cast<std::uint64_t>(static_cast<VkPipeline>(pipeline));
    if (BindSkipEnabled() && g_bound_pipeline.load(std::memory_order_relaxed) == handle &&
        handle != 0) {
        g_bind_skipped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_bound_pipeline.store(handle, std::memory_order_relaxed);
    g_bind_issued.fetch_add(1, std::memory_order_relaxed);
    return true;
}
} // namespace Vulkan
