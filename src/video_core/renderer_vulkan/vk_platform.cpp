// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Include the vulkan platform specific header
#if defined(ANDROID)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(_WIN64)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#else
#define VK_USE_PLATFORM_WAYLAND_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <fmt/ranges.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace Vulkan {

static const char* const VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
static const char* const CRASH_DIAGNOSTIC_LAYER_NAME = "VK_LAYER_LUNARG_crash_diagnostic";

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data) {

    spdlog::level level{};
    switch (severity) {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
        level = spdlog::level::err;
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
        level = spdlog::level::info;
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        level = spdlog::level::debug;
        break;
    default:
        level = spdlog::level::info;
    }

    LOG_GENERIC(Common::Log::Class::Render_Vulkan, level, "{}: {}",
                callback_data->pMessageIdName ? callback_data->pMessageIdName : "<null>",
                callback_data->pMessage ? callback_data->pMessage : "<null>");

    return VK_FALSE;
}

vk::SurfaceKHR CreateSurface(vk::Instance instance, const Frontend::WindowSDL& emu_window) {
    const auto& window_info = emu_window.GetWindowInfo();
    vk::SurfaceKHR surface{};

#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if (window_info.type == Frontend::WindowSystemType::Windows) {
        const vk::Win32SurfaceCreateInfoKHR win32_ci = {
            .hinstance = nullptr,
            .hwnd = static_cast<HWND>(window_info.render_surface),
        };

        if (instance.createWin32SurfaceKHR(&win32_ci, nullptr, &surface) != vk::Result::eSuccess) {
            LOG_CRITICAL(Render_Vulkan, "Failed to initialize Win32 surface");
            UNREACHABLE();
        }
    }
#elif defined(VK_USE_PLATFORM_XLIB_KHR) || defined(VK_USE_PLATFORM_WAYLAND_KHR)
    if (window_info.type == Frontend::WindowSystemType::X11) {
        const vk::XlibSurfaceCreateInfoKHR xlib_ci = {
            .dpy = static_cast<Display*>(window_info.display_connection),
            .window = reinterpret_cast<Window>(window_info.render_surface),
        };

        if (instance.createXlibSurfaceKHR(&xlib_ci, nullptr, &surface) != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Xlib surface");
            UNREACHABLE();
        }
    } else if (window_info.type == Frontend::WindowSystemType::Wayland) {
        if (EmulatorSettings.IsRenderdocEnabled()) {
            LOG_ERROR(Render_Vulkan,
                      "RenderDoc is not compatible with Wayland, use an X11 window instead.");
        }

        const vk::WaylandSurfaceCreateInfoKHR wayland_ci = {
            .display = static_cast<wl_display*>(window_info.display_connection),
            .surface = static_cast<wl_surface*>(window_info.render_surface),
        };

        if (instance.createWaylandSurfaceKHR(&wayland_ci, nullptr, &surface) !=
            vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Wayland surface");
            UNREACHABLE();
        }
    }
#elif defined(VK_USE_PLATFORM_METAL_EXT)
    if (window_info.type == Frontend::WindowSystemType::Metal) {
        const vk::MetalSurfaceCreateInfoEXT macos_ci = {
            .pLayer = static_cast<const CAMetalLayer*>(window_info.render_surface),
        };

        if (instance.createMetalSurfaceEXT(&macos_ci, nullptr, &surface) != vk::Result::eSuccess) {
            LOG_CRITICAL(Render_Vulkan, "Failed to initialize MacOS surface");
            UNREACHABLE();
        }
    }
#endif

    if (!surface) {
        LOG_CRITICAL(Render_Vulkan, "Presentation not supported on this platform");
        UNREACHABLE();
    }

    return surface;
}

static auto GetLayerExtensions(std::vector<const char*>&& extensions,
                               const std::vector<const char*>& layers) {
    auto all_missing_vk_settings = true;

    for (const auto& layer_name : layers) {
        const auto [layer_properties_result, layer_extensions] =
            vk::enumerateInstanceExtensionProperties(std::string(layer_name));
        if (layer_properties_result != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "Failed to query extension properties of {}: {}", layer_name,
                      vk::to_string(layer_properties_result));
        }
        auto found = false;
        for (const auto& extension : layer_extensions) {
            if (extension.extensionName == std::string_view(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME)) {
                found = true;
                all_missing_vk_settings = false;
                break;
            }
        }
        if (!found) {
            LOG_ERROR(Render_Vulkan, "Settings for layer {} not available.", layer_name);
        }
    }

    if (!all_missing_vk_settings) {
        extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    }

    return extensions;
}

std::vector<const char*> GetInstanceExtensions(Frontend::WindowSystemType window_type,
                                               bool enable_debug_utils) {
    const auto [properties_result, properties] = vk::enumerateInstanceExtensionProperties();
    if (properties_result != vk::Result::eSuccess || properties.empty()) {
        LOG_ERROR(Render_Vulkan, "Failed to query extension properties: {}",
                  vk::to_string(properties_result));
        return {};
    }

    // Add the windowing system specific extension
    std::vector<const char*> extensions;
    extensions.reserve(7);

    switch (window_type) {
    case Frontend::WindowSystemType::Headless:
        break;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    case Frontend::WindowSystemType::Windows:
        extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        break;
#elif defined(VK_USE_PLATFORM_XLIB_KHR) || defined(VK_USE_PLATFORM_WAYLAND_KHR)
    case Frontend::WindowSystemType::X11:
        extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
        break;
    case Frontend::WindowSystemType::Wayland:
        extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
        break;
#elif defined(VK_USE_PLATFORM_METAL_EXT)
    case Frontend::WindowSystemType::Metal:
        extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
        break;
#endif
    default:
        LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
        break;
    }

    if (window_type != Frontend::WindowSystemType::Headless) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    }

    if (EmulatorSettings.IsHdrAllowed()) {
        extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }

    if (enable_debug_utils) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Sanitize extension list
    std::erase_if(extensions, [&](const char* extension) -> bool {
        const auto it =
            std::find_if(properties.begin(), properties.end(), [extension](const auto& prop) {
                return std::strcmp(extension, prop.extensionName) == 0;
            });

        if (it == properties.end()) {
            LOG_INFO(Render_Vulkan, "Candidate instance extension {} is not available", extension);
            return true;
        }
        return false;
    });

    return extensions;
}

std::vector<const char*> GetInstanceLayers(bool enable_validation, bool enable_crash_diagnostic) {
    const auto [properties_result, properties] = vk::enumerateInstanceLayerProperties();
    if (properties_result != vk::Result::eSuccess || properties.empty()) {
        LOG_ERROR(Render_Vulkan, "Failed to query layer properties: {}",
                  vk::to_string(properties_result));
        return {};
    }

    std::vector<const char*> layers;
    layers.reserve(2);

    if (enable_validation) {
        layers.push_back(VALIDATION_LAYER_NAME);
    }
    if (enable_crash_diagnostic) {
        layers.push_back(CRASH_DIAGNOSTIC_LAYER_NAME);
    }

    // Sanitize layer list
    std::erase_if(layers, [&](const char* layer) -> bool {
        const auto it = std::ranges::find_if(properties, [layer](const auto& prop) {
            return std::strcmp(layer, prop.layerName) == 0;
        });
        if (it == properties.end()) {
            LOG_ERROR(Render_Vulkan, "Requested layer {} is not available", layer);
            return true;
        }
        return false;
    });

    return layers;
}

vk::UniqueInstance CreateInstance(Frontend::WindowSystemType window_type, bool enable_validation,
                                  bool enable_crash_diagnostic) {
    LOG_INFO(Render_Vulkan, "Creating vulkan instance");

#if defined(__APPLE__)
    // Initialize the environment with the path to the included ICD, so that the loader will
    // find it.
    static const auto icd_path = [] {
        char path[PATH_MAX];
        u32 length = PATH_MAX;
        _NSGetExecutablePath(path, &length);
        return std::filesystem::path(path).parent_path();
    }();
    setenv("VK_DRIVER_FILES", icd_path.c_str(), true);
#endif

    static vk::detail::DynamicLoader dl;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

    const auto [available_version_result, available_version] =
        VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion
            ? vk::enumerateInstanceVersion()
            : vk::ResultValue(vk::Result::eSuccess, VK_API_VERSION_1_0);
    ASSERT_MSG(available_version_result == vk::Result::eSuccess,
               "Failed to query Vulkan API version: {}", vk::to_string(available_version_result));
    ASSERT_MSG(available_version >= TargetVulkanApiVersion,
               "Vulkan {}.{} is required, but only {}.{} is supported by instance!",
               VK_VERSION_MAJOR(TargetVulkanApiVersion), VK_VERSION_MINOR(TargetVulkanApiVersion),
               VK_VERSION_MAJOR(available_version), VK_VERSION_MINOR(available_version));

    const auto layers = GetInstanceLayers(enable_validation, enable_crash_diagnostic);
    const auto extensions = GetLayerExtensions(GetInstanceExtensions(window_type, true), layers);

    const vk::ApplicationInfo application_info = {
        .pApplicationName = "shadPS4",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "shadPS4 Vulkan",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = available_version,
    };

    const std::string extensions_string = fmt::format("{}", fmt::join(extensions, ", "));
    const std::string layers_string = fmt::format("{}", fmt::join(layers, ", "));
    LOG_INFO(Render_Vulkan, "Enabled instance extensions: {}", extensions_string);
    LOG_INFO(Render_Vulkan, "Enabled instance layers: {}", layers_string);

    // Validation settings
    vk::Bool32 enable_core = EmulatorSettings.IsVkValidationCoreEnabled() ? vk::True : vk::False;
    vk::Bool32 enable_sync = EmulatorSettings.IsVkValidationSyncEnabled() ? vk::True : vk::False;
    vk::Bool32 enable_gpuav = EmulatorSettings.IsVkValidationGpuEnabled() ? vk::True : vk::False;

    // Crash diagnostics settings
    static const auto crash_diagnostic_path =
        Common::FS::GetUserPathString(Common::FS::PathType::LogDir);
    const char* log_path = crash_diagnostic_path.c_str();
    vk::Bool32 enable_force_barriers = vk::True;

    const std::array layer_setings = {
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "validate_core",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_core,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "validate_sync",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_sync,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "syncval_submit_time_validation",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_sync,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_enable",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_descriptor_checks",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_buffers_validation",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_indirect_draws_buffers",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_indirect_dispatches_buffers",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_indirect_trace_rays_buffers",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = VALIDATION_LAYER_NAME,
            .pSettingName = "gpuav_buffer_copies",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_gpuav,
        },
        vk::LayerSettingEXT{
            .pLayerName = "lunarg_crash_diagnostic",
            .pSettingName = "output_path",
            .type = vk::LayerSettingTypeEXT::eString,
            .valueCount = 1,
            .pValues = &log_path,
        },
        vk::LayerSettingEXT{
            .pLayerName = "lunarg_crash_diagnostic",
            .pSettingName = "sync_after_commands",
            .type = vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = &enable_force_barriers,
        },
    };

    vk::StructureChain<vk::InstanceCreateInfo, vk::LayerSettingsCreateInfoEXT> instance_ci_chain = {
        vk::InstanceCreateInfo{
            .pApplicationInfo = &application_info,
            .enabledLayerCount = static_cast<u32>(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<u32>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        },
        vk::LayerSettingsCreateInfoEXT{
            .settingCount = layer_setings.size(),
            .pSettings = layer_setings.data(),
        },
    };

    auto [instance_result, instance] = vk::createInstanceUnique(instance_ci_chain.get());
    ASSERT_MSG(instance_result == vk::Result::eSuccess, "Failed to create instance: {}",
               vk::to_string(instance_result));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    return std::move(instance);
}

// [R8 ADDRBIND] SHADPS4_ADDRBIND=1: lock-free ring of driver bind/unbind events, written from
// the driver callback thread, read once at DEVFAULT. Answers whether the faulting shader-heap VA
// was ever bound (H1 wild jump) or was unbound just before the fault (H3 WDDM residency churn).
bool AddrBindEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_ADDRBIND");
        return e && e[0] == '1';
    }();
    return on;
}

namespace AddrBind {
struct Entry {
    u64 base;
    u64 size;
    u32 unbind;
    u32 obj_type;
    u64 handle; // [ATTRIB] VkObject handle of pObjects[0]
    u64 tick;   // [ATTRIB] draw-scheduler tick current at the event (g_draw_tick)
    u64 seq;    // [ATTRIB] delivery number, so the dump can order bind/unbind pairs
};
// [ATTRIB] 06.09: the ring used to drop every base >= 0x4'0000'0000 - i.e. every buffer/image
// allocation - so a DEVFAULT ReadInvalid in that space (0x46..., 0x4d44fa000) could never be
// attributed. Keep everything; the ring is large enough for the churn (~300 reg/unreg per 30
// ticks) and DumpCovering names the object(s) whose [base, base+size) covers a fault address.
constexpr u32 kRingSize = 65536; // power of two
static std::array<Entry, kRingSize> ring{};
static std::atomic<u32> head{0};
static std::atomic<u64> total_events{0}; // all deliveries
std::atomic<u64> unbind_shader_region{0};
std::atomic<u64> g_draw_tick{0}; // set by Rasterizer::Draw/Dispatch (scheduler.CurrentTick())

struct CoverInfo {
    u64 base;
    u64 size;
    bool found;
};
CoverInfo DumpCovering(u64 addr) {
    CoverInfo info{};
    const u32 h = head.load(std::memory_order_relaxed);
    const u32 total = std::min<u32>(h, kRingSize);
    u32 printed = 0;
    for (u32 n = 1; n <= total && printed < 24; ++n) {
        const Entry& e = ring[(h - n) & (kRingSize - 1)];
        if (e.base == 0 || addr < e.base || addr >= e.base + e.size) {
            continue;
        }
        LOG_CRITICAL(Render_Vulkan,
                     "ADDRBIND-COVER {:#x}: {} base={:#x} size={:#x} objtype={} handle={:#x} "
                     "tick={} seq={} (age {} events)",
                     addr, e.unbind ? "unbind" : "bind", e.base, e.size, e.obj_type, e.handle,
                     e.tick, e.seq, h - (h - n));
        ++printed;
        if (!info.found) {
            info = CoverInfo{e.base, e.size, true};
        }
    }
    if (printed == 0) {
        LOG_CRITICAL(Render_Vulkan, "ADDRBIND-COVER {:#x}: no bind/unbind in the ring covers it",
                     addr);
    }
    return info;
}

u64 TotalEvents() {
    return total_events.load(std::memory_order_relaxed);
}

// [BINDLOG] the VA history of one Vulkan object handle (newest first): where the image a
// draw actually bound lives, next to the fault address and the unbinds that cover it.
static std::mutex live_mutex;
static std::unordered_map<u64, std::pair<u64, u64>> live_by_handle;

void DumpHandle(u64 handle) {
    {
        std::scoped_lock lk{live_mutex};
        if (const auto it = live_by_handle.find(handle); it != live_by_handle.end()) {
            LOG_CRITICAL(Render_Vulkan, "ADDRBIND-LIVE {:#x}: base={:#x} size={:#x} end={:#x}",
                         handle, it->second.first, it->second.second,
                         it->second.first + it->second.second);
        } else {
            LOG_CRITICAL(Render_Vulkan, "ADDRBIND-LIVE {:#x}: not bound now", handle);
        }
    }
    const u32 h = head.load(std::memory_order_relaxed);
    const u32 total = std::min<u32>(h, kRingSize);
    u32 printed = 0;
    for (u32 n = 1; n <= total && printed < 8; ++n) {
        const Entry& e = ring[(h - n) & (kRingSize - 1)];
        if (e.handle != handle) {
            continue;
        }
        LOG_CRITICAL(Render_Vulkan,
                     "ADDRBIND-HANDLE {:#x}: {} base={:#x} size={:#x} objtype={} tick={} seq={} "
                     "(age {} events)",
                     handle, e.unbind ? "unbind" : "bind", e.base, e.size, e.obj_type, e.tick,
                     e.seq, n);
        ++printed;
    }
    if (printed == 0) {
        LOG_CRITICAL(Render_Vulkan, "ADDRBIND-HANDLE {:#x}: no event in the ring", handle);
    }
}

void DumpRing() {
    const u32 h = head.load(std::memory_order_relaxed);
    const u32 total = std::min<u32>(h, kRingSize);
    LOG_CRITICAL(Render_Vulkan, "ADDRBIND-SUM: unbind_shader_region={} ring_total={} events={}",
                 unbind_shader_region.load(std::memory_order_relaxed), h,
                 total_events.load(std::memory_order_relaxed));
    if (h == 0) {
        // Distinguishes "no events arrived" from "dump not wired up".
        LOG_CRITICAL(Render_Vulkan, "ADDRBIND-RING: empty");
        return;
    }
    u32 printed = 0;
    for (u32 n = 1; n <= total && printed < 256; ++n) {
        const Entry& e = ring[(h - n) & (kRingSize - 1)];
        if (e.base == 0) {
            continue;
        }
        LOG_CRITICAL(Render_Vulkan,
                     "ADDRBIND[{}]: {} base={:#x} size={:#x} objtype={} handle={:#x} tick={}", n,
                     e.unbind ? "unbind" : "bind", e.base, e.size, e.obj_type, e.handle, e.tick);
        ++printed;
    }
}
} // namespace AddrBind

static VKAPI_ATTR VkBool32 VKAPI_CALL AddrBindCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT, vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT* cb_data, void*) {
    // Walk the pNext chain honestly for the address-binding payload.
    const auto* chain = static_cast<const vk::BaseInStructure*>(cb_data->pNext);
    const vk::DeviceAddressBindingCallbackDataEXT* bind = nullptr;
    while (chain != nullptr) {
        if (chain->sType == vk::StructureType::eDeviceAddressBindingCallbackDataEXT) {
            bind = reinterpret_cast<const vk::DeviceAddressBindingCallbackDataEXT*>(chain);
            break;
        }
        chain = chain->pNext;
    }
    if (bind == nullptr) {
        return VK_FALSE;
    }
    const u64 base = bind->baseAddress;
    // Verification: the first 4 deliveries are logged UNCONDITIONALLY (before any filter) -
    // at startup pipeline creation must shower binds; silence here means the driver emits none.
    const u64 ev = AddrBind::total_events.fetch_add(1, std::memory_order_relaxed);
    if (ev < 4) {
        LOG_CRITICAL(Render_Vulkan, "ADDRBIND event[{}]: {} base={:#x} size={:#x}", ev,
                     bind->bindingType == vk::DeviceAddressBindingTypeEXT::eUnbind ? "unbind"
                                                                                   : "bind",
                     base, bind->size);
    }
    const bool unbind = bind->bindingType == vk::DeviceAddressBindingTypeEXT::eUnbind;
    if (unbind && base >= 0x2'0000'0000ULL && base < 0x2'0010'0000ULL) {
        AddrBind::unbind_shader_region.fetch_add(1, std::memory_order_relaxed);
    }
    const u32 i = AddrBind::head.fetch_add(1, std::memory_order_relaxed) & (AddrBind::kRingSize - 1);
    const u32 obj_type =
        cb_data->objectCount ? static_cast<u32>(cb_data->pObjects[0].objectType) : 0u;
    const u64 handle = cb_data->objectCount ? cb_data->pObjects[0].objectHandle : 0ull;
    AddrBind::ring[i] = {base,   bind->size, unbind ? 1u : 0u, obj_type, handle,
                         AddrBind::g_draw_tick.load(std::memory_order_relaxed), ev};
    // [BINDLOG] live handle -> VA map: the ring forgets binds older than 65536 events (level
    // load), the map does not. Objects are keyed by handle; a handle value is reused by the
    // driver only after the previous object's unbind erased it.
    if (handle != 0) {
        std::scoped_lock lk{AddrBind::live_mutex};
        if (unbind) {
            AddrBind::live_by_handle.erase(handle);
        } else {
            AddrBind::live_by_handle[handle] = {base, bind->size};
        }
    }
    return VK_FALSE;
}

vk::UniqueDebugUtilsMessengerEXT CreateAddrBindMessenger(vk::Instance instance) {
    const vk::DebugUtilsMessengerCreateInfoEXT ci = {
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding,
        .pfnUserCallback = AddrBindCallback,
    };
    auto [res, m] = instance.createDebugUtilsMessengerEXTUnique(ci);
    if (res != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "ADDRBIND: messenger failed: {}", vk::to_string(res));
        return {};
    }
    return std::move(m);
}

vk::UniqueDebugUtilsMessengerEXT CreateDebugCallback(vk::Instance instance) {
    const vk::DebugUtilsMessengerCreateInfoEXT msg_ci = {
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = DebugUtilsCallback,
    };
    auto [messenger_result, messenger] = instance.createDebugUtilsMessengerEXTUnique(msg_ci);
    ASSERT_MSG(messenger_result == vk::Result::eSuccess, "Failed to create debug callback: {}",
               vk::to_string(messenger_result));
    return std::move(messenger);
}

} // namespace Vulkan
