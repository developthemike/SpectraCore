// Pure Vulkan portability / extension-selection logic shared by
// VulkanContext::create_instance(), VulkanContext::create_device(), and
// list_gpus().  Has no Vulkan header dependency so unit tests always exercise
// it, including on headless Linux builds.
#pragma once

#include <string>
#include <vector>

namespace spectro {

inline constexpr const char* kVkKhrPortabilityEnumerationExtension      = "VK_KHR_portability_enumeration";
inline constexpr const char* kVkKhrGetPhysicalDeviceProperties2Extension = "VK_KHR_get_physical_device_properties2";
inline constexpr const char* kVkKhrPortabilitySubsetExtension           = "VK_KHR_portability_subset";
inline constexpr const char* kVkKhrSwapchainExtension                   = "VK_KHR_swapchain";
inline constexpr const char* kVkExtDebugUtilsExtension                  = "VK_EXT_debug_utils";

struct VulkanInstancePlan {
    std::vector<std::string> extensions;
    bool                     enumerate_portability = false;
};

// Plans the instance extensions and portability flag for vkCreateInstance.
// Enables VK_KHR_portability_enumeration (and the enumerate-portability flag)
// whenever the loader offers it, which is required for MoltenVK on macOS.
// Also enables VK_KHR_get_physical_device_properties2 when available alongside
// portability enumeration, as required by the Vulkan 1.0 spec whenever a device
// enables VK_KHR_portability_subset.
VulkanInstancePlan plan_vulkan_instance(const std::vector<std::string>& required_extensions,
                                        const std::vector<std::string>& available_extensions,
                                        bool                            enable_debug_utils);

// Plans the device extensions for vkCreateDevice: always requests
// VK_KHR_swapchain, and adds VK_KHR_portability_subset whenever the physical
// device exposes it (required by the Vulkan spec on portability drivers such as
// MoltenVK).
std::vector<std::string> plan_vulkan_device_extensions(const std::vector<std::string>& available_extensions);

}  // namespace spectro
