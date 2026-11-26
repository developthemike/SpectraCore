#include "gfx/vk_portability.hpp"

#include <algorithm>
#include <string_view>

namespace spectro {

namespace {

bool contains(const std::vector<std::string>& list, std::string_view name)
{
    return std::any_of(list.begin(), list.end(), [name](const std::string& entry) { return entry == name; });
}

void add_unique(std::vector<std::string>& list, std::string_view name)
{
    if (!contains(list, name))
        list.emplace_back(name);
}

}  // namespace

VulkanInstancePlan plan_vulkan_instance(const std::vector<std::string>& required_extensions,
                                        const std::vector<std::string>& available_extensions,
                                        bool                            enable_debug_utils)
{
    VulkanInstancePlan plan;
    for (const std::string& ext : required_extensions)
        add_unique(plan.extensions, ext);

    if (enable_debug_utils)
        add_unique(plan.extensions, kVkExtDebugUtilsExtension);

    if (contains(available_extensions, kVkKhrPortabilityEnumerationExtension)) {
        add_unique(plan.extensions, kVkKhrPortabilityEnumerationExtension);
        plan.enumerate_portability = true;
        // VK_KHR_portability_subset depends on VK_KHR_get_physical_device_properties2
        // on Vulkan 1.0 instances; enable it when the loader supports it.
        if (contains(available_extensions, kVkKhrGetPhysicalDeviceProperties2Extension))
            add_unique(plan.extensions, kVkKhrGetPhysicalDeviceProperties2Extension);
    }

    return plan;
}

std::vector<std::string> plan_vulkan_device_extensions(const std::vector<std::string>& available_extensions)
{
    std::vector<std::string> enabled;
    enabled.emplace_back(kVkKhrSwapchainExtension);
    if (contains(available_extensions, kVkKhrPortabilitySubsetExtension))
        enabled.emplace_back(kVkKhrPortabilitySubsetExtension);
    return enabled;
}

}  // namespace spectro
