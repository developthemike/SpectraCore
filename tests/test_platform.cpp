#include <algorithm>
#include <string>
#include <vector>

#include "app/options.hpp"
#include "gfx/vk_portability.hpp"
#include "test_framework.hpp"
#include "util/platform.hpp"
#include "util/thread_util.hpp"

using namespace spectro;

namespace {

bool has_ext(const std::vector<std::string>& exts, const char* name)
{
    return std::find(exts.begin(), exts.end(), name) != exts.end();
}

size_t count_ext(const std::vector<std::string>& exts, const char* name)
{
    return static_cast<size_t>(std::count(exts.begin(), exts.end(), name));
}

}  // namespace

TEST(vulkan_instance_plan_enables_portability_on_moltenvk)
{
    const std::vector<std::string> required  = {"VK_KHR_surface", "VK_EXT_metal_surface"};
    const std::vector<std::string> available = {
        "VK_KHR_surface",
        "VK_EXT_metal_surface",
        kVkKhrPortabilityEnumerationExtension,
        kVkKhrGetPhysicalDeviceProperties2Extension,
        kVkExtDebugUtilsExtension,
    };

    const VulkanInstancePlan plan = plan_vulkan_instance(required, available, false);
    CHECK(plan.enumerate_portability);
    CHECK(has_ext(plan.extensions, "VK_KHR_surface"));
    CHECK(has_ext(plan.extensions, "VK_EXT_metal_surface"));
    CHECK(has_ext(plan.extensions, kVkKhrPortabilityEnumerationExtension));
    CHECK(has_ext(plan.extensions, kVkKhrGetPhysicalDeviceProperties2Extension));
    CHECK(!has_ext(plan.extensions, kVkExtDebugUtilsExtension));
}

TEST(vulkan_instance_plan_enables_debug_utils_without_duplicates)
{
    const std::vector<std::string> required = {
        "VK_KHR_surface",
        "VK_EXT_metal_surface",
        kVkKhrPortabilityEnumerationExtension,  // e.g. already supplied by a newer GLFW
    };
    const std::vector<std::string> available = {
        "VK_KHR_surface",
        "VK_EXT_metal_surface",
        kVkKhrPortabilityEnumerationExtension,
        kVkKhrGetPhysicalDeviceProperties2Extension,
        kVkExtDebugUtilsExtension,
    };

    const VulkanInstancePlan plan = plan_vulkan_instance(required, available, true);
    CHECK(plan.enumerate_portability);
    CHECK_EQ(count_ext(plan.extensions, kVkKhrPortabilityEnumerationExtension), size_t{1});
    CHECK_EQ(count_ext(plan.extensions, kVkKhrGetPhysicalDeviceProperties2Extension), size_t{1});
    CHECK_EQ(count_ext(plan.extensions, kVkExtDebugUtilsExtension), size_t{1});
}

TEST(vulkan_instance_plan_skips_portability_when_unavailable)
{
    const std::vector<std::string> required  = {"VK_KHR_surface", "VK_KHR_xcb_surface"};
    const std::vector<std::string> available = {"VK_KHR_surface", "VK_KHR_xcb_surface"};

    const VulkanInstancePlan plan = plan_vulkan_instance(required, available, false);
    CHECK(!plan.enumerate_portability);
    CHECK_EQ(plan.extensions.size(), size_t{2});
    CHECK(!has_ext(plan.extensions, kVkKhrPortabilityEnumerationExtension));
    CHECK(!has_ext(plan.extensions, kVkKhrGetPhysicalDeviceProperties2Extension));
}

TEST(vulkan_instance_plan_handles_portability_without_properties2)
{
    const std::vector<std::string> required  = {"VK_KHR_surface"};
    const std::vector<std::string> available = {"VK_KHR_surface", kVkKhrPortabilityEnumerationExtension};

    const VulkanInstancePlan plan = plan_vulkan_instance(required, available, false);
    CHECK(plan.enumerate_portability);
    CHECK(has_ext(plan.extensions, kVkKhrPortabilityEnumerationExtension));
    CHECK(!has_ext(plan.extensions, kVkKhrGetPhysicalDeviceProperties2Extension));
}

TEST(vulkan_device_plan_enables_portability_subset_when_exposed)
{
    const std::vector<std::string> moltenvk = {kVkKhrSwapchainExtension, kVkKhrPortabilitySubsetExtension};
    const auto                     enabled  = plan_vulkan_device_extensions(moltenvk);
    CHECK(has_ext(enabled, kVkKhrSwapchainExtension));
    CHECK(has_ext(enabled, kVkKhrPortabilitySubsetExtension));
    CHECK_EQ(enabled.size(), size_t{2});
}

TEST(vulkan_device_plan_skips_portability_subset_on_conformant_driver)
{
    const std::vector<std::string> mesa    = {kVkKhrSwapchainExtension, "VK_KHR_maintenance1"};
    const auto                     enabled = plan_vulkan_device_extensions(mesa);
    CHECK(has_ext(enabled, kVkKhrSwapchainExtension));
    CHECK(!has_ext(enabled, kVkKhrPortabilitySubsetExtension));
    CHECK_EQ(enabled.size(), size_t{1});
}

TEST(platform_modifier_conventions)
{
    CHECK(primary_modifier_held(true, false, HostPlatform::Linux));
    CHECK(!primary_modifier_held(false, true, HostPlatform::Linux));
    CHECK(!primary_modifier_held(false, false, HostPlatform::Linux));
    CHECK(std::string(primary_modifier_name(HostPlatform::Linux)) == "Ctrl");

    CHECK(primary_modifier_held(false, true, HostPlatform::MacOS));
    CHECK(primary_modifier_held(true, false, HostPlatform::MacOS));
    CHECK(!primary_modifier_held(false, false, HostPlatform::MacOS));
    CHECK(std::string(primary_modifier_name(HostPlatform::MacOS)) == "Cmd");
}

TEST(platform_window_backend_validation)
{
    CHECK(validate_window_platform("", HostPlatform::MacOS).empty());
    CHECK(!validate_window_platform("x11", HostPlatform::MacOS).empty());
    CHECK(!validate_window_platform("wayland", HostPlatform::MacOS).empty());

    CHECK(validate_window_platform("", HostPlatform::Linux).empty());
    CHECK(validate_window_platform("x11", HostPlatform::Linux).empty());
    CHECK(validate_window_platform("wayland", HostPlatform::Linux).empty());

    // End-to-end check through parse_options() for the host platform.
    Options     options;
    std::string error;
    char        prog[] = "spectrogram";
    char        flag[] = "--x11";
    char*       argv[] = {prog, flag};
    const bool  ok     = parse_options(2, argv, options, error);
    if (current_platform() == HostPlatform::MacOS) {
        CHECK(!ok);
        CHECK(!error.empty());
    } else {
        CHECK(ok);
        CHECK(options.platform == "x11");
    }
}

TEST(thread_utilities_work_on_this_platform)
{
    set_current_thread_name("spectro-test");
    const double t0 = current_thread_cpu_seconds();
    CHECK(t0 >= 0.0);

    volatile double acc = 0.0;
    for (int i = 0; i < 200000; ++i)
        acc += static_cast<double>(i) * 1e-6;
    (void)acc;

    const double t1 = current_thread_cpu_seconds();
    CHECK(t1 >= t0);
}
