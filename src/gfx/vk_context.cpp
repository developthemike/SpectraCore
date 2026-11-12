#include "gfx/vk_context.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <GLFW/glfw3.h>

#include "util/log.hpp"

namespace spectro {

namespace {

std::atomic<uint32_t> g_validation_errors{0};
std::atomic<uint32_t> g_validation_warnings{0};

const char* result_name(VkResult result)
{
    switch (result) {
    case VK_SUCCESS:                        return "VK_SUCCESS";
    case VK_NOT_READY:                      return "VK_NOT_READY";
    case VK_TIMEOUT:                        return "VK_TIMEOUT";
    case VK_INCOMPLETE:                     return "VK_INCOMPLETE";
    case VK_SUBOPTIMAL_KHR:                 return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_HOST_MEMORY:       return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:    return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:              return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:        return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:    return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:         return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_SURFACE_LOST_KHR:         return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:          return "VK_ERROR_OUT_OF_DATE_KHR";
    default:                                return "unknown VkResult";
    }
}

const char* device_type_name(VkPhysicalDeviceType type)
{
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "software renderer";
    default:                                     return "other";
    }
}

int device_type_score(VkPhysicalDeviceType type)
{
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
    default:                                     return 1;
    }
}

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* /*user*/)
{
    const char* message = data && data->pMessage ? data->pMessage : "(no message)";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        g_validation_errors.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("[vulkan validation] %s", message);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        g_validation_warnings.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("[vulkan validation] %s", message);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info()
{
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debug_callback;
    return info;
}

VkApplicationInfo application_info()
{
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "spectrogram";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName        = "spectrogram";
    app.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    // Only core Vulkan 1.0 functionality is used, so this runs on any driver.
    app.apiVersion         = VK_API_VERSION_1_0;
    return app;
}

}  // namespace

void vk_check(VkResult result, const char* what)
{
    if (result < 0)
        throw std::runtime_error(std::string("Vulkan: ") + what + " failed (" + result_name(result) + ")");
}

VulkanContext::VulkanContext(GLFWwindow* window_, const VulkanOptions& options)
    : window(window_), vsync_(options.vsync)
{
    try {
        create_instance(options.validation);
        vk_check(glfwCreateWindowSurface(instance, window, nullptr, &surface), "creating the window surface");
        pick_physical_device(options.gpu);
        create_device();
        if (!recreate_swapchain())
            throw std::runtime_error("Vulkan: the window has no drawable area");
    } catch (...) {
        destroy();
        throw;
    }
}

VulkanContext::~VulkanContext()
{
    destroy();
}

void VulkanContext::destroy()
{
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        destroy_swapchain_views();
        if (swapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(instance, surface, nullptr);
    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy)
            destroy(instance, debug_messenger_, nullptr);
    }
    if (instance != VK_NULL_HANDLE)
        vkDestroyInstance(instance, nullptr);

    swapchain = VK_NULL_HANDLE;
    command_pool = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    surface = VK_NULL_HANDLE;
    debug_messenger_ = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}

uint32_t VulkanContext::validation_errors() const
{
    return g_validation_errors.load(std::memory_order_relaxed);
}

uint32_t VulkanContext::validation_warnings() const
{
    return g_validation_warnings.load(std::memory_order_relaxed);
}

void VulkanContext::create_instance(bool want_validation)
{
    uint32_t     glfw_count      = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_count);
    if (!glfw_extensions) {
        throw std::runtime_error(
            "Vulkan cannot present to a window on this system (no Vulkan driver with surface support was found)");
    }
    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_count);
    std::vector<const char*> layers;

    bool validation = false;
    if (want_validation) {
        uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> available(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, available.data());
        for (const VkLayerProperties& layer : available) {
            if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
                validation = true;
        }
        if (validation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            LOG_INFO("Vulkan validation layers enabled");
        } else {
            LOG_WARN("--validate: the Vulkan validation layers are not installed; continuing without them");
        }
    }

    const VkApplicationInfo            app   = application_info();
    VkDebugUtilsMessengerCreateInfoEXT debug = debug_messenger_info();

    VkInstanceCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo        = &app;
    info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    info.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    info.ppEnabledLayerNames     = layers.data();
    if (validation)
        info.pNext = &debug;  // also covers instance creation and destruction
    vk_check(vkCreateInstance(&info, nullptr, &instance), "creating the instance");

    if (validation) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create)
            vk_check(create(instance, &debug, nullptr, &debug_messenger_), "creating the debug messenger");
    }
}

void VulkanContext::pick_physical_device(const std::string& selection)
{
    uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerating devices");
    std::vector<VkPhysicalDevice> devices(count);
    vk_check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerating devices");
    devices.resize(count);
    if (devices.empty())
        throw std::runtime_error("No Vulkan-capable graphics device was found");

    struct Candidate {
        VkPhysicalDevice           device;
        VkPhysicalDeviceProperties properties;
        uint32_t                   family = 0;
        bool                       usable = false;
        std::string                reason;
    };
    std::vector<Candidate> candidates;

    for (VkPhysicalDevice gpu : devices) {
        Candidate c{};
        c.device = gpu;
        vkGetPhysicalDeviceProperties(gpu, &c.properties);

        // A queue family that can both draw and present to our window.
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, families.data());
        bool found_family = false;
        for (uint32_t i = 0; i < family_count && !found_family; ++i) {
            VkBool32 can_present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &can_present);
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && can_present) {
                c.family     = i;
                found_family = true;
            }
        }

        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extension_count, extensions.data());
        const bool has_swapchain = std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
            return std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
        });

        uint32_t format_count = 0, mode_count = 0;
        if (found_family && has_swapchain) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &format_count, nullptr);
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &mode_count, nullptr);
        }

        if (!found_family)
            c.reason = "cannot present to this window";
        else if (!has_swapchain)
            c.reason = "no swapchain support";
        else if (format_count == 0 || mode_count == 0)
            c.reason = "no usable surface format";
        else
            c.usable = true;
        candidates.push_back(c);
    }

    const Candidate* chosen = nullptr;
    if (!selection.empty()) {
        const bool numeric = std::all_of(selection.begin(), selection.end(),
                                         [](unsigned char ch) { return std::isdigit(ch) != 0; });
        if (numeric) {
            const size_t index = static_cast<size_t>(std::strtoul(selection.c_str(), nullptr, 10));
            if (index >= candidates.size())
                throw std::runtime_error("--gpu " + selection + ": there are only " +
                                         std::to_string(candidates.size()) + " devices (see --list-gpus)");
            if (!candidates[index].usable)
                throw std::runtime_error(std::string("--gpu ") + selection + ": " +
                                         candidates[index].properties.deviceName + " " + candidates[index].reason);
            chosen = &candidates[index];
        } else {
            const std::string needle = to_lower(selection);
            for (const Candidate& c : candidates) {
                if (c.usable && to_lower(c.properties.deviceName).find(needle) != std::string::npos) {
                    chosen = &c;
                    break;
                }
            }
            if (!chosen)
                throw std::runtime_error("--gpu " + selection + ": no usable device matches (see --list-gpus)");
        }
    } else {
        // Prefer a discrete GPU, then integrated, then anything else; keep the
        // system's ordering among equals.
        for (const Candidate& c : candidates) {
            if (!c.usable)
                continue;
            if (!chosen ||
                device_type_score(c.properties.deviceType) > device_type_score(chosen->properties.deviceType))
                chosen = &c;
        }
        if (!chosen) {
            std::string detail;
            for (const Candidate& c : candidates)
                detail += std::string("\n  ") + c.properties.deviceName + ": " + c.reason;
            throw std::runtime_error("No graphics device can render to this window:" + detail);
        }
    }

    physical_device   = chosen->device;
    device_properties = chosen->properties;
    queue_family      = chosen->family;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    LOG_INFO("Rendering on %s (%s, Vulkan %u.%u.%u)", device_properties.deviceName,
             device_type_name(device_properties.deviceType), VK_VERSION_MAJOR(device_properties.apiVersion),
             VK_VERSION_MINOR(device_properties.apiVersion), VK_VERSION_PATCH(device_properties.apiVersion));
}

void VulkanContext::create_device()
{
    const float             priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount       = 1;
    queue_info.pQueuePriorities = &priority;

    const char*              extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceFeatures features{};  // no optional features are needed

    VkDeviceCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount    = 1;
    info.pQueueCreateInfos       = &queue_info;
    info.enabledExtensionCount   = 1;
    info.ppEnabledExtensionNames = extensions;
    info.pEnabledFeatures        = &features;
    vk_check(vkCreateDevice(physical_device, &info, nullptr, &device), "creating the device");
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pool{};
    pool.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = queue_family;
    vk_check(vkCreateCommandPool(device, &pool, nullptr, &command_pool), "creating the command pool");
}

void VulkanContext::destroy_swapchain_views()
{
    for (VkImageView view : swapchain_views)
        vkDestroyImageView(device, view, nullptr);
    swapchain_views.clear();
    swapchain_images.clear();
}

bool VulkanContext::recreate_swapchain()
{
    int fb_width = 0, fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    if (fb_width <= 0 || fb_height <= 0)
        return false;

    vk_check(vkDeviceWaitIdle(device), "waiting for the device");

    VkSurfaceCapabilitiesKHR caps{};
    vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps),
             "querying the surface");

    VkExtent2D size = caps.currentExtent;
    if (size.width == UINT32_MAX) {
        // The surface takes whatever size we give it (Wayland).
        size.width  = std::clamp(static_cast<uint32_t>(fb_width), caps.minImageExtent.width, caps.maxImageExtent.width);
        size.height = std::clamp(static_cast<uint32_t>(fb_height), caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (size.width == 0 || size.height == 0)
        return false;

    // The format is chosen once and then kept, since the render pass depends on it.
    if (swapchain == VK_NULL_HANDLE) {
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, formats.data()),
                 "querying surface formats");
        if (formats.empty())
            throw std::runtime_error("Vulkan: the surface offers no pixel formats");

        // An 8-bit UNORM target: shaders (and Dear ImGui) write display-ready
        // sRGB values and no conversion happens on output.
        surface_format = formats[0];
        bool found     = false;
        for (const VkFormat wanted : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
            for (const VkSurfaceFormatKHR& f : formats) {
                if (!found && f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    surface_format = f;
                    found          = true;
                }
            }
        }
        if (!found)
            LOG_WARN("No 8-bit UNORM surface format available; colours may look washed out");
    }

    present_mode = VK_PRESENT_MODE_FIFO_KHR;  // always available; synchronised to the display
    if (!vsync_) {
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, nullptr);
        std::vector<VkPresentModeKHR> modes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, modes.data());
        const auto has = [&modes](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
        if (has(VK_PRESENT_MODE_MAILBOX_KHR))
            present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
        else if (has(VK_PRESENT_MODE_IMMEDIATE_KHR))
            present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0)
        image_count = std::min(image_count, caps.maxImageCount);
    min_image_count = std::max<uint32_t>(2, caps.minImageCount);

    VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & composite)) {
        for (const VkCompositeAlphaFlagBitsKHR option :
             {VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
              VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
            if (caps.supportedCompositeAlpha & option) {
                composite = option;
                break;
            }
        }
    }

    swapchain_readable = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;

    VkSwapchainCreateInfoKHR info{};
    info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface          = surface;
    info.minImageCount    = image_count;
    info.imageFormat      = surface_format.format;
    info.imageColorSpace  = surface_format.colorSpace;
    info.imageExtent      = size;
    info.imageArrayLayers = 1;
    info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (swapchain_readable)
        info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // for screenshots
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform     = caps.currentTransform;
    info.compositeAlpha   = composite;
    info.presentMode      = present_mode;
    info.clipped          = VK_TRUE;
    info.oldSwapchain     = swapchain;

    VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
    vk_check(vkCreateSwapchainKHR(device, &info, nullptr, &new_swapchain), "creating the swapchain");

    destroy_swapchain_views();
    if (swapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = new_swapchain;
    extent    = size;

    uint32_t count = 0;
    vk_check(vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr), "querying swapchain images");
    swapchain_images.resize(count);
    vk_check(vkGetSwapchainImagesKHR(device, swapchain, &count, swapchain_images.data()),
             "querying swapchain images");
    swapchain_images.resize(count);

    swapchain_views.reserve(count);
    for (VkImage image : swapchain_images) {
        VkImageViewCreateInfo view{};
        view.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image                       = image;
        view.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        view.format                      = surface_format.format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        VkImageView handle = VK_NULL_HANDLE;
        vk_check(vkCreateImageView(device, &view, nullptr, &handle), "creating a swapchain image view");
        swapchain_views.push_back(handle);
    }
    return true;
}

uint32_t VulkanContext::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags properties) const
{
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return UINT32_MAX;
}

GpuBuffer VulkanContext::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags properties, bool map) const
{
    GpuBuffer buffer;
    buffer.size = size;

    VkBufferCreateInfo info{};
    info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size        = size;
    info.usage       = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &info, nullptr, &buffer.buffer), "creating a buffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
    const uint32_t type = find_memory_type(requirements.memoryTypeBits, properties);
    if (type == UINT32_MAX) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        throw std::runtime_error("Vulkan: no suitable memory type for a buffer");
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = requirements.size;
    alloc.memoryTypeIndex = type;
    vk_check(vkAllocateMemory(device, &alloc, nullptr, &buffer.memory), "allocating buffer memory");
    vk_check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "binding buffer memory");

    if (map)
        vk_check(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "mapping a buffer");
    return buffer;
}

void VulkanContext::destroy_buffer(GpuBuffer& buffer) const
{
    if (buffer.buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, buffer.memory, nullptr);  // also unmaps
    buffer = {};
}

GpuImage VulkanContext::create_image(uint32_t width, uint32_t height, VkFormat format,
                                     VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                                     VkImageAspectFlags aspect) const
{
    GpuImage image;
    image.format = format;
    image.width  = width;
    image.height = height;

    VkImageCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType     = VK_IMAGE_TYPE_2D;
    info.format        = format;
    info.extent        = {width, height, 1};
    info.mipLevels     = 1;
    info.arrayLayers   = 1;
    info.samples       = samples;
    info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    info.usage         = usage;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vk_check(vkCreateImage(device, &info, nullptr, &image.image), "creating an image");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image.image, &requirements);

    // Multisample targets are never read back: let tile-based GPUs skip the allocation.
    uint32_t type = UINT32_MAX;
    if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
        type = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    if (type == UINT32_MAX)
        type = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX)
        type = find_memory_type(requirements.memoryTypeBits, 0);
    if (type == UINT32_MAX) {
        vkDestroyImage(device, image.image, nullptr);
        throw std::runtime_error("Vulkan: no suitable memory type for an image");
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = requirements.size;
    alloc.memoryTypeIndex = type;
    vk_check(vkAllocateMemory(device, &alloc, nullptr, &image.memory), "allocating image memory");
    vk_check(vkBindImageMemory(device, image.image, image.memory, 0), "binding image memory");

    VkImageViewCreateInfo view{};
    view.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image                       = image.image;
    view.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    view.format                      = format;
    view.subresourceRange.aspectMask = aspect;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device, &view, nullptr, &image.view), "creating an image view");
    return image;
}

void VulkanContext::destroy_image(GpuImage& image) const
{
    if (image.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, image.view, nullptr);
    if (image.image != VK_NULL_HANDLE)
        vkDestroyImage(device, image.image, nullptr);
    if (image.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, image.memory, nullptr);
    image = {};
}

VkCommandBuffer VulkanContext::begin_setup_commands() const
{
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool        = command_pool;
    alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &alloc, &cmd), "allocating a command buffer");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &begin), "beginning a command buffer");
    return cmd;
}

void VulkanContext::end_setup_commands(VkCommandBuffer cmd) const
{
    vk_check(vkEndCommandBuffer(cmd), "ending a command buffer");
    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vk_check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "submitting setup commands");
    vk_check(vkQueueWaitIdle(queue), "waiting for setup commands");
    vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

bool VulkanContext::format_supports(VkFormat format, VkFormatFeatureFlags features) const
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
    return (properties.optimalTilingFeatures & features) == features;
}

int list_gpus()
{
    // Use the same instance extensions as the application itself: drivers (Mesa's
    // device-select layer in particular) order the devices differently once
    // they know which window system is in use, and the indices printed here
    // must be the ones --gpu accepts.
    uint32_t                 glfw_count      = 0;
    const char**             glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_count);
    std::vector<const char*> extensions;
    if (glfw_extensions)
        extensions.assign(glfw_extensions, glfw_extensions + glfw_count);

    const VkApplicationInfo app = application_info();
    VkInstanceCreateInfo    info{};
    info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo        = &app;
    info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "No Vulkan implementation was found on this system.\n");
        return 1;
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    devices.resize(count);

    // The device the application would pick by itself (see pick_physical_device).
    int preferred = -1;
    std::vector<VkPhysicalDeviceProperties> properties(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        vkGetPhysicalDeviceProperties(devices[i], &properties[i]);
        if (preferred < 0 || device_type_score(properties[i].deviceType) >
                                 device_type_score(properties[static_cast<size_t>(preferred)].deviceType))
            preferred = static_cast<int>(i);
    }

    if (devices.empty())
        std::printf("No Vulkan devices found.\n");
    for (size_t i = 0; i < devices.size(); ++i) {
        const VkPhysicalDeviceProperties& p = properties[i];
        std::printf("%zu: %s  [%s, Vulkan %u.%u.%u]%s\n", i, p.deviceName, device_type_name(p.deviceType),
                    VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
                    VK_VERSION_PATCH(p.apiVersion), static_cast<int>(i) == preferred ? "  <- default" : "");
    }
    if (!devices.empty())
        std::printf("\nSelect one with --gpu <index> or --gpu <part of the name>.\n");

    vkDestroyInstance(instance, nullptr);
    return devices.empty() ? 1 : 0;
}

}  // namespace spectro
