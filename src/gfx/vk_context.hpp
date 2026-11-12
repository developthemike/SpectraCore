// Vulkan instance, device and swapchain for one GLFW window, plus small
// helpers for creating buffers and images.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

struct GLFWwindow;

namespace spectro {

struct VulkanOptions {
    bool        validation = false;  // enable VK_LAYER_KHRONOS_validation when installed
    bool        vsync      = true;
    std::string gpu;                 // index or part of a device name; empty = choose automatically
};

// Throws std::runtime_error (with the failing call) if `result` is an error.
void vk_check(VkResult result, const char* what);

struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
    void*          mapped = nullptr;  // non-null for host-visible buffers created with map = true
};

struct GpuImage {
    VkImage        image  = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view   = VK_NULL_HANDLE;
    VkFormat       format = VK_FORMAT_UNDEFINED;
    uint32_t       width  = 0;
    uint32_t       height = 0;
};

class VulkanContext {
public:
    VulkanContext(GLFWwindow* window, const VulkanOptions& options);
    ~VulkanContext();

    VulkanContext(const VulkanContext&)            = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // (Re)creates the swapchain to match the window's framebuffer.  Waits for
    // the device to go idle first.  Returns false if the window currently has
    // no area (minimised), in which case there is no swapchain to render to.
    bool recreate_swapchain();

    GpuBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                            bool map) const;
    void      destroy_buffer(GpuBuffer& buffer) const;

    GpuImage create_image(uint32_t width, uint32_t height, VkFormat format, VkSampleCountFlagBits samples,
                          VkImageUsageFlags usage, VkImageAspectFlags aspect) const;
    void     destroy_image(GpuImage& image) const;

    // For one-off setup work (initial uploads): records, submits and waits.
    VkCommandBuffer begin_setup_commands() const;
    void            end_setup_commands(VkCommandBuffer cmd) const;

    bool format_supports(VkFormat format, VkFormatFeatureFlags features) const;

    // Validation layer messages seen so far (always 0 without --validate).
    uint32_t validation_errors() const;
    uint32_t validation_warnings() const;
    bool     validation_active() const { return debug_messenger_ != VK_NULL_HANDLE; }

    // --- Instance / device ---
    GLFWwindow*                      window          = nullptr;
    VkInstance                       instance        = VK_NULL_HANDLE;
    VkSurfaceKHR                     surface         = VK_NULL_HANDLE;
    VkPhysicalDevice                 physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties       device_properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkDevice                         device       = VK_NULL_HANDLE;
    uint32_t                         queue_family = 0;
    VkQueue                          queue        = VK_NULL_HANDLE;
    VkCommandPool                    command_pool = VK_NULL_HANDLE;  // resettable command buffers

    // --- Swapchain ---
    VkSwapchainKHR           swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR       surface_format{};
    VkPresentModeKHR         present_mode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D               extent{};
    uint32_t                 min_image_count = 2;
    std::vector<VkImage>     swapchain_images;
    std::vector<VkImageView> swapchain_views;
    bool                     swapchain_readable = false;  // images can be copied from (screenshots)

private:
    void destroy();
    void create_instance(bool want_validation);
    void pick_physical_device(const std::string& selection);
    void create_device();
    void destroy_swapchain_views();
    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags properties) const;

    bool                     vsync_ = true;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
};

// Prints the Vulkan devices on this machine, numbered the way --gpu <index>
// counts them.  GLFW must be initialised.  Returns a process exit code.
int list_gpus();

}  // namespace spectro
