// Dear ImGui integration: context, style, and the GLFW + Vulkan backends.
#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

struct GLFWwindow;

namespace spectro {

class VulkanContext;

class ImGuiLayer {
public:
    ImGuiLayer(GLFWwindow* window, VulkanContext& context, VkRenderPass render_pass,
               VkSampleCountFlagBits samples);
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&)            = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    // Starts a new user-interface frame.
    void begin_frame();
    // Draws the frame finished with ImGui::Render() into the current render pass.
    void render(VkCommandBuffer cmd);

    // Factor applied to user-interface sizes on high-DPI displays whose window
    // coordinates are physical pixels (X11, Windows).  1 where the windowing
    // system scales for us (Wayland, macOS).
    float ui_scale() const { return ui_scale_; }

    // Framebuffer pixels per user-interface coordinate unit.
    float pixel_ratio() const;

private:
    float ui_scale_ = 1.0f;
};

}  // namespace spectro
