#include "ui/ui.hpp"

#include <algorithm>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "gfx/vk_context.hpp"
#include "util/log.hpp"

namespace spectro {

namespace {

void check_vk_result(VkResult result)
{
    if (result < 0)
        LOG_ERROR("Dear ImGui: a Vulkan call failed (VkResult %d)", static_cast<int>(result));
}

void apply_style(float scale)
{
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowRounding    = 6.0f;
    style.ChildRounding     = 4.0f;
    style.FrameRounding     = 4.0f;
    style.GrabRounding      = 4.0f;
    style.PopupRounding     = 4.0f;
    style.TabRounding       = 4.0f;
    style.WindowBorderSize  = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.WindowPadding     = ImVec2(12.0f, 10.0f);
    style.FramePadding      = ImVec2(7.0f, 4.0f);
    style.ItemSpacing       = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);
    style.IndentSpacing     = 14.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 10.0f;
    style.WindowTitleAlign  = ImVec2(0.0f, 0.5f);

    ImVec4* c = style.Colors;
    const ImVec4 accent(0.18f, 0.62f, 0.86f, 1.0f);
    c[ImGuiCol_WindowBg]             = ImVec4(0.055f, 0.065f, 0.085f, 0.90f);
    c[ImGuiCol_PopupBg]              = ImVec4(0.070f, 0.080f, 0.105f, 0.97f);
    c[ImGuiCol_Border]               = ImVec4(1.0f, 1.0f, 1.0f, 0.10f);
    c[ImGuiCol_TitleBg]              = ImVec4(0.070f, 0.085f, 0.115f, 0.95f);
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.085f, 0.110f, 0.150f, 0.98f);
    c[ImGuiCol_TitleBgCollapsed]     = ImVec4(0.060f, 0.070f, 0.095f, 0.80f);
    c[ImGuiCol_FrameBg]              = ImVec4(1.0f, 1.0f, 1.0f, 0.065f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(1.0f, 1.0f, 1.0f, 0.11f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    c[ImGuiCol_Header]               = ImVec4(1.0f, 1.0f, 1.0f, 0.07f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(accent.x, accent.y, accent.z, 0.30f);
    c[ImGuiCol_HeaderActive]         = ImVec4(accent.x, accent.y, accent.z, 0.45f);
    c[ImGuiCol_Button]               = ImVec4(1.0f, 1.0f, 1.0f, 0.085f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(accent.x, accent.y, accent.z, 0.45f);
    c[ImGuiCol_ButtonActive]         = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    c[ImGuiCol_SliderGrab]           = ImVec4(accent.x, accent.y, accent.z, 0.85f);
    c[ImGuiCol_SliderGrabActive]     = ImVec4(0.35f, 0.76f, 0.97f, 1.0f);
    c[ImGuiCol_CheckMark]            = ImVec4(0.35f, 0.76f, 0.97f, 1.0f);
    c[ImGuiCol_Separator]            = ImVec4(1.0f, 1.0f, 1.0f, 0.10f);
    c[ImGuiCol_Text]                 = ImVec4(0.90f, 0.92f, 0.95f, 1.0f);
    c[ImGuiCol_TextDisabled]         = ImVec4(0.55f, 0.59f, 0.66f, 1.0f);

    style.ScaleAllSizes(scale);
    style.FontSizeBase = 15.0f;
    style.FontScaleDpi = scale;
}

}  // namespace

ImGuiLayer::ImGuiLayer(GLFWwindow* window, VulkanContext& context, VkRenderPass render_pass,
                       VkSampleCountFlagBits samples)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // do not scatter imgui.ini files around
    io.LogFilename = nullptr;
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    // Chains to (and keeps) any GLFW callbacks installed before this point.
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // After the backend is initialised it knows whether we run on Wayland,
    // where the compositor does the scaling and this returns 1.
    ui_scale_ = std::clamp(ImGui_ImplGlfw_GetContentScaleForWindow(window), 1.0f, 4.0f);
    apply_style(ui_scale_);
    io.Fonts->AddFontDefaultVector();

    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion         = VK_API_VERSION_1_0;
    info.Instance           = context.instance;
    info.PhysicalDevice     = context.physical_device;
    info.Device             = context.device;
    info.QueueFamily        = context.queue_family;
    info.Queue              = context.queue;
    info.DescriptorPoolSize = 16;  // the backend creates its own small pool for font textures
    info.MinImageCount      = context.min_image_count;
    info.ImageCount         = std::max(static_cast<uint32_t>(context.swapchain_images.size()), context.min_image_count);
    info.PipelineInfoMain.RenderPass  = render_pass;
    info.PipelineInfoMain.Subpass     = 0;
    info.PipelineInfoMain.MSAASamples = samples;
    info.CheckVkResultFn    = check_vk_result;
    ImGui_ImplVulkan_Init(&info);
}

ImGuiLayer::~ImGuiLayer()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::begin_frame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(VkCommandBuffer cmd)
{
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data && draw_data->DisplaySize.x > 0.0f && draw_data->DisplaySize.y > 0.0f)
        ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
}

float ImGuiLayer::pixel_ratio() const
{
    const float ratio = ImGui::GetIO().DisplayFramebufferScale.x;
    return ratio > 0.0f ? ratio : 1.0f;
}

}  // namespace spectro
