#include "gfx/renderer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <GLFW/glfw3.h>

#include "dsp/spectrum_engine.hpp"
#include "gfx/colormap.hpp"
#include "gfx/png_writer.hpp"
#include "util/log.hpp"

#include "shaders/line_frag_spv.h"
#include "shaders/line_vert_spv.h"
#include "shaders/surface_frag_spv.h"
#include "shaders/surface_vert_spv.h"

namespace spectro {

namespace {

// Mirrors the "Globals" uniform block in shaders/common.glsl (std140).
struct GlobalsUbo {
    float   view_proj[16];
    float   eye[4];
    float   origin[4];
    float   scale[4];
    float   level[4];
    int32_t grid[4];
    float   misc[4];
    float   viewport[4];
    float   background[4];
    float   light[4];
};
static_assert(sizeof(GlobalsUbo) == 208, "GlobalsUbo must match the std140 layout in common.glsl");

// Parts of the mesh, selected by a push constant (see surface.vert).
enum SurfacePart : int32_t { kPartSurface = 0, kPartFront = 1, kPartBack = 2, kPartLeft = 3, kPartRight = 4 };

constexpr float kNormalTilt     = 14.0f;  // how strongly level steps tilt the shading normal
constexpr float kWallBrightness = 0.62f;
constexpr float kAmbient        = 0.42f;

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
                   VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                   VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask               = src_access;
    barrier.dstAccessMask               = dst_access;
    barrier.oldLayout                   = old_layout;
    barrier.newLayout                   = new_layout;
    barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                       = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

}  // namespace

Renderer::Renderer(VulkanContext& context, const Config& config)
    : context_(context), columns_(config.columns), rows_(config.rows)
{
    const VkPhysicalDeviceLimits& limits = context_.device_properties.limits;
    if (columns_ < 2 || rows_ < kMaxRowsPerFrame)
        throw std::invalid_argument("Renderer: the spectrogram is too small");
    if (static_cast<uint32_t>(columns_) > limits.maxImageDimension2D ||
        static_cast<uint32_t>(rows_) > limits.maxImageDimension2D) {
        throw std::runtime_error("This GPU cannot hold a " + std::to_string(columns_) + " x " +
                                 std::to_string(rows_) + " spectrogram texture (limit " +
                                 std::to_string(limits.maxImageDimension2D) + "); reduce --bins / --history");
    }

    // Anti-aliasing: the largest supported sample count not above the request.
    const VkSampleCountFlags supported = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
    int samples = std::clamp(config.msaa, 1, 8);
    while (samples > 1 && !(supported & static_cast<VkSampleCountFlags>(samples)))
        samples /= 2;
    samples_ = static_cast<VkSampleCountFlagBits>(samples);
    if (samples != config.msaa)
        LOG_INFO("Using %dx anti-aliasing (%dx is not supported by this GPU)", samples, config.msaa);

    depth_format_ = VK_FORMAT_UNDEFINED;
    for (const VkFormat format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D16_UNORM}) {
        if (context_.format_supports(format, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            depth_format_ = format;
            break;
        }
    }
    if (depth_format_ == VK_FORMAT_UNDEFINED)
        throw std::runtime_error("Vulkan: no supported depth buffer format");

    try {
        create_render_pass();
        create_descriptor_layout();
        create_pipelines();
        create_textures();
        create_frames();
        create_swapchain_resources();
    } catch (...) {
        destroy();
        throw;
    }
}

Renderer::~Renderer()
{
    destroy();
}

void Renderer::destroy()
{
    VkDevice device = context_.device;
    if (device == VK_NULL_HANDLE)
        return;
    vkDeviceWaitIdle(device);

    destroy_swapchain_resources();
    context_.destroy_buffer(readback_);

    for (Frame& frame : frames_) {
        if (frame.fence != VK_NULL_HANDLE)
            vkDestroyFence(device, frame.fence, nullptr);
        if (frame.image_available != VK_NULL_HANDLE)
            vkDestroySemaphore(device, frame.image_available, nullptr);
        if (frame.cmd != VK_NULL_HANDLE)
            vkFreeCommandBuffers(device, context_.command_pool, 1, &frame.cmd);
        context_.destroy_buffer(frame.globals);
        context_.destroy_buffer(frame.row_staging);
        context_.destroy_buffer(frame.lines);
        frame = {};
    }

    if (descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
    if (nearest_sampler_ != VK_NULL_HANDLE)
        vkDestroySampler(device, nearest_sampler_, nullptr);
    if (linear_sampler_ != VK_NULL_HANDLE)
        vkDestroySampler(device, linear_sampler_, nullptr);
    context_.destroy_image(spectra_);
    context_.destroy_image(colormaps_);

    if (surface_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device, surface_pipeline_, nullptr);
    if (line_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device, line_pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    if (descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, descriptor_layout_, nullptr);
    if (render_pass_ != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, render_pass_, nullptr);

    descriptor_pool_ = VK_NULL_HANDLE;
    nearest_sampler_ = linear_sampler_ = VK_NULL_HANDLE;
    surface_pipeline_ = line_pipeline_ = VK_NULL_HANDLE;
    pipeline_layout_   = VK_NULL_HANDLE;
    descriptor_layout_ = VK_NULL_HANDLE;
    render_pass_       = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void Renderer::create_render_pass()
{
    const bool msaa = samples_ != VK_SAMPLE_COUNT_1_BIT;

    VkAttachmentDescription attachments[3] = {};
    // 0: colour (multisampled when anti-aliasing, otherwise the swapchain image itself)
    attachments[0].format         = context_.surface_format.format;
    attachments[0].samples        = samples_;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = msaa ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // 1: depth
    attachments[1].format         = depth_format_;
    attachments[1].samples        = samples_;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // 2: the swapchain image the multisampled colour is resolved into
    attachments[2].format         = context_.surface_format.format;
    attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    const VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference resolve_ref{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;
    subpass.pResolveAttachments     = msaa ? &resolve_ref : nullptr;

    // What this render pass must wait for before it clears its attachments:
    //  - the swapchain image being acquired (the submit waits for that semaphore
    //    at the colour-output stage), and
    //  - the previous frame being done with the multisample colour and depth
    //    images, which every frame shares.  Their last writes happen at the
    //    colour-output and late-fragment-test stages of that frame.
    VkSubpassDependency dependency{};
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0;
    dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo info{};
    info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = msaa ? 3u : 2u;
    info.pAttachments    = attachments;
    info.subpassCount    = 1;
    info.pSubpasses      = &subpass;
    info.dependencyCount = 1;
    info.pDependencies   = &dependency;
    vk_check(vkCreateRenderPass(context_.device, &info, nullptr, &render_pass_), "creating the render pass");
}

void Renderer::create_descriptor_layout()
{
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding         = 0;  // Globals
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding         = 1;  // spectra, read by the vertex shader
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[2].binding         = 2;  // colour maps
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 3;
    layout.pBindings    = bindings;
    vk_check(vkCreateDescriptorSetLayout(context_.device, &layout, nullptr, &descriptor_layout_),
             "creating the descriptor set layout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset     = 0;
    push.size       = sizeof(int32_t);

    VkPipelineLayoutCreateInfo info{};
    info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount         = 1;
    info.pSetLayouts            = &descriptor_layout_;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges    = &push;
    vk_check(vkCreatePipelineLayout(context_.device, &info, nullptr, &pipeline_layout_),
             "creating the pipeline layout");
}

VkShaderModule Renderer::create_shader(const uint32_t* code, size_t bytes)
{
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = bytes;
    info.pCode    = code;
    VkShaderModule module = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(context_.device, &info, nullptr, &module), "creating a shader module");
    return module;
}

void Renderer::create_pipelines()
{
    VkDevice device = context_.device;

    const VkShaderModule surface_vert = create_shader(surface_vert_spv, sizeof(surface_vert_spv));
    const VkShaderModule surface_frag = create_shader(surface_frag_spv, sizeof(surface_frag_spv));
    const VkShaderModule line_vert    = create_shader(line_vert_spv, sizeof(line_vert_spv));
    const VkShaderModule line_frag    = create_shader(line_frag_spv, sizeof(line_frag_spv));

    // State shared by both pipelines -------------------------------------------
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;  // set dynamically: the window can be resized
    viewport.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;  // the surface is also visible from below
    // With this winding the top of the surface is the front face (surface.frag
    // darkens back faces).
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = samples_;

    const VkDynamicState             dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dynamic_states;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = "main";

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_attachment;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkGraphicsPipelineCreateInfo info{};
    info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount          = 2;
    info.pStages             = stages;
    info.pVertexInputState   = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState      = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState   = &multisample;
    info.pDepthStencilState  = &depth;
    info.pColorBlendState    = &blend;
    info.pDynamicState       = &dynamic;
    info.layout              = pipeline_layout_;
    info.renderPass          = render_pass_;
    info.subpass             = 0;

    // Surface: opaque, depth tested, geometry generated in the vertex shader ----
    stages[0].module       = surface_vert;
    stages[1].module       = surface_frag;
    depth.depthTestEnable  = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp   = VK_COMPARE_OP_LESS;
    VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &surface_pipeline_);

    // Lines: alpha blended, depth tested but not written, one instance per segment
    if (result >= 0) {
        stages[0].module       = line_vert;
        stages[1].module       = line_frag;
        depth.depthWriteEnable = VK_FALSE;
        depth.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        blend_attachment.blendEnable         = VK_TRUE;
        blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
        blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;

        VkVertexInputBindingDescription binding{};
        binding.binding   = 0;
        binding.stride    = sizeof(LineInstance);
        binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        VkVertexInputAttributeDescription attributes[3] = {};
        attributes[0].location = 0;
        attributes[0].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[0].offset   = offsetof(LineInstance, start);
        attributes[1].location = 1;
        attributes[1].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[1].offset   = offsetof(LineInstance, end);
        attributes[2].location = 2;
        attributes[2].format   = VK_FORMAT_R8G8B8A8_UNORM;
        attributes[2].offset   = offsetof(LineInstance, color);

        vertex_input.vertexBindingDescriptionCount   = 1;
        vertex_input.pVertexBindingDescriptions      = &binding;
        vertex_input.vertexAttributeDescriptionCount = 3;
        vertex_input.pVertexAttributeDescriptions    = attributes;

        result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &line_pipeline_);
    }

    vkDestroyShaderModule(device, surface_vert, nullptr);
    vkDestroyShaderModule(device, surface_frag, nullptr);
    vkDestroyShaderModule(device, line_vert, nullptr);
    vkDestroyShaderModule(device, line_frag, nullptr);
    vk_check(result, "creating the graphics pipelines");
}

void Renderer::create_textures()
{
    VkDevice device = context_.device;

    // Spectra: one float per bin and row, holding dBFS.  The level scale is
    // applied in the shader, so changing it re-maps the entire history at once.
    spectra_ = context_.create_image(static_cast<uint32_t>(columns_), static_cast<uint32_t>(rows_),
                                     VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
                                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT);

    // Colour maps: 256 entries wide, one map per row.
    const int        map_count = colormap_count();
    const int        entries   = colormap_entries();
    const size_t     map_bytes = static_cast<size_t>(entries) * static_cast<size_t>(map_count) * 4;
    colormaps_ = context_.create_image(static_cast<uint32_t>(entries), static_cast<uint32_t>(map_count),
                                       VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT,
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                       VK_IMAGE_ASPECT_COLOR_BIT);
    GpuBuffer staging = context_.create_buffer(
        map_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    colormap_fill_rgba(static_cast<uint8_t*>(staging.mapped));

    VkCommandBuffer cmd = context_.begin_setup_commands();

    // Start with an empty history: every bin at the floor level.
    image_barrier(cmd, spectra_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkClearColorValue clear{};
    clear.float32[0] = SpectrumEngine::kFloorDb;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, spectra_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    image_barrier(cmd, spectra_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

    image_barrier(cmd, colormaps_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent                 = {static_cast<uint32_t>(entries), static_cast<uint32_t>(map_count), 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, colormaps_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image_barrier(cmd, colormaps_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    context_.end_setup_commands(cmd);
    context_.destroy_buffer(staging);

    VkSamplerCreateInfo sampler{};
    sampler.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter    = VK_FILTER_NEAREST;
    sampler.minFilter    = VK_FILTER_NEAREST;
    sampler.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device, &sampler, nullptr, &nearest_sampler_), "creating a sampler");
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    vk_check(vkCreateSampler(device, &sampler, nullptr, &linear_sampler_), "creating a sampler");
}

void Renderer::create_frames()
{
    VkDevice device = context_.device;

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kFramesInFlight},
    };
    VkDescriptorPoolCreateInfo pool{};
    pool.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets       = kFramesInFlight;
    pool.poolSizeCount = 2;
    pool.pPoolSizes    = pool_sizes;
    vk_check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool_), "creating the descriptor pool");

    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkDeviceSize row_bytes     = static_cast<VkDeviceSize>(columns_) * sizeof(float);

    for (Frame& frame : frames_) {
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool        = context_.command_pool;
        alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device, &alloc, &frame.cmd), "allocating a command buffer");

        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // the first wait must not block
        vk_check(vkCreateFence(device, &fence, nullptr, &frame.fence), "creating a fence");

        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vk_check(vkCreateSemaphore(device, &semaphore, nullptr, &frame.image_available), "creating a semaphore");

        frame.globals     = context_.create_buffer(sizeof(GlobalsUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, host, true);
        frame.row_staging = context_.create_buffer(row_bytes * kMaxRowsPerFrame, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                   host, true);
        frame.lines       = context_.create_buffer(sizeof(LineInstance) * kMaxLines,
                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host, true);

        VkDescriptorSetAllocateInfo set{};
        set.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set.descriptorPool     = descriptor_pool_;
        set.descriptorSetCount = 1;
        set.pSetLayouts        = &descriptor_layout_;
        vk_check(vkAllocateDescriptorSets(device, &set, &frame.descriptors), "allocating a descriptor set");

        const VkDescriptorBufferInfo globals_info{frame.globals.buffer, 0, sizeof(GlobalsUbo)};
        const VkDescriptorImageInfo  spectra_info{nearest_sampler_, spectra_.view,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo  colormap_info{linear_sampler_, colormaps_.view,
                                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet writes[3] = {};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = frame.descriptors;
            writes[i].dstBinding      = i;
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo    = &globals_info;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo     = &spectra_info;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo     = &colormap_info;
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }
}

void Renderer::create_swapchain_resources()
{
    VkDevice         device = context_.device;
    const VkExtent2D extent = context_.extent;
    const bool       msaa   = samples_ != VK_SAMPLE_COUNT_1_BIT;

    if (msaa) {
        msaa_color_ = context_.create_image(extent.width, extent.height, context_.surface_format.format, samples_,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                                            VK_IMAGE_ASPECT_COLOR_BIT);
    }
    depth_ = context_.create_image(extent.width, extent.height, depth_format_, samples_,
                                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                                   VK_IMAGE_ASPECT_DEPTH_BIT);

    for (VkImageView swapchain_view : context_.swapchain_views) {
        VkImageView views[3];
        uint32_t    count = 0;
        if (msaa) {
            views[count++] = msaa_color_.view;
            views[count++] = depth_.view;
            views[count++] = swapchain_view;
        } else {
            views[count++] = swapchain_view;
            views[count++] = depth_.view;
        }
        VkFramebufferCreateInfo info{};
        info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass      = render_pass_;
        info.attachmentCount = count;
        info.pAttachments    = views;
        info.width           = extent.width;
        info.height          = extent.height;
        info.layers          = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        vk_check(vkCreateFramebuffer(device, &info, nullptr, &framebuffer), "creating a framebuffer");
        framebuffers_.push_back(framebuffer);

        // A "rendering finished" semaphore per swapchain image: the presentation
        // engine may hold on to it until that image is acquired again.
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore handle = VK_NULL_HANDLE;
        vk_check(vkCreateSemaphore(device, &semaphore, nullptr, &handle), "creating a semaphore");
        render_finished_.push_back(handle);
    }

    glfwGetFramebufferSize(context_.window, &last_fb_width_, &last_fb_height_);
}

void Renderer::destroy_swapchain_resources()
{
    VkDevice device = context_.device;
    for (VkFramebuffer framebuffer : framebuffers_)
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    framebuffers_.clear();
    for (VkSemaphore semaphore : render_finished_)
        vkDestroySemaphore(device, semaphore, nullptr);
    render_finished_.clear();
    context_.destroy_image(msaa_color_);
    context_.destroy_image(depth_);
}

bool Renderer::recreate_swapchain()
{
    // Remember the size we reacted to even if the swapchain cannot be created
    // (zero area), so that we do not retry every frame for nothing.
    glfwGetFramebufferSize(context_.window, &last_fb_width_, &last_fb_height_);
    if (!context_.recreate_swapchain())  // waits for the device to go idle
        return false;
    destroy_swapchain_resources();
    create_swapchain_resources();
    needs_recreate_ = false;
    return true;
}

// ---------------------------------------------------------------------------
// Per frame
// ---------------------------------------------------------------------------

bool Renderer::begin_frame()
{
    assert(!frame_active_);

    int width = 0, height = 0;
    glfwGetFramebufferSize(context_.window, &width, &height);
    if (width <= 0 || height <= 0)
        return false;  // minimised
    if (needs_recreate_ || width != last_fb_width_ || height != last_fb_height_) {
        if (!recreate_swapchain())
            return false;
    }

    Frame& frame = frames_[frame_index_];
    vk_check(vkWaitForFences(context_.device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "waiting for a frame");

    for (int attempt = 0; attempt < 2; ++attempt) {
        const VkResult result = vkAcquireNextImageKHR(context_.device, context_.swapchain, UINT64_MAX,
                                                      frame.image_available, VK_NULL_HANDLE, &image_index_);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            if (!recreate_swapchain())
                return false;
            continue;
        }
        vk_check(result, "acquiring a swapchain image");
        if (result == VK_SUBOPTIMAL_KHR)
            needs_recreate_ = true;  // still presentable; fix it up next frame

        frame.pending_rows = 0;
        frame_active_      = true;
        return true;
    }
    return false;
}

int Renderer::upload_rows(const float* rows, int count)
{
    assert(frame_active_);
    Frame& frame = frames_[frame_index_];
    count        = std::min(count, kMaxRowsPerFrame - frame.pending_rows);
    if (count <= 0)
        return 0;

    const size_t row_bytes = static_cast<size_t>(columns_) * sizeof(float);
    std::memcpy(static_cast<uint8_t*>(frame.row_staging.mapped) + static_cast<size_t>(frame.pending_rows) * row_bytes,
                rows, static_cast<size_t>(count) * row_bytes);

    if (frame.pending_rows == 0)
        frame.first_texture_row = (head_row_ + 1) % rows_;
    frame.pending_rows += count;
    head_row_ = (head_row_ + count) % rows_;
    return count;
}

// Copies this frame's new spectra from the staging buffer into the ring-buffer
// texture.  The rows are consecutive, so they form at most two rectangles.
void Renderer::record_row_upload(VkCommandBuffer cmd, Frame& frame)
{
    if (frame.pending_rows == 0)
        return;

    image_barrier(cmd, spectra_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy regions[2] = {};
    uint32_t          region_count = 0;
    int               first        = frame.first_texture_row;
    int               remaining    = frame.pending_rows;
    VkDeviceSize      offset       = 0;
    while (remaining > 0 && region_count < 2) {
        const int          chunk = std::min(remaining, rows_ - first);
        VkBufferImageCopy& r     = regions[region_count++];
        r.bufferOffset                = offset;
        r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.imageSubresource.layerCount = 1;
        r.imageOffset                 = {0, first, 0};
        r.imageExtent                 = {static_cast<uint32_t>(columns_), static_cast<uint32_t>(chunk), 1};
        offset += static_cast<VkDeviceSize>(chunk) * static_cast<VkDeviceSize>(columns_) * sizeof(float);
        remaining -= chunk;
        first = 0;
    }
    vkCmdCopyBufferToImage(cmd, frame.row_staging.buffer, spectra_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           region_count, regions);

    image_barrier(cmd, spectra_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
}

void Renderer::write_globals(Frame& frame, const SceneParams& scene)
{
    const float column_width = scene.plot_width / static_cast<float>(columns_);
    const float span         = std::max(scene.db_ceiling - scene.db_floor, 1e-3f);
    const Vec3  light        = normalize(Vec3{-0.42f, 0.74f, 0.52f});

    GlobalsUbo u{};
    std::memcpy(u.view_proj, scene.view_proj.m, sizeof(u.view_proj));
    u.eye[0] = scene.eye.x;
    u.eye[1] = scene.eye.y;
    u.eye[2] = scene.eye.z;
    u.eye[3] = scene.orthographic ? 0.0f : 1.0f;

    u.origin[0] = -0.5f * scene.plot_width + scene.column_phase * column_width;
    u.origin[1] = 0.0f;
    u.origin[2] = 0.5f * scene.plot_depth;

    u.scale[0] = column_width;
    u.scale[1] = scene.plot_height;
    u.scale[2] = scene.plot_depth / static_cast<float>(rows_ - 1);
    u.scale[3] = kNormalTilt;

    u.level[0] = scene.db_floor;
    u.level[1] = 1.0f / span;
    u.level[2] = std::clamp(scene.relief, 0.0f, 1.0f);
    u.level[3] = std::clamp(scene.fog, 0.0f, 1.0f);

    u.grid[0] = columns_;
    u.grid[1] = rows_;
    u.grid[2] = head_row_;

    const int map = std::clamp(scene.colormap, 0, colormap_count() - 1);
    u.misc[0]     = std::clamp(scene.scroll_phase, 0.0f, 1.0f);
    u.misc[1]     = (static_cast<float>(map) + 0.5f) / static_cast<float>(colormap_count());
    u.misc[2]     = kWallBrightness;

    u.viewport[0] = static_cast<float>(context_.extent.width);
    u.viewport[1] = static_cast<float>(context_.extent.height);
    u.viewport[2] = 1.0f / u.viewport[0];
    u.viewport[3] = 1.0f / u.viewport[1];

    u.background[0] = scene.background[0];
    u.background[1] = scene.background[1];
    u.background[2] = scene.background[2];
    u.background[3] = 1.0f;

    u.light[0] = light.x;
    u.light[1] = light.y;
    u.light[2] = light.z;
    u.light[3] = kAmbient;

    std::memcpy(frame.globals.mapped, &u, sizeof(u));
}

void Renderer::end_frame(const SceneParams& scene, const LineInstance* lines, int line_count,
                         const std::function<void(VkCommandBuffer)>& draw_overlay)
{
    assert(frame_active_);
    Frame&           frame  = frames_[frame_index_];
    VkCommandBuffer  cmd    = frame.cmd;
    const VkExtent2D extent = context_.extent;

    write_globals(frame, scene);
    line_count = std::clamp(line_count, 0, kMaxLines);
    if (line_count > 0)
        std::memcpy(frame.lines.mapped, lines, static_cast<size_t>(line_count) * sizeof(LineInstance));

    vk_check(vkResetCommandBuffer(cmd, 0), "resetting a command buffer");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &begin), "beginning a command buffer");

    record_row_upload(cmd, frame);

    VkClearValue clear_values[3] = {};
    clear_values[0].color        = {{scene.background[0], scene.background[1], scene.background[2], 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo pass{};
    pass.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass        = render_pass_;
    pass.framebuffer       = framebuffers_[image_index_];
    pass.renderArea.extent = extent;
    pass.clearValueCount   = samples_ != VK_SAMPLE_COUNT_1_BIT ? 3u : 2u;
    pass.pClearValues      = clear_values;
    vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height),
                              0.0f, 1.0f};
    const VkRect2D   scissor{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &frame.descriptors, 0,
                            nullptr);

    // The surface: one instanced triangle strip per row of the history.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, surface_pipeline_);
    const auto draw_part = [&](int32_t part, int vertices, int instances) {
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(part), &part);
        vkCmdDraw(cmd, static_cast<uint32_t>(vertices), static_cast<uint32_t>(instances), 0, 0);
    };
    draw_part(kPartSurface, 2 * columns_, rows_);
    if (scene.walls) {
        draw_part(kPartFront, 2 * columns_, 1);
        draw_part(kPartBack, 2 * columns_, 1);
        draw_part(kPartLeft, 2 * (rows_ + 1), 1);
        draw_part(kPartRight, 2 * (rows_ + 1), 1);
    }

    if (line_count > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline_);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &frame.lines.buffer, &offset);
        vkCmdDraw(cmd, 4, static_cast<uint32_t>(line_count), 0, 0);
    }

    if (draw_overlay)
        draw_overlay(cmd);

    vkCmdEndRenderPass(cmd);

    const std::string screenshot = std::move(screenshot_path_);
    screenshot_path_.clear();
    const bool capture = !screenshot.empty() && context_.swapchain_readable;
    if (!screenshot.empty() && !capture)
        LOG_ERROR("Cannot take a screenshot: this display does not allow reading back its images");
    if (capture)
        record_screenshot_copy(cmd);

    vk_check(vkEndCommandBuffer(cmd), "ending a command buffer");

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo               submit{};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &frame.image_available;
    submit.pWaitDstStageMask    = &wait_stage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &render_finished_[image_index_];
    vk_check(vkResetFences(context_.device, 1, &frame.fence), "resetting a fence");
    vk_check(vkQueueSubmit(context_.queue, 1, &submit, frame.fence), "submitting a frame");

    if (capture) {
        vk_check(vkWaitForFences(context_.device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "waiting for a frame");
        save_screenshot(screenshot);
    }

    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &render_finished_[image_index_];
    present.swapchainCount     = 1;
    present.pSwapchains        = &context_.swapchain;
    present.pImageIndices      = &image_index_;
    const VkResult result      = vkQueuePresentKHR(context_.queue, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        needs_recreate_ = true;
    else
        vk_check(result, "presenting a frame");

    frame_active_ = false;
    frame_index_  = (frame_index_ + 1) % kFramesInFlight;
    ++frames_rendered_;
}

// ---------------------------------------------------------------------------
// Screenshots
// ---------------------------------------------------------------------------

void Renderer::record_screenshot_copy(VkCommandBuffer cmd)
{
    const VkExtent2D   extent = context_.extent;
    const VkDeviceSize bytes  = static_cast<VkDeviceSize>(extent.width) * extent.height * 4;
    if (readback_.size < bytes) {
        // Safe to replace: a previous screenshot waited for its frame to finish.
        context_.destroy_buffer(readback_);
        readback_ = context_.create_buffer(
            bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    }

    VkImage image = context_.swapchain_images[image_index_];
    image_barrier(cmd, image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent                 = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.buffer, 1, &region);

    image_barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_ACCESS_TRANSFER_READ_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void Renderer::save_screenshot(const std::string& path)
{
    const VkExtent2D extent = context_.extent;
    const VkFormat   format = context_.surface_format.format;
    const bool       bgr    = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    const bool       rgb    = format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB;
    if (!bgr && !rgb) {
        LOG_ERROR("Cannot save a screenshot: unsupported surface format %d", static_cast<int>(format));
        return;
    }

    const size_t         pixels = static_cast<size_t>(extent.width) * extent.height;
    const uint8_t*       src    = static_cast<const uint8_t*>(readback_.mapped);
    std::vector<uint8_t> out(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        out[3 * i + 0] = src[4 * i + (bgr ? 2 : 0)];
        out[3 * i + 1] = src[4 * i + 1];
        out[3 * i + 2] = src[4 * i + (bgr ? 0 : 2)];
    }
    if (write_png(path, out.data(), static_cast<int>(extent.width), static_cast<int>(extent.height)))
        LOG_INFO("Saved screenshot %s (%u x %u)", path.c_str(), extent.width, extent.height);
    else
        LOG_ERROR("Could not write screenshot %s", path.c_str());
}

}  // namespace spectro
