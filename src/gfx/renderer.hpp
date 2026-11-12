// Draws the spectrogram: a height-field surface whose geometry and colour come
// straight from a ring-buffer texture of spectra, plus anti-aliased lines and
// an overlay (the user interface) on top.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/vk_context.hpp"
#include "util/math3d.hpp"

namespace spectro {

// One line segment.  The memory layout is the vertex format of the line pipeline.
struct LineInstance {
    float    start[3];
    float    width;       // pixels
    float    end[3];
    float    depth_bias;  // moves the line towards the camera, in depth-buffer units
    uint32_t color;       // R, G, B, A bytes in memory order
};
static_assert(sizeof(LineInstance) == 36, "LineInstance must match the line pipeline's vertex layout");

constexpr uint32_t pack_rgba(uint32_t r, uint32_t g, uint32_t b, uint32_t a = 255)
{
    return (a << 24) | (b << 16) | (g << 8) | r;
}

struct SceneParams {
    Mat4 view_proj;
    Vec3 eye;                    // camera position, or for an orthographic view the direction towards the camera
    bool orthographic = false;

    // World-space size of the plot.  It is centred on the origin in X and Z and
    // sits on the Y = 0 plane; the newest row is at +Z.
    float plot_width  = 2.0f;
    float plot_height = 0.5f;
    float plot_depth  = 2.4f;
    float column_phase = 0.0f;   // position of column 0 within its cell, 0..1 (see SpectrumEngine::column_frequency)

    float db_floor     = -100.0f;  // level drawn at height 0 and the bottom of the colour map
    float db_ceiling   = -20.0f;   // level drawn at full height and the top of the colour map
    float scroll_phase = 0.0f;     // 0..1 progress towards the next row (smooth scrolling)
    int   colormap     = 0;
    float relief       = 0.4f;     // 0 = flat colour, 1 = fully lit
    float fog          = 0.5f;     // how strongly old rows fade into the background
    bool  walls        = true;
    float background[3] = {0.02f, 0.025f, 0.035f};
};

class Renderer {
public:
    struct Config {
        int columns = 2048;  // spectrogram width (frequency bins)
        int rows    = 512;   // spectrogram depth (history)
        int msaa    = 4;     // requested sample count
    };

    static constexpr int kMaxRowsPerFrame = 64;
    static constexpr int kMaxLines        = 1 << 15;

    Renderer(VulkanContext& context, const Config& config);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    // ---- Frame protocol: begin_frame, [upload_rows], end_frame ---------------

    // Waits for a free frame slot and a swapchain image, recreating the
    // swapchain if the window changed.  Returns false when nothing can be drawn
    // right now (window minimised); skip the frame in that case.
    bool begin_frame();

    // Appends spectra (dBFS, `columns` values each, oldest first) to the history.
    // At most kMaxRowsPerFrame rows are accepted per frame; returns how many.
    int upload_rows(const float* rows, int count);

    // Records and submits the frame.  `draw_overlay` is called inside the render
    // pass, after the scene, to draw the user interface.
    void end_frame(const SceneParams& scene, const LineInstance* lines, int line_count,
                   const std::function<void(VkCommandBuffer)>& draw_overlay);

    // --------------------------------------------------------------------------

    // Saves the next rendered frame as a PNG.
    void request_screenshot(const std::string& path) { screenshot_path_ = path; }
    bool screenshot_supported() const { return context_.swapchain_readable; }

    VkRenderPass          render_pass() const { return render_pass_; }
    VkSampleCountFlagBits sample_count() const { return samples_; }
    VkExtent2D            extent() const { return context_.extent; }
    int                   columns() const { return columns_; }
    int                   rows() const { return rows_; }
    int                   head_row() const { return head_row_; }  // texture row of the newest spectrum
    uint64_t              frames_rendered() const { return frames_rendered_; }

    static constexpr int kFramesInFlight = 2;

private:
    struct Frame {
        VkCommandBuffer cmd             = VK_NULL_HANDLE;
        VkFence         fence           = VK_NULL_HANDLE;
        VkSemaphore     image_available = VK_NULL_HANDLE;
        VkDescriptorSet descriptors     = VK_NULL_HANDLE;
        GpuBuffer       globals;      // uniform buffer
        GpuBuffer       row_staging;  // spectra waiting to be copied into the texture
        GpuBuffer       lines;        // line instances
        int             pending_rows      = 0;
        int             first_texture_row = 0;
    };

    void create_render_pass();
    void create_descriptor_layout();
    void create_pipelines();
    void create_textures();
    void create_frames();
    void create_swapchain_resources();
    void destroy_swapchain_resources();
    bool recreate_swapchain();
    void destroy();

    void record_row_upload(VkCommandBuffer cmd, Frame& frame);
    void write_globals(Frame& frame, const SceneParams& scene);
    void record_screenshot_copy(VkCommandBuffer cmd);
    void save_screenshot(const std::string& path);

    VkShaderModule create_shader(const uint32_t* code, size_t bytes);

    VulkanContext& context_;
    const int      columns_;
    const int      rows_;

    VkSampleCountFlagBits samples_      = VK_SAMPLE_COUNT_1_BIT;
    VkFormat              depth_format_ = VK_FORMAT_D32_SFLOAT;

    VkRenderPass          render_pass_       = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout_   = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool_   = VK_NULL_HANDLE;
    VkPipeline            surface_pipeline_  = VK_NULL_HANDLE;
    VkPipeline            line_pipeline_     = VK_NULL_HANDLE;

    GpuImage  spectra_;                 // R32F ring buffer: one spectrum per row
    GpuImage  colormaps_;               // RGBA8: one colour map per row
    VkSampler nearest_sampler_ = VK_NULL_HANDLE;
    VkSampler linear_sampler_  = VK_NULL_HANDLE;

    // Recreated together with the swapchain
    GpuImage                   msaa_color_;
    GpuImage                   depth_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkSemaphore>   render_finished_;  // one per swapchain image
    int                        last_fb_width_  = 0;
    int                        last_fb_height_ = 0;
    bool                       needs_recreate_ = false;

    Frame    frames_[kFramesInFlight];
    int      frame_index_  = 0;
    uint32_t image_index_  = 0;
    bool     frame_active_ = false;

    int      head_row_        = 0;  // texture row holding the newest spectrum
    uint64_t frames_rendered_ = 0;

    std::string screenshot_path_;
    GpuBuffer   readback_;
};

}  // namespace spectro
