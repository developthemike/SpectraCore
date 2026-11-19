#include "app/app.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gfx/vk_context.hpp"  // pulls in vulkan.h, which must precede glfw3.h

#include <GLFW/glfw3.h>
#include <imgui.h>

#include "app/pipeline.hpp"
#include "app/signals.hpp"
#include "audio/heard_channels.hpp"
#include "audio/listener.hpp"
#include "audio/transcriber.hpp"
#include "audio/trunk_follower.hpp"
#if SPECTRO_HAVE_AUDIO
#include "audio/audio_output.hpp"
#endif
#include "gfx/camera.hpp"
#include "gfx/colormap.hpp"
#include "gfx/renderer.hpp"
#include "radio/sim_source.hpp"
#include "ui/axes.hpp"
#include "ui/ui.hpp"
#include "util/log.hpp"
#include "util/units.hpp"

namespace spectro {

namespace {

using Clock = std::chrono::steady_clock;

constexpr float kPlotWidth   = 2.0f;  // world units; the frequency span always maps onto this
constexpr int   kMaxPending  = 256;   // rows buffered between the DSP thread and the screen
constexpr int   kPopBatch    = 64;
constexpr float kFloorDb     = SpectrumEngine::kFloorDb;

constexpr int kFftSizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

// Listening: channel widths on offer, and the squelch slider's "off" position.
constexpr double kNarrowWidthsHz[] = {8e3, 10e3, 12.5e3, 16e3, 20e3, 25e3};
constexpr double kWideWidthsHz[]   = {100e3, 150e3, 200e3, 250e3};
constexpr float  kSquelchSliderOff = -120.0f;

// A transcript Whisper gave less confidence than this is shown dimmed.
[[maybe_unused]] constexpr float kUnsureTranscript = 0.5f;

// The audio plot: the spans on offer, and how far back to look for a zero
// crossing to steady the short ones on.
constexpr double      kScopeSpans[]        = {0.02, 0.1, 1.0, 5.0};
constexpr const char* kScopeSpanNames[]    = {"20 ms", "0.1 s", "1 s", "5 s"};
constexpr double      kScopeTriggerSeconds = 0.02;

// Automatic level scale: the noise floor sits this far above the bottom, the
// strongest signal this far below the top.
constexpr float kRangeFloorMargin = 10.0f;
constexpr float kRangeHeadroom    = 10.0f;
constexpr float kRangeMinSpan     = 40.0f;

struct ViewSettings {
    float db_floor     = -110.0f;
    float db_ceiling   = -10.0f;
    bool  track_levels = false;  // keep adjusting the level scale to the signal

    int   colormap    = 0;
    float plot_height = 0.55f;
    float plot_depth  = 2.4f;
    float relief      = 0.45f;
    float fog         = 0.55f;
    bool  walls       = true;
    bool  grid        = true;
    bool  trace       = true;
    bool  max_hold    = false;
    bool  labels      = true;

    bool paused     = false;
    bool show_panel = true;
    bool show_help  = false;
};

// The trunk follower's clock, in seconds.
double follow_clock()
{
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

void glfw_error_callback(int code, const char* description)
{
    // Wayland reports several harmless "feature not available" conditions here.
    LOG_DEBUG("GLFW (%d): %s", code, description ? description : "");
}

struct GlfwSession {
    explicit GlfwSession(const std::string& platform)
    {
        glfwSetErrorCallback(glfw_error_callback);
#ifdef GLFW_PLATFORM
        if (platform == "x11")
            glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        else if (platform == "wayland")
            glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
#else
        (void)platform;
#endif
        if (!glfwInit())
            throw std::runtime_error("Could not initialise the windowing system (is a display available?)");
    }
    ~GlfwSession() { glfwTerminate(); }
    GlfwSession(const GlfwSession&)            = delete;
    GlfwSession& operator=(const GlfwSession&) = delete;
};

struct WindowDeleter {
    void operator()(GLFWwindow* window) const
    {
        if (window)
            glfwDestroyWindow(window);
    }
};

class App {
public:
    explicit App(const Options& options);
    ~App();

    int run();

private:
    void frame();

    // Data flow
    void pump_rows();
    void push_pending(const float* row, bool gap);
    void show_rows(float dt);
    void note_shown_row(const float* row);

    // Interaction
    void handle_input();
    void apply_view(ViewPreset preset);
    void tune_to(double hz);
    void set_gain(double db);
    void change_fft_size(int direction);
    void auto_range();
    void update_level_tracking(float dt);
    void take_screenshot(const std::string& path);
    void toggle_fullscreen();
    void run_self_test(double elapsed);
    void probe_cursor(float column_fraction, int row_depth, bool verify);
    void self_test_listen(bool wide);
    void self_test_check_audio(bool wide);
    void self_test_scan(int stage);
    void self_test_heard(int stage);
    void update_cursor_readout();
    bool surface_under(float window_x, float window_y, Vec3& hit, Vec3* entry = nullptr) const;
    double frequency_at(float x) const;
    float surface_height(float x, float z, float* db_out = nullptr, float* age_rows_out = nullptr) const;
    void toast(const std::string& text);
    void update_title();

    // Listening
    void set_listening(bool on);
    void toggle_listening();
    void start_scanning();
    void skip_signal();
    void stop_scanning();
    bool scanning() const { return scanner_ && scanner_->active(); }  // for signals, in the spectrum
    bool scan_locked() const;  // scanning and on a signal
    // P25 only, with control channels from --control-channels: scanning goes
    // through them instead (the trunk follower hunts), and follows the calls
    // of the one heard.
    bool channel_scan_wanted() const { return listen_.p25_only && !follower_.control_channels().empty(); }
    bool channel_scanning() const { return follower_.hunting(); }
    void start_channel_scan();
    void use_control_channels(const std::vector<double>& listed);
    double control_channels_centre(double hz) const;
    void listen_to(double hz, double search_hz);
    void select_channel(double hz, double search_hz);
    void listen_to_heard(double hz, int phase2_slot = -1);
    void follow_to(const FollowAction& action);
    void set_following(bool on);
    [[maybe_unused]] void draw_heard_channels();  // only with audio
    bool choose_channel_at(float window_x, float window_y, bool find_carrier);
    void update_listening();
    bool listen_in_band() const;
    double listen_bandwidth() const;  // as used: limited to what the stream can carry
    bool   listen_wide() const;

    // Drawing
    void draw_panel();
    void draw_status();
    void draw_help();
    void draw_axis_labels();
    void draw_listen_window();
    [[maybe_unused]] void draw_audio_plot();  // only with audio
    [[maybe_unused]] void draw_dmr_status();  // likewise
    [[maybe_unused]] void draw_p25_status();  // likewise
    [[maybe_unused]] void draw_p25_phase2_status();  // likewise
    [[maybe_unused]] void draw_following();  // likewise
    [[maybe_unused]] void draw_channel_scan();  // likewise
    void draw_transcript_window();
    std::string           digital_filter_name() const;
    void draw_listen_label();
    void build_lines();
    void add_listen_box(float pixel_ratio);
    void add_scan_marks(float pixel_ratio);
    SceneParams scene_params() const;
    Mat4        view_projection() const;
    float       aspect() const;
    float       plot_aspect() const;
    float       column_phase() const;

    Options  options_;
    // Speech to text of the voice the listener decodes.  (Before the pipeline:
    // the listener hands it the voice until the pipeline stops.)
    std::unique_ptr<Transcriber> transcriber_;
    Pipeline pipeline_;

    std::unique_ptr<GlfwSession>               glfw_;
    std::unique_ptr<GLFWwindow, WindowDeleter> window_;
    std::unique_ptr<VulkanContext>             vk_;
    std::unique_ptr<Renderer>                  renderer_;
    std::unique_ptr<ImGuiLayer>                imgui_;

    const int columns_;
    const int history_;

    // Rows on their way from the DSP thread to the screen
    std::vector<float>   incoming_;
    std::vector<RowInfo> incoming_info_;
    std::vector<float>   pending_;       // FIFO of kMaxPending rows
    std::vector<uint8_t> pending_gap_;
    int                  pending_count_ = 0;
    uint64_t             next_row_index_ = 0;
    bool                 have_row_index_ = false;
    bool                 skipped_while_paused_ = false;
    uint64_t             rows_shown_ = 0;
    uint64_t             rows_missed_ = 0;
    double               scroll_pos_ = -1.0;   // continuous display position, in rows
    float                scroll_phase_ = 0.0f;
    double               row_period_ = 0.02;   // signal time per row
    int                  fft_size_;

    // A copy of what the spectrogram texture holds, for the cursor readout
    std::vector<float> history_data_;  // history_ rows of columns_ values; same ring layout as the texture

    // What is under the mouse cursor
    struct CursorReadout {
        bool   valid = false;
        Vec3   world;
        double frequency_hz = 0.0;
        float  level_db     = 0.0f;
        double seconds_ago  = 0.0;
    } cursor_;

    // --self-test bookkeeping
    int      self_test_step_   = 0;
    bool     self_test_failed_ = false;
    bool     probe_active_     = false;  // the readout follows the probe point instead of the mouse
    float    probe_x_ = 0.0f, probe_y_ = 0.0f;
    int      probe_frames_     = 0;      // frames until the readout at the probe point is checked
    double   probe_expected_hz_ = 0.0;
    uint64_t frames_drawn_     = 0;

    // The newest spectrum on screen, and statistics derived from it
    std::vector<float> latest_row_;
    std::vector<float> max_hold_;
    std::vector<float> scratch_row_;
    bool               have_row_ = false;
    float              noise_estimate_ = std::numeric_limits<float>::quiet_NaN();
    float              peak_estimate_  = std::numeric_limits<float>::quiet_NaN();
    int                stats_rows_ = 0;
    bool               initial_auto_range_ = true;
    bool               range_animating_ = false;
    float              range_target_floor_ = 0.0f, range_target_ceiling_ = 0.0f;
    float              tracking_timer_ = 0.0f;

    ViewSettings view_;
    Camera       camera_;
    ViewPreset   preset_ = ViewPreset::Perspective;
    float        panel_fraction_ = 0.0f;  // part of the window width covered by the docked panel
    float        lens_shift_     = 0.0f;  // smoothed horizontal shift of the picture, in NDC
    // A preset view keeps framing the plot (window resized, panel toggled, plot
    // reshaped) until the user moves the camera.
    bool         camera_untouched_ = true;
    float        framed_aspect_ = 0.0f, framed_height_ = 0.0f, framed_depth_ = 0.0f;

    std::vector<LineInstance> lines_;
    std::vector<AxisLabel>    labels_;

    double tune_mhz_ = 0.0;
    int    gain_db_  = 0;

    // The channel being listened to (FM audio)
    struct Listening {
        bool   on        = false;  // audio is wanted
        bool   selected  = false;  // a channel has been chosen
        double frequency = 0.0;    // Hz
        double bandwidth = Listener::kDefaultBandwidth;      // also chooses narrowband or wideband FM
        double deemphasis = FmDemodulator::kDefaultDeemphasis;  // wideband FM, seconds
        float  volume    = 0.5f;   // 0..1
        float  boost_db  = 0.0f;   // beyond full volume, 0 ... Listener::kMaxBoostDb
        float  squelch   = FmDemodulator::kSquelchOff;  // dBFS
        bool   dmr_only  = false;  // play DMR voice only (and scanning, stay only for it)
        bool   p25_only  = false;  // likewise P25; with both, both
        bool   follow    = false;  // on a P25 control channel, follow the calls it grants
        double dmr_hold  = Scanner::kDefaultDmrHold;  // scanning, DMR/P25 only: wait for a call this long
        float  scan_threshold = std::numeric_limits<float>::quiet_NaN();  // dBFS; NaN: from the noise floor
        double scan_hang      = Scanner::kDefaultHang;
    } listen_;
    std::unique_ptr<Scanner> scanner_;
    Scanner::State           logged_scan_state_ = Scanner::State::Off;
    std::string              logged_signal_;  // what the terminal was last told the channel carries
    std::string              logged_dmr_[2];  // and the DMR call on each slot
    std::string              logged_p25_;     // and the P25 call
    std::string              logged_p25_site_;     // and a control channel's system
    std::string              logged_p25_phases_;   // and its voice channels' kinds
    std::vector<std::string> logged_p25_grants_;   // and its grants (the latest few)
    HeardChannels            heard_;          // channels DMR voice has been heard on
    std::vector<Transcript>  transcripts_;      // what was said, oldest first
    uint64_t                 transcript_serial_ = 0;  // the latest of them
    bool                     show_transcript_   = true;
    TrunkFollower            follower_;       // following a P25 system's calls
    size_t                   hunt_tries_ = 0; // control channels tried since one was last heard
    std::string              logged_p25p2_[2];  // the P25 Phase 2 call on each slot, as logged

    // The audio plot in the listening window
    int                scope_span_       = 2;   // index into kScopeSpans
    int                scope_shown_span_ = -1;  // what scope_lo_ / scope_hi_ hold
    std::vector<float> scope_samples_, scope_mins_, scope_maxs_;  // read from the listener
    std::vector<float> scope_lo_, scope_hi_;  // per pixel column; lo > hi where there is nothing
    double       listen_mhz_ = 0.0;
    ListenStatus listen_status_;
    bool         choosing_ = false;  // a Ctrl + drag across the plot is in progress
#if SPECTRO_HAVE_AUDIO
    std::unique_ptr<AudioOutput> audio_;
#endif

    Clock::time_point start_time_;
    Clock::time_point last_frame_time_;
    float             frame_time_avg_ = 1.0f / 60.0f;
    bool              auto_screenshot_done_ = false;

    std::string       toast_text_;
    Clock::time_point toast_until_;

    bool fullscreen_ = false;
    int  windowed_x_ = 100, windowed_y_ = 100, windowed_w_ = 1600, windowed_h_ = 900;
};

// ---------------------------------------------------------------------------
// Setup and teardown
// ---------------------------------------------------------------------------

App::App(const Options& options)
    : options_(options),
      pipeline_(options),
      columns_(options.columns),
      history_(options.history),
      fft_size_(options.spectrum.fft_size)
{
    // Start the radio first: if it is missing there is no point opening a window.
    pipeline_.start();

    glfw_ = std::make_unique<GlfwSession>(options_.platform);
    if (!glfwVulkanSupported())
        throw std::runtime_error("No Vulkan driver was found on this system");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifdef GLFW_WAYLAND_APP_ID
    glfwWindowHintString(GLFW_WAYLAND_APP_ID, "spectrogram");
#endif
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "spectrogram");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "spectrogram");

    GLFWmonitor* monitor = nullptr;
    int          width = options_.width, height = options_.height;
    if (options_.fullscreen) {
        monitor = glfwGetPrimaryMonitor();
        if (const GLFWvidmode* mode = monitor ? glfwGetVideoMode(monitor) : nullptr) {
            width  = mode->width;
            height = mode->height;
        }
        fullscreen_ = monitor != nullptr;
    }
    windowed_w_ = options_.width;
    windowed_h_ = options_.height;
    window_.reset(glfwCreateWindow(width, height, "Spectrogram", monitor, nullptr));
    if (!window_)
        throw std::runtime_error("Could not create a window");

    VulkanOptions vk_options;
    vk_options.validation = options_.validation;
    vk_options.vsync      = options_.vsync;
    vk_options.gpu        = options_.gpu;
    vk_ = std::make_unique<VulkanContext>(window_.get(), vk_options);

    Renderer::Config config;
    config.columns = columns_;
    config.rows    = history_;
    config.msaa    = options_.msaa;
    renderer_ = std::make_unique<Renderer>(*vk_, config);
    imgui_    = std::make_unique<ImGuiLayer>(window_.get(), *vk_, renderer_->render_pass(),
                                             renderer_->sample_count());

    const size_t row = static_cast<size_t>(columns_);
    incoming_.resize(row * kPopBatch);
    incoming_info_.resize(kPopBatch);
    pending_.resize(row * kMaxPending);
    pending_gap_.resize(kMaxPending);
    history_data_.assign(row * static_cast<size_t>(history_), kFloorDb);
    latest_row_.assign(row, kFloorDb);
    max_hold_.assign(row, kFloorDb);
    scratch_row_.resize(row);
    lines_.reserve(static_cast<size_t>(2 * columns_ + 512));

    // View defaults and command-line overrides
    view_.show_panel = !options_.hide_ui;
    if (const int map = colormap_from_key(options_.colormap); map >= 0)
        view_.colormap = map;
    else
        LOG_WARN("Unknown colour map '%s'; using %s", options_.colormap.c_str(), colormap_name(0));
    if (!std::isnan(options_.db_min))
        view_.db_floor = static_cast<float>(options_.db_min);
    if (!std::isnan(options_.db_max))
        view_.db_ceiling = static_cast<float>(options_.db_max);
    if (view_.db_ceiling < view_.db_floor + 5.0f)
        view_.db_ceiling = view_.db_floor + 5.0f;
    initial_auto_range_ = std::isnan(options_.db_min) || std::isnan(options_.db_max);

    tune_mhz_ = pipeline_.source().frequency() / 1e6;
    gain_db_  = static_cast<int>(std::lround(pipeline_.source().gain()));
    // The panel is docked at the left when the first frame is drawn; frame the
    // plot for that layout straight away.
    if (view_.show_panel) {
        int window_width = 0, window_height = 0;
        glfwGetWindowSize(window_.get(), &window_width, &window_height);
        panel_fraction_ = lens_shift_ =
            std::clamp(342.0f * imgui_->ui_scale() / static_cast<float>(std::max(window_width, 1)), 0.0f, 0.6f);
    }
    apply_view(options_.view == "top" ? ViewPreset::Top : options_.view == "front" ? ViewPreset::Front
                                                                                 : ViewPreset::Perspective);
    update_title();

    // Listening.  The self-test exercises the whole audio path too, but
    // silently unless a volume was asked for.
    const double volume = !std::isnan(options_.volume) ? options_.volume : options_.self_test ? 0.0 : kDefaultVolume;
    listen_.volume      = static_cast<float>(volume / 100.0);
    listen_.boost_db    = static_cast<float>(options_.boost_db);
    listen_.bandwidth   = options_.listen_bandwidth;
    listen_.deemphasis  = options_.deemphasis_us * 1e-6;
    listen_.scan_hang   = options_.scan_hang;
    listen_.dmr_only    = options_.dmr_only;
    listen_.p25_only    = options_.p25_only;
    listen_.dmr_hold    = options_.dmr_hold;
    set_following(options_.follow_calls);
    use_control_channels(options_.control_channels);
    if (!std::isnan(options_.scan_threshold))
        listen_.scan_threshold = static_cast<float>(options_.scan_threshold);
    scanner_ = std::make_unique<Scanner>(pipeline_.listener());
    if (!std::isnan(options_.squelch))
        listen_.squelch = static_cast<float>(options_.squelch);
    // Speech to text: the model loads on the transcriber's own thread.
    if (!options_.transcribe_model.empty()) {
        if (!Transcriber::available()) {
            LOG_WARN("--transcribe: this build has no speech to text (configure with -DSPECTRO_WITH_WHISPER=ON)");
        } else {
            if (!pipeline_.listener().status().vocoder)
                LOG_WARN("--transcribe: this build has no vocoder, so there is no voice to transcribe "
                         "(configure with -DSPECTRO_WITH_MBELIB=ON)");
            Transcriber::Settings settings;
            settings.model    = options_.transcribe_model;
            settings.language = options_.transcribe_language;
            settings.file     = options_.transcript_file;
            transcriber_      = std::make_unique<Transcriber>(settings);
            pipeline_.listener().set_voice_sink(transcriber_.get());
        }
    }
    if (options_.scan)
        start_scanning();
    if (!std::isnan(options_.listen_frequency)) {
        listen_to(options_.listen_frequency, 0.0);
        if (!listen_in_band()) {
            const IqSource& source = pipeline_.source();
            LOG_WARN("--listen %s is outside the band being received (%s ... %s); tune there to hear it",
                     format_frequency(options_.listen_frequency).c_str(),
                     format_frequency(source.frequency() - 0.5 * source.sample_rate()).c_str(),
                     format_frequency(source.frequency() + 0.5 * source.sample_rate()).c_str());
        }
    }

    start_time_      = Clock::now();
    last_frame_time_ = start_time_;
}

App::~App()
{
    // The sound device first: it reads from the pipeline's audio FIFO.
    set_listening(false);
#if SPECTRO_HAVE_AUDIO
    audio_.reset();
#endif
    // Stop the radio before tearing down the display.
    pipeline_.stop();
    if (vk_ && vk_->device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(vk_->device);
    imgui_.reset();
    renderer_.reset();
    vk_.reset();
    window_.reset();
    glfw_.reset();
}

int App::run()
{
    GLFWwindow* window = window_.get();
    while (!glfwWindowShouldClose(window) && !quit_requested()) {
        glfwPollEvents();
        frame();

        const double elapsed = std::chrono::duration<double>(Clock::now() - start_time_).count();
        if (options_.self_test)
            run_self_test(elapsed);
        if (options_.duration > 0.0 && elapsed >= options_.duration)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    const double           run_time = std::chrono::duration<double>(Clock::now() - start_time_).count();
    const AcquisitionStats acq      = pipeline_.acquisition().stats();
    LOG_INFO("Received %.1f M samples; lost by the device/USB: %llu; discarded by the host: %llu",
             static_cast<double>(acq.samples) / 1e6, static_cast<unsigned long long>(acq.lost_samples),
             static_cast<unsigned long long>(acq.ring_dropped));
    LOG_INFO("Drew %llu frames in %.1f s (%.1f per second); %llu of %llu rows were not shown",
             static_cast<unsigned long long>(frames_drawn_), run_time,
             run_time > 0.0 ? static_cast<double>(frames_drawn_) / run_time : 0.0,
             static_cast<unsigned long long>(rows_missed_ + pipeline_.engine().stats().rows_dropped),
             static_cast<unsigned long long>(pipeline_.engine().stats().rows_emitted +
                                             pipeline_.engine().stats().rows_dropped));

    const ListenStatus audio = pipeline_.listener().status();
    if (audio.audio_samples > 0) {
        LOG_INFO("Audio: %.1f s demodulated; gaps: %llu; discarded for the sound card: %llu; samples missed: %llu",
                 static_cast<double>(audio.audio_samples) / Listener::kAudioRate,
                 static_cast<unsigned long long>(audio.underruns), static_cast<unsigned long long>(audio.overruns),
                 static_cast<unsigned long long>(audio.dropped_iq));
    }

    int exit_code = 0;
    if (vk_->validation_active()) {
        LOG_INFO("Vulkan validation: %u error(s), %u warning(s)", vk_->validation_errors(),
                 vk_->validation_warnings());
        if (vk_->validation_errors() > 0)
            exit_code = 3;
    }
    if (options_.self_test) {
        const bool ok = !acq.failed && !self_test_failed_ && frames_drawn_ > 100 &&
                        vk_->validation_errors() == 0 && acq.lost_samples == 0 && acq.ring_dropped == 0;
        std::printf("Self-test %s\n", ok ? "PASSED" : "FAILED");
        if (!ok)
            exit_code = 4;
    }
    return exit_code;
}

float App::aspect() const
{
    const VkExtent2D extent = renderer_->extent();
    return extent.height > 0 ? static_cast<float>(extent.width) / static_cast<float>(extent.height) : 1.0f;
}

// Aspect ratio of the part of the window that the control panel leaves free.
float App::plot_aspect() const
{
    return aspect() * std::max(0.3f, 1.0f - panel_fraction_);
}

// The camera transform, with the picture shifted sideways so that the plot is
// centred in the area next to the control panel rather than behind it.
Mat4 App::view_projection() const
{
    Mat4 shift;
    shift.at(0, 3) = lens_shift_;  // clip.x += shift * clip.w
    return shift * camera_.projection(aspect()) * camera_.view();
}

// Where column 0 sits within its cell; see SpectrumEngine::column_frequency().
float App::column_phase() const
{
    const float ratio = std::max(1.0f, static_cast<float>(fft_size_) / static_cast<float>(columns_));
    return 0.5f - 0.5f / ratio;
}

void App::update_title()
{
    char title[160];
    std::snprintf(title, sizeof(title), "Spectrogram - %s - %s",
                  format_frequency(pipeline_.source().frequency()).c_str(),
                  pipeline_.source().info().kind.c_str());
    glfwSetWindowTitle(window_.get(), title);
}

// Skipping the last signal of the list goes round again from the first.
// Scanning control channels, it tries the next of them.
void App::skip_signal()
{
    if (channel_scanning()) {
        const FollowAction action = follower_.skip(follow_clock());
        if (action.tune)
            follow_to(action);
        return;
    }
    if (scanner_ && scanner_->skip()) {
        LOG_INFO("Scanner: end of the list, starting again from the bottom");
        toast("End of the list: back to the first signal");
    }
}

void App::toast(const std::string& text)
{
    toast_text_  = text;
    toast_until_ = Clock::now() + std::chrono::milliseconds(2500);
}

// ---------------------------------------------------------------------------
// Rows: DSP thread -> pending FIFO -> screen
// ---------------------------------------------------------------------------

void App::push_pending(const float* row, bool gap)
{
    const size_t width = static_cast<size_t>(columns_);
    if (pending_count_ == kMaxPending) {
        // The display fell far behind (window hidden, ...): drop the oldest row.
        std::move(pending_.begin() + static_cast<std::ptrdiff_t>(width), pending_.end(), pending_.begin());
        std::move(pending_gap_.begin() + 1, pending_gap_.end(), pending_gap_.begin());
        --pending_count_;
        ++rows_missed_;
    }
    float* dst = pending_.data() + static_cast<size_t>(pending_count_) * width;
    if (row)
        std::copy_n(row, width, dst);
    else
        std::fill_n(dst, width, kFloorDb);
    pending_gap_[static_cast<size_t>(pending_count_)] = gap ? 1 : 0;
    ++pending_count_;
}

void App::pump_rows()
{
    SpectrumEngine& engine = pipeline_.engine();
    for (;;) {
        const int n = engine.pop_rows(incoming_.data(), kPopBatch, incoming_info_.data());
        for (int i = 0; i < n; ++i) {
            const RowInfo& info = incoming_info_[static_cast<size_t>(i)];
            if (!info.gap) {
                row_period_ = info.period_s;
                fft_size_   = info.fft_size;
            }

            // The scanner sees every row, also while the display is paused.
            if (scanning()) {
                const IqSource&    source = pipeline_.source();
                const SpectrumView view{source.frequency(), source.sample_rate(), columns_, info.fft_size};
                scanner_->on_row(incoming_.data() + static_cast<size_t>(i) * static_cast<size_t>(columns_), info,
                                 view);
            }

            // Rows the DSP thread had to drop leave a hole in the numbering.
            const bool hole = have_row_index_ && info.index != next_row_index_;
            next_row_index_ = info.index + 1;
            have_row_index_ = true;

            if (view_.paused) {
                skipped_while_paused_ = true;
                continue;
            }
            if (hole || skipped_while_paused_) {
                push_pending(nullptr, true);  // mark the break in time
                skipped_while_paused_ = false;
            }
            push_pending(incoming_.data() + static_cast<size_t>(i) * static_cast<size_t>(columns_), info.gap);
        }
        if (n < kPopBatch)
            break;
    }
}

void App::note_shown_row(const float* row)
{
    const size_t width = static_cast<size_t>(columns_);
    std::copy_n(row, width, latest_row_.begin());
    have_row_ = true;

    float peak = kFloorDb;
    for (size_t c = 0; c < width; ++c) {
        max_hold_[c] = std::max(max_hold_[c], row[c]);
        peak         = std::max(peak, row[c]);
    }

    // Noise floor: a low percentile of the row, followed downwards immediately
    // and upwards only slowly, so that bursts filling the whole band do not
    // count as "floor".  Peak: the mirror image.
    std::copy_n(row, width, scratch_row_.begin());
    const auto tenth = scratch_row_.begin() + static_cast<std::ptrdiff_t>(width / 10);
    std::nth_element(scratch_row_.begin(), tenth, scratch_row_.end());
    const float floor = *tenth;
    const float leak  = static_cast<float>(2.0 * row_period_);  // 2 dB per second
    noise_estimate_   = std::isnan(noise_estimate_) ? floor : std::min(floor, noise_estimate_ + leak);
    peak_estimate_    = std::isnan(peak_estimate_) ? peak : std::max(peak, peak_estimate_ - leak);
    ++stats_rows_;
}

// Moves rows from the pending FIFO to the renderer at a steady pace.
//
// Rows arrive in small bursts (USB transfers, thread scheduling, one batch per
// frame), but the waterfall should glide.  So the display follows a continuous
// position `scroll_pos_` that advances at the nominal row rate and is gently
// steered to stay one row behind the newest data; its fractional part becomes
// the sub-row scroll offset in the vertex shader.
void App::show_rows(float dt)
{
    const double available = static_cast<double>(rows_shown_) + pending_count_;
    const double rate      = row_period_ > 0.0 ? 1.0 / row_period_ : 50.0;
    const double target    = available - 1.0;

    scroll_pos_ += static_cast<double>(dt) * rate;
    const double error = target - scroll_pos_;
    if (error > 6.0 || error < -1.5)
        scroll_pos_ = target;  // far off (start-up, after a pause or a stall): jump
    else
        scroll_pos_ += error * (1.0 - std::exp(-static_cast<double>(dt) * 5.0));
    scroll_pos_ = std::clamp(scroll_pos_, -1.0, available);

    const double wanted  = std::min(available, std::floor(scroll_pos_) + 1.0);
    int          to_show = static_cast<int>(std::clamp(wanted - static_cast<double>(rows_shown_), 0.0,
                                                       static_cast<double>(pending_count_)));
    to_show = std::min(to_show, Renderer::kMaxRowsPerFrame);

    if (to_show > 0) {
        const size_t width    = static_cast<size_t>(columns_);
        const int    before   = renderer_->head_row();
        const int    accepted = renderer_->upload_rows(pending_.data(), to_show);
        for (int i = 0; i < accepted; ++i) {
            const float* row = pending_.data() + static_cast<size_t>(i) * width;
            // Keep our copy of the texture in step with it, row for row.
            const int texture_row = (before + 1 + i) % history_;
            std::copy_n(row, width, history_data_.begin() +
                                        static_cast<std::ptrdiff_t>(static_cast<size_t>(texture_row) * width));
            if (!pending_gap_[static_cast<size_t>(i)])
                note_shown_row(row);
        }
        std::move(pending_.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(accepted) * width),
                  pending_.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(pending_count_) * width),
                  pending_.begin());
        std::move(pending_gap_.begin() + accepted, pending_gap_.begin() + pending_count_, pending_gap_.begin());
        pending_count_ -= accepted;
        rows_shown_ += static_cast<uint64_t>(accepted);
    }

    // Never scroll past the newest row that is actually on screen.
    scroll_pos_   = std::min(scroll_pos_, static_cast<double>(rows_shown_));
    scroll_phase_ = static_cast<float>(std::clamp(scroll_pos_ - (static_cast<double>(rows_shown_) - 1.0), 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

void App::apply_view(ViewPreset preset)
{
    preset_           = preset;
    camera_           = make_view(preset, kPlotWidth, view_.plot_height, view_.plot_depth, plot_aspect());
    camera_untouched_ = true;
    framed_aspect_    = plot_aspect();
    framed_height_    = view_.plot_height;
    framed_depth_     = view_.plot_depth;
}

void App::tune_to(double hz)
{
    IqSource& source = pipeline_.source();
    if (!source.info().tunable) {
        toast("This source cannot be tuned");
        return;
    }
    hz = std::clamp(hz, source.info().frequency_range.min, source.info().frequency_range.max);
    if (!source.set_frequency(hz)) {
        toast("Tuning failed");
        tune_mhz_ = source.frequency() / 1e6;
        return;
    }
    // The history now shows another band: mark the break and restart max hold.
    pipeline_.engine().mark_gap();
    std::fill(max_hold_.begin(), max_hold_.end(), kFloorDb);
    tune_mhz_ = source.frequency() / 1e6;
    update_title();
}

void App::set_gain(double db)
{
    IqSource& source = pipeline_.source();
    if (!source.info().has_gain)
        return;
    db = std::clamp(db, source.info().gain_range.min, source.info().gain_range.max);
    if (!source.set_gain(db))
        toast("Setting the gain failed");
    gain_db_ = static_cast<int>(std::lround(source.gain()));
}

void App::change_fft_size(int direction)
{
    SpectrumParams params = pipeline_.engine().params();
    const int next = direction > 0 ? params.fft_size * 2 : params.fft_size / 2;
    if (next < kFftSizes[0] || next > kFftSizes[std::size(kFftSizes) - 1])
        return;
    params.fft_size = next;
    pipeline_.engine().set_params(params);
    toast("FFT size " + std::to_string(next));
}

// Fits the level scale to what is on air: the noise floor a little above the
// bottom, the strongest signal a little below the top.
void App::auto_range()
{
    if (std::isnan(noise_estimate_) || std::isnan(peak_estimate_))
        return;
    float floor   = std::floor(noise_estimate_ - kRangeFloorMargin);
    float ceiling = std::ceil(std::max(peak_estimate_ + kRangeHeadroom, floor + kRangeMinSpan));
    floor   = std::clamp(floor, -190.0f, 0.0f);
    ceiling = std::clamp(ceiling, floor + 10.0f, 20.0f);

    range_target_floor_   = floor;
    range_target_ceiling_ = ceiling;
    range_animating_      = true;
}

void App::update_level_tracking(float dt)
{
    // One automatic fit shortly after start-up, once there is something to fit to.
    if (initial_auto_range_ && stats_rows_ >= 20) {
        initial_auto_range_ = false;
        auto_range();
        if (!std::isnan(options_.db_min))
            range_target_floor_ = static_cast<float>(options_.db_min);
        if (!std::isnan(options_.db_max))
            range_target_ceiling_ = static_cast<float>(options_.db_max);
    }

    if (view_.track_levels && !initial_auto_range_) {
        tracking_timer_ += dt;
        if (tracking_timer_ >= 0.5f && !std::isnan(noise_estimate_)) {
            tracking_timer_ = 0.0f;
            const float floor   = std::floor(noise_estimate_ - kRangeFloorMargin);
            const float ceiling = std::ceil(std::max(peak_estimate_ + kRangeHeadroom, floor + kRangeMinSpan));
            // Only react to real changes so the scale does not breathe.
            if (std::fabs(floor - view_.db_floor) > 3.0f || std::fabs(ceiling - view_.db_ceiling) > 3.0f)
                auto_range();
        }
    }

    if (range_animating_) {
        const float k = 1.0f - std::exp(-dt * 7.0f);
        view_.db_floor += (range_target_floor_ - view_.db_floor) * k;
        view_.db_ceiling += (range_target_ceiling_ - view_.db_ceiling) * k;
        if (std::fabs(range_target_floor_ - view_.db_floor) < 0.05f &&
            std::fabs(range_target_ceiling_ - view_.db_ceiling) < 0.05f) {
            view_.db_floor   = range_target_floor_;
            view_.db_ceiling = range_target_ceiling_;
            range_animating_ = false;
        }
    }
}

void App::take_screenshot(const std::string& path)
{
    if (!renderer_->screenshot_supported()) {
        toast("Screenshots are not supported on this display");
        return;
    }
    std::string file = path;
    if (file.empty()) {
        char        name[64];
        std::time_t now = std::time(nullptr);
        std::tm     local{};
#if defined(_WIN32)
        localtime_s(&local, &now);
#else
        localtime_r(&now, &local);
#endif
        std::strftime(name, sizeof(name), "spectrogram-%Y%m%d-%H%M%S.png", &local);
        file = name;
    }
    renderer_->request_screenshot(file);
    toast("Saved " + file);
}

void App::toggle_fullscreen()
{
    GLFWwindow* window = window_.get();
    if (!fullscreen_) {
        GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode    = monitor ? glfwGetVideoMode(monitor) : nullptr;
        if (!mode)
            return;
        glfwGetWindowPos(window, &windowed_x_, &windowed_y_);
        glfwGetWindowSize(window, &windowed_w_, &windowed_h_);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        fullscreen_ = true;
    } else {
        glfwSetWindowMonitor(window, nullptr, windowed_x_, windowed_y_, windowed_w_, windowed_h_, 0);
        fullscreen_ = false;
    }
}

// A scripted tour through views, window sizes and settings.  Together with
// --validate this checks that every code path the user can reach from the
// interface works on this machine's graphics stack.
void App::run_self_test(double elapsed)
{
    struct Step {
        double      at;
        const char* what;
    };
    static const Step kSteps[] = {
        {1.0, "top view"},          {1.5, "small window"},    {2.0, "front view"},
        {2.5, "FFT size 16384"},    {3.0, "large window"},    {3.5, "3D view"},
        {4.0, "next colour map"},   {4.5, "FFT size 1024, Blackman-Harris, peak detector"},
        {5.0, "walls and grid off"}, {5.5, "tiny window"},    {6.0, "retune"},
        {6.5, "gain"},              {7.0, "pause"},           {7.6, "resume"},
        {8.0, "hide the panel"},    {8.5, "max hold"},        {9.0, "restore the defaults"},
        {10.0, "top view"},         {10.6, "cursor readout in the top view"},
        {11.0, "listen to a signal (Ctrl + click)"},          {11.6, "cursor readout in the 3D view"},
        {12.6, "check the audio"},  {13.0, "listen to a broadcast station (wideband FM)"},
        {14.6, "check the wideband audio"},                   {15.0, "scan for signals"},
        {16.6, "check the scanner, then skip"},               {17.8, "check the next signal, then skip"},
        {19.8, "check the digital signal"},                   {20.2, "go back to a channel DMR voice was heard on"},
        {21.4, "check that channel"},                         {21.8, "finish"},
    };
    constexpr int kStepCount = static_cast<int>(std::size(kSteps));
    if (self_test_step_ >= kStepCount || elapsed < kSteps[self_test_step_].at)
        return;

    GLFWwindow*     window = window_.get();
    SpectrumEngine& engine = pipeline_.engine();
    SpectrumParams  params = engine.params();
    LOG_INFO("self-test: %s", kSteps[self_test_step_].what);

    switch (self_test_step_) {
    case 0:  apply_view(ViewPreset::Top); break;
    case 1:  glfwSetWindowSize(window, 1000, 640); break;
    case 2:  apply_view(ViewPreset::Front); break;
    case 3:  params.fft_size = 16384; engine.set_params(params); break;
    case 4:  glfwSetWindowSize(window, 1500, 880); break;
    case 5:  apply_view(ViewPreset::Perspective); break;
    case 6:  view_.colormap = (view_.colormap + 1) % colormap_count(); break;
    case 7:  params.fft_size = 1024; params.detector = Detector::Peak; params.window = WindowType::BlackmanHarris;
             engine.set_params(params); break;
    case 8:  view_.walls = false; view_.grid = false; break;
    case 9:  glfwSetWindowSize(window, 480, 360); break;
    case 10: tune_to(pipeline_.source().frequency() + 0.25 * pipeline_.source().sample_rate()); break;
    case 11: set_gain(pipeline_.source().gain() + 6.0); break;
    case 12: view_.paused = true; break;
    case 13: view_.paused = false; break;
    case 14: view_.show_panel = false; break;
    case 15: view_.max_hold = true; view_.walls = true; view_.grid = true; break;
    case 16: glfwSetWindowSize(window, options_.width, options_.height); view_.show_panel = true;
             params.fft_size = 4096; params.detector = Detector::Average; params.window = WindowType::Hann;
             engine.set_params(params);
             tune_to(pipeline_.source().frequency() - 0.25 * pipeline_.source().sample_rate());
             set_gain(pipeline_.source().gain() - 6.0);
             apply_view(ViewPreset::Perspective); break;
    case 17: apply_view(ViewPreset::Top); break;
    case 18: probe_cursor(0.3f, 20, true); break;
    case 19: self_test_listen(false); break;
    case 20: apply_view(ViewPreset::Perspective); probe_cursor(0.62f, 60, false);
             if (!options_.screenshot_path.empty()) take_screenshot(options_.screenshot_path);
             break;
    case 21: self_test_check_audio(false); break;
    case 22: apply_view(ViewPreset::Top); self_test_listen(true); break;
    case 23: self_test_check_audio(true); apply_view(ViewPreset::Perspective); break;
    case 24: self_test_scan(0); break;
    case 25: self_test_scan(1); break;
    case 26: self_test_scan(2); break;
    case 27: self_test_scan(3); break;
    case 28: self_test_heard(0); break;
    case 29: self_test_heard(1); break;
    default: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
    }
    ++self_test_step_;
}

// Self-test helper: makes the cursor readout look at a chosen point of the
// surface, as if the mouse were over it.  With `verify`, the readout computed a
// few frames later must name the frequency of that point - which checks the
// whole picking path (ray, plot placement, column mapping) against the
// rendering path.
void App::probe_cursor(float column_fraction, int row_depth, bool verify)
{
    const float column_width = kPlotWidth / static_cast<float>(columns_);
    const float column       = std::round(column_fraction * static_cast<float>(columns_ - 1));
    const float x            = -0.5f * kPlotWidth + (column + column_phase()) * column_width;
    const float z            = 0.5f * view_.plot_depth -
                               (static_cast<float>(row_depth) + scroll_phase_) * view_.plot_depth /
                                   static_cast<float>(history_ - 1);
    const Vec3  point{x, surface_height(x, z), z};

    ImGuiIO& io = ImGui::GetIO();
    float    sx = 0.0f, sy = 0.0f;
    if (!project_to_screen(view_projection(), point, io.DisplaySize.x, io.DisplaySize.y, sx, sy)) {
        LOG_ERROR("self-test: the probe point is not on screen");
        self_test_failed_ = true;
        return;
    }
    // (Not via the real mouse position: the windowing backend overwrites that
    // every frame while the window has the focus.)
    probe_active_ = true;
    probe_x_      = sx;
    probe_y_      = sy;
    if (verify) {
        probe_expected_hz_ = pipeline_.source().frequency() +
                             SpectrumEngine::column_frequency(column, columns_, fft_size_,
                                                              pipeline_.source().sample_rate());
        probe_frames_ = 3;
    }
}

// Self-test: the audio path, from choosing a signal in the plot to the sound
// device.  It "clicks" on the plot in the top view, in the middle of a display
// column.  With the simulator that is the column showing its narrowband voice
// channel or its broadcast station, whose carrier lies somewhere within the
// column: the receiver has to find it, and what it then plays is known.  On a
// radio any column will do to exercise the path.
void App::self_test_listen(bool wide)
{
#if SPECTRO_HAVE_AUDIO
    Listener& listener = pipeline_.listener();
    if (!listener.available() || (wide && !listener.supports(FmMode::Wide)))
        return;
    const IqSource& source = pipeline_.source();
    const double    fs     = source.sample_rate();
    double          offset = (wide ? 0.3 : 0.2) * fs;
    if (options_.source.kind == "sim" && (!wide || fs >= SimSource::kBroadcastMinRate)) {
        offset = options_.source.frequency - source.frequency() +
                 (wide ? SimSource::broadcast_station(fs).offset_hz : SimSource::voice_channel(fs).offset_hz);
    }
    listen_.bandwidth = wide ? FmDemodulator::kWideDefaultBandwidth : Listener::kDefaultBandwidth;
    listener.set_bandwidth(listen_.bandwidth);

    const float column_width = kPlotWidth / static_cast<float>(columns_);
    const float column       = std::round(static_cast<float>((offset / fs + 0.5) * columns_) - column_phase());
    const float x            = -0.5f * kPlotWidth + (column + column_phase()) * column_width;
    const float z            = 0.5f * view_.plot_depth -
                               (20.0f + scroll_phase_) * view_.plot_depth / static_cast<float>(history_ - 1);

    const ImGuiIO& io = ImGui::GetIO();
    float          sx = 0.0f, sy = 0.0f;
    if (!project_to_screen(view_projection(), {x, surface_height(x, z), z}, io.DisplaySize.x, io.DisplaySize.y, sx, sy) ||
        !choose_channel_at(sx, sy, true)) {
        LOG_ERROR("self-test: could not choose a signal in the plot");
        self_test_failed_ = true;
    }
#else
    (void)wide;
#endif
}

void App::self_test_check_audio(bool wide)
{
#if SPECTRO_HAVE_AUDIO
    Listener& listener = pipeline_.listener();
    if (!listener.available() || (wide && !listener.supports(FmMode::Wide)))
        return;
    const ListenStatus status = listener.status();
    LOG_INFO("self-test: %s audio of %.5f MHz: level %.1f dBFS, loudness %.2f, %.2f s demodulated, %llu gap(s), "
             "audio thread %.1f %% of a core, %.0f ms buffered (sound card period %.0f ms), on %s",
             status.mode == FmMode::Wide ? "wideband" : "narrowband", status.frequency / 1e6,
             static_cast<double>(status.level_dbfs), static_cast<double>(status.audio_rms),
             static_cast<double>(status.audio_samples) / Listener::kAudioRate,
             static_cast<unsigned long long>(status.underruns), static_cast<double>(status.load) * 100.0,
             1e3 * static_cast<double>(listener.audio().fill()) / Listener::kAudioRate,
             1e3 * static_cast<double>(listener.audio().largest_read()) / Listener::kAudioRate,
             audio_ ? audio_->description().c_str() : "nothing");

    // The audio plot shows what is played: the last half second of it must be
    // as loud as the listener says, times the volume and any boost - but no
    // louder than a full-scale tone, where the limiter takes over.  (Silent,
    // as the self-test is unless a volume is given.)
    std::vector<float> tail(static_cast<size_t>(Listener::kAudioRate / 2));
    const size_t       plotted = listener.scope().latest(tail.data(), tail.size());
    double             power   = 0.0;
    for (size_t i = 0; i < plotted; ++i)
        power += static_cast<double>(tail[i]) * tail[i];
    const double plotted_rms = plotted > 0 ? std::sqrt(power / static_cast<double>(plotted)) : 0.0;
    const double played_gain = static_cast<double>(listen_.volume) * static_cast<double>(listen_.volume) *
                               std::pow(10.0, static_cast<double>(listen_.boost_db) / 20.0);
    const double played_rms  = std::min(static_cast<double>(status.audio_rms) * played_gain, 1.0 / std::sqrt(2.0));

    bool ok = listen_.on && status.in_band && !status.searching && status.dropped_iq == 0 &&
              (status.mode == FmMode::Wide) == wide &&
              status.audio_samples >= static_cast<uint64_t>(Listener::kAudioRate) / 2 && plotted == tail.size() &&
              std::fabs(plotted_rms - played_rms) < 0.25 * played_rms + 0.02;
    const double fs = pipeline_.source().sample_rate();
    if (ok && options_.source.kind == "sim" && (!wide || fs >= SimSource::kBroadcastMinRate)) {
        // The carrier was found, and the tune comes out as loud as it should.
        double carrier = 0.0, level = 0.0, amplitude = 0.0;
        if (wide) {
            const SimSource::BroadcastStation station = SimSource::broadcast_station(fs);
            carrier   = station.offset_hz;
            level     = station.level_dbfs;
            amplitude = station.deviation_hz / FmDemodulator::kWideDeviation;  // pre- and de-emphasis cancel
        } else {
            // Every note like a 1 kHz tone of the channel's nominal deviation.
            const SimSource::VoiceChannel voice = SimSource::voice_channel(fs);
            carrier   = voice.offset_hz;
            level     = voice.level_dbfs;
            amplitude = voice.deviation_hz / (0.4 * listen_bandwidth());
        }
        const double loudness = amplitude / std::sqrt(2.0);
        ok = status.squelch_open && std::fabs(status.frequency - (options_.source.frequency + carrier)) < 2000.0 &&
             std::fabs(status.level_dbfs - level) < 2.0 && std::fabs(status.audio_rms - loudness) < 0.25 * loudness;
    }
    if (!ok) {
        LOG_ERROR("self-test: the audio is not what it should be");
        self_test_failed_ = true;
    }
    if (!wide)
        return;  // the wideband test follows
    set_listening(false);
    listen_.bandwidth = Listener::kDefaultBandwidth;
    listener.set_bandwidth(listen_.bandwidth);
#else
    (void)wide;
#endif
}

// Self-test: scanning.  The simulator's lowest signal above -50 dBFS is a
// steady carrier, the next one up its narrowband voice channel, and the next
// its DMR-like digital station: the scanner must stop on the first, classify
// it as narrowband, after a skip find and play the second, and after another
// one find the third and recognise it as DMR.  On a radio there is no telling what is on the
// air, so it only has to keep running.
// Self-test helper: goes back to the first channel in the list of those DMR
// voice was heard on, as a click on it would - with the radio tuned away
// first, so that it has to be retuned - and then checks that it is there.
void App::self_test_heard(int stage)
{
#if SPECTRO_HAVE_AUDIO
    if (!pipeline_.listener().available())
        return;
    const bool sim = options_.source.kind == "sim";
    if (heard_.channels().empty()) {
        if (sim) {
            LOG_ERROR("self-test: no DMR voice was heard");
            self_test_failed_ = true;
        } else {
            LOG_INFO("self-test: no DMR voice heard, nothing to go back to");
        }
        return;
    }
    const HeardChannel& channel = heard_.channels().front();
    IqSource&           source  = pipeline_.source();
    if (stage == 0) {
        LOG_INFO("self-test: %zu channel(s) with DMR voice; %.5f MHz: %d call(s), the last: %s",
                 heard_.channels().size(), channel.frequency / 1e6, channel.calls, channel.last_call.c_str());
        if (source.info().tunable)
            tune_to(channel.frequency + source.sample_rate());  // the channel out of the band
        listen_to_heard(channel.frequency);
        return;
    }
    const ListenStatus ls = pipeline_.listener().status();
    LOG_INFO("self-test: radio at %.5f MHz, listening to %.5f MHz (%s), DMR %s, colour code %d", source.frequency() / 1e6,
             ls.frequency / 1e6, ls.in_band ? "in the band" : "outside the band", ls.dmr.present ? "decoded" : "not decoded",
             ls.dmr.color_code);
    bool ok = listen_.on && ls.in_band && std::fabs(ls.frequency - channel.frequency) < 1.0;
    if (sim)
        ok = ok && ls.dmr.present && ls.dmr.color_code == SimSource::kDmrColorCode;
    if (!ok) {
        LOG_ERROR("self-test: going back to the channel did not work");
        self_test_failed_ = true;
    }
    set_listening(false);
#else
    (void)stage;
#endif
}

void App::self_test_scan(int stage)
{
#if SPECTRO_HAVE_AUDIO
    if (!pipeline_.listener().available())
        return;
    const bool sim = options_.source.kind == "sim";
    if (stage == 0) {
        if (sim)
            listen_.scan_threshold = -50.0f;
        start_scanning();
        if (!scanning()) {
            LOG_ERROR("self-test: scanning did not start");
            self_test_failed_ = true;
        }
        return;
    }

    const Scanner::Status st = scanner_->status();
    const ListenStatus    ls = pipeline_.listener().status();
    static const char* const kStates[] = {"off", "searching", "receiving", "holding"};
    LOG_INFO("self-test: scanner %s; %.5f MHz, %s FM, %.0f kHz wide, peak %.1f dBFS (threshold %.0f, floor %.0f "
             "dBFS); %zu signal(s) above the threshold; loudness %.2f",
             kStates[static_cast<int>(st.state)], ls.frequency / 1e6, st.wide ? "wideband" : "narrowband",
             st.occupied_hz / 1e3, static_cast<double>(st.level_dbfs), static_cast<double>(listen_.scan_threshold),
             static_cast<double>(st.floor_dbfs), scanner_->candidates().size(), static_cast<double>(ls.audio_rms));

    bool ok = scanning();
    if (sim) {
        const double fs       = pipeline_.source().sample_rate();
        const double expected = options_.source.frequency + (stage == 1   ? SimSource::carriers(fs)[0].offset_hz
                                                             : stage == 2 ? SimSource::voice_channel(fs).offset_hz
                                                                          : SimSource::digital_stations(fs)[0].offset_hz);
        ok = ok && st.state == Scanner::State::Receiving && !st.wide && !ls.searching &&
             std::fabs(ls.frequency - expected) < 2000.0;
        // The digital station: recognised for what it is.
        if (ok && stage == 3) {
            LOG_INFO("self-test: the digital station is: %s; DMR %s, colour code %d, %llu bursts decoded",
                     describe(ls.signal).c_str(), ls.dmr.present ? "decoded" : "not decoded", ls.dmr.color_code,
                     static_cast<unsigned long long>(ls.dmr.data_bursts + ls.dmr.voice_bursts));
            ok = ls.signal.kind == SignalKind::Fsk4 && ls.signal.protocol == "DMR" && ls.dmr.present &&
                 ls.dmr.repeater && ls.dmr.color_code == SimSource::kDmrColorCode;
        }
        if (ok && stage == 2) {
            // The tune, as loud as a 1 kHz tone of the channel's nominal deviation.
            const double loudness = SimSource::voice_channel(fs).deviation_hz / (0.4 * ls.bandwidth) / std::sqrt(2.0);
            ok = std::fabs(ls.audio_rms - loudness) < 0.25 * loudness;
        }
    }
    if (!ok) {
        LOG_ERROR("self-test: the scanner is not where it should be");
        self_test_failed_ = true;
    }
    if (stage < 3)
        scanner_->skip();
    else
        set_listening(false);
#else
    (void)stage;
#endif
}

// Height of the displayed surface above the floor at a point of the plot, the
// same way the vertex shader computes it (see shaders/surface.vert).
float App::surface_height(float x, float z, float* db_out, float* age_rows_out) const
{
    const float column_width = kPlotWidth / static_cast<float>(columns_);
    const float row_depth    = view_.plot_depth / static_cast<float>(history_ - 1);
    const float x0           = -0.5f * kPlotWidth + column_phase() * column_width;

    const float cf = std::clamp((x - x0) / column_width, 0.0f, static_cast<float>(columns_ - 1));
    const int   c0 = std::min(static_cast<int>(cf), columns_ - 2);
    const float fx = cf - static_cast<float>(c0);

    // Depth in rows; the newest row is held up to `scroll_phase_` rows deep.
    const float depth = std::clamp((0.5f * view_.plot_depth - z) / row_depth, 0.0f, static_cast<float>(history_ - 1));
    const float age   = std::max(depth - scroll_phase_, 0.0f);
    const int   a0    = std::min(static_cast<int>(age), history_ - 1);
    const int   a1    = std::min(a0 + 1, history_ - 1);
    const float fa    = std::min(age - static_cast<float>(a0), 1.0f);

    const int   head     = renderer_->head_row();
    const float inv_span = 1.0f / std::max(view_.db_ceiling - view_.db_floor, 1e-3f);
    const auto  db_at    = [&](int column, int age_rows) {
        const int row = ((head - age_rows) % history_ + history_) % history_;
        return history_data_[static_cast<size_t>(row) * static_cast<size_t>(columns_) + static_cast<size_t>(column)];
    };
    const auto level_of = [&](float db) { return std::clamp((db - view_.db_floor) * inv_span, 0.0f, 1.0f); };

    const float d00 = db_at(c0, a0), d10 = db_at(c0 + 1, a0);
    const float d01 = db_at(c0, a1), d11 = db_at(c0 + 1, a1);
    const float level = (level_of(d00) * (1.0f - fx) + level_of(d10) * fx) * (1.0f - fa) +
                        (level_of(d01) * (1.0f - fx) + level_of(d11) * fx) * fa;

    if (db_out) {
        // Report the nearest actual sample rather than an interpolated value.
        *db_out = db_at(fx < 0.5f ? c0 : c0 + 1, fa < 0.5f ? a0 : a1);
    }
    if (age_rows_out)
        *age_rows_out = age;
    return level * view_.plot_height;
}

// Finds the point of the surface under a position in the window by marching
// the view ray across the height field.  False if the ray misses the surface.
// `entry`, if given, receives the point where the ray enters the plot's
// bounding box - set whenever the ray passes through the box at all.
bool App::surface_under(float window_x, float window_y, Vec3& hit, Vec3* entry) const
{
    const ImGuiIO& io = ImGui::GetIO();
    if (io.DisplaySize.x <= 0.0f || io.DisplaySize.y <= 0.0f)
        return false;

    const float ndc_x = 2.0f * window_x / io.DisplaySize.x - 1.0f - lens_shift_;
    const float ndc_y = 2.0f * window_y / io.DisplaySize.y - 1.0f;
    Vec3 origin, dir;
    camera_.ray(ndc_x, ndc_y, aspect(), origin, dir);

    // Clip the ray to the box that contains the surface.
    const float column_width = kPlotWidth / static_cast<float>(columns_);
    const float row_depth    = view_.plot_depth / static_cast<float>(history_ - 1);
    const float x_min        = -0.5f * kPlotWidth + column_phase() * column_width;
    const float x_max        = x_min + static_cast<float>(columns_ - 1) * column_width;
    const float box_min[3]   = {x_min, 0.0f, -0.5f * view_.plot_depth};
    const float box_max[3]   = {x_max, view_.plot_height, 0.5f * view_.plot_depth};
    const float o[3]         = {origin.x, origin.y, origin.z};
    const float d[3]         = {dir.x, dir.y, dir.z};
    float       t_enter = 0.0f, t_exit = 1e9f;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(d[axis]) < 1e-9f) {
            if (o[axis] < box_min[axis] || o[axis] > box_max[axis])
                return false;
            continue;
        }
        float t0 = (box_min[axis] - o[axis]) / d[axis];
        float t1 = (box_max[axis] - o[axis]) / d[axis];
        if (t0 > t1)
            std::swap(t0, t1);
        t_enter = std::max(t_enter, t0);
        t_exit  = std::min(t_exit, t1);
    }
    if (t_enter > t_exit)
        return false;
    if (entry)
        *entry = origin + dir * t_enter;

    // March in steps of at most half a grid cell, then refine by bisection.
    const float step_x = std::fabs(d[0]) > 1e-6f ? 0.5f * column_width / std::fabs(d[0]) : 1e9f;
    const float step_z = std::fabs(d[2]) > 1e-6f ? 0.5f * row_depth / std::fabs(d[2]) : 1e9f;
    const float step   = std::max(std::min({step_x, step_z, 0.01f}), (t_exit - t_enter) / 20000.0f);

    const auto below = [&](float t) {
        const Vec3 p = origin + dir * t;
        return p.y <= surface_height(p.x, p.z);
    };
    float t_hit = -1.0f;
    if (below(t_enter)) {
        t_hit = t_enter;  // entered through a wall
    } else {
        float previous = t_enter;
        for (float t = t_enter + step; t <= t_exit + step; t += step) {
            const float clamped = std::min(t, t_exit);
            if (below(clamped)) {
                float lo = previous, hi = clamped;
                for (int i = 0; i < 12; ++i) {
                    const float mid = 0.5f * (lo + hi);
                    (below(mid) ? hi : lo) = mid;
                }
                t_hit = hi;
                break;
            }
            previous = clamped;
            if (clamped >= t_exit)
                break;
        }
    }
    if (t_hit < 0.0f)
        return false;

    hit = origin + dir * t_hit;
    return true;
}

// The frequency shown at a position along the plot's frequency axis.
double App::frequency_at(float x) const
{
    const IqSource& source       = pipeline_.source();
    const float     column_width = kPlotWidth / static_cast<float>(columns_);
    const float     x_min        = -0.5f * kPlotWidth + column_phase() * column_width;
    const double    column       = (x - x_min) / column_width;
    return source.frequency() + SpectrumEngine::column_frequency(column, columns_, fft_size_, source.sample_rate());
}

// What is under the mouse cursor: a marker on the surface and its values.
void App::update_cursor_readout()
{
    cursor_.valid = false;
    const ImGuiIO& io = ImGui::GetIO();
    if (!have_row_)
        return;

    float mouse_x = probe_x_, mouse_y = probe_y_;
    if (!probe_active_) {
        if (io.WantCaptureMouse || io.MousePos.x < 0.0f || io.MousePos.y < 0.0f)
            return;
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
            ImGui::IsMouseDown(ImGuiMouseButton_Middle))
            return;  // the mouse is busy moving the camera
        mouse_x = io.MousePos.x;
        mouse_y = io.MousePos.y;
    }

    Vec3 hit;
    if (!surface_under(mouse_x, mouse_y, hit))
        return;
    float      db = kFloorDb, age_rows = 0.0f;
    const float height = surface_height(hit.x, hit.z, &db, &age_rows);
    if (db <= kFloorDb + 1.0f)
        return;  // empty history or a gap marker: nothing to read

    cursor_.valid        = true;
    cursor_.world        = {hit.x, height, hit.z};
    cursor_.frequency_hz = frequency_at(hit.x);
    cursor_.level_db     = db;
    cursor_.seconds_ago  = static_cast<double>(age_rows) * row_period_;

    // Marker on the surface, and the values next to the cursor.
    float sx = 0.0f, sy = 0.0f;
    if (project_to_screen(view_projection(), cursor_.world, io.DisplaySize.x, io.DisplaySize.y, sx, sy)) {
        ImDrawList* draw = ImGui::GetBackgroundDrawList();
        const float r    = 4.5f * imgui_->ui_scale();
        draw->AddCircle(ImVec2(sx, sy), r + 1.0f, IM_COL32(0, 0, 0, 200), 0, 2.5f);
        draw->AddCircle(ImVec2(sx, sy), r, IM_COL32(255, 255, 255, 240), 0, 1.5f);
    }
    if (probe_active_)
        ImGui::SetNextWindowPos(ImVec2(mouse_x + 18.0f, mouse_y + 18.0f));
    ImGui::BeginTooltip();
    ImGui::Text("%.4f MHz", cursor_.frequency_hz / 1e6);
    ImGui::Text("%.1f dBFS", static_cast<double>(cursor_.level_db));
    ImGui::TextDisabled("%.2f s ago", cursor_.seconds_ago);
    ImGui::EndTooltip();
}

void App::handle_input()
{
    ImGuiIO& io = ImGui::GetIO();

    // ---- Mouse: choose a channel, orbit, pan, zoom ----
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
        choosing_ = false;
    if (!io.WantCaptureMouse) {
        const bool pan_drag = ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f) ||
                              ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f) ||
                              (io.KeyShift && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f));
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.KeyCtrl && !io.KeyShift) {
            // Ctrl + click: listen to the signal under the cursor.
            choosing_ = true;
            choose_channel_at(io.MousePos.x, io.MousePos.y, true);
        } else if (choosing_) {
            // ... and dragging on from there slides the channel along.  (Only
            // once the mouse has clearly moved: a click is rarely perfectly still.)
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f))
                choose_channel_at(io.MousePos.x, io.MousePos.y, false);
        } else if (pan_drag) {
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) {
                camera_.pan(io.MouseDelta.x, io.MouseDelta.y, io.DisplaySize.y);
                camera_untouched_ = false;
            }
        } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) {
                camera_.orthographic = false;  // rotating leaves the flat 2D views
                camera_.orbit(io.MouseDelta.x, io.MouseDelta.y);
                camera_untouched_ = false;
            }
        }
        if (io.MouseWheel != 0.0f) {
            camera_.zoom(io.MouseWheel);
            camera_untouched_ = false;
        }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && !io.KeyCtrl)
            apply_view(preset_);
    }

    // ---- Keyboard ----
    if (io.WantCaptureKeyboard)
        return;

    const auto pressed = [](ImGuiKey key, bool repeat = false) { return ImGui::IsKeyPressed(key, repeat); };

    if (pressed(ImGuiKey_Space)) {
        view_.paused = !view_.paused;
        toast(view_.paused ? "Paused" : "Running");
    }
    if (pressed(ImGuiKey_H))
        view_.show_panel = !view_.show_panel;
    if (pressed(ImGuiKey_F1))
        view_.show_help = !view_.show_help;
    if (pressed(ImGuiKey_1))
        apply_view(ViewPreset::Perspective);
    if (pressed(ImGuiKey_2))
        apply_view(ViewPreset::Top);
    if (pressed(ImGuiKey_3))
        apply_view(ViewPreset::Front);
    if (pressed(ImGuiKey_R))
        apply_view(preset_);
    if (pressed(ImGuiKey_A))
        auto_range();
    if (pressed(ImGuiKey_C)) {
        const int count = colormap_count();
        view_.colormap  = (view_.colormap + (io.KeyShift ? count - 1 : 1)) % count;
        toast(std::string("Colour map: ") + colormap_name(view_.colormap));
    }
    if (pressed(ImGuiKey_G))
        view_.grid = !view_.grid;
    if (pressed(ImGuiKey_W))
        view_.walls = !view_.walls;
    if (pressed(ImGuiKey_T))
        view_.trace = !view_.trace;
    if (pressed(ImGuiKey_M)) {
        view_.max_hold = !view_.max_hold;
        if (view_.max_hold)
            std::fill(max_hold_.begin(), max_hold_.end(), kFloorDb);
        toast(view_.max_hold ? "Max hold on" : "Max hold off");
    }
    if (pressed(ImGuiKey_L))
        toggle_listening();
    if (pressed(ImGuiKey_S)) {
        if (scanning() || channel_scanning()) {
            stop_scanning();
            toast("Scanning off: staying on this channel");
        } else {
            start_scanning();
        }
    }
    if (pressed(ImGuiKey_N) && (scanning() || channel_scanning()))
        skip_signal();
    if (pressed(ImGuiKey_P) || pressed(ImGuiKey_F12))
        take_screenshot({});
    if (pressed(ImGuiKey_F11) || pressed(ImGuiKey_F))
        toggle_fullscreen();
    if (pressed(ImGuiKey_Escape) && fullscreen_)
        toggle_fullscreen();
    if (pressed(ImGuiKey_Q) && io.KeyCtrl)
        request_quit();

    // Tuning: arrow keys step by 1/20 of the span (Shift: 1/200, Ctrl: half the span).
    const double span = pipeline_.source().sample_rate();
    const double step = io.KeyCtrl ? span * 0.5 : io.KeyShift ? span / 200.0 : span / 20.0;
    if (pressed(ImGuiKey_LeftArrow, true))
        tune_to(pipeline_.source().frequency() - step);
    if (pressed(ImGuiKey_RightArrow, true))
        tune_to(pipeline_.source().frequency() + step);
    if (pressed(ImGuiKey_UpArrow, true))
        set_gain(pipeline_.source().gain() + (io.KeyShift ? 5.0 : 1.0));
    if (pressed(ImGuiKey_DownArrow, true))
        set_gain(pipeline_.source().gain() - (io.KeyShift ? 5.0 : 1.0));

    if (pressed(ImGuiKey_LeftBracket))
        change_fft_size(-1);
    if (pressed(ImGuiKey_RightBracket))
        change_fft_size(+1);
    if (pressed(ImGuiKey_Comma, true) || pressed(ImGuiKey_Period, true)) {
        SpectrumParams params = pipeline_.engine().params();
        params.rows_per_sec   = std::clamp(params.rows_per_sec * (pressed(ImGuiKey_Period, true) ? 1.25 : 0.8),
                                           2.0, 1000.0);
        pipeline_.engine().set_params(params);
    }
}

// ---------------------------------------------------------------------------
// Listening
// ---------------------------------------------------------------------------

void App::set_listening(bool on)
{
    if (on == listen_.on)
        return;
#if SPECTRO_HAVE_AUDIO
    Listener& listener = pipeline_.listener();
    if (!on) {
        stop_scanning();
        listener.set_enabled(false);
        if (audio_)
            audio_->stop();
        listen_.on = false;
        return;
    }
    if (!listener.available()) {
        toast("The sample rate is too low to hold an FM channel");
        return;
    }
    if (!audio_)
        audio_ = std::make_unique<AudioOutput>(listener.audio(), Listener::kAudioRate);
    std::string message;
    const bool  started = audio_->start(message);
    if (!message.empty()) {
        LOG_WARN("%s", message.c_str());
        toast(message);
    }
    if (!started)
        return;
    listener.set_bandwidth(listen_.bandwidth);
    listener.set_deemphasis(listen_.deemphasis);
    listener.set_volume(listen_.volume);
    listener.set_boost(listen_.boost_db);
    listener.set_squelch(listen_.squelch);
    listener.set_dmr_only(listen_.dmr_only);
    listener.set_p25_only(listen_.p25_only);
    listener.set_enabled(true);
    listen_.on = true;
#else
    toast("This build has no audio support");
#endif
}

// The L key and the "Listen" check box.  Turning the audio on starts scanning
// for signals; Ctrl + click chooses one instead.
void App::toggle_listening()
{
    if (listen_.on)
        set_listening(false);
    else
        start_scanning();
}

// Listens by scanning: the scanner finds the signals, the listener plays them.
// (P25 only, with control channels to go through: see start_channel_scan().)
void App::start_scanning()
{
    if (scanning() || channel_scanning())
        return;
    if (channel_scan_wanted()) {
        start_channel_scan();
        return;
    }
    // Without a threshold the scanner waits until it knows the noise floor
    // (see update_listening()), and takes nothing before.
    constexpr float kNothingYet = 1e9f;
    scanner_->set_threshold(std::isnan(listen_.scan_threshold) ? kNothingYet : listen_.scan_threshold);
    scanner_->set_hang(listen_.scan_hang);
    scanner_->set_dmr_only(listen_.dmr_only);
    scanner_->set_p25_only(listen_.p25_only);
    scanner_->set_dmr_hold(listen_.dmr_hold);
    scanner_->start();  // mutes the listener until it finds something
    set_listening(true);
    if (!listen_.on) {
        scanner_->stop();
        return;
    }
    listen_.selected = true;
    toast("Scanning for signals (S: stop, N: next)");
}

void App::stop_scanning()
{
    if (scanner_)
        scanner_->stop();  // the listener stays where it is
    follower_.stop_hunting();  // likewise; a control channel it is on is still followed
}

// Scanning the control channels: the trunk follower tries them in turn until
// one is heard, follows its calls, and when it is lost looks again.
void App::start_channel_scan()
{
    set_listening(true);
    if (!listen_.on)
        return;
    listen_.selected = true;
    set_following(true);
    hunt_tries_               = 0;
    const FollowAction action = follower_.hunt(listen_.frequency, follow_clock());
    if (follower_.state() == TrunkFollower::State::Hunting)
        LOG_INFO("P25: looking for a control channel, %zu listed", follower_.control_channels().size());
    if (action.tune)
        follow_to(action);
    toast("Scanning the control channels (S: stop, N: next)");
}

// The control channels of --control-channels: those this radio can receive
// are the ones scanning goes through.
void App::use_control_channels(const std::vector<double>& listed)
{
    if (listed.empty())
        return;
    const IqSource&   source = pipeline_.source();
    const SourceInfo& info   = source.info();
    const double      fs     = source.sample_rate();
    std::vector<double> usable;
    for (const double hz : listed) {
        // A radio that cannot be tuned receives only the band it is on.
        const bool reachable = info.tunable ? hz >= info.frequency_range.min && hz <= info.frequency_range.max
                                            : std::fabs(hz - source.frequency()) + 0.5 * Listener::kDefaultBandwidth <=
                                                  0.45 * fs;
        if (reachable)
            usable.push_back(hz);
        else
            LOG_WARN("--control-channels: %s is out of this radio's reach; left out", format_frequency(hz).c_str());
    }
    follower_.set_control_channels(usable);
    if (usable.empty()) {
        LOG_WARN("--control-channels: none of them can be received here");
        return;
    }
    std::string list;
    for (const double hz : usable) {
        char text[32];
        std::snprintf(text, sizeof(text), "%s%.5f", list.empty() ? "" : ", ", hz / 1e6);
        list += text;
    }
    LOG_INFO("P25: %zu control channel(s) to look for: %s MHz%s", usable.size(), list.c_str(),
             options_.p25_only ? "" : " (scanning goes through them with P25 only)");
}

// A centre frequency for the radio that holds `hz` and every control channel
// of the list - so that looking through them needs no more retuning - with
// none of them on the DC spike in the middle; NaN if there is none.
double App::control_channels_centre(double hz) const
{
    const std::vector<double>& listed = follower_.control_channels();
    if (listed.empty())
        return std::numeric_limits<double>::quiet_NaN();
    const auto [lowest, highest] = std::minmax_element(listed.begin(), listed.end());
    const double middle = 0.5 * (std::min(*lowest, hz) + std::max(*highest, hz));
    const double fs     = pipeline_.source().sample_rate();
    const double width  = Listener::kDefaultBandwidth;
    for (const double shift : {0.0, 1.0, -1.0, 2.0, -2.0}) {
        const double centre = middle + shift * width;
        const auto   fits   = [&](double f) {
            return std::fabs(f - centre) + 0.5 * width <= 0.45 * fs && std::fabs(f - centre) >= width;
        };
        if (fits(hz) && std::all_of(listed.begin(), listed.end(), fits))
            return centre;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

bool App::scan_locked() const
{
    if (!scanning())
        return false;
    const Scanner::State state = scanner_->status().state;
    return state == Scanner::State::Receiving || state == Scanner::State::Holding;
}

// Listens to a channel DMR voice was heard on: narrowband, and with the radio
// retuned first if the channel is outside the band it receives (the channel
// then a quarter of the band above the centre, clear of the DC spike).
void App::listen_to_heard(double hz, int phase2_slot)
{
    IqSource&    source = pipeline_.source();
    const double fs     = source.sample_rate();
    const double width  = Listener::kDefaultBandwidth;
    if (std::fabs(hz - source.frequency()) + 0.5 * width > 0.45 * fs) {
        if (!source.info().tunable) {
            toast("That channel is outside the band received, and this source cannot be tuned");
            return;
        }
        tune_to(hz - 0.25 * fs);
    }
    listen_.bandwidth = width;
    pipeline_.listener().set_bandwidth(width);
    listen_to(hz, 0.0);
    pipeline_.listener().set_p25_phase2_slot(phase2_slot);
    char text[64];
    if (phase2_slot >= 0)
        std::snprintf(text, sizeof(text), "Listening to %.5f MHz, slot %d", hz / 1e6, phase2_slot + 1);
    else
        std::snprintf(text, sizeof(text), "Listening to %.5f MHz", hz / 1e6);
    toast(text);
}

// Following a P25 system's calls: on or off.
void App::set_following(bool on)
{
    listen_.follow = on;
    follower_.set_enabled(on);
}

// Where the follower says to go: a call's voice channel, back to the control
// channel, or the next control channel of the list to try.  If it lies
// outside the band received the radio is retuned - where possible so that
// the control channel fits too, and the next call needs no retune; or so
// that all the control channels of the list do, and the next try needs none.
void App::follow_to(const FollowAction& action)
{
    IqSource&    source = pipeline_.source();
    const double fs     = source.sample_rate();
    const double width  = Listener::kDefaultBandwidth;
    const auto   fits   = [&](double centre, double hz) { return std::fabs(hz - centre) + 0.5 * width <= 0.45 * fs; };
    const TrunkFollower::State state = follower_.state();
    if (!fits(source.frequency(), action.frequency)) {
        if (!source.info().tunable) {
            toast("The call is outside the band received, and this source cannot be tuned: not following calls");
            set_following(false);
            return;
        }
        // Otherwise the channel a quarter of the band above the centre, clear of the DC spike.
        double centre = action.frequency - 0.25 * fs;
        if (state == TrunkFollower::State::Voice) {
            const double control = follower_.control_frequency();
            const double both    = 0.5 * (action.frequency + control);
            if (fits(both, action.frequency) && fits(both, control))
                centre = both;
        } else if (const double all = control_channels_centre(action.frequency); !std::isnan(all)) {
            centre = all;
        }
        tune_to(centre);
    }
    listen_.bandwidth = width;
    pipeline_.listener().set_bandwidth(width);
    select_channel(action.frequency, 0.0);  // (without ending a scan of the control channels)
    pipeline_.listener().set_p25_phase2_slot(action.slot);
    switch (state) {
    case TrunkFollower::State::Voice:
        LOG_INFO("P25: %s", follower_.describe().c_str());
        break;
    case TrunkFollower::State::Hunting:
        LOG_DEBUG("P25: %s", follower_.describe().c_str());
        if (++hunt_tries_ == follower_.control_channels().size() + 1)
            LOG_INFO("P25: no control channel heard on any of the %zu listed; still looking",
                     follower_.control_channels().size());
        break;
    default:
        LOG_INFO("P25: back to the control channel at %.5f MHz", action.frequency / 1e6);
        break;
    }
}

// Selects the channel to listen to and turns the audio on.  With `search_hz`
// the receiver looks for the carrier that far around `hz` and centres on it.
// Choosing a channel ends scanning.
void App::listen_to(double hz, double search_hz)
{
    stop_scanning();
    select_channel(hz, search_hz);
}

// The same, but scanning (of the control channels) goes on: the follower's.
void App::select_channel(double hz, double search_hz)
{
    listen_.selected  = true;
    listen_.frequency = hz;
    listen_mhz_       = hz / 1e6;
    pipeline_.listener().tune(hz, search_hz);
    pipeline_.listener().set_p25_phase2_slot(-1);  // whichever speaks, unless the caller says
    set_listening(true);
}

// Listens to what the plot shows at a position in the window.  A click means
// "that signal", but the display is only as fine as its columns and the hand
// not much finer, so with `find_carrier` the receiver looks for the carrier
// nearby; without it the channel is placed exactly where the cursor is.
bool App::choose_channel_at(float window_x, float window_y, bool find_carrier)
{
    // Above the trace in the front view the ray skims over the surface without
    // touching it; where it enters the plot is then the frequency meant.
    constexpr float kNone = -1e30f;
    Vec3            hit, entry{kNone, 0.0f, 0.0f};
    if (!surface_under(window_x, window_y, hit, &entry)) {
        if (entry.x == kNone)
            return false;
        hit = entry;
    }
    const double column_hz = pipeline_.source().sample_rate() / columns_;
    listen_to(frequency_at(hit.x), find_carrier ? std::max(0.5 * column_hz, 0.25 * listen_bandwidth()) : 0.0);
    return true;
}

void App::update_listening()
{
    listen_status_ = pipeline_.listener().status();
    // Tell the terminal what kind of signal the channel carries, when that changes.
    {
        const SignalClass& signal = listen_status_.signal;
        if (!listen_.on || signal.kind == SignalKind::Analysing) {
            logged_signal_.clear();
        } else {
            char key[160];
            std::snprintf(key, sizeof(key), "%.0f %d %s %.0f %.1f %d", listen_status_.frequency,
                          static_cast<int>(signal.kind), signal.protocol.c_str(), signal.symbol_rate, signal.ctcss_hz,
                          signal.stereo);
            if (logged_signal_ != key) {
                logged_signal_ = key;
                const std::string typical = typical_systems(signal);
                LOG_INFO("Signal at %.5f MHz: %s (SNR %.0f dB)%s%s", listen_status_.frequency / 1e6,
                         describe(signal).c_str(), static_cast<double>(signal.snr_db),
                         typical.empty() ? "" : " - typical of ", typical.c_str());
            }
        }
    }
    // Remember the channels DMR or P25 voice comes through on.
    const bool settled = listen_.on && listen_status_.in_band && !listen_status_.searching;
    heard_.update(settled ? listen_status_.frequency : std::numeric_limits<double>::quiet_NaN(), listen_status_.dmr,
                  listen_status_.p25, std::chrono::duration<double>(Clock::now().time_since_epoch()).count(),
                  listen_status_.p25p2);

    // And of DMR calls as they start and as their callers become known.
    for (int s = 0; s < 2; ++s) {
        const DmrCall& call = listen_status_.dmr.slot[s];
        std::string    key;
        if (listen_.on && listen_status_.dmr.present && call.active)
            key = describe_dmr_slot(listen_status_.dmr, s);
        if (key != logged_dmr_[s]) {
            if (!key.empty())
                LOG_INFO("DMR at %.5f MHz, time slot %d: %s", listen_status_.frequency / 1e6, s + 1, key.c_str());
            logged_dmr_[s] = key;
        }
    }
    {
        std::string key;
        if (listen_.on && listen_status_.p25.present && listen_status_.p25.call.active)
            key = describe_p25(listen_status_.p25);
        if (key != logged_p25_) {
            if (!key.empty())
                LOG_INFO("P25 at %.5f MHz (NAC 0x%03X): %s", listen_status_.frequency / 1e6,
                         static_cast<unsigned>(std::max(listen_status_.p25.nac, 0)), key.c_str());
            logged_p25_ = key;
        }
        // P25 Phase 2: the call on each slot.
        const P25Phase2Status& p2 = listen_status_.p25p2;
        for (int s = 0; s < 2; ++s) {
            std::string slot_key;
            if (listen_.on && p2.present && p2.slot[s].active)
                slot_key = describe_p25_phase2_slot(p2, s);
            if (slot_key != logged_p25p2_[s]) {
                if (!slot_key.empty())
                    LOG_INFO("P25 Phase 2 at %.5f MHz, slot %d: %s", listen_status_.frequency / 1e6, s + 1,
                             slot_key.c_str());
                logged_p25p2_[s] = slot_key;
            }
        }
        // A control channel: its system once, then each call it grants.
        const P25Status& p25 = listen_status_.p25;
        if (listen_.on && p25.present && p25.site.system >= 0 && p25.site.wacn >= 0 && p25.site.site >= 0) {
            char site[160];
            std::snprintf(site, sizeof(site), "system 0x%03X, WACN 0x%05X, NAC 0x%03X, RFSS %d, site %d",
                          static_cast<unsigned>(p25.site.system), static_cast<unsigned>(std::max(p25.site.wacn, 0)),
                          static_cast<unsigned>(std::max(p25.nac, 0)), p25.site.rfss, p25.site.site);
            // Once per system - and again if more of its channel plans turn
            // up, but not as they are heard afresh on coming back to it.
            const std::string phases = p25.site.voice_phases();
            if (logged_p25_site_ != site || phases.size() > logged_p25_phases_.size()) {
                LOG_INFO("P25 control channel at %.5f MHz: %s; voice in %s", listen_status_.frequency / 1e6, site,
                         phases.empty() ? "?" : phases.c_str());
                logged_p25_site_   = site;
                logged_p25_phases_ = phases;
            }
            for (const P25Grant& g : p25.grants) {
                if (g.frequency <= 0.0)
                    continue;
                char grant[160];
                std::snprintf(grant, sizeof(grant), "%s%s on %.5f MHz%s%s", p25_grant_destination(g).c_str(),
                              g.source ? (" from " + std::to_string(g.source)).c_str() : "", g.frequency / 1e6,
                              g.slot >= 0 ? (", Phase 2 slot " + std::to_string(g.slot % 2 + 1)).c_str() : ", Phase 1",
                              g.encrypted ? ", encrypted" : "");
                if (std::find(logged_p25_grants_.begin(), logged_p25_grants_.end(), grant) != logged_p25_grants_.end())
                    continue;
                LOG_INFO("P25 grant: %s", grant);
                logged_p25_grants_.push_back(grant);
                if (logged_p25_grants_.size() > 32)
                    logged_p25_grants_.erase(logged_p25_grants_.begin());
            }
        }
    }
    if (listen_.on) {
        listen_.frequency = listen_status_.frequency;  // a search moves it onto the carrier
        if (scanning())
            listen_.bandwidth = listen_status_.bandwidth;  // the scanner chose it
    }
    if (scanning()) {
        // A log of what the scanner stops on.
        const Scanner::Status scan = scanner_->status();
        if (scan.state == Scanner::State::Receiving && logged_scan_state_ == Scanner::State::Searching) {
            LOG_INFO("Scanner: %s FM at %.5f MHz (%.0f kHz wide, peak %.1f dBFS)",
                     scan.wide ? "wideband" : "narrowband", scan.frequency / 1e6, scan.occupied_hz / 1e3,
                     static_cast<double>(scan.level_dbfs));
        } else if (scan.state == Scanner::State::Searching && (logged_scan_state_ == Scanner::State::Receiving ||
                                                               logged_scan_state_ == Scanner::State::Holding)) {
            LOG_INFO("Scanner: moving on from %.5f MHz", scan.frequency / 1e6);
        }
        logged_scan_state_ = scan.state;
    } else {
        logged_scan_state_ = Scanner::State::Off;
    }
    if (scanning() && std::isnan(listen_.scan_threshold)) {
        // No threshold given: some way above the noise, once that is known.
        const Scanner::Status scan = scanner_->status();
        if (!scan.warming_up && scan.floor_dbfs > -190.0f) {
            listen_.scan_threshold = std::round(scan.floor_dbfs + Scanner::kDefaultThresholdAboveFloor);
            scanner_->set_threshold(listen_.scan_threshold);
        }
    }
    // What was said, as it is transcribed: for the window, and the log.
    if (transcriber_) {
        for (Transcript& t : transcriber_->since(transcript_serial_)) {
            transcript_serial_ = t.serial;
            LOG_INFO("Transcript: %s", format_transcript(t).c_str());
            transcripts_.push_back(std::move(t));
        }
        if (transcripts_.size() > Transcriber::kKept)
            transcripts_.erase(transcripts_.begin(), transcripts_.end() - static_cast<ptrdiff_t>(Transcriber::kKept));
    }
    // Following a P25 system's calls (not while scanning for signals: the
    // scanner chooses the channels then).
    if (listen_.on && !scanning() && follower_.enabled()) {
        const TrunkFollower::State before = follower_.state();
        const FollowAction         action =
            follower_.update(settled ? listen_status_.frequency : std::numeric_limits<double>::quiet_NaN(),
                             listen_status_.p25, listen_status_.p25p2, follow_clock());
        const TrunkFollower::State after = follower_.state();
        if (before == TrunkFollower::State::Hunting && after == TrunkFollower::State::Control) {
            hunt_tries_ = 0;
            LOG_INFO("P25: control channel at %.5f MHz (%d of %zu listed): following its calls",
                     follower_.control_frequency() / 1e6, follower_.listed_index() + 1,
                     follower_.control_channels().size());
        } else if (before == TrunkFollower::State::Control && after == TrunkFollower::State::Hunting) {
            hunt_tries_ = 0;
            LOG_INFO("P25: the control channel at %.5f MHz has gone quiet: looking for one",
                     follower_.control_frequency() / 1e6);
        }
        if (action.tune)
            follow_to(action);
    }
}

// Whether the channel lies inside the band being received (and displayed).
bool App::listen_in_band() const
{
    const IqSource& source = pipeline_.source();
    return std::fabs(listen_.frequency - source.frequency()) + 0.5 * listen_bandwidth() <=
           0.5 * source.sample_rate();
}

double App::listen_bandwidth() const
{
    return FmDemodulator::usable_bandwidth(pipeline_.source().sample_rate(), Listener::kAudioRate, listen_.bandwidth);
}

bool App::listen_wide() const
{
    return FmDemodulator::mode_for(listen_bandwidth()) == FmMode::Wide;
}

// The channel being played, as a box around its part of the plot: as wide as
// the channel filter, as deep as the history, as tall as the level scale.  It
// is bright while there is something to hear and dim while the squelch mutes.
void App::add_listen_box(float pixel_ratio)
{
    if (!listen_.on || (scanning() && !scan_locked()))
        return;
    const IqSource& source = pipeline_.source();
    const double    span   = source.sample_rate();
    const double    lo     = (listen_.frequency - 0.5 * listen_bandwidth() - source.frequency()) / span;
    const double    hi     = (listen_.frequency + 0.5 * listen_bandwidth() - source.frequency()) / span;
    if (hi <= -0.5 || lo >= 0.5)
        return;  // not on the display at all

    const float x0 = static_cast<float>(std::max(lo, -0.5)) * kPlotWidth;
    const float x1 = static_cast<float>(std::min(hi, 0.5)) * kPlotWidth;
    // A hair outside the data, like the frame, so that its lines do not fight
    // the surface's walls for the same depth.
    const float eps = 0.003f * kPlotWidth;
    const float z0  = 0.5f * view_.plot_depth + eps;
    const float z1  = -0.5f * view_.plot_depth - eps;
    const float y0  = 0.0f, y1 = view_.plot_height;

    const bool audible = listen_in_band() && listen_status_.squelch_open &&
                         (!scanning() || scanner_->status().state == Scanner::State::Receiving);
    const uint32_t color   = audible ? pack_rgba(110, 255, 190, 255) : pack_rgba(150, 200, 180, 190);
    const float    width   = (audible ? 1.8f : 1.3f) * pixel_ratio;
    const auto     edge    = [&](Vec3 a, Vec3 b) {
        LineInstance line{};
        line.start[0]   = a.x;
        line.start[1]   = a.y;
        line.start[2]   = a.z;
        line.width      = width;
        line.end[0]     = b.x;
        line.end[1]     = b.y;
        line.end[2]     = b.z;
        line.depth_bias = 1e-4f;
        line.color      = color;
        lines_.push_back(line);
    };
    for (const float x : {x0, x1}) {
        for (const float y : {y0, y1})
            edge({x, y, z0}, {x, y, z1});  // along the time axis
    }
    for (const float z : {z0, z1}) {
        edge({x0, y0, z}, {x1, y0, z});
        edge({x0, y1, z}, {x1, y1, z});
        edge({x0, y0, z}, {x0, y1, z});
        edge({x1, y0, z}, {x1, y1, z});
    }
}

// While scanning: the threshold as a plane through the plot, so that the
// signals poking through it are the ones the scanner will take, and a bar
// under each of them along the front edge.
void App::add_scan_marks(float pixel_ratio)
{
    if (!scanning() || std::isnan(listen_.scan_threshold))
        return;
    const float span = view_.db_ceiling - view_.db_floor;
    const float level = (listen_.scan_threshold - view_.db_floor) / std::max(span, 1e-3f);
    if (level < 0.0f || level > 1.0f)
        return;  // off the level scale

    const auto add = [&](Vec3 a, Vec3 b, uint32_t color, float width) {
        LineInstance line{};
        line.start[0]   = a.x;
        line.start[1]   = a.y;
        line.start[2]   = a.z;
        line.width      = width;
        line.end[0]     = b.x;
        line.end[1]     = b.y;
        line.end[2]     = b.z;
        line.depth_bias = 1e-4f;
        line.color      = color;
        lines_.push_back(line);
    };
    const float    eps   = 0.003f * kPlotWidth;
    const float    y     = level * view_.plot_height;
    const float    xl    = -0.5f * kPlotWidth - eps, xr = 0.5f * kPlotWidth + eps;
    const float    zf    = 0.5f * view_.plot_depth + eps, zb = -0.5f * view_.plot_depth - eps;
    const uint32_t plane = pack_rgba(255, 190, 80, 150);
    add({xl, y, zf}, {xr, y, zf}, plane, 1.2f * pixel_ratio);
    add({xl, y, zb}, {xr, y, zb}, plane, 1.0f * pixel_ratio);
    add({xl, y, zf}, {xl, y, zb}, plane, 1.0f * pixel_ratio);
    add({xr, y, zf}, {xr, y, zb}, plane, 1.0f * pixel_ratio);

    const IqSource& source = pipeline_.source();
    const auto      x_of   = [&](double hz) {
        const double f = std::clamp((hz - source.frequency()) / source.sample_rate(), -0.5, 0.5);
        return static_cast<float>(f) * kPlotWidth;
    };
    for (const DetectedSignal& s : scanner_->candidates()) {
        // At least a few pixels wide, so that narrow signals show too.
        float x0 = x_of(s.low_hz), x1 = x_of(s.high_hz);
        const float min_width = 0.004f * kPlotWidth;
        if (x1 - x0 < min_width) {
            const float mid = 0.5f * (x0 + x1);
            x0 = mid - 0.5f * min_width;
            x1 = mid + 0.5f * min_width;
        }
        add({x0, 0.0f, zf}, {x1, 0.0f, zf}, pack_rgba(255, 190, 80, 235), 3.0f * pixel_ratio);
    }
}

// Names the box: the frequency being played, above its front edge.
void App::draw_listen_label()
{
    if (!listen_.on || !view_.labels || (scanning() && !scan_locked()))
        return;
    const IqSource& source = pipeline_.source();
    const double    center = (listen_.frequency - source.frequency()) / source.sample_rate();
    if (std::fabs(center) > 0.5)
        return;

    const ImGuiIO& io = ImGui::GetIO();
    const Vec3     anchor{static_cast<float>(center) * kPlotWidth, view_.plot_height, 0.5f * view_.plot_depth};
    float          x = 0.0f, y = 0.0f;
    if (!project_to_screen(view_projection(), anchor, io.DisplaySize.x, io.DisplaySize.y, x, y))
        return;

    char              text[96];
    std::string       kind = short_name(listen_status_.signal);
    const DmrStatus&  dmr  = listen_status_.dmr;
    if (dmr.present) {
        // The call being heard, or else any call.
        int slot = dmr.playing;
        for (int s = 0; s < 2 && slot < 0; ++s)
            slot = dmr.slot[s].active ? s : -1;
        kind = "DMR";
        if (slot >= 0) {
            const std::string to = dmr_destination(dmr.slot[slot]);
            kind += " TS" + std::to_string(slot + 1) + (to.empty() ? "" : " " + to);
            if (dmr.slot[slot].encrypted)
                kind += " (encrypted)";
        }
    }
    const P25Status& p25 = listen_status_.p25;
    if (p25.present && !dmr.present) {
        kind = "P25";
        if (p25.call.active) {
            const std::string to = p25_destination(p25.call);
            kind += to.empty() ? "" : " " + to;
            if (p25.call.encrypted)
                kind += " (encrypted)";
        }
    }
    std::snprintf(text, sizeof(text), "%s %.5f MHz%s%s", listen_wide() ? "WFM" : "NFM", listen_.frequency / 1e6,
                  kind.empty() ? "" : "  ", kind.c_str());
    const bool wanted = (listen_status_.dmr_only && dmr.present) || (listen_status_.p25_only && p25.present);
    const bool filtered = listen_status_.dmr_only || listen_status_.p25_only;
    const bool   audible = listen_in_band() && (listen_status_.squelch_open || dmr.present || p25.present) &&
                         (!filtered || wanted) &&
                         (!scanning() || scanner_->status().state == Scanner::State::Receiving);
    const ImVec2 size    = ImGui::CalcTextSize(text);
    const float  line    = ImGui::GetFontSize();
    const ImVec2 pad(0.45f * line, 0.2f * line);
    const ImVec2 pos(std::round(x - 0.5f * size.x), std::round(y - size.y - pad.y - 0.5f * line));
    ImDrawList*  draw = ImGui::GetBackgroundDrawList();
    draw->AddRectFilled(ImVec2(pos.x - pad.x, pos.y - pad.y), ImVec2(pos.x + size.x + pad.x, pos.y + size.y + pad.y),
                        IM_COL32(8, 12, 16, 190), 0.3f * line);
    draw->AddText(pos, audible ? IM_COL32(110, 255, 190, 255) : IM_COL32(150, 200, 180, 220), text);
}

// ---------------------------------------------------------------------------
// User interface
// ---------------------------------------------------------------------------

void App::draw_panel()
{
    IqSource&              source = pipeline_.source();
    SpectrumEngine&        engine = pipeline_.engine();
    const SourceInfo&      info   = source.info();
    const AcquisitionStats acq    = pipeline_.acquisition().stats();
    const EngineStats      es     = engine.stats();
    const float            scale  = imgui_->ui_scale();

    // Fixed width; the height follows the content and scrolls in a short window.
    const float panel_width = 330.0f * scale;
    const float max_height  = std::max(120.0f * scale, ImGui::GetIO().DisplaySize.y - 24.0f * scale);
    ImGui::SetNextWindowPos(ImVec2(12.0f * scale, 12.0f * scale), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(panel_width, 0.0f), ImVec2(panel_width, max_height));
    if (!ImGui::Begin("Spectrogram", &view_.show_panel,
                      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    // While the panel sits at the left edge, the plot is centred in what remains.
    {
        const ImVec2 pos  = ImGui::GetWindowPos();
        const ImVec2 size = ImGui::GetWindowSize();
        const float  display_width = ImGui::GetIO().DisplaySize.x;
        if (pos.x < 40.0f * scale && display_width > 1.0f)
            panel_fraction_ = std::clamp((pos.x + size.x) / display_width, 0.0f, 0.6f);
    }
    ImGui::PushItemWidth(-110.0f * scale);

    // ---- Radio ----
    ImGui::SeparatorText("Radio");
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", info.description.c_str());
    ImGui::PopStyleColor();

    ImGui::BeginDisabled(!info.tunable);
    const double step_mhz = source.sample_rate() / 20.0 / 1e6;
    if (ImGui::InputDouble("MHz", &tune_mhz_, step_mhz, step_mhz * 10.0, "%.4f",
                           ImGuiInputTextFlags_EnterReturnsTrue))
        tune_to(tune_mhz_ * 1e6);
    if (!ImGui::IsItemActive() && !ImGui::IsItemFocused())
        tune_mhz_ = source.frequency() / 1e6;
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!info.has_gain);
    if (ImGui::SliderInt("Gain", &gain_db_, static_cast<int>(std::floor(info.gain_range.min)),
                         static_cast<int>(std::ceil(info.gain_range.max)), "%d dB", ImGuiSliderFlags_AlwaysClamp))
        set_gain(gain_db_);
    ImGui::EndDisabled();

    ImGui::Text("Rate   %s  (measured %.3f M)", format_si(source.sample_rate(), "S/s").c_str(),
                acq.measured_rate / 1e6);
    const uint64_t lost = acq.lost_samples + acq.ring_dropped;
    if (lost > 0 || acq.discontinuities > 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f), "Lost   %llu samples in %llu break(s)",
                           static_cast<unsigned long long>(lost),
                           static_cast<unsigned long long>(acq.discontinuities));
    } else {
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f), "Lost   0 samples");
    }
    ImGui::Text("Load   DSP %.0f %%   buffer %.1f %%", static_cast<double>(es.cpu_load) * 100.0,
                acq.ring_fill * 100.0);

    // ---- Spectrum ----
    ImGui::SeparatorText("Spectrum");
    SpectrumParams params  = engine.params();
    bool           changed = false;

    char preview[64];
    std::snprintf(preview, sizeof(preview), "%d", params.fft_size);
    if (ImGui::BeginCombo("FFT size", preview)) {
        for (const int size : kFftSizes) {
            char label[64];
            std::snprintf(label, sizeof(label), "%d", size);
            if (ImGui::Selectable(label, size == params.fft_size)) {
                params.fft_size = size;
                changed         = true;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::Text("Resolution bandwidth %s%s", format_si(es.rbw_hz, "Hz").c_str(),
                es.fft_size != params.fft_size ? "  (preparing...)" : "");
    ImGui::PopStyleColor();

    if (ImGui::BeginCombo("Window", window_name(params.window))) {
        for (int i = 0; i < static_cast<int>(WindowType::Count); ++i) {
            const WindowType type = static_cast<WindowType>(i);
            if (ImGui::Selectable(window_name(type), type == params.window)) {
                params.window = type;
                changed       = true;
            }
        }
        ImGui::EndCombo();
    }

    static const char* const kOverlapNames[] = {"None", "50 %", "75 %"};
    static const int         kOverlapValues[] = {0, 50, 75};
    int overlap_index = params.overlap_pct >= 63 ? 2 : params.overlap_pct >= 25 ? 1 : 0;
    if (ImGui::Combo("Overlap", &overlap_index, kOverlapNames, 3)) {
        params.overlap_pct = kOverlapValues[overlap_index];
        changed            = true;
    }

    int detector = static_cast<int>(params.detector);
    static const char* const kDetectorNames[] = {"Average (RMS)", "Peak"};
    if (ImGui::Combo("Detector", &detector, kDetectorNames, 2)) {
        params.detector = static_cast<Detector>(detector);
        changed         = true;
    }

    float speed = static_cast<float>(params.rows_per_sec);
    if (ImGui::SliderFloat("Speed", &speed, 2.0f, 1000.0f, "%.0f rows/s",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        params.rows_per_sec = speed;
        changed             = true;
    }
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::Text("History %.1f s  (%.1f ms per row)", (history_ - 1) * row_period_, row_period_ * 1e3);
    ImGui::PopStyleColor();

    if (ImGui::Checkbox("Remove DC offset", &params.dc_block))
        changed = true;
    if (changed)
        engine.set_params(params);

    // ---- Level scale ----
    ImGui::SeparatorText("Level");
    float floor = view_.db_floor, ceiling = view_.db_ceiling;
    if (ImGui::DragFloatRange2("Range", &floor, &ceiling, 0.5f, -200.0f, 20.0f, "%.0f dB", "%.0f dB",
                               ImGuiSliderFlags_AlwaysClamp)) {
        if (ceiling < floor + 5.0f)
            ceiling = floor + 5.0f;
        view_.db_floor      = floor;
        view_.db_ceiling    = ceiling;
        range_animating_    = false;
        initial_auto_range_ = false;
    }
    if (ImGui::Button("Auto"))
        auto_range();
    ImGui::SameLine();
    ImGui::Checkbox("Track", &view_.track_levels);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Keep fitting the scale to the noise floor and the strongest signal");

    // ---- Display ----
    ImGui::SeparatorText("Display");
    if (ImGui::BeginCombo("Colours", colormap_name(view_.colormap))) {
        for (int i = 0; i < colormap_count(); ++i) {
            if (ImGui::Selectable(colormap_name(i), i == view_.colormap))
                view_.colormap = i;
        }
        ImGui::EndCombo();
    }
    ImGui::SliderFloat("Height", &view_.plot_height, 0.05f, 1.5f, "%.2f");
    ImGui::SliderFloat("Depth", &view_.plot_depth, 0.6f, 6.0f, "%.1f");
    ImGui::SliderFloat("Relief", &view_.relief, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Fade", &view_.fog, 0.0f, 1.0f, "%.2f");

    ImGui::Checkbox("Walls", &view_.walls);
    ImGui::SameLine();
    ImGui::Checkbox("Grid", &view_.grid);
    ImGui::SameLine();
    ImGui::Checkbox("Labels", &view_.labels);
    ImGui::Checkbox("Trace", &view_.trace);
    ImGui::SameLine();
    if (ImGui::Checkbox("Max hold", &view_.max_hold) && view_.max_hold)
        std::fill(max_hold_.begin(), max_hold_.end(), kFloorDb);
    if (view_.max_hold) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset"))
            std::fill(max_hold_.begin(), max_hold_.end(), kFloorDb);
    }

    // ---- View ----
    ImGui::SeparatorText("View");
    if (ImGui::Button("3D"))
        apply_view(ViewPreset::Perspective);
    ImGui::SameLine();
    if (ImGui::Button("Top (2D)"))
        apply_view(ViewPreset::Top);
    ImGui::SameLine();
    if (ImGui::Button("Front"))
        apply_view(ViewPreset::Front);
    ImGui::SameLine();
    ImGui::Checkbox("Pause", &view_.paused);

    if (ImGui::Button("Screenshot"))
        take_screenshot({});
    ImGui::SameLine();
    if (ImGui::Button("Help (F1)"))
        view_.show_help = !view_.show_help;
#if SPECTRO_HAVE_AUDIO
    ImGui::SameLine();
    bool listening = listen_.on;
    if (ImGui::Checkbox("Listen", &listening))
        toggle_listening();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Play FM audio, narrowband or wideband (L).  It scans for signals\n"
                          "above a threshold; Ctrl + click a signal in the plot to choose one.");
#endif

    ImGui::PopItemWidth();
    ImGui::End();
}

// The controls of the channel being listened to, in a window of their own
// that is only there while the audio is on.
void App::draw_listen_window()
{
#if SPECTRO_HAVE_AUDIO
    if (!listen_.on)
        return;
    const ImGuiIO& io     = ImGui::GetIO();
    const float    scale  = imgui_->ui_scale();
    const float    margin = 12.0f * scale;
    const float    width  = 300.0f * scale;
    Listener&      listener = pipeline_.listener();

    // Top right, below the status box, where it hides the least of the plot.
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - margin, 132.0f * scale), ImGuiCond_FirstUseEver,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(width, 0.0f), ImVec2(width, io.DisplaySize.y));
    // (The "###" part keeps the window the same while its title changes.)
    const bool wide = listen_wide();
    bool       open = true;
    if (ImGui::Begin(wide ? "Listen: wideband FM###listen" : "Listen: narrowband FM###listen", &open,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::PushItemWidth(-80.0f * scale);

        // ---- Scanning ----
        // (The "###" part keeps the check box the same while its label changes.)
        bool scan = scanning() || channel_scanning();
        if (ImGui::Checkbox(channel_scan_wanted() ? "Scan control channels (S)###scan" : "Scan for signals (S)###scan",
                            &scan)) {
            if (scan)
                start_scanning();
            else
                stop_scanning();
        }
        if (channel_scan_wanted() && ImGui::IsItemHovered()) {
            std::string listed;
            for (const double hz : follower_.control_channels()) {
                char line[32];
                std::snprintf(line, sizeof(line), "\n  %.5f MHz", hz / 1e6);
                listed += line;
            }
            ImGui::SetTooltip("P25 only, with control channels from --control-channels: try each in turn until\n"
                              "one is heard to be a control channel, follow its calls, and when it goes quiet\n"
                              "look again, from the next.  In this order:%s",
                              listed.c_str());
        }
        if (channel_scanning()) {
            draw_channel_scan();
            ImGui::Separator();
        }
        if (scanning()) {
            const Scanner::Status st = scanner_->status();
            const ImVec4 green(0.45f, 0.85f, 0.55f, 1.0f), amber(1.0f, 0.75f, 0.3f, 1.0f);
            switch (st.state) {
            case Scanner::State::Searching:
                if (st.warming_up || std::isnan(listen_.scan_threshold))
                    ImGui::TextDisabled("Looking at the spectrum...");
                else
                    ImGui::TextColored(amber, "Searching: %zu signal(s) above the threshold",
                                       scanner_->candidates().size());
                break;
            case Scanner::State::Receiving:
                if (st.voice)
                    ImGui::TextColored(green, "%s call: staying until it ends", st.p25 && !st.dmr ? "P25" : "DMR");
                else if (scanner_->filtered() && st.wanted)
                    ImGui::TextColored(amber, "%s, no call: moving on in %.1f s", st.p25 && !st.dmr ? "P25" : "DMR",
                                       st.wait_left);
                else if (scanner_->filtered())
                    ImGui::TextColored(amber, "Not %s: moving on in %.1f s", digital_filter_name().c_str(),
                                       st.wait_left);
                else
                    ImGui::TextColored(green, "Receiving %s FM, %.0f kHz wide", st.wide ? "wideband" : "narrowband",
                                       st.occupied_hz / 1e3);
                break;
            case Scanner::State::Holding:
                ImGui::TextColored(amber, "Signal gone: moving on in %.1f s", st.hold_left);
                break;
            case Scanner::State::Off:
                break;
            }

            float threshold = std::isnan(listen_.scan_threshold) ? -60.0f : listen_.scan_threshold;
            ImGui::BeginDisabled(std::isnan(listen_.scan_threshold));
            if (ImGui::SliderFloat("Threshold", &threshold, -160.0f, 0.0f, "%.0f dBFS", ImGuiSliderFlags_AlwaysClamp)) {
                listen_.scan_threshold = threshold;
                scanner_->set_threshold(threshold);
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("A signal's peak on the plot must reach this level (the amber plane).\n"
                                  "Noise floor: %.0f dBFS", static_cast<double>(st.floor_dbfs));
            float hang = static_cast<float>(listen_.scan_hang);
            if (ImGui::SliderFloat("Hang", &hang, 0.0f, 10.0f, "%.1f s", ImGuiSliderFlags_AlwaysClamp)) {
                listen_.scan_hang = hang;
                scanner_->set_hang(hang);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How long to wait for a signal that has stopped before moving on");

            ImGui::BeginDisabled(!scan_locked());
            if (ImGui::Button("Skip (N)"))
                skip_signal();
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Move on to the next signal, leaving this one out until the end of the list;\n"
                                  "skipping the last one starts again from the first");
            if (scanner_->lockouts() > 0) {
                ImGui::SameLine();
                ImGui::TextDisabled("%zu skipped", scanner_->lockouts());
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear"))
                    scanner_->clear_lockouts();
            }
            ImGui::Separator();
        }

        // Typed frequencies are taken literally; the step buttons move by
        // 500 Hz, or by a 12.5 kHz channel with Ctrl held.
        if (ImGui::InputDouble("MHz", &listen_mhz_, 0.0005, 0.0125, "%.5f", ImGuiInputTextFlags_EnterReturnsTrue))
            listen_to(listen_mhz_ * 1e6, 0.0);
        if (!ImGui::IsItemActive() && !ImGui::IsItemFocused())
            listen_mhz_ = listen_.frequency / 1e6;

        // The width also chooses the kind of FM.  Only what the stream can
        // carry is offered.
        char label[48];
        std::snprintf(label, sizeof(label), "%g kHz  %s", listen_bandwidth() / 1e3, wide ? "WFM" : "NFM");
        if (ImGui::BeginCombo("Width", label)) {
            const auto group = [&](const char* title, FmMode mode, const double* widths, size_t count) {
                if (!listener.supports(mode))
                    return;
                ImGui::TextDisabled("%s", title);
                for (size_t i = 0; i < count; ++i) {
                    if (widths[i] > listener.widest(mode) + 1.0)
                        continue;
                    std::snprintf(label, sizeof(label), "  %g kHz", widths[i] / 1e3);
                    if (ImGui::Selectable(label, widths[i] == listen_.bandwidth)) {
                        listen_.bandwidth = widths[i];
                        listener.set_bandwidth(widths[i]);
                    }
                }
            };
            group("Narrowband FM (voice)", FmMode::Narrow, kNarrowWidthsHz, std::size(kNarrowWidthsHz));
            group("Wideband FM (broadcast)", FmMode::Wide, kWideWidthsHz, std::size(kWideWidthsHz));
            ImGui::EndCombo();
        }

        if (wide) {
            static const char* const kDeemphasisNames[] = {"75 \xC2\xB5s (the Americas)", "50 \xC2\xB5s (elsewhere)"};
            int choice = listen_.deemphasis < 62.5e-6 ? 1 : 0;
            if (ImGui::Combo("De-emph.", &choice, kDeemphasisNames, 2)) {
                listen_.deemphasis = choice == 1 ? 50e-6 : 75e-6;
                listener.set_deemphasis(listen_.deemphasis);
            }
        }

        int volume = static_cast<int>(std::lround(listen_.volume * 100.0f));
        if (ImGui::SliderInt("Volume", &volume, 0, 100, "%d %%", ImGuiSliderFlags_AlwaysClamp)) {
            listen_.volume = static_cast<float>(volume) / 100.0f;
            listener.set_volume(listen_.volume);
        }

        int boost = static_cast<int>(std::lround(listen_.boost_db));
        if (ImGui::SliderInt("Boost", &boost, 0, static_cast<int>(Listener::kMaxBoostDb), boost == 0 ? "off" : "+%d dB",
                             ImGuiSliderFlags_AlwaysClamp)) {
            listen_.boost_db = static_cast<float>(boost);
            listener.set_boost(listen_.boost_db);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Amplify the audio beyond full volume, analog or digital, for quiet voice.\n"
                              "What would then go past full scale is turned down rather than clipped.");

        float squelch = std::max(listen_.squelch, kSquelchSliderOff);
        if (ImGui::SliderFloat("Squelch", &squelch, kSquelchSliderOff, 0.0f,
                               squelch <= kSquelchSliderOff ? "off" : "%.0f dBFS", ImGuiSliderFlags_AlwaysClamp)) {
            listen_.squelch = squelch <= kSquelchSliderOff + 0.5f ? FmDemodulator::kSquelchOff : squelch;
            listener.set_squelch(listen_.squelch);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Mute the audio while the channel is below this level");

        const char* const no_vocoder =
            listen_status_.vocoder ? "" : "\nThis build has no vocoder, so the voice is silent too (see the README).";
        if (ImGui::Checkbox("DMR only", &listen_.dmr_only)) {
            listener.set_dmr_only(listen_.dmr_only);
            if (scanner_)
                scanner_->set_dmr_only(listen_.dmr_only);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Play only DMR voice: analog audio and other digital modes stay silent.\n"
                              "Scanning, move on from anything that is not DMR, and from a DMR channel\n"
                              "without a call once the hold time is up.  With P25 only too, both.%s",
                              no_vocoder);
        ImGui::SameLine();
        if (ImGui::Checkbox("P25 only", &listen_.p25_only)) {
            listener.set_p25_only(listen_.p25_only);
            if (scanner_)
                scanner_->set_p25_only(listen_.p25_only);
            // With control channels listed, it also chooses what scanning goes through.
            if ((scanning() && channel_scan_wanted()) || (channel_scanning() && !channel_scan_wanted())) {
                stop_scanning();
                start_scanning();
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Play only P25 voice, Phase 1 and 2: analog audio and other digital modes stay\n"
                              "silent.  Scanning, move on from anything that is not P25, and from a P25 channel\n"
                              "without a call once the hold time is up.  With DMR only too, both.\n"
                              "With control channels from --control-channels, scanning goes through those\n"
                              "instead, and follows the calls of the one heard.%s",
                              no_vocoder);
        bool follow = listen_.follow;
        if (ImGui::Checkbox("Follow calls", &follow))
            set_following(follow);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("On a P25 trunked system's control channel: go to the voice channel of each call\n"
                              "it grants - Phase 1, or Phase 2 and its slot - and back when the call ends.\n"
                              "Calls that are encrypted are left out.  Not while scanning for signals;\n"
                              "scanning control channels turns it on.");
        if (transcriber_) {
            ImGui::SameLine();
            ImGui::Checkbox("Transcript", &show_transcript_);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Show what is said in the DMR and P25 calls heard, as text\n"
                                  "(speech to text with Whisper, on this computer)");
        }
        draw_following();
        if ((listen_.dmr_only || listen_.p25_only) && scanning()) {
            float hold = static_cast<float>(listen_.dmr_hold);
            if (ImGui::SliderFloat("Hold", &hold, 1.0f, 60.0f, "%.0f s",
                                   ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic)) {
                listen_.dmr_hold = hold;
                scanner_->set_dmr_hold(hold);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How long to stay on a DMR or P25 channel waiting for a call (and for the\n"
                                  "next over after one ends) before moving on");
        }
        if (listen_.dmr_only || listen_.p25_only)
            draw_heard_channels();

        if (scanning() && !scan_locked()) {
            // Nothing to report until the scanner finds something.
        } else if (!listen_in_band()) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f), "Outside the received band");
        } else {
            ImGui::Text("Level  %.1f dBFS", static_cast<double>(listen_status_.level_dbfs));
            ImGui::SameLine();
            if (listen_status_.searching)
                ImGui::TextDisabled("  finding the carrier");
            else if (listen_status_.dmr.present)
                ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f), "  decoding DMR");
            else if (listen_status_.p25.present || listen_status_.p25p2.present)
                ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f), "  decoding P25");
            else if (listen_status_.dmr_only || listen_status_.p25_only)
                ImGui::TextDisabled("  silent: not %s", digital_filter_name().c_str());
            else if (listen_status_.squelch_open)
                ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f), "  playing");
            else
                ImGui::TextDisabled("  squelched");
        }
        // What kind of signal it is.
        if (listen_in_band() && !(scanning() && !scan_locked())) {
            const SignalClass& signal = listen_status_.signal;
            const ImVec4       colour = signal.digital() ? ImVec4(0.45f, 0.78f, 1.0f, 1.0f)
                                        : signal.kind == SignalKind::Analog || signal.kind == SignalKind::Broadcast
                                            ? ImVec4(0.45f, 0.85f, 0.55f, 1.0f)
                                            : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
            ImGui::PushStyleColor(ImGuiCol_Text, colour);
            ImGui::TextWrapped("%s", describe(signal).c_str());
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("What the signal looks like to the demodulator, judged half a second at a time:\n"
                                  "analog audio, or digital modulation with its symbol rate and deviation.\n"
                                  "DMR, P25 and POCSAG are named when their sync patterns are found.");
            if (signal.kind != SignalKind::Analysing && signal.kind != SignalKind::Noise)
                ImGui::TextDisabled("SNR %.0f dB", static_cast<double>(signal.snr_db));
            const std::string typical = typical_systems(signal);
            if (!typical.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("Typical of %s", typical.c_str());
                ImGui::PopStyleColor();
            }
            draw_dmr_status();
            draw_p25_status();
            draw_p25_phase2_status();
        }

        ImGui::SeparatorText("Audio");
        draw_audio_plot();

        if (listen_status_.underruns > 0 || listen_status_.dropped_iq > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f), "Gaps in the audio: %llu",
                               static_cast<unsigned long long>(listen_status_.underruns));
            if (listen_status_.dropped_iq > 0) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f), "  samples missed: %llu",
                                   static_cast<unsigned long long>(listen_status_.dropped_iq));
            }
        }
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        if (audio_)
            ImGui::TextWrapped("%s", audio_->description().c_str());
        ImGui::TextWrapped("Ctrl + click or drag in the plot to choose a signal");
        ImGui::PopStyleColor();

        ImGui::PopItemWidth();
    }
    ImGui::End();
    if (!open)
        set_listening(false);
#endif
}

// The channels DMR voice has been heard on; a click goes back to one.
void App::draw_heard_channels()
{
    ImGui::SeparatorText("Voice heard on");
    const std::vector<HeardChannel>& channels = heard_.channels();
    if (channels.empty()) {
        ImGui::TextDisabled("No DMR or P25 voice heard yet");
        return;
    }
    const double now     = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    const auto   ago     = [](double seconds) {
        char text[24];
        if (seconds < 2.0)
            std::snprintf(text, sizeof(text), "now");
        else if (seconds < 120.0)
            std::snprintf(text, sizeof(text), "%.0f s", seconds);
        else if (seconds < 7200.0)
            std::snprintf(text, sizeof(text), "%.0f min", seconds / 60.0);
        else
            std::snprintf(text, sizeof(text), "%.0f h", seconds / 3600.0);
        return std::string(text);
    };
    const ImVec4 secret(1.0f, 0.55f, 0.25f, 1.0f);
    const float  line  = ImGui::GetTextLineHeightWithSpacing();
    const float  shown = static_cast<float>(std::min<size_t>(channels.size(), 8));
    double       chosen = std::numeric_limits<double>::quiet_NaN();
    if (ImGui::BeginTable("heard", 4, ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0.0f, (shown + 1.4f) * line))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("MHz");
        ImGui::TableSetupColumn("Calls");
        ImGui::TableSetupColumn("Heard");
        ImGui::TableSetupColumn("Last", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < channels.size(); ++i) {
            const HeardChannel& c = channels[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            char label[40];
            std::snprintf(label, sizeof(label), "%.5f##heard%zu", c.frequency / 1e6, i);
            const bool here = listen_.on && std::fabs(listen_.frequency - c.frequency) < HeardChannels::kSameChannelHz;
            if (ImGui::Selectable(label, here, ImGuiSelectableFlags_SpanAllColumns))
                chosen = c.frequency;
            if (ImGui::IsItemHovered()) {
                if (c.system == "P25")
                    ImGui::SetTooltip("Listen here%s\nP25, NAC 0x%03X; the last call:\n%s", scanning() ? " (ends scanning)" : "",
                                      static_cast<unsigned>(std::max(c.nac, 0)), c.last_call.c_str());
                else if (c.system == "P25 Phase 2")
                    ImGui::SetTooltip("Listen here%s\nP25 Phase 2; the last call, on slot %d:\n%s",
                                      scanning() ? " (ends scanning)" : "", c.slot + 1, c.last_call.c_str());
                else
                    ImGui::SetTooltip("Listen here%s\nDMR, colour code %d; the last call, on time slot %d:\n%s",
                                      scanning() ? " (ends scanning)" : "", c.color_code, c.slot + 1, c.last_call.c_str());
            }
            ImGui::TableNextColumn();
            ImGui::Text("%d", c.calls);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(ago(now - c.last_heard).c_str());
            ImGui::TableNextColumn();
            const std::string last = c.system + " " + (c.talkgroup.empty() ? (c.encrypted ? "encrypted" : "-") : c.talkgroup);
            if (c.encrypted)
                ImGui::TextColored(secret, "%s", last.c_str());
            else
                ImGui::TextUnformatted(last.c_str());
        }
        ImGui::EndTable();
    }
    if (ImGui::SmallButton("Clear##heard"))
        heard_.clear();
    if (!std::isnan(chosen))
        listen_to_heard(chosen);
}

// What was said in the calls heard, as Whisper made it out: the latest at the top.
void App::draw_transcript_window()
{
#if SPECTRO_HAVE_AUDIO
    if (!transcriber_ || !listen_.on || !show_transcript_)
        return;
    const ImGuiIO& io     = ImGui::GetIO();
    const float    scale  = imgui_->ui_scale();
    const float    margin = 12.0f * scale;
    const ImVec4   amber(1.0f, 0.75f, 0.3f, 1.0f), green(0.45f, 0.85f, 0.55f, 1.0f);

    // Bottom right, beside the listening window.
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 2.0f * margin - 300.0f * scale, io.DisplaySize.y - 3.0f * margin),
                            ImGuiCond_FirstUseEver, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2(500.0f * scale, 240.0f * scale), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Transcript", &show_transcript_, ImGuiWindowFlags_NoCollapse)) {
        const TranscriberStatus st = transcriber_->status();
        bool                    on = st.enabled;
        if (ImGui::Checkbox("Transcribe", &on))
            transcriber_->set_enabled(on);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Speech to text of the DMR and P25 voice heard, an over at a time once it ends:\n"
                              "%s, on this computer.  Off: what is said is let go.",
                              st.model.c_str());
        ImGui::SameLine();
        switch (st.state) {
        case TranscriberStatus::State::Loading:
            ImGui::TextDisabled("loading %s...", st.model.c_str());
            break;
        case TranscriberStatus::State::Failed:
            ImGui::PushStyleColor(ImGuiCol_Text, amber);
            ImGui::TextWrapped("%s", st.error.c_str());
            ImGui::PopStyleColor();
            break;
        case TranscriberStatus::State::Ready:
            if (!listen_status_.vocoder)
                ImGui::TextColored(amber, "no vocoder in this build: no voice to hear");
            else if (st.busy || st.waiting > 0)
                ImGui::TextColored(green, "transcribing%s", st.waiting > 0 ? (" (" + std::to_string(st.waiting) + " waiting)").c_str() : "");
            else
                ImGui::TextDisabled("%s", st.model.c_str());
            if (st.speed > 0.0 && ImGui::IsItemHovered())
                ImGui::SetTooltip("%.1f times as fast as the speech, lately", st.speed);
            break;
        }
        if (st.dropped > 0) {
            ImGui::SameLine();
            ImGui::TextColored(amber, "%llu let go", static_cast<unsigned long long>(st.dropped));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Overs that came faster than they could be transcribed");
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(transcripts_.empty());
        if (ImGui::SmallButton("Copy")) {
            std::string all;
            for (const Transcript& t : transcripts_)
                all += format_transcript(t) + "\n";
            ImGui::SetClipboardText(all.c_str());
            toast("Transcript copied");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear##transcript"))
            transcripts_.clear();
        ImGui::EndDisabled();

        if (transcripts_.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::TextWrapped("Nothing yet.  What is said in the DMR and P25 calls heard appears here, "
                               "each over once it ends.");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("transcripts", 3,
                                     ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit |
                                         ImGuiTableFlags_BordersInnerV,
                                     ImVec2(0.0f, ImGui::GetContentRegionAvail().y))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Time");
            ImGui::TableSetupColumn("Who");
            ImGui::TableSetupColumn("Said", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (size_t i = transcripts_.size(); i-- > 0;) {
                const Transcript& t = transcripts_[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                const auto when = static_cast<std::time_t>(std::floor(t.start));
                std::tm    local{};
                localtime_r(&when, &local);
                char clock[16];
                std::strftime(clock, sizeof(clock), "%H:%M:%S", &local);
                ImGui::TextUnformatted(clock);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%.5f MHz, %s\n%.1f s of speech%s", t.frequency / 1e6,
                                      transcript_channel(t).c_str(), t.seconds,
                                      t.part > 0 ? " (a long over, continued)" : "");
                ImGui::TableNextColumn();
                const std::string who = transcript_speaker(t);
                ImGui::TextUnformatted(who.empty() ? "?" : who.c_str());
                ImGui::TableNextColumn();
                const bool unsure = t.confidence < kUnsureTranscript;
                if (unsure)
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("%s", t.text.c_str());
                if (unsure) {
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Whisper was not sure of this (%.0f %%)",
                                          100.0 * static_cast<double>(t.confidence));
                }
            }
            ImGui::EndTable();
        }
    }
    ImGui::End();
#endif
}

// "DMR", "P25" or "DMR or P25": what DMR only and P25 only let through.
std::string App::digital_filter_name() const
{
    if (listen_.dmr_only && listen_.p25_only)
        return "DMR or P25";
    return listen_.p25_only ? "P25" : "DMR";
}

// P25 on the channel: its network access code, and the call.
void App::draw_p25_status()
{
    const P25Status& p25 = listen_status_.p25;
    if (!p25.present)
        return;
    ImGui::SeparatorText("P25");
    ImGui::Text("NAC 0x%03X%s", static_cast<unsigned>(std::max(p25.nac, 0)), p25.control ? ", control channel" : "");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The network access code tells the systems on a frequency apart.\n"
                          "A control channel carries a trunked system's signalling; its calls are\n"
                          "on other frequencies: click one below, or tick Follow calls.");
    // A control channel: its system, and the calls it grants.
    const P25Site& site = p25.site;
    if (site.system >= 0) {
        std::string text = "System 0x";
        char        part[48];
        std::snprintf(part, sizeof(part), "%03X", static_cast<unsigned>(site.system));
        text += part;
        if (site.wacn >= 0) {
            std::snprintf(part, sizeof(part), ", WACN 0x%05X", static_cast<unsigned>(site.wacn));
            text += part;
        }
        if (site.site >= 0) {
            std::snprintf(part, sizeof(part), ", RFSS %d, site %d", site.rfss, site.site);
            text += part;
        }
        ImGui::TextWrapped("%s", text.c_str());
        const std::string phases = site.voice_phases();
        if (!phases.empty()) {
            ImGui::TextDisabled("Voice in %s", phases.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("From the system's channel plans: FDMA ones carry Phase 1 voice, TDMA ones\n"
                                  "Phase 2 - two calls to a channel, scrambled with a key made from the\n"
                                  "system's IDs above, which is why Phase 2 is only read after visiting here.");
        }
    }
    if (!p25.grants.empty()) {
        ImGui::TextDisabled("Calls granted:");
        double chosen      = std::numeric_limits<double>::quiet_NaN();
        int    chosen_slot = -1;
        for (size_t i = 0; i < p25.grants.size(); ++i) {
            const P25Grant& g   = p25.grants[i];
            const double    age = std::max(0.0, p25.time - g.time);
            char            label[128];
            if (g.frequency > 0.0)
                std::snprintf(label, sizeof(label), "%s%s  %.5f%s  %.0f s##grant%zu", p25_grant_destination(g).c_str(),
                              g.encrypted ? " (enc.)" : "", g.frequency / 1e6,
                              g.slot < 0 ? "" : g.slot % 2 == 0 ? " slot 1" : " slot 2", age, i);
            else
                std::snprintf(label, sizeof(label), "%s  channel %X-%03X  %.0f s##grant%zu", p25_grant_destination(g).c_str(),
                              g.channel >> 12, g.channel & 0xFFFu, age, i);
            const bool known = g.frequency > 0.0;
            ImGui::BeginDisabled(!known);
            if (ImGui::Selectable(label)) {
                chosen      = g.frequency;
                chosen_slot = g.slot >= 0 ? g.slot % 2 : -1;
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                const std::string from = g.source ? " from " + std::to_string(g.source) : "";
                if (known && g.slot < 0)
                    ImGui::SetTooltip("%s%s on %.5f MHz (Phase 1)\nClick to listen there", p25_grant_destination(g).c_str(),
                                      from.c_str(), g.frequency / 1e6);
                else if (known)
                    ImGui::SetTooltip("%s%s on %.5f MHz, slot %d (Phase 2)\nClick to listen there",
                                      p25_grant_destination(g).c_str(), from.c_str(), g.frequency / 1e6, g.slot % 2 + 1);
                else
                    ImGui::SetTooltip("%s%s: the channel plan for channel %X-%03X has not been heard yet",
                                      p25_grant_destination(g).c_str(), from.c_str(), g.channel >> 12, g.channel & 0xFFFu);
            }
        }
        if (!std::isnan(chosen))
            listen_to_heard(chosen, chosen_slot);
    }

    const P25Call& c      = p25.call;
    const ImVec4   colour = !c.active      ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled)
                            : c.encrypted ? ImVec4(1.0f, 0.55f, 0.25f, 1.0f)
                            : listen_status_.vocoder ? ImVec4(0.45f, 0.85f, 0.55f, 1.0f)
                                                     : ImVec4(0.45f, 0.78f, 1.0f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextWrapped("%s%s", describe_p25(p25).c_str(),
                       c.active && !c.encrypted && listen_status_.vocoder ? "  (playing)" : "");
    ImGui::PopStyleColor();
    if (c.active && c.voice && !c.encrypted && !listen_status_.vocoder) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.55f, 0.25f, 1.0f));
        ImGui::TextWrapped("Voice not decoded: built without the IMBE vocoder");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Build with -DSPECTRO_WITH_MBELIB=ON to hear P25 voice (see the README).");
    }
}

// P25 Phase 2 on the channel: the system (if known, the voice can be read)
// and the call on each slot, and which to hear.
void App::draw_p25_phase2_status()
{
    const P25Phase2Status& p2 = listen_status_.p25p2;
    if (!p2.present)
        return;
    ImGui::SeparatorText("P25 Phase 2");
    const ImVec4 heard(0.45f, 0.85f, 0.55f, 1.0f), call(0.45f, 0.78f, 1.0f, 1.0f), secret(1.0f, 0.55f, 0.25f, 1.0f);
    if (p2.keyed) {
        ImGui::TextWrapped("System 0x%03X, WACN 0x%05X, NAC 0x%03X%s", static_cast<unsigned>(p2.system.system),
                           static_cast<unsigned>(p2.system.wacn), static_cast<unsigned>(p2.system.nac),
                           p2.recovered ? " (from its signalling)" : "");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Phase 2 is scrambled with a sequence made from these.  %s",
                              p2.recovered ? "They were worked out from a clean signalling burst on this channel."
                                           : "They came from the system's control channel.");
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, secret);
        ImGui::TextWrapped("Scrambled: working out the system");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Phase 2 is scrambled with a sequence made from the system's WACN, system ID and\n"
                              "NAC.  They are worked out from the first signalling burst that comes through\n"
                              "without errors - usually within a second of a call starting; a noisy channel\n"
                              "takes longer.  The system's control channel, once heard, gives them too.\n"
                              "Until then calls are seen, but not who makes them, and not heard.");
    }
    bool voice = false;
    for (int s = 0; s < 2; ++s) {
        const P25Call& c = p2.slot[s];
        voice |= c.active && c.voice && !c.encrypted && p2.keyed;
        const ImVec4 colour = !c.active      ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled)
                              : c.encrypted ? secret
                              : p2.playing == s && listen_status_.vocoder ? heard
                                                                          : call;
        ImGui::PushStyleColor(ImGuiCol_Text, colour);
        ImGui::TextWrapped("Slot %d  %s%s", s + 1, describe_p25_phase2_slot(p2, s).c_str(),
                           c.active && p2.playing == s && !c.encrypted && listen_status_.vocoder ? "  (playing)" : "");
        ImGui::PopStyleColor();
    }
    // Which slot to hear.
    int slot = listen_status_.p25p2_slot + 1;
    ImGui::TextDisabled("Hear");
    ImGui::SameLine();
    bool changed = ImGui::RadioButton("either", &slot, 0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Whichever slot speaks first; when its call ends, the next");
    ImGui::SameLine();
    changed |= ImGui::RadioButton("slot 1", &slot, 1);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("slot 2", &slot, 2);
    if (changed)
        pipeline_.listener().set_p25_phase2_slot(slot - 1);
    if (voice && !listen_status_.vocoder) {
        ImGui::PushStyleColor(ImGuiCol_Text, secret);
        ImGui::TextWrapped("Voice not decoded: built without the AMBE+2 vocoder");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Build with -DSPECTRO_WITH_MBELIB=ON to hear P25 Phase 2 voice (see the README).");
    }
}

// Scanning the control channels: the one being tried, or the one heard.
void App::draw_channel_scan()
{
    const ImVec4 green(0.45f, 0.85f, 0.55f, 1.0f), amber(1.0f, 0.75f, 0.3f, 1.0f);
    const size_t count = follower_.control_channels().size();
    const int    index = follower_.listed_index();
    if (follower_.state() == TrunkFollower::State::Hunting)
        ImGui::TextColored(amber, "Trying %.5f MHz (%d of %zu)", follower_.trying_frequency() / 1e6, index + 1, count);
    else if (index >= 0)
        ImGui::TextColored(green, "Control channel %d of %zu heard", index + 1, count);
    else
        ImGui::TextColored(green, "On a control channel");
    if (ImGui::Button("Skip (N)"))
        skip_signal();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Try the next control channel of the list");
}

// Following a P25 system's calls: where the follower is.
void App::draw_following()
{
    if (!listen_.follow || follower_.state() == TrunkFollower::State::Hunting)
        return;  // (looking for a control channel: the scan says where)
    if (scanning()) {
        ImGui::TextDisabled("Following calls: paused while scanning");
        return;
    }
    const ImVec4 colour = follower_.state() == TrunkFollower::State::Voice ? ImVec4(0.45f, 0.85f, 0.55f, 1.0f)
                                                                           : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextWrapped("%s", follower_.describe().c_str());
    ImGui::PopStyleColor();
}

// DMR on the channel: the repeater or radios, and the call on each time slot.
void App::draw_dmr_status()
{
    const DmrStatus& dmr = listen_status_.dmr;
    if (!dmr.present)
        return;
    ImGui::SeparatorText("DMR");
    ImGui::Text("%s, colour code %d", dmr.repeater ? "Repeater" : "Radios direct", dmr.color_code);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A repeater sends both time slots all the time; radios talking directly send one.\n"
                          "The colour code tells repeaters on the same frequency apart.");
    const ImVec4 heard(0.45f, 0.85f, 0.55f, 1.0f), call(0.45f, 0.78f, 1.0f, 1.0f), secret(1.0f, 0.55f, 0.25f, 1.0f);
    bool         voice = false;
    for (int s = 0; s < 2; ++s) {
        const DmrCall& c = dmr.slot[s];
        voice |= c.active && c.voice && !c.encrypted;
        const ImVec4 colour = !c.active      ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled)
                              : c.encrypted ? secret
                              : dmr.playing == s && listen_status_.vocoder ? heard
                                                                           : call;
        ImGui::PushStyleColor(ImGuiCol_Text, colour);
        ImGui::TextWrapped("TS%d  %s%s", s + 1, describe_dmr_slot(dmr, s).c_str(),
                           c.active && dmr.playing == s && !c.encrypted && listen_status_.vocoder ? "  (playing)" : "");
        ImGui::PopStyleColor();
    }
    if (voice && !listen_status_.vocoder) {
        ImGui::PushStyleColor(ImGuiCol_Text, secret);
        ImGui::TextWrapped("Voice not decoded: built without the AMBE+2 vocoder");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Build with -DSPECTRO_WITH_MBELIB=ON to hear DMR voice (see the README).");
    }
}

// A time series of the audio being played: the waveform on the short spans,
// steadied on a rising zero crossing so that a tone stands still, and the
// envelope - the lowest and highest value per pixel - on the long ones.  It
// shows the audio as played, after the volume control and any boost, so +/-1
// is the sound card's full scale (which the limiter keeps it within), and it
// freezes while the display is paused.
void App::draw_audio_plot()
{
    const AudioScope& scope = pipeline_.listener().scope();
    const float       scale = imgui_->ui_scale();

    for (int i = 0; i < static_cast<int>(std::size(kScopeSpans)); ++i) {
        if (i > 0)
            ImGui::SameLine();
        ImGui::RadioButton(kScopeSpanNames[i], &scope_span_, i);
    }

    const float  width  = std::max(ImGui::GetContentRegionAvail().x, 50.0f);
    const float  height = 84.0f * scale;
    const ImVec2 p0     = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + width, p0.y + height);
    ImGui::InvisibleButton("##audio plot", ImVec2(width, height));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The audio as played, after the volume control and any boost.\n+/-1 is full scale.");

    // Read what the plot needs: per pixel column, the lowest and highest value.
    const int columns = std::max(1, static_cast<int>(width));
    if (!view_.paused || static_cast<int>(scope_lo_.size()) != columns || scope_shown_span_ != scope_span_) {
        scope_shown_span_ = scope_span_;
        scope_lo_.assign(static_cast<size_t>(columns), std::numeric_limits<float>::infinity());
        scope_hi_.assign(static_cast<size_t>(columns), -std::numeric_limits<float>::infinity());
        const auto put = [&](double age, double span, float lo, float hi) {  // age: 0 = newest .. span
            const int c = columns - 1 - static_cast<int>(age / span * columns);
            if (c < 0)
                return;
            scope_lo_[static_cast<size_t>(c)] = std::min(scope_lo_[static_cast<size_t>(c)], lo);
            scope_hi_[static_cast<size_t>(c)] = std::max(scope_hi_[static_cast<size_t>(c)], hi);
        };

        const double rate = Listener::kAudioRate;
        const double span = kScopeSpans[scope_span_];
        if (span <= 0.1) {
            const size_t want    = static_cast<size_t>(std::llround(span * rate));
            const size_t trigger = static_cast<size_t>(std::llround(kScopeTriggerSeconds * rate));
            scope_samples_.resize(want + trigger);
            const size_t got = scope.latest(scope_samples_.data(), want + trigger);
            // End on the latest rising zero crossing, if there is one recently.
            size_t end = got;
            for (size_t i = got; i-- > 1 && i + trigger > got;) {
                if (scope_samples_[i - 1] < 0.0f && scope_samples_[i] >= 0.0f) {
                    if (i >= want)
                        end = i;
                    break;
                }
            }
            for (size_t j = end > want ? end - want : 0; j < end; ++j)
                put(static_cast<double>(end - 1 - j), static_cast<double>(want), scope_samples_[j], scope_samples_[j]);
        } else {
            const size_t want = static_cast<size_t>(std::llround(span * rate / AudioScope::kBlock));
            scope_mins_.resize(want);
            scope_maxs_.resize(want);
            const size_t got = scope.latest_blocks(scope_mins_.data(), scope_maxs_.data(), want);
            for (size_t k = 0; k < got; ++k)
                put(static_cast<double>(got - 1 - k), static_cast<double>(want), scope_mins_[k], scope_maxs_[k]);
        }
    }

    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float mid  = 0.5f * (p0.y + p1.y);
    const float half = 0.5f * height - 2.0f * scale;
    draw->AddRectFilled(p0, p1, IM_COL32(9, 13, 18, 255), 3.0f * scale);
    draw->AddLine(ImVec2(p0.x, mid - 0.5f * half), ImVec2(p1.x, mid - 0.5f * half), IM_COL32(255, 255, 255, 18));
    draw->AddLine(ImVec2(p0.x, mid + 0.5f * half), ImVec2(p1.x, mid + 0.5f * half), IM_COL32(255, 255, 255, 18));
    draw->AddLine(ImVec2(p0.x, mid), ImVec2(p1.x, mid), IM_COL32(255, 255, 255, 40));

    const bool  audible = listen_in_band() && listen_status_.squelch_open &&
                          (!scanning() || scanner_->status().state == Scanner::State::Receiving);
    const ImU32 color   = audible ? IM_COL32(110, 255, 190, 255) : IM_COL32(140, 160, 170, 200);
    for (int c = 0; c < columns; ++c) {
        const float lo = scope_lo_[static_cast<size_t>(c)], hi = scope_hi_[static_cast<size_t>(c)];
        if (lo > hi)
            continue;  // nothing here (yet)
        const float top    = mid - std::clamp(hi, -1.0f, 1.0f) * half;
        const float bottom = mid - std::clamp(lo, -1.0f, 1.0f) * half;
        const float x      = p0.x + static_cast<float>(c);
        draw->AddRectFilled(ImVec2(x, top), ImVec2(x + 1.0f, std::max(bottom, top + 1.0f)), color);
    }
    draw->AddRect(p0, p1, IM_COL32(255, 255, 255, 40), 3.0f * scale);

    // The span, on a backing so that it stays legible over loud audio.
    const char*  label = view_.paused ? "paused" : kScopeSpanNames[scope_span_];
    const ImVec2 size  = ImGui::CalcTextSize(label);
    const ImVec2 at(p1.x - size.x - 5.0f * scale, p1.y - size.y - 3.0f * scale);
    draw->AddRectFilled(ImVec2(at.x - 3.0f * scale, at.y - 1.0f * scale), ImVec2(p1.x - 2.0f * scale, p1.y - 2.0f * scale),
                        IM_COL32(9, 13, 18, 220), 2.0f * scale);
    draw->AddText(at, IM_COL32(170, 180, 195, 255), label);
}

void App::draw_status()
{
    const ImGuiIO&         io     = ImGui::GetIO();
    IqSource&              source = pipeline_.source();
    const AcquisitionStats acq    = pipeline_.acquisition().stats();
    const float            scale  = imgui_->ui_scale();
    const float            margin = 12.0f * scale;

    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;

    // Top right: what we are looking at.
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - margin, margin), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.55f);
    if (ImGui::Begin("##status", nullptr, flags)) {
        ImGui::Text("%s   %s", format_frequency(source.frequency()).c_str(),
                    format_si(source.sample_rate(), "S/s").c_str());
        if (have_row_) {
            const auto   peak_it  = std::max_element(latest_row_.begin(), latest_row_.end());
            const int    peak_col = static_cast<int>(peak_it - latest_row_.begin());
            const double peak_hz  = source.frequency() + SpectrumEngine::column_frequency(
                                                             peak_col, columns_, fft_size_, source.sample_rate());
            ImGui::Text("Peak %.4f MHz  %.1f dBFS", peak_hz / 1e6, static_cast<double>(*peak_it));
        } else {
            ImGui::TextUnformatted("Waiting for samples...");
        }
        const uint64_t lost = acq.lost_samples + acq.ring_dropped;
        ImGui::Text("%.0f fps", 1.0 / static_cast<double>(frame_time_avg_));
        ImGui::SameLine();
        if (lost > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f), "  %llu samples lost",
                               static_cast<unsigned long long>(lost));
        else
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f), "  no samples lost");
        if (listen_.on && scanning() && !scan_locked())
            ImGui::TextUnformatted("Audio: scanning for signals");
        else if (listen_.on)
            ImGui::Text("Audio %.5f MHz%s", listen_.frequency / 1e6, scanning() ? "  (scanning)" : "");
        if (view_.paused)
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f), "PAUSED (Space to resume)");
        if (!view_.show_panel)
            ImGui::TextDisabled("H: controls   F1: help");
    }
    ImGui::End();

    // Top centre: problems.
    if (acq.failed) {
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, margin), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.45f, 0.08f, 0.08f, 0.92f));
        if (ImGui::Begin("##error", nullptr, flags)) {
            ImGui::Text("The sample stream stopped: %s", acq.error.c_str());
            ImGui::TextUnformatted("Restart the program once the problem is fixed.");
        }
        ImGui::End();
        ImGui::PopStyleColor();
    }

    // Bottom centre: transient messages.
    if (!toast_text_.empty() && Clock::now() < toast_until_) {
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y - margin * 2.0f),
                                ImGuiCond_Always, ImVec2(0.5f, 1.0f));
        ImGui::SetNextWindowBgAlpha(0.75f);
        if (ImGui::Begin("##toast", nullptr, flags))
            ImGui::TextUnformatted(toast_text_.c_str());
        ImGui::End();
    }
}

void App::draw_help()
{
    if (!view_.show_help)
        return;
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Appearing,
                            ImVec2(0.5f, 0.5f));
    if (ImGui::Begin("Help", &view_.show_help, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        static const struct { const char* keys; const char* action; } kHelp[] = {
            {"Left drag", "Rotate the view"},
            {"Right / middle drag, Shift + drag", "Move the view"},
            {"Mouse wheel", "Zoom"},
            {"Double click, R", "Reset the view"},
            {"Ctrl + click or drag", "Listen to the signal under the cursor (FM)"},
            {"L", "Audio on / off (scans for signals)"},
            {"S", "Scanning on / off"},
            {"N", "Skip to the next signal (or control channel) while scanning"},
            {"1 / 2 / 3", "3D view / top view (2D waterfall) / front view (spectrum)"},
            {"Left / Right", "Tune down / up (Shift: fine, Ctrl: half the span)"},
            {"Up / Down", "Gain up / down (Shift: 5 dB steps)"},
            {"[ / ]", "Halve / double the FFT size"},
            {", / .", "Slower / faster waterfall"},
            {"A", "Fit the level scale to the signal"},
            {"C / Shift + C", "Next / previous colour map"},
            {"M", "Max hold trace"},
            {"G / W / T", "Toggle grid / walls / live trace"},
            {"Space", "Pause"},
            {"P, F12", "Save a screenshot"},
            {"F, F11", "Full screen"},
            {"H", "Show / hide the control panel"},
            {"Ctrl + Q", "Quit"},
        };
        if (ImGui::BeginTable("help", 2, ImGuiTableFlags_SizingFixedFit)) {
            for (const auto& row : kHelp) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.45f, 0.78f, 0.97f, 1.0f), "%s", row.keys);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.action);
            }
            ImGui::EndTable();
        }
    }
    ImGui::End();
}

void App::draw_axis_labels()
{
    if (!view_.labels || labels_.empty())
        return;

    const ImGuiIO& io        = ImGui::GetIO();
    ImDrawList*    draw      = ImGui::GetBackgroundDrawList();
    const float    line      = ImGui::GetFontSize();
    const Mat4     view_proj = view_projection();

    const ImU32 tick_color   = IM_COL32(214, 220, 232, 255);
    const ImU32 title_color  = IM_COL32(150, 160, 180, 255);
    const ImU32 shadow_color = IM_COL32(0, 0, 0, 200);

    // Per axis, the rectangle of the last tick label drawn: labels that would
    // run into it are skipped.
    ImVec2 last_min[3], last_max[3];
    bool   have_last[3] = {false, false, false};

    for (const AxisLabel& label : labels_) {
        float x = 0.0f, y = 0.0f;
        if (!project_to_screen(view_proj, label.anchor, io.DisplaySize.x, io.DisplaySize.y, x, y))
            continue;

        // The direction on screen in which "away from the plot" points here.
        float dir_x = label.fallback_x, dir_y = label.fallback_y;
        float ox = 0.0f, oy = 0.0f;
        const Vec3 probe = label.anchor + label.outward * 0.1f;
        if (project_to_screen(view_proj, probe, io.DisplaySize.x, io.DisplaySize.y, ox, oy)) {
            const float len = std::hypot(ox - x, oy - y);
            if (len > 3.0f) {  // otherwise it points at the viewer: keep the fallback
                dir_x = (ox - x) / len;
                dir_y = (oy - y) / len;
            }
        }

        // Centre the text box so that its near side is `distance` beyond the anchor.
        const ImVec2 size   = ImGui::CalcTextSize(label.text.c_str());
        const float  reach  = label.distance * line + 0.5f * (std::fabs(dir_x) * size.x + std::fabs(dir_y) * size.y);
        const ImVec2 pos(std::round(x + dir_x * reach - 0.5f * size.x), std::round(y + dir_y * reach - 0.5f * size.y));
        const ImVec2 max(pos.x + size.x, pos.y + size.y);
        if (max.x < 0.0f || max.y < 0.0f || pos.x > io.DisplaySize.x || pos.y > io.DisplaySize.y)
            continue;

        const int axis = std::clamp(label.axis, 0, 2);
        if (!label.title) {
            const float pad_x = 0.5f * line, pad_y = 0.1f * line;
            if (have_last[axis] && pos.x < last_max[axis].x + pad_x && max.x > last_min[axis].x - pad_x &&
                pos.y < last_max[axis].y + pad_y && max.y > last_min[axis].y - pad_y)
                continue;
            last_min[axis]  = pos;
            last_max[axis]  = max;
            have_last[axis] = true;
        }

        draw->AddText(ImVec2(pos.x + 1.0f, pos.y + 1.0f), shadow_color, label.text.c_str());
        draw->AddText(pos, label.title ? title_color : tick_color, label.text.c_str());
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

SceneParams App::scene_params() const
{
    SceneParams scene;
    scene.view_proj    = view_projection();
    scene.orthographic = camera_.orthographic;
    scene.eye          = camera_.orthographic ? camera_.to_eye() : camera_.eye();
    scene.plot_width   = kPlotWidth;
    scene.plot_height  = view_.plot_height;
    scene.plot_depth   = view_.plot_depth;
    scene.column_phase = column_phase();
    scene.db_floor     = view_.db_floor;
    scene.db_ceiling   = view_.db_ceiling;
    scene.scroll_phase = scroll_phase_;
    scene.colormap     = view_.colormap;
    scene.relief       = view_.relief;
    scene.fog          = view_.fog;
    scene.walls        = view_.walls;
    return scene;
}

void App::build_lines()
{
    lines_.clear();
    labels_.clear();

    const ImGuiIO& io          = ImGui::GetIO();
    const float    pixel_ratio = imgui_->pixel_ratio() * imgui_->ui_scale();

    AxesInput axes;
    axes.plot_width      = kPlotWidth;
    axes.plot_height     = view_.plot_height;
    axes.plot_depth      = view_.plot_depth;
    axes.center_hz       = pipeline_.source().frequency();
    axes.span_hz         = pipeline_.source().sample_rate();
    axes.db_floor        = view_.db_floor;
    axes.db_ceiling      = view_.db_ceiling;
    axes.history_seconds = (history_ - 1) * row_period_;
    axes.camera          = camera_;
    axes.view_proj       = view_projection();
    axes.viewport_width  = io.DisplaySize.x;
    axes.viewport_height = io.DisplaySize.y;
    axes.pixel_ratio     = pixel_ratio;
    axes.show_grid       = view_.grid;
    build_axes(axes, lines_, labels_);
    add_listen_box(pixel_ratio);
    add_scan_marks(pixel_ratio);

    // Spectrum traces along the front edge of the plot.
    if (!have_row_)
        return;
    const float column_width = kPlotWidth / static_cast<float>(columns_);
    const float x0           = -0.5f * kPlotWidth + column_phase() * column_width;
    const float z            = 0.5f * view_.plot_depth;
    const float inv_span     = 1.0f / std::max(view_.db_ceiling - view_.db_floor, 1e-3f);
    const auto  height_of    = [&](float db) {
        return std::clamp((db - view_.db_floor) * inv_span, 0.0f, 1.0f) * view_.plot_height;
    };
    const auto add_trace = [&](const std::vector<float>& row, uint32_t color, float width, float bias) {
        float prev_y = height_of(row[0]);
        for (int c = 1; c < columns_; ++c) {
            const float y = height_of(row[static_cast<size_t>(c)]);
            LineInstance line{};
            line.start[0]   = x0 + static_cast<float>(c - 1) * column_width;
            line.start[1]   = prev_y;
            line.start[2]   = z;
            line.width      = width;
            line.end[0]     = x0 + static_cast<float>(c) * column_width;
            line.end[1]     = y;
            line.end[2]     = z;
            line.depth_bias = bias;
            line.color      = color;
            lines_.push_back(line);
            prev_y = y;
        }
    };
    if (view_.max_hold)
        add_trace(max_hold_, pack_rgba(255, 196, 64, 215), 1.2f * pixel_ratio, 5e-5f);
    if (view_.trace)
        add_trace(latest_row_, pack_rgba(255, 255, 255, 235), 1.4f * pixel_ratio, 8e-5f);
}

// ---------------------------------------------------------------------------
// One frame
// ---------------------------------------------------------------------------

void App::frame()
{
    // Nothing to draw while minimised (where the window system tells us):
    // keep the pipeline drained and do not spin.
    if (glfwGetWindowAttrib(window_.get(), GLFW_ICONIFIED) || !renderer_->begin_frame()) {
        pump_rows();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        last_frame_time_ = Clock::now();
        return;
    }

    const auto  now = Clock::now();
    const float dt  = std::clamp(std::chrono::duration<float>(now - last_frame_time_).count(), 0.0f, 0.25f);
    last_frame_time_ = now;
    frame_time_avg_ += (std::max(dt, 1e-4f) - frame_time_avg_) * 0.05f;

    pump_rows();
    show_rows(dt);
    update_level_tracking(dt);

    imgui_->begin_frame();
    handle_input();
    update_listening();
    panel_fraction_ = 0.0f;
    if (view_.show_panel) {
        draw_panel();  // sets panel_fraction_ while docked at the left edge
        draw_listen_window();
        draw_transcript_window();
    }
    lens_shift_ += (panel_fraction_ - lens_shift_) * (1.0f - std::exp(-dt * 10.0f));
    if (camera_untouched_ && (std::fabs(plot_aspect() - framed_aspect_) > 1e-3f ||
                              view_.plot_height != framed_height_ || view_.plot_depth != framed_depth_))
        apply_view(preset_);
    draw_status();
    draw_help();
    update_cursor_readout();
    if (probe_frames_ > 0 && --probe_frames_ == 0) {
        const double tolerance = 1.5 * pipeline_.source().sample_rate() / columns_;
        if (!cursor_.valid || std::fabs(cursor_.frequency_hz - probe_expected_hz_) > tolerance) {
            LOG_ERROR("self-test: cursor readout %s, expected %.4f MHz",
                      cursor_.valid ? (std::to_string(cursor_.frequency_hz / 1e6) + " MHz").c_str() : "missing",
                      probe_expected_hz_ / 1e6);
            self_test_failed_ = true;
        } else {
            LOG_INFO("self-test: cursor readout %.4f MHz, %.1f dBFS, %.2f s ago (expected %.4f MHz)",
                     cursor_.frequency_hz / 1e6, static_cast<double>(cursor_.level_db), cursor_.seconds_ago,
                     probe_expected_hz_ / 1e6);
        }
    }
    build_lines();
    draw_axis_labels();
    draw_listen_label();
    ImGui::Render();
    ++frames_drawn_;

    const double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (!options_.self_test && !options_.screenshot_path.empty() && !auto_screenshot_done_ &&
        elapsed >= options_.screenshot_after) {
        auto_screenshot_done_ = true;
        if (renderer_->screenshot_supported())
            renderer_->request_screenshot(options_.screenshot_path);
        else
            LOG_ERROR("--screenshot: this display does not allow reading back its images");
    }

    renderer_->end_frame(scene_params(), lines_.data(), static_cast<int>(lines_.size()),
                         [this](VkCommandBuffer cmd) { imgui_->render(cmd); });
}

}  // namespace

int run_gui(const Options& options)
{
    App app(options);
    return app.run();
}

int run_list_gpus(const Options& options)
{
    GlfwSession glfw(options.platform);  // so that the listing sees the same window system
    return list_gpus();
}

}  // namespace spectro
