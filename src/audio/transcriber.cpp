#include "audio/transcriber.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <numbers>
#include <utility>

#include "util/log.hpp"
#include "util/thread_util.hpp"

#if SPECTRO_HAVE_WHISPER
#include <whisper.h>
#endif

namespace spectro {

namespace {

constexpr double kWhisperRate    = 16000.0;
constexpr int    kHalfTaps       = 16;     // of the interpolation filter, either side
constexpr float  kTargetRms      = 0.1f;   // -20 dBFS
constexpr float  kMaxGain        = 30.0f;  // +30 dB at most
constexpr double kTailSeconds    = 0.3;    // of silence after the speech, to let its last word end
constexpr double kShortestSeconds = 1.0;   // Whisper is unsure of less
constexpr int    kMostRepeats    = 3;      // more of the same word in a row is Whisper stuck in a loop

bool alphanumeric(unsigned char c)
{
    // Any letter or digit - and any character beyond ASCII, which is a letter
    // of some other alphabet more often than not.
    return std::isalnum(c) || c >= 0x80;
}

#if SPECTRO_HAVE_WHISPER
constexpr int kBeamSize = 5;

// whisper.cpp tells a lot as it loads: only its warnings and errors are worth passing on.
void whisper_log(ggml_log_level level, const char* text, void*)
{
    if (level != GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_ERROR) {
        return;
    }
    std::string line(text ? text : "");
    while (!line.empty() && (line.back() == '\n' || line.back() == ' '))
        line.pop_back();
    if (!line.empty())
        LOG_WARN("whisper: %s", line.c_str());
}

bool whisper_abort(void* data)
{
    return static_cast<std::atomic<bool>*>(data)->load(std::memory_order_relaxed);
}
#endif

}  // namespace

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

std::string transcript_speaker(const Transcript& t)
{
    std::string who = t.destination;
    if (t.source != 0)
        who += (who.empty() ? "from " : " from ") + std::to_string(t.source);
    return who;
}

std::string transcript_channel(const Transcript& t)
{
    return t.slot >= 0 ? t.system + " slot " + std::to_string(t.slot + 1) : t.system;
}

std::string format_transcript(const Transcript& t)
{
    const auto  seconds = static_cast<std::time_t>(std::floor(t.start));
    std::tm     local{};
    localtime_r(&seconds, &local);
    char when[32];
    std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &local);
    const std::string who = transcript_speaker(t);
    char              head[192];
    std::snprintf(head, sizeof(head), "%s  %.5f MHz  %s%s%s  (%.1f s)", when, t.frequency / 1e6,
                  transcript_channel(t).c_str(), who.empty() ? "" : "  ", who.c_str(), t.seconds);
    return std::string(head) + ": " + t.text;
}

std::string clean_transcript(std::string_view text)
{
    // Whisper's notes of what is not speech: [BLANK_AUDIO], (static), *beep*.
    std::string kept;
    kept.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        const char c       = text[i];
        const char closing = c == '[' ? ']' : c == '(' ? ')' : c == '{' ? '}' : c == '*' ? '*' : 0;
        if (closing) {
            const size_t end = text.find(closing, i + 1);
            if (end != std::string_view::npos) {
                i = end;
                kept += ' ';
                continue;
            }
        }
        if (text.compare(i, 3, "\xE2\x99\xAA") == 0 || text.compare(i, 3, "\xE2\x99\xAB") == 0) {
            i += 2;  // music: U+266A, U+266B
            continue;
        }
        kept += c;
    }
    // One space between words, none around.
    std::string out;
    out.reserve(kept.size());
    bool space = false;
    for (const char c : kept) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            space = !out.empty();
            continue;
        }
        if (space && c != '.' && c != ',' && c != '?' && c != '!')
            out += ' ';
        space = false;
        out += c;
    }
    if (std::none_of(out.begin(), out.end(), [](char c) { return alphanumeric(static_cast<unsigned char>(c)); }))
        return {};

    // On speech it cannot make out, Whisper can get stuck saying the same
    // word over and over ("white, white, white, white, white"): once is enough.
    const auto key = [](std::string_view word) {
        std::string k;
        for (const char c : word) {
            if (alphanumeric(static_cast<unsigned char>(c)))
                k += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return k;
    };
    std::vector<std::string_view> words;
    for (size_t at = 0; at < out.size();) {
        const size_t end = std::min(out.find(' ', at), out.size());
        words.emplace_back(out.data() + at, end - at);
        at = end + 1;
    }
    std::string tidy;
    for (size_t i = 0; i < words.size();) {
        size_t run = 1;
        while (i + run < words.size() && !key(words[i]).empty() && key(words[i + run]) == key(words[i]))
            ++run;
        const size_t keep = run > static_cast<size_t>(kMostRepeats) ? 1 : run;
        for (size_t k = run - keep; k < run; ++k)
            tidy += (tidy.empty() ? "" : " ") + std::string(words[i + k]);
        i += run;
    }
    return tidy;
}

std::vector<float> speech_for_whisper(const float* speech, size_t count)
{
    // Twice the rate: the samples in between from a windowed sinc (a half-band
    // filter; the vocoders' speech has nothing above 4 kHz to fold back).
    static const std::vector<float> taps = [] {
        std::vector<float> c(kHalfTaps);
        double             sum = 0.0;
        for (int j = 0; j < kHalfTaps; ++j) {
            const double t      = j + 0.5;
            const double sinc   = std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
            const double x      = t / (kHalfTaps + 0.5);  // -1 ... 1 across the filter
            const double window = 0.42 + 0.5 * std::cos(std::numbers::pi * x) + 0.08 * std::cos(2.0 * std::numbers::pi * x);
            c[j] = static_cast<float>(sinc * window);
            sum += 2.0 * c[j];
        }
        for (float& v : c)
            v = static_cast<float>(v / sum);
        return c;
    }();

    double power = 0.0;
    float  peak  = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        power += static_cast<double>(speech[i]) * speech[i];
        peak = std::max(peak, std::fabs(speech[i]));
    }
    const auto rms  = static_cast<float>(count > 0 ? std::sqrt(power / static_cast<double>(count)) : 0.0);
    float      gain = rms > 0.0f ? std::min(kTargetRms / rms, kMaxGain) : 1.0f;
    if (peak * gain > 0.99f)
        gain = 0.99f / peak;

    const auto at = [&](ptrdiff_t k) { return k >= 0 && static_cast<size_t>(k) < count ? speech[k] : 0.0f; };
    const size_t length = std::max(2 * count + static_cast<size_t>(kTailSeconds * kWhisperRate),
                                   static_cast<size_t>(kShortestSeconds * kWhisperRate));
    std::vector<float> out(length, 0.0f);
    for (size_t k = 0; k < count; ++k) {
        out[2 * k] = gain * speech[k];
        float between = 0.0f;
        for (int j = 0; j < kHalfTaps; ++j)
            between += taps[j] * (at(static_cast<ptrdiff_t>(k) - j) + at(static_cast<ptrdiff_t>(k) + 1 + j));
        out[2 * k + 1] = gain * between;
    }
    return out;
}

// ---------------------------------------------------------------------------
// The transcriber
// ---------------------------------------------------------------------------

struct Transcriber::Engine {
#if SPECTRO_HAVE_WHISPER
    whisper_context* context = nullptr;
    ~Engine()
    {
        if (context)
            whisper_free(context);
    }
#endif
};

bool Transcriber::available()
{
    return SPECTRO_HAVE_WHISPER != 0;
}

Transcriber::Transcriber(Settings settings) : settings_(std::move(settings)), engine_(std::make_unique<Engine>())
{
    status_.model = std::filesystem::path(settings_.model).filename().string();
    if (!settings_.file.empty()) {
        file_ = std::fopen(settings_.file.c_str(), "a");
        if (!file_)
            LOG_WARN("Could not open %s to write the transcripts to: %s", settings_.file.c_str(), std::strerror(errno));
    }
    thread_ = std::thread([this] { run(); });
}

Transcriber::~Transcriber()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    abort_.store(true, std::memory_order_relaxed);
    wake_.notify_all();
    if (thread_.joinable())
        thread_.join();
    if (file_)
        std::fclose(file_);
}

void Transcriber::set_enabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    status_.enabled = enabled;
    if (!enabled) {
        queue_.clear();
        if (!status_.busy)
            idle_.notify_all();
    }
}

void Transcriber::voice_over(VoiceOver&& over)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_ || !status_.enabled || status_.state == TranscriberStatus::State::Failed)
        return;
    if (queue_.size() >= kMaxWaiting) {
        queue_.pop_front();
        ++status_.dropped;
    }
    queue_.push_back(std::move(over));
    wake_.notify_one();
}

TranscriberStatus Transcriber::status() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    TranscriberStatus s = status_;
    s.waiting           = queue_.size();
    return s;
}

std::vector<Transcript> Transcriber::since(uint64_t serial) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Transcript> out;
    for (const Transcript& t : done_) {
        if (t.serial > serial)
            out.push_back(t);
    }
    return out;
}

bool Transcriber::wait_idle(double timeout) const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_.wait_for(lock, std::chrono::duration<double>(timeout), [&] {
        return status_.state == TranscriberStatus::State::Failed ||
               (status_.state == TranscriberStatus::State::Ready && queue_.empty() && !status_.busy);
    });
}

void Transcriber::run()
{
    set_current_thread_name("spectro-whisper");
#if SPECTRO_HAVE_WHISPER
    static std::once_flag log_once;
    std::call_once(log_once, [] { whisper_log_set(whisper_log, nullptr); });

    whisper_context_params params = whisper_context_default_params();
    engine_->context              = whisper_init_from_file_with_params(settings_.model.c_str(), params);
    whisper_context* const ctx    = engine_->context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ctx) {
            status_.state = TranscriberStatus::State::Failed;
            status_.error = "could not load the Whisper model " + settings_.model;
            queue_.clear();
            idle_.notify_all();
            LOG_ERROR("Speech to text: %s", status_.error.c_str());
            return;
        }
        status_.state = TranscriberStatus::State::Ready;
    }
    const bool  multilingual = whisper_is_multilingual(ctx) != 0;
    const char* language     = multilingual ? settings_.language.c_str() : "en";
    if (!multilingual && settings_.language != "en")
        LOG_WARN("Speech to text: %s knows only English; use a model without .en for %s", status_.model.c_str(),
                 settings_.language.c_str());
    const int threads = settings_.threads > 0
                            ? settings_.threads
                            : std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 1, 8);
    LOG_INFO("Speech to text: %s loaded (%s, %d threads)", status_.model.c_str(),
             multilingual ? language : "English", threads);
    const whisper_token end_of_text = whisper_token_eot(ctx);

    for (;;) {
        VoiceOver over;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_)
                return;
            over = std::move(queue_.front());
            queue_.pop_front();
            status_.busy = true;
        }

        double power = 0.0;
        for (const float v : over.speech)
            power += static_cast<double>(v) * v;
        const double rms = over.speech.empty() ? 0.0 : std::sqrt(power / static_cast<double>(over.speech.size()));
        if (rms < kQuietRms) {
            finish(over, {}, 0.0f);  // silence: the call was muted, or there is no vocoder
            continue;
        }

        const std::vector<float> samples = speech_for_whisper(over.speech.data(), over.speech.size());
        whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
        wp.beam_search.beam_size     = kBeamSize;
        wp.n_threads                 = threads;
        wp.language                  = language;
        wp.translate                 = false;
        wp.no_context                = true;  // each over on its own: another talker, another call
        wp.no_timestamps             = true;
        wp.print_special             = false;
        wp.print_progress            = false;
        wp.print_realtime            = false;
        wp.print_timestamps          = false;
        wp.suppress_blank            = true;
        wp.suppress_nst              = true;
        wp.abort_callback            = whisper_abort;
        wp.abort_callback_user_data  = &abort_;

        const auto  began = std::chrono::steady_clock::now();
        std::string text;
        double      p_sum = 0.0;
        int         p_count = 0;
        if (whisper_full(ctx, wp, samples.data(), static_cast<int>(samples.size())) == 0) {
            const int segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < segments; ++i) {
                text += whisper_full_get_segment_text(ctx, i);
                const int tokens = whisper_full_n_tokens(ctx, i);
                for (int k = 0; k < tokens; ++k) {
                    if (whisper_full_get_token_id(ctx, i, k) >= end_of_text)
                        continue;  // timestamps and other special tokens
                    p_sum += whisper_full_get_token_p(ctx, i, k);
                    ++p_count;
                }
            }
        }
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const double speed = took > 0.0 ? over.seconds() / took : 0.0;
            status_.speed      = status_.speed > 0.0 ? 0.7 * status_.speed + 0.3 * speed : speed;
        }
        finish(over, clean_transcript(text), p_count > 0 ? static_cast<float>(p_sum / p_count) : 0.0f);
    }
#else
    std::lock_guard<std::mutex> lock(mutex_);
    status_.state = TranscriberStatus::State::Failed;
    status_.error = "this build has no speech to text (SPECTRO_WITH_WHISPER=OFF)";
    queue_.clear();
    idle_.notify_all();
#endif
}

void Transcriber::finish(const VoiceOver& over, std::string text, float confidence)
{
    Transcript t;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        status_.busy = false;
        if (text.empty()) {
            ++status_.silent;
        } else {
            ++status_.transcribed;
            t.serial      = ++serial_;
            t.start       = over.start;
            t.seconds     = over.seconds();
            t.frequency   = over.frequency;
            t.system      = over.system;
            t.slot        = over.slot;
            t.destination = over.destination;
            t.source      = over.source;
            t.part        = over.part;
            t.text        = std::move(text);
            t.confidence  = confidence;
            done_.push_back(t);
            if (done_.size() > kKept)
                done_.pop_front();
        }
        if (queue_.empty())
            idle_.notify_all();
    }
    if (file_ && t.serial != 0) {
        std::fprintf(file_, "%s\n", format_transcript(t).c_str());
        std::fflush(file_);
    }
}

}  // namespace spectro
