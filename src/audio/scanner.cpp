#include "audio/scanner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace spectro {

namespace {

// A signal's extent: columns standing this far above the noise floor.
constexpr float kClearOfFloorDb = 6.0f;
// Its occupied width: columns within this of its peak.
constexpr float kOccupiedDb = 26.0f;
// Stretches separated by a dip this narrow are one signal.
constexpr double kMergeGapHz = 2e3;

float to_db(float power) { return 10.0f * std::log10(std::max(power, 1e-30f)); }
float to_power(float db) { return std::pow(10.0f, db / 10.0f); }

}  // namespace

float spectrum_noise_floor(const float* power, int columns)
{
    if (columns <= 0)
        return 0.0f;
    std::vector<float> sorted(power, power + columns);
    const auto         middle = sorted.begin() + columns / 2;
    std::nth_element(sorted.begin(), middle, sorted.end());
    return *middle;
}

std::vector<DetectedSignal> find_signals(const float* power, const SpectrumView& view, float floor_power)
{
    std::vector<DetectedSignal> signals;
    const int    columns   = view.columns;
    const double column_hz = view.column_hz();
    const float  clear     = floor_power * to_power(kClearOfFloorDb);
    const int    merge_gap = std::max(1, static_cast<int>(std::lround(kMergeGapHz / column_hz)));

    int c = 0;
    while (c < columns) {
        if (power[c] < clear) {
            ++c;
            continue;
        }
        // A stretch above the floor, bridging dips of up to merge_gap columns.
        const int first = c;
        int       last  = c;
        int       gap   = 0;
        for (++c; c < columns; ++c) {
            if (power[c] >= clear) {
                last = c;
                gap  = 0;
            } else if (++gap > merge_gap) {
                break;
            }
        }
        c = last + 1;

        int peak = first;
        for (int k = first; k <= last; ++k) {
            if (power[k] > power[peak])
                peak = k;
        }
        // Occupied width and power centre, from the columns near the peak.
        const float near_peak = std::max(clear, power[peak] / to_power(kOccupiedDb));
        int         low = peak, high = peak;
        double      weight = 0.0, moment = 0.0;
        for (int k = first; k <= last; ++k) {
            if (power[k] < near_peak)
                continue;
            low  = std::min(low, k);
            high = std::max(high, k);
            weight += power[k];
            moment += static_cast<double>(power[k]) * k;
        }

        DetectedSignal s;
        s.frequency_hz = view.frequency(weight > 0.0 ? moment / weight : peak);
        s.occupied_hz  = (high - low + 1) * column_hz;
        s.low_hz       = view.frequency(first) - 0.5 * column_hz;
        s.high_hz      = view.frequency(last) + 0.5 * column_hz;
        s.peak_dbfs    = to_db(power[peak]);
        s.wide         = s.occupied_hz >= kWideFmMinOccupied;
        signals.push_back(s);
    }

    std::sort(signals.begin(), signals.end(),
              [](const DetectedSignal& a, const DetectedSignal& b) { return a.peak_dbfs > b.peak_dbfs; });
    return signals;
}

double listen_bandwidth_for(const DetectedSignal& signal, double column_hz)
{
    if (signal.wide)
        return 200e3;
    // The measured width is only good to a column or so.
    const double width = std::max(0.0, signal.occupied_hz - column_hz);
    if (width <= 11e3)
        return 12.5e3;
    if (width <= 18e3)
        return 16e3;
    return 25e3;
}

Scanner::Scanner(ScanReceiver& receiver) : receiver_(receiver) {}

void Scanner::set_hang(double seconds)
{
    hang_ = std::clamp(seconds, 0.0, 60.0);
}

void Scanner::start()
{
    if (active())
        return;
    state_        = State::Searching;
    resume_above_ = -1e300;
    pending_      = false;
    restart_spectrum();
    receiver_.scan_mute(true);
    status_       = {};
    status_.state = state_;
}

void Scanner::stop()
{
    if (!active())
        return;
    state_ = State::Off;
    receiver_.scan_mute(false);
    candidates_.clear();
    status_       = {};
    status_.state = state_;
}

void Scanner::restart_spectrum()
{
    smoothed_.clear();
    warmup_left_ = 1.5 * kSmoothingSeconds;
}

void Scanner::skip()
{
    if (state_ != State::Receiving && state_ != State::Holding)
        return;
    const double half = 0.5 * std::max(locked_bw_, status_.occupied_hz);
    lockouts_.push_back({locked_hz_ - half, locked_hz_ + half});
    release();
}

bool Scanner::locked_out(double frequency) const
{
    return std::any_of(lockouts_.begin(), lockouts_.end(),
                       [frequency](const Range& r) { return frequency >= r.low && frequency <= r.high; });
}

void Scanner::lock(const DetectedSignal& signal, const SpectrumView& view)
{
    locked_hz_ = signal.frequency_hz;
    locked_bw_ = listen_bandwidth_for(signal, view.column_hz());
    // Let the receiver find the carrier: the display is only as fine as its columns.
    const double search = std::max(0.5 * view.column_hz(), 0.25 * locked_bw_);
    receiver_.scan_tune(locked_hz_, locked_bw_, search);
    receiver_.scan_mute(false);
    state_ = State::Receiving;

    status_.frequency   = locked_hz_;
    status_.bandwidth   = locked_bw_;
    status_.occupied_hz = signal.occupied_hz;
    status_.wide        = signal.wide;
    status_.level_dbfs  = signal.peak_dbfs;
    status_.hold_left   = 0.0;
}

void Scanner::release()
{
    // Go on from the top of this signal.
    resume_above_ = locked_hz_ + 0.5 * locked_bw_;
    state_        = State::Searching;
    pending_      = false;
    receiver_.scan_mute(true);
}

// The strongest column within the channel being received.
float Scanner::channel_peak(const SpectrumView& view) const
{
    const double reach = 0.5 * locked_bw_ + 0.5 * view.column_hz();
    float        best  = -200.0f;
    for (int c = 0; c < view.columns; ++c) {
        if (std::fabs(view.frequency(c) - locked_hz_) <= reach)
            best = std::max(best, smoothed_[static_cast<size_t>(c)]);
    }
    return best;
}

void Scanner::on_row(const float* row_dbfs, const RowInfo& info, const SpectrumView& view)
{
    if (!active())
        return;
    if (info.gap) {
        // The radio was retuned (or the stream broke): what was there is gone.
        restart_spectrum();
        candidates_.clear();
        if (state_ == State::Receiving || state_ == State::Holding)
            release();
        status_.state      = state_;
        status_.warming_up = true;
        return;
    }
    if (static_cast<int>(smoothed_.size()) != view.columns || info.fft_size != fft_size_) {
        fft_size_ = info.fft_size;
        restart_spectrum();
    }

    // Smooth the spectrum over about a tenth of a second.
    const double period = info.period_s > 0.0 ? info.period_s : 0.02;
    if (smoothed_.empty()) {
        smoothed_.assign(row_dbfs, row_dbfs + view.columns);
    } else {
        const float alpha = static_cast<float>(1.0 - std::exp(-period / kSmoothingSeconds));
        for (int c = 0; c < view.columns; ++c) {
            float& s = smoothed_[static_cast<size_t>(c)];
            s += alpha * (row_dbfs[c] - s);
        }
    }
    warmup_left_ -= period;
    status_.warming_up = warmup_left_ > 0.0;
    if (status_.warming_up)
        return;

    // Signals above the threshold that could be FM channels.
    power_.resize(smoothed_.size());
    for (size_t c = 0; c < smoothed_.size(); ++c)
        power_[c] = to_power(smoothed_[c]);
    floor_power_       = spectrum_noise_floor(power_.data(), view.columns);
    status_.floor_dbfs = to_db(floor_power_);
    candidates_.clear();
    for (const DetectedSignal& s : find_signals(power_.data(), view, floor_power_)) {
        if (s.peak_dbfs < threshold_ || s.occupied_hz > kFmMaxOccupied || locked_out(s.frequency_hz))
            continue;
        const double half = 0.5 * listen_bandwidth_for(s, view.column_hz());
        if (std::fabs(s.frequency_hz - view.centre_hz) + half > 0.5 * view.sample_rate)
            continue;  // would not fit into the band
        candidates_.push_back(s);
    }

    switch (state_) {
    case State::Searching: {
        // The next signal up from where the last one was, wrapping around.
        const DetectedSignal* next = nullptr;
        for (const DetectedSignal& s : candidates_) {
            if (s.frequency_hz > resume_above_ && (!next || s.frequency_hz < next->frequency_hz))
                next = &s;
        }
        if (!next) {
            for (const DetectedSignal& s : candidates_) {
                if (!next || s.frequency_hz < next->frequency_hz)
                    next = &s;
            }
        }
        if (!next) {
            pending_ = false;
            break;
        }
        // It has to stay put for a moment: a sweep or a click is not a signal.
        const double tolerance = std::max(2.0 * view.column_hz(), 0.25 * listen_bandwidth_for(*next, view.column_hz()));
        if (pending_ && std::fabs(next->frequency_hz - pending_hz_) <= tolerance) {
            pending_for_ += period;
        } else {
            pending_     = true;
            pending_for_ = period;
        }
        pending_hz_ = next->frequency_hz;
        if (pending_for_ >= kConfirmSeconds)
            lock(*next, view);
        break;
    }

    case State::Receiving:
    case State::Holding: {
        // Follow the receiver onto the carrier it found.
        const double settled = receiver_.scan_frequency();
        if (!std::isnan(settled))
            locked_hz_ = settled;
        status_.frequency  = locked_hz_;
        status_.level_dbfs = channel_peak(view);

        if (state_ == State::Receiving) {
            if (status_.level_dbfs < threshold_ - kHysteresisDb) {
                state_            = State::Holding;
                status_.hold_left = hang_;
                receiver_.scan_mute(true);
            }
        } else if (status_.level_dbfs >= threshold_) {
            state_            = State::Receiving;
            status_.hold_left = 0.0;
            receiver_.scan_mute(false);
        } else {
            status_.hold_left -= period;
            if (status_.hold_left <= 0.0)
                release();
        }
        break;
    }

    case State::Off:
        break;
    }
    status_.state = state_;
}

}  // namespace spectro
