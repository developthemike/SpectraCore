// Abstract source of complex baseband samples.
//
// Every source delivers the bladeRF's native sample format, SC16 Q11:
// interleaved int16_t I/Q pairs where full scale is +/-2048.  Sources with
// another native format (e.g. the RTL-SDR's 8 bits) convert to it.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace spectro {

struct ValueRange {
    double min  = 0.0;
    double max  = 0.0;
    double step = 0.0;
};

struct SourceConfig {
    std::string kind        = "bladerf";  // "bladerf", "rtlsdr", "sim" or "file"
    double      frequency   = 2.437e9;    // Hz
    double      sample_rate = 20e6;       // samples per second
    double      bandwidth   = 0.0;        // Hz; 0 selects the sample rate
    double      gain        = 30.0;       // dB
    bool        agc         = false;

    // radios
    std::string device;        // bladeRF: libbladeRF device identifier; RTL-SDR: serial number
                               // or index; empty = first found
    int         channel = 0;   // bladeRF: RX channel index
    int         ppm     = 0;   // RTL-SDR: crystal frequency error to correct, parts per million
    bool        verbose = false;

    // file playback
    std::string file_path;
    std::string file_format = "sc16";  // "sc16" (bladeRF Q11), "cs16" (full scale), "cf32", "cs8", "cu8"
    bool        file_loop   = true;

    // simulator
    unsigned sim_seed = 1;
};

struct SourceInfo {
    std::string kind;         // short name, e.g. "bladeRF"
    std::string description;  // one line describing the device or file
    bool        tunable  = false;
    bool        has_gain = false;
    ValueRange  frequency_range;
    ValueRange  gain_range;
};

struct ReadStatus {
    bool     discontinuity = false;  // samples were lost before this block
    uint64_t lost_samples  = 0;      // how many, if known
};

class IqSource {
public:
    virtual ~IqSource() = default;

    // Opens the device or file and starts the stream.  Throws std::runtime_error
    // with a message suitable for showing to the user.
    virtual void start() = 0;
    virtual void stop()  = 0;

    // Blocks until samples are available, then writes at most max_samples IQ
    // pairs to `iq`.  Returns the number written, 0 on a timeout (the caller
    // should simply retry), or -1 on an unrecoverable error (see last_error()).
    virtual int read(int16_t* iq, int max_samples, ReadStatus& status) = 0;

    virtual const SourceInfo& info() const = 0;
    virtual double            sample_rate() const = 0;  // actual rate, Hz
    virtual double            frequency() const   = 0;  // centre frequency, Hz
    virtual double            bandwidth() const   = 0;  // analog filter bandwidth, Hz
    virtual double            gain() const        = 0;  // dB

    // Thread-safe; may be called while another thread is inside read().
    virtual bool set_frequency(double hz) { (void)hz; return false; }
    virtual bool set_gain(double db)      { (void)db; return false; }

    virtual std::string last_error() const { return {}; }

    // A good number of samples to request per read() call.
    virtual int preferred_chunk() const { return 16384; }
};

// Creates the source selected by config.kind.  Throws std::runtime_error for an
// unknown kind or one that this build does not support.
std::unique_ptr<IqSource> make_source(const SourceConfig& config);

// Whether this build can talk to bladeRF / RTL-SDR hardware.
bool bladerf_supported();
bool rtlsdr_supported();

}  // namespace spectro
