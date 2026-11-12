#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "test_framework.hpp"
#include "util/iq_ring.hpp"

using spectro::IqRing;

TEST(ring_capacity_rounds_up_to_power_of_two)
{
    IqRing ring(3000);
    CHECK_EQ(ring.capacity(), size_t{4096});
    CHECK_EQ(ring.readable(), size_t{0});
}

TEST(ring_write_then_read_back)
{
    IqRing ring(1024);
    auto   span = ring.write_span(100);
    CHECK_EQ(span.samples, size_t{100});
    for (int i = 0; i < 200; ++i)
        span.data[i] = static_cast<int16_t>(i);
    ring.commit(100);
    CHECK_EQ(ring.readable(), size_t{100});

    const auto spans = ring.peek(100);
    CHECK_EQ(spans.first_count, size_t{100});
    CHECK_EQ(spans.second_count, size_t{0});
    bool ok = true;
    for (int i = 0; i < 200; ++i)
        ok = ok && spans.first[i] == static_cast<int16_t>(i);
    CHECK(ok);

    ring.consume(60);
    CHECK_EQ(ring.readable(), size_t{40});
    CHECK_EQ(ring.peek(40).first[0], int16_t{120});
}

TEST(ring_wraps_around)
{
    IqRing ring(1024);
    // Advance both positions close to the end of the buffer.
    {
        auto span = ring.write_span(1000);
        CHECK_EQ(span.samples, size_t{1000});
        ring.commit(1000);
        ring.consume(1000);
    }
    // 24 samples fit before the end; the rest must come from a second span.
    int16_t next = 0;
    size_t  written = 0;
    while (written < 100) {
        auto span = ring.write_span(100 - written);
        CHECK(span.samples > 0);
        if (span.samples == 0)
            return;
        for (size_t i = 0; i < 2 * span.samples; ++i)
            span.data[i] = next++;
        ring.commit(span.samples);
        written += span.samples;
    }
    const auto spans = ring.peek(100);
    CHECK_EQ(spans.first_count, size_t{24});
    CHECK_EQ(spans.second_count, size_t{76});
    bool    ok       = true;
    int16_t expected = 0;
    for (size_t i = 0; i < 2 * spans.first_count; ++i)
        ok = ok && spans.first[i] == expected++;
    for (size_t i = 0; i < 2 * spans.second_count; ++i)
        ok = ok && spans.second[i] == expected++;
    CHECK(ok);
}

TEST(ring_reports_full)
{
    IqRing ring(1024);
    ring.commit(ring.write_span(1024).samples);
    CHECK_EQ(ring.readable(), size_t{1024});
    CHECK_EQ(ring.write_span(10).samples, size_t{0});
    ring.consume(5);
    CHECK_EQ(ring.write_span(10).samples, size_t{5});
}

TEST(ring_close_wakes_waiting_consumer)
{
    IqRing            ring(1024);
    std::atomic<bool> result{true};
    std::thread       consumer([&] { result.store(ring.wait_readable(10)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ring.close();
    consumer.join();
    CHECK(result.load() == false);

    // Data that is already buffered can still be read after closing.
    IqRing ring2(1024);
    ring2.commit(ring2.write_span(10).samples);
    ring2.close();
    CHECK(ring2.wait_readable(10));
    CHECK(!ring2.wait_readable(11));
}

TEST(ring_concurrent_stream_is_lossless)
{
    // A producer pushes a known sequence through a small ring, in awkward chunk
    // sizes, while a consumer reads overlapping frames the way the DSP thread does.
    constexpr size_t kTotal = 2'000'000;
    constexpr size_t kFrame = 512;
    constexpr size_t kHop   = 256;
    IqRing           ring(4096);

    std::thread producer([&] {
        uint32_t value = 0;
        size_t   sent  = 0;
        size_t   chunk = 1;
        while (sent < kTotal) {
            auto span = ring.write_span(std::min(chunk, kTotal - sent));
            if (span.samples == 0) {
                std::this_thread::yield();
                continue;
            }
            for (size_t i = 0; i < span.samples; ++i) {
                span.data[2 * i]     = static_cast<int16_t>(value & 0x7fff);
                span.data[2 * i + 1] = static_cast<int16_t>((value >> 15) & 0x7fff);
                ++value;
            }
            ring.commit(span.samples);
            sent += span.samples;
            chunk = chunk % 777 + 1;
        }
        ring.close();
    });

    uint64_t position = 0;  // index of the first sample of the current frame
    uint64_t errors   = 0;
    uint64_t frames   = 0;
    while (ring.wait_readable(kFrame)) {
        const auto spans = ring.peek(kFrame);
        for (size_t i = 0; i < kFrame; ++i) {
            const int16_t* s = i < spans.first_count ? spans.first + 2 * i
                                                     : spans.second + 2 * (i - spans.first_count);
            const uint32_t expected = static_cast<uint32_t>(position + i);
            if (s[0] != static_cast<int16_t>(expected & 0x7fff) ||
                s[1] != static_cast<int16_t>((expected >> 15) & 0x7fff))
                ++errors;
        }
        ring.consume(kHop);
        position += kHop;
        ++frames;
    }
    producer.join();

    CHECK_EQ(errors, uint64_t{0});
    CHECK_EQ(frames, uint64_t{(kTotal - kFrame) / kHop + 1});
}
