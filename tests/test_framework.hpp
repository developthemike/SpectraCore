// A deliberately tiny test framework: TEST() registers a function, CHECK*()
// record failures without aborting the test.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace testing {

struct TestCase {
    const char*           name;
    std::function<void()> body;
};

inline std::vector<TestCase>& registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

inline int& failure_count()
{
    static int failures = 0;
    return failures;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline void report_failure(const char* file, int line, const std::string& message)
{
    ++failure_count();
    std::printf("    FAILED %s:%d: %s\n", file, line, message.c_str());
}

}  // namespace testing

#define TEST(name)                                                              \
    static void test_##name();                                                  \
    static ::testing::Registrar registrar_##name(#name, test_##name);           \
    static void test_##name()

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            ::testing::report_failure(__FILE__, __LINE__, "CHECK(" #condition ")"); \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        const auto va_ = (a);                                                   \
        const auto vb_ = (b);                                                   \
        if (!(va_ == vb_))                                                      \
            ::testing::report_failure(__FILE__, __LINE__,                       \
                std::string("CHECK_EQ(" #a ", " #b ") -> ") + std::to_string(va_) + " vs " + std::to_string(vb_)); \
    } while (0)

#define CHECK_NEAR(a, b, tolerance)                                             \
    do {                                                                        \
        const double va_ = static_cast<double>(a);                              \
        const double vb_ = static_cast<double>(b);                              \
        if (!(std::fabs(va_ - vb_) <= static_cast<double>(tolerance)))          \
            ::testing::report_failure(__FILE__, __LINE__,                       \
                std::string("CHECK_NEAR(" #a ", " #b ") -> ") + std::to_string(va_) + " vs " + std::to_string(vb_) + \
                " (tolerance " + std::to_string(static_cast<double>(tolerance)) + ")"); \
    } while (0)
