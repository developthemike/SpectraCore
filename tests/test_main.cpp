#include <cstdio>
#include <cstring>

#include "test_framework.hpp"
#include "util/log.hpp"

int main(int argc, char** argv)
{
    // Optional substring filter: spectro_tests <pattern>
    const char* filter = argc > 1 ? argv[1] : nullptr;

    spectro::log_set_level(spectro::LogLevel::Warn);

    int run = 0, failed = 0;
    for (const auto& test : testing::registry()) {
        if (filter && !std::strstr(test.name, filter))
            continue;
        const int before = testing::failure_count();
        std::printf("[ RUN  ] %s\n", test.name);
        std::fflush(stdout);
        test.body();
        const bool ok = testing::failure_count() == before;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", test.name);
        ++run;
        if (!ok)
            ++failed;
    }

    std::printf("\n%d test(s) run, %d failed\n", run, failed);
    return failed == 0 ? 0 : 1;
}
