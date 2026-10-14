// src/core/host_clock.cpp —— 主机侧计时（std::chrono）
//
// 每个 rep 用 steady_clock 夹住 (inner 次 launch + 一次同步)，返回单次耗时。
#include "bench/detail/clock.hpp"

#include <chrono>
#include <cstddef>
#include <cuda_runtime.h>

namespace bench::detail {

// 用 steady_clock 不用 system_clock：后者会被 NTP 校时拨动。
using host_clock = std::chrono::steady_clock;

std::vector<double> time_host(config const& cfg,
                              callback const& launch,
                              callback const& setup,
                              callback const& cleanup)
{
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(cfg.reps));

    for (int r = 0; r < cfg.reps; r++) {
        setup();

        const auto t0 = host_clock::now();
        for (int j = 0; j < cfg.inner; j++) launch();   // inner 次挤进同一个窗口
        cudaDeviceSynchronize();
        const auto t1 = host_clock::now();

        cleanup();

        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count()
                          / cfg.inner);
    }
    return samples;
}

}  // namespace bench::detail
