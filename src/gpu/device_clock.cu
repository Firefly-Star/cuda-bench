// src/gpu/device_clock.cu —— 设备侧计时（cudaEvent）
//
// 每个 rep 用一对事件夹住 inner 次 launch，返回除以 inner 之后的单次耗时。
#include "bench/detail/clock.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace bench::detail {

std::vector<double> time_device(config const& cfg,
                                callback const& launch,
                                callback const& setup,
                                callback const& cleanup)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(cfg.reps));

    for (int r = 0; r < cfg.reps; r++) {
        setup();
        cudaEventRecord(e0, 0);
        // inner 次 launch 挤进【同一个】窗口：固定开销只付一次，
        // 而不是被 reps 平均掉——这是 inner 和 reps 的本质区别。
        for (int j = 0; j < cfg.inner; j++) launch();
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cleanup();

        float span = 0.0f;                       // 事件精度就是 float，别指望更多
        cudaEventElapsedTime(&span, e0, e1);
        samples.push_back(span / cfg.inner);     // 报单次耗时，好和 inner=1 直接比
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return samples;
}

}  // namespace bench::detail
