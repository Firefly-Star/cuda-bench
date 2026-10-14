// src/core/run.cpp —— 测量的主流程
//
// 整个框架的核心，也是唯一「本来在头文件里、现在搬出来了」的那部分：
//
//     ① 绑核  →  ② 设线程数  →  ③ 预热  →  ④ 预跑  →  ⑤ 计时  →  ⑥ 统计并打印
//
// 它故意【不是模板】。原来这些逻辑写在 measure.cuh 里，靠 if constexpr 把
// 不用的分支删掉；代价是整段逻辑都必须在头文件里，任何一个使用它的程序都要
// 重新编译它。现在编译期信息只保留 measure.hpp 那一层薄壳要用的部分，
// 这里全部是运行时判断——代价是每次测量几个分支，而不是每个 rep 一次。
#include "bench/detail/run.hpp"

#include "bench/affinity.hpp"
#include "bench/detail/clock.hpp"
#include "bench/detail/gpu_internal.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
  #include <omp.h>
#endif

namespace bench::detail {

namespace {

// ① + ② 绑核、设 OpenMP 线程数。返回一段说明，写进日志。
//
// 这两件事必须放在最前面，有两个原因：
//   · OpenMP 只在【第一个并行区】初始化时读 OMP_PROC_BIND / OMP_PLACES /
//     OMP_NUM_THREADS，而第一个并行区就发生在后面 warm_up 的 launch 里。
//   · sched_setaffinity 只作用于【调用线程】，所以要赶在 worker 线程被创建
//     之前先把自己的掩码设好，新线程才会继承它。
std::string setup_affinity(config const& cfg)
{
    std::string desc;

    if (cfg.pin_cores > 0) {
        std::vector<int> phys = physical_cpus();
        if (phys.empty()) {
            desc = "绑核失败(读不到 CPU 拓扑)";
        } else {
            if (static_cast<int>(phys.size()) > cfg.pin_cores)
                phys.resize(static_cast<std::size_t>(cfg.pin_cores));

            if (pin_to_cpus(phys)) {
                desc = "绑核 " + std::to_string(phys.size()) + " 物理核[";
                for (std::size_t i = 0; i < phys.size(); i++)
                    desc += (i ? " " : "") + std::to_string(phys[i]);
                desc += "]";
            } else {
                desc = "绑核失败(sched_setaffinity 被拒)";
            }
        }
    }

    // 线程数放在绑核【之后】。不显式设置时，libgomp 的默认线程数是按当前亲和
    // 掩码里的可用 CPU 数算出来的——掩码变小，默认线程数就跟着变小。
#if defined(_OPENMP)
    if (cfg.omp_threads > 0) omp_set_num_threads(cfg.omp_threads);

    // 报告【实际生效】的线程数，而不是只报用户写了什么。不报的话，
    // pin_cores=8 会让默认线程数从 16 悄悄变成 8，很容易误判。
    desc += (desc.empty() ? "" : " + ");
    desc += "OpenMP " + std::to_string(omp_get_max_threads()) + " 线程";
#endif

    return desc;
}

// ③ + ④ 预热和预跑。两者都不计时。
void warm_up(config const& cfg,
             callback const& launch,
             callback const& setup,
             callback const& cleanup)
{
    // 预热：把 GPU 频率从空闲拉到满载。为什么 CPU 侧一般该关掉它，见
    // detail/gpu_internal.hpp。
    if (cfg.preheat_on) preheat(cfg.preheat_seconds);

    // 预跑：有的 kernel 第一次慢，是因为指令 cache / 页表还没热，跟频率无关。
    // 这几步不计时，但必须做——否则第一轮就被算进统计里了。
    for (int i = 0; i < cfg.warmup_runs; i++) {
        setup();
        launch();
        cleanup();
    }

    // 按时长预跑。warmup_runs 是【计数】口径，同一个数字在不同 workload 上含义
    // 完全不同——0.18 ms 的活儿跑 100 次只有 18 ms，而频率爬升、功耗/温度预算、
    // 页表建立这些都是【时间】过程（ms ~ s 级），18 ms 走不到。
    if (cfg.warmup_seconds > 0.0f) {
        const auto t0 = std::chrono::steady_clock::now();
        do {
            setup();
            launch();
            cleanup();
        } while (std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count() < cfg.warmup_seconds);
    }

    // 就算一次预跑都没有，也要同步一下：preheat 的活儿还在队列里，
    // 留到计时循环里会被算进第一个 rep。
    cudaDeviceSynchronize();
}

// ⑥ 把「这次是怎么测的」写清楚，省得看结果的人猜。
void print_result(config const& cfg, std::string const& env_desc, stats const& st)
{
    std::ostringstream how;

    if (cfg.preheat_on) how << "预热 " << cfg.preheat_seconds << "s + ";
    else                how << "不预热 + ";

    // 两种预热分开报。旧版只检查 warmup_runs，于是
    // {.warmup_runs = 0, .warmup_seconds = 1.0f} 会打印「不预跑」，
    // 而按时长预跑其实是开着的。
    if (cfg.warmup_runs > 0) {
        how << "预跑 " << cfg.warmup_runs << " 次 + ";
    } else if (cfg.warmup_seconds <= 0.0f) {
        how << "不预跑 + ";
    }
    if (cfg.warmup_seconds > 0.0f) {
        how << "按时长预跑 " << cfg.warmup_seconds << "s + ";
    }

    if (cfg.inner > 1) how << "每次计时内循环 " << cfg.inner << " 次 + ";

    how << "计时 " << cfg.reps << " 次"
        << (cfg.clock == TimingSource::Device ? "，cudaEvent" : "，chrono");

    std::string how_str = how.str();
    if (!env_desc.empty()) how_str = env_desc + " + " + how_str;

    *cfg.out << cfg.name
             << ": min "    << st.min
             << "  median " << st.median
             << "  p95 "    << st.p95
             << "  mean "   << st.mean
             << "  max "    << st.max
             << " ms   [抖动(p95-min)/min " << st.jitter() * 100.0 << "%]"
             << "   （" << how_str << "）" << std::endl;
}

}  // namespace

stats run_measure(config const& cfg,
                  callback const& launch,
                  callback const& setup,
                  callback const& cleanup)
{
    const std::string env_desc = setup_affinity(cfg);

    warm_up(cfg, launch, setup, cleanup);

    std::vector<double> samples = (cfg.clock == TimingSource::Device)
                                      ? time_device(cfg, launch, setup, cleanup)
                                      : time_host(cfg, launch, setup, cleanup);

    const stats st = compute_stats(std::move(samples));

    if (cfg.out != nullptr) print_result(cfg, env_desc, st);
    return st;
}

}  // namespace bench::detail
