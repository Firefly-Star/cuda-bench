// bench/measure.hpp —— 测量入口（对外公开）
//
// 这是框架对外最主要的头。里面只有【薄模板壳】：把编译期的 options 翻成运行时
// 配置、把回调擦成 std::function，然后交给 detail::run_measure。
//
// 真正的逻辑（绑核 → 预热 → 预跑 → 计时 → 统计 → 打印）在 src/core/run.cpp。
//
//   bench::measure(launch);                                   // 只要均值
//   bench::measure<bench::options{.reps = 200}>(launch, "名字");   // 打印完整分布
//   bench::stats st = bench::measure_stats(launch, setup, cleanup, "名字");
//
// 为什么这一层必须是模板：options 是【非类型模板参数】（C++20 的 class NTTP），
// 模板的定义必须在使用点可见。这是语言规则，绕不过去。
//
// 但除了这一层，框架里【没有别的东西】必须是模板——原来和它捆在一起的那些
// `if constexpr` 分支（预热开关、内循环、用哪口钟）现在都是运行时判断，
// 代价是每次测量的几个分支，而不是每个 rep 一次。
//
// 注意本头 include 了 <cuda_runtime.h>，因为 measure_kernel 要用 dim3。
// 这是「整个框架都经 nvcc 编译」这个选择的直接后果。
#pragma once

#include "bench/options.hpp"
#include "bench/stats.hpp"
#include "bench/detail/config.hpp"
#include "bench/detail/run.hpp"

#include <cuda_runtime.h>   // dim3（measure_kernel 用）

#include <functional>
#include <string>
#include <utility>

namespace bench {

// 空操作。占住 setup / cleanup 的位置，给「没有准备工作」的测量用。
struct noop { void operator()() const {} };

// 测量一个「启动 kernel 的动作」，返回完整的样本分布。
//
// 内部自动做六件事：绑核 → 预热 → 预跑 → 计时 → 统计 → 打印。
//
//   · launch  每次计时都跑，被测的就是它。
//   · setup   在计时窗口【之前】跑（每个 rep 一次）。放准备工作、冲 cache、
//             以及累加型 kernel 需要的清零。
//   · cleanup 在计时窗口【之后】跑（每个 rep 一次）。放校验、还原。
//
// 三个回调都只在本次调用期间被使用，不会被存下来，所以按引用捕获局部变量是安全的。
template <options Opt = options{}, typename F1, typename F2 = noop, typename F3 = noop,
          typename = decltype(std::declval<F2&>()()),
          typename = decltype(std::declval<F3&>()())>
inline stats measure_stats(F1&& launch, F2&& setup = noop{}, F3&& cleanup = noop{},
                           std::string name = "")
{
    // 这个函数体就是整个框架里唯一「必须」是模板的部分，而它只干两件事：
    //   1. 把 Opt 的字段读出来，做成运行时 config
    //   2. 把三个回调擦成 std::function
    // 剩下的全部交给非模板的 run_measure。
    //
    // 三个 lambda 按引用捕获调用者栈上的 launch / setup / cleanup。run_measure
    // 是同步调用，不会把它们存下来，所以没有生命周期问题。
    return detail::run_measure(
        detail::make_config<Opt>(std::move(name)),
        [&] { launch(); },
        [&] { setup(); },
        [&] { cleanup(); });
}

// 同上，但只返回均值（单位 ms）。
//
// ⚠️ 别拿这个返回值当「代表性的性能」用。均值会被个别离群点单方面拉偏，
// 而那个离群点可能只是一次进程被抢占。跨实现比较请看 min / median——
// 所以多数时候你应该用 measure_stats()。
//
// 保留它主要是为了向后兼容历史日志（那些日志里的「平均」就是这个数）。
template <options Opt = options{}, typename F1, typename F2 = noop, typename F3 = noop,
          typename = decltype(std::declval<F2&>()()),
          typename = decltype(std::declval<F3&>()())>
inline double measure(F1&& launch, F2&& setup = noop{}, F3&& cleanup = noop{},
                      std::string name = "")
{
    return measure_stats<Opt>(launch, setup, cleanup, std::move(name)).mean;
}

// 没有 setup / cleanup 的简写：bench::measure(launch, "名字");
//
// 这个重载必须存在，否则 measure(launch, "名字") 会被上面那个抢走
// （F2 = const char[5]，对模板来说是完美匹配），然后在里面调 "名字"() 报错。
// 上面两个 typename 约束就是为它准备的。
template <options Opt = options{}, typename F1>
inline double measure(F1&& launch, std::string name)
{
    return measure_stats<Opt>(launch, noop{}, noop{}, std::move(name)).mean;
}

// 一次 kernel 发射的简写。不需要 setup / cleanup 时用。
//
//     bench::measure_kernel<bench::options{.reps = 100}>(myKernel, grid, block, a, b, n);
//
// 需要动态共享内存、或者一次要测一串 kernel，就别用它，直接把整个序列写成
// lambda 传给 measure()。
template <options Opt = options{}, typename Kernel, typename... Args>
inline double measure_kernel(Kernel kernel, dim3 grid, dim3 block, Args... args)
{
    return measure<Opt>([=] { kernel<<<grid, block>>>(args...); });
}

}  // namespace bench
