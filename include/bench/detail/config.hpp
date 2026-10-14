// bench/detail/config.hpp —— 内部：把编译期的 options 翻成运行时的 config
//
// 【内部头】公共头不会 include 它里面的实现细节；用户也不该直接用。
//
// 为什么要这一层：
// measure() 必须是个模板（options 是非类型模板参数），但除了「把 Opt 的字段读出来」
// 之外，计时循环本身【不需要】是模板。把它翻成一个运行时结构之后，真正的逻辑
// 就能放进 .cpp / .cu，而不是全堆在头文件里。
#pragma once

#include "bench/options.hpp"

#include <functional>
#include <string>
#include <utility>

namespace bench::detail {

// 被测动作。launch / setup / cleanup 都擦成这个类型。
//
// 擦除的代价：每次调用多一次间接跳转（约 2 ns）。相对动辄毫秒级的 rep
// 完全可以忽略——换来的是计时循环不必是模板。
//
// 擦除【不影响被测代码本身的速度】：lambda 里的计算照常全速跑，只是外面套了
// 一层 std::function。它也不会阻止编译器优化那个计算，只是不能在调用点内联。
using callback = std::function<void()>;

// options 的运行时镜像。字段一一对应，顺序也一致，方便聚合初始化。
struct config {
    TimingSource clock;
    bool          preheat_on;
    float         preheat_seconds;
    int           warmup_runs;
    float         warmup_seconds;
    int           reps;
    int           inner;
    int           pin_cores;
    int           omp_threads;
    std::ostream* out;
    std::string   name;
};

// 编译期 -> 运行时。这个模板本身很小，留在头里是必要的。
template <options Opt>
inline config make_config(std::string name)
{
    return config{
        Opt.clock,        Opt.preheat_on,  Opt.preheat_seconds,
        Opt.warmup_runs,  Opt.warmup_seconds, Opt.reps,
        Opt.inner,        Opt.pin_cores,   Opt.omp_threads,
        Opt.out,          std::move(name),
    };
}

}  // namespace bench::detail
