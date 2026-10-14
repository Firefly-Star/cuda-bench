// bench/options.hpp —— 测量参数（对外公开）
//
// 这个文件只有「怎么测」的旋钮，不含任何实现。
//
// ⚠️ 一条必须记住的约束：std::ostream* out 必须是 options 的【最后一个成员】。
//    原因写在它前面那段注释里（nvcc cudafe++ 的一个坑）。往里加字段时，
//    新字段只能加在 out 之前。
#pragma once

#include <iostream>
#include <ostream>

namespace bench {

// 从哪一侧读时间戳。
//
//   Host   —— std::chrono::steady_clock。时间戳由主机打，量的是
//             「主机说开始 → 主机确知干完」的墙钟时间，把发射开销和一次
//             同步的往返也算进去了。纯 CPU 的活儿用这个。
//   Device —— cudaEvent。时间戳由 GPU 自己打，量的是 GPU 时间线那一段，
//             主机在干什么（发射、分配、printf）不算进去。
//
//             注意它【对纯 CPU 负载没有意义】：那时 GPU 时间线上什么都没有，
//             你量到的是一个空区间。反过来 Host 对 GPU 负载是有意义的，
//             只是多含一层固定开销（kernel 越长占比越小）。
enum class TimingSource { Host, Device };

// 测量参数：所有「怎么测」的旋钮都在这里，全部有默认值。
//
// 它同时是【编译期策略】—— 用法是把它当模板参数塞给 measure()：
//
//     bench::measure(launch);                                    // 全默认
//     bench::measure<bench::options{.reps = 200}>(launch);
//     bench::measure<bench::options{.preheat_on = false, .warmup_runs = 0}>(launch, "冷");
//     bench::measure<bench::options{.clock = bench::TimingSource::Host}>(launch, "墙钟");
//
// 编译期传参的代价是这些值不能再从 argv 里读——要换个数得重新编译。
//
// 能放进来的（C++20 structural type 的要求）：
//     enum / bool / int / float / double / 指针
// 放不进来的（都实测过）：
//     std::string            —— 不是字面量类型
//     const char* 指向字面量  —— 非类型模板参数不能取字符串字面量的地址
//   所以 name 只能留在函数参数里。
//
// C++20 的指定初始化器必须按【声明顺序】写，可以跳过但不能倒过来：
//     options{.clock = ..., .reps = 200}   ✓
//     options{.reps = 200, .clock = ...}   ✗ out-of-order initializers are nonstandard
//
// 当前声明顺序（加字段时请同步更新这一行）：
//     clock, preheat_on, preheat_seconds, warmup_runs, warmup_seconds,
//     reps, inner, pin_cores, omp_threads, out
struct options {
    // 两组「预热」别混，它们的负载性质和执行顺序都不一样：
    //   preheat_*  ——【假负载】（GPU 上是个烧显存的 kernel）。只干一件事：把 GPU
    //                 从空闲频率升到满载。对 CPU 侧毫无意义，而且它会把 cache
    //                 灌满无关数据（实测残留 0.1~0.3 ms 的扰动）。
    //   warmup_*   ——【真实 workload】预跑（setup + launch + cleanup）。热身的是
    //                 指令 cache / 页表 / 首次分配 / 分支预测器，CPU 和 GPU 都有用。
    // 顺序永远是 preheat → warmup → 计时：real workload 放在最后，正好把假负载
    // 留下的 cache 状态盖掉。
    TimingSource  clock           = TimingSource::Device;  // 用哪口钟
    bool          preheat_on      = true;                  // 要不要预热
    float         preheat_seconds = 1.5f;                   // 烤多久（preheat_on=false 时不用）
    int           warmup_runs     = 10;                     // 烤完再预跑几次（0 = 不预跑）
    float         warmup_seconds  = 0.0f;                   // 再按时长预跑：至少跑这么久（0 = 不用）
    int           reps            = 50;                     // 计时循环跑几次（报的是分布，不是只报平均）

    // inner：把 launch 连续发射 inner 次，但【只记一次时间】，报的数是除以
    //        inner 之后的单次耗时。
    //
    //        它和 reps 是两件事，别混：
    //          reps  —— 重复【测量】，用来刻画分布（min/median/p95）
    //          inner —— 重复【工作】，用来摊薄固定开销，拉长单个计时窗口
    //        区别在于固定开销（发射、同步、计时器分辨率）在 inner 里只被付一次，
    //        而不是被平均掉；偶尔被抢占一次的相对占比也按 inner 缩小。
    //
    //        ⚠️ 前提：launch 必须是【幂等】的。累加型 kernel（比如
    //        C[i][k] += ... 那种矩阵乘）连跑 inner 次会算错——那种情况
    //        要改用更大的问题规模 N。
    int           inner           = 1;

    // pin_cores：0 = 不绑核；N > 0 = 把本进程绑到前 N 个【物理核】上，
    //            每核只取一个逻辑 CPU，避开 SMT 兄弟。
    //
    //            它只决定【能用哪些核】，【不决定开几个线程】——线程数用
    //            下面的 omp_threads。早先的实现把两者绑在一起（绑 8 核 →
    //            强制 8 线程），代价是连带丢掉 SMT，很难查。
    //
    //            绑核是【粘性】的，measure() 不会还原——一次运行里的所有变体
    //            必须在同一套核上才可比。
    int           pin_cores       = 0;

    // omp_threads：0 = 不干预，让 OpenMP 自己定（libgomp 会按【当前亲和掩码】
    //              里的可用 CPU 数来定，所以 pin_cores=8 时通常就是 8）；
    //              N > 0 = 显式 omp_set_num_threads(N)。
    //
    //              想让 16 线程铺在 8 个物理核上（吃 SMT）：
    //                  .pin_cores = 8, .omp_threads = 16
    //              想一核一线程（可预测性优先）：
    //                  .pin_cores = 8, .omp_threads = 8
    //
    //              只在编译时带了 -fopenmp（有 _OPENMP）时才起作用。
    //              ⚠️ 反过来，编译时【忘了】-fopenmp 不会报错，但 #pragma omp
    //              parallel for 会被静默忽略、OpenMP 版本退化成单线程
    //              （实测慢 3 倍）。这是个没有报错的坑。
    int           omp_threads     = 0;

    // 必须是最后一个成员。
    //
    // 这不是风格问题，是 nvcc 13.3 的 cudafe++ 的一个 bug（已实测定位）：
    // 当一个类被当作【非类型模板参数】、成员里有【指针】、而指针【后面】还有
    // 初始化值非零的成员时，cudafe++ 会报一句完全莫名其妙的错：
    //     error: invalid conversion from 'int' to 'std::ostream*'
    // 而且错误位置指向每一个 template <options Opt = options{}> 的【声明行】，
    // 跟真正出问题的成员毫无关系。g++ 单独编译同一份文件不会报这个错。
    //
    // 实测边界（35 行最小复现，只改这一个值就能翻转）：
    //     ...; std::ostream* out = &std::cout; int x = 0;   → 编译通过
    //     ...; std::ostream* out = &std::cout; int x = 1;   → 编译失败
    //     ...; int x = 1; std::ostream* out = &std::cout;   → 编译通过 ← 本文的写法
    // 结论：out 放最后，其余字段全都放它前面。
    std::ostream* out             = &std::cout;             // 结果打印到哪
};

}  // namespace bench
