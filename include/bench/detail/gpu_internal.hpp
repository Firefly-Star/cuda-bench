// bench/detail/gpu_internal.hpp —— 内部：GPU 侧的假负载
//
// 【内部头】这个文件刻意不被任何公共头引用，所以 preheat 在公共 API 里
// 是【不可见】的：用户 include bench/bench.hpp 拿不到它。
//
// 唯一的使用方是 src/core/run.cpp（它按 config::preheat_on 决定要不要调），
// 以及 GPU 侧的 flush（见 bench/flush.hpp）。
//
// 放在 include/bench/detail/ 而不是 src/ 里，是因为 run.cpp 和 preheat.cu 都要
// 看到这个声明；而 CMake 只需要暴露一个 include 目录。想做到物理上也不可见，
// 就得把这层藏进 PIMPL——对一个学习用的框架不值得。
#pragma once

namespace bench::detail {

// 在 GPU 上烧 preheat_seconds 秒【纯计算】负载，把 GPU 从空闲频率拉到满载。
//
// 为什么需要：GPU 空闲时降到 300 MHz，一有负载才逐步爬到满载。这个爬升是
// 【台阶】不是斜坡——实测冷态 12.0 ms、热态 6.86 ms，差 75%，而且门槛附近不稳定。
// 不预热的话，你测的是冷启动的数。
//
// 为什么不能用「读数组写数组」那种循环：那烤的是显存带宽，SM 反而没压满。
// 实现里所有变量都在寄存器里，专门把 SM 频率拉起来。
//
// 为什么对 CPU 侧毫无意义：CPU 没有这种「一有负载就爬频率」的台阶（有也是
// 另一种机制），而且假负载会把 cache 灌满无关数据（实测残留 0.1~0.3 ms 的扰动）。
// CPU 侧靠 warmup_* 用真实 workload 热身就够了。
//
// 实现在 src/gpu/preheat.cu。
void preheat(float seconds = 1.5f);

}  // namespace bench::detail
