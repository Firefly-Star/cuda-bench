// bench/detail/clock.hpp —— 内部：两口钟
//
// 【内部头】实现在各自的源文件里，CPU / GPU 分开放：
//     time_host   -> src/core/host_clock.cpp   （std::chrono，与 GPU 无关）
//     time_device -> src/gpu/device_clock.cu   （cudaEvent）
//
// 两者都返回【每个 rep 一个样本】，除以 inner 之后的值。统计交给 compute_stats。
//
// 为什么拆成两个函数而不是一个带 if constexpr 的模板：见 detail/config.hpp 的说明。
#pragma once

#include "bench/detail/config.hpp"

#include <vector>

namespace bench::detail {

// 主机侧：每个 rep 用 steady_clock 夹住 (inner 次 launch + 一次同步)。
//
// 结尾那个同步不能省。launch() 对 GPU 只是把命令塞进队列就返回（微秒级），
// 不等同步的话测到的是「发射有多快」，不是「活儿干完花了多久」——34 ms 的 kernel
// 会报成 0.01 ms，差三个数量级。
//
// 所以这个数必然比事件的大一点：多了发射开销 + 一次同步的往返。kernel 越长，
// 这层固定开销占比越小。
//
// 注意它同步的是【整个设备】，而事件只等它记的那条流。多流程序里会把别的流
// 也算进来。
std::vector<double> time_host(config const& cfg,
                              callback const& launch,
                              callback const& setup,
                              callback const& cleanup);

// 设备侧：每个 rep 用一对事件夹住 inner 次 launch。
//
// 窗口里只有 launch 发射的活儿：setup() 在 e0 之前入队、cleanup() 在 e1 之后
// 入队，同一条流上顺序执行，两头的耗时都落在窗口外。
//
// 一个要注意的地方：cudaEventSynchronize 让 GPU 在两次 rep 之间空转。e0 到达 GPU
// 时队列是空的，GPU 打完时间戳就干等主机把 kernel 发过来，这段等待也算进
// (e0, e1)。kernel 长（≳1 ms）时可以忽略；想彻底去掉就别在循环里同步（给每个
// rep 各开一对事件，循环里只入队，循环结束后同步一次再逐个读耗时）。
std::vector<double> time_device(config const& cfg,
                                callback const& launch,
                                callback const& setup,
                                callback const& cleanup);

}  // namespace bench::detail
