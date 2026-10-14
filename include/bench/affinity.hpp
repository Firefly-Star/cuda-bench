// bench/affinity.hpp —— 绑核（对外公开）
//
// 为什么需要：本机 16 个逻辑 CPU / 8 个物理核（SMT）。不绑核时调度器会把
// 线程随机撒在 16 个逻辑 CPU 上，于是「这一轮有多少对线程正好撞在同一个物理核上」
// 是随机的。实测同一个二进制跨运行能差 2.7 倍。
//
// 顺带解决一个更隐蔽的问题：不绑核时线程会在核之间【迁移】，而 L1/L2 是每核
// 私有的，迁移一次就等于把私有 cache 全丢了重来。
//
// 绑核能做和不能做的：
//   · 能：把本进程、以及它之后创建的所有线程（含 OpenMP worker）限制在一个固定
//         的逻辑 CPU 集合内。
//   · 不能：阻止【别的】进程用那些核。真正独占要靠内核命令行 isolcpus + 搬 IRQ
//         亲和性 + 搬 unbound workqueue。而且在 WSL2 里那些只能挡住 guest 内部的
//         进程，挡不住宿主 Windows 的调度器。
#pragma once

#include <vector>

namespace bench {

// 每个物理核挑一个逻辑 CPU（避开 SMT 兄弟），返回升序的逻辑 CPU 编号。
//
// 判断谁和谁是兄弟靠 sysfs 的 thread_siblings_list：同一物理核的兄弟共享同一个
// 串（本机是 "0-1" / "2-3" / ...）。每组只取编号最小的那个，于是本机得到
// {0, 2, 4, 6, 8, 10, 12, 14}——正好 8 个物理核。
//
// 读不到 sysfs（非 Linux、或容器里没挂）时返回空，由调用方按「不支持」处理，
// 而不是猜一个可能错的集合。
std::vector<int> physical_cpus();

// 把【当前线程】绑到 cpus。新线程从创建者继承掩码，所以之后 spawn 的 OpenMP
// worker 天然也落在这个集合内。
//
// 另外会设 OMP_PROC_BIND=close / OMP_PLACES=cores（用 overwrite=0，用户自己设过
// 的环境变量优先）。这两个变量必须在【第一个并行区之前】设好——OpenMP 运行时
// 只在第一次初始化时读它们。
//
// 这里【故意不碰线程数】：绑核和开几个线程是两件事，线程数见
// options::omp_threads。
//
// 绑核是【粘性】的，不会自动还原。
bool pin_to_cpus(std::vector<int> const& cpus);

}  // namespace bench
