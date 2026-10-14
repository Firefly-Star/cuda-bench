// src/cpu/affinity.cpp —— 绑核（CPU 专属）
//
// 实现放在这里，对外只有 include/bench/affinity.hpp 里的两个声明。
//
// Linux 之外（Windows 原生）没有 sched_setaffinity，整条路径会被编译掉，
// pin_to_cpus 返回 false；调用方按「不支持」处理，而不是崩掉。
#include "bench/affinity.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>

#if defined(__linux__)
  #include <sched.h>
  #include <unistd.h>
  #if defined(CPU_SETSIZE)
    #define BENCH_HAVE_AFFINITY 1
  #endif
#endif

namespace bench {

std::vector<int> physical_cpus()
{
    std::vector<int> cpus;
#if defined(__linux__)
    std::vector<std::string> seen;   // 已经见过的兄弟组

    for (int i = 0; i < 256; i++) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i)
                        + "/topology/thread_siblings_list");
        std::string s;
        if (!(f >> s) || s.empty()) {
            if (i == 0) return {};   // 连 cpu0 都没有 → sysfs 没挂，别白跑 256 次
            continue;                // 编号用完了，不是错误
        }
        if (std::find(seen.begin(), seen.end(), s) != seen.end()) continue;   // 兄弟，跳过
        seen.push_back(s);
        cpus.push_back(i);           // 每组取编号最小的那个
    }
#endif
    return cpus;
}

bool pin_to_cpus(std::vector<int> const& cpus)
{
#if defined(BENCH_HAVE_AFFINITY)
    if (cpus.empty()) return false;

    cpu_set_t set;
    CPU_ZERO(&set);
    int usable = 0;
    for (int c : cpus) {
        if (c >= 0 && c < CPU_SETSIZE) { CPU_SET(c, &set); usable++; }
    }
    if (usable == 0) return false;

    // sched_setaffinity(0, ...) 作用于【调用线程】，不是整个进程。
    // 这就是为什么必须赶在 worker 线程被创建之前调（见 src/core/run.cpp）。
    if (sched_setaffinity(0, sizeof(set), &set) != 0) return false;

    // overwrite=0：用户自己设过的环境变量优先。
    // 必须在第一个并行区之前设好，OpenMP 运行时只在初始化时读它们。
    setenv("OMP_PROC_BIND", "close", 0);
    setenv("OMP_PLACES", "cores", 0);
    return true;
#else
    (void)cpus;
    return false;   // 非 Linux / 没有 sched_setaffinity
#endif
}

}  // namespace bench
