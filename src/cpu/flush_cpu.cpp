// src/cpu/flush_cpu.cpp —— 冲 CPU cache（CPU 专属）
//
// 两个版本，能力和代价完全不同，详见 include/bench/flush.hpp：
//   flush_cpu_cache —— 扫荡法，近似，便宜、不精确
//   flush_cpu_lines —— CLFLUSH，精确，只冲自己那几块
#include "bench/flush.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace bench {

std::size_t cpu_llc_bytes()
{
    std::size_t best = 0;
    for (int i = 0; i < 16; i++) {
        std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/size");
        std::string s;
        if (!(f >> s) || s.empty()) continue;   // 索引用完就没了，不是错误

        std::size_t mul = 1;
        switch (s.back()) {                     // 格式形如 "32K" / "16384K"
            case 'K': case 'k': mul = 1024;          s.pop_back(); break;
            case 'M': case 'm': mul = 1024 * 1024;   s.pop_back(); break;
            default: break;
        }
        if (s.empty()) continue;
        best = std::max(best, static_cast<std::size_t>(std::stoull(s) * mul));
    }
    return best;
}

void flush_cpu_cache(std::size_t bytes)
{
    // 2 倍 LLC：保证每个 cache set 至少被扫两轮，跟替换策略无关。
    // （「保证」是设计意图，不是定理——见 flush.hpp 里那段实测记录。）
    static std::vector<char> scratch = [] {
        std::size_t b = cpu_llc_bytes() * 2;
        if (b < (std::size_t(16) << 20)) b = std::size_t(16) << 20;   // 读不到 sysfs 时的下限
        return std::vector<char>(b);
    }();

    // bytes == 0 用整块（默认）；想研究「冲多少才够」可以传小一点，
    // 但超过缓冲容量就按容量算（缓冲只分配一次，不会为每次调用重新 malloc）。
    const std::size_t n_bytes =
        (bytes == 0 || bytes > scratch.size()) ? scratch.size() : bytes;

    // 每 64 字节读一次（一条 cache line 一次），用 8 字节宽读省掉 7/8 的指令
    auto* p = reinterpret_cast<unsigned long long*>(scratch.data());
    const std::size_t n = n_bytes / sizeof(unsigned long long);

    // volatile 出口：不然整段会被优化掉（读一个没用的 buffer 是死代码）
    static volatile unsigned long long sink = 0;
    unsigned long long acc = sink;
    for (std::size_t i = 0; i < n; i += 8) acc ^= p[i];
    sink = acc;
}

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

void flush_cpu_lines(void const* p, std::size_t bytes)
{
    constexpr std::uintptr_t line = 64;
    const std::uintptr_t first = reinterpret_cast<std::uintptr_t>(p) & ~(line - 1);
    const std::uintptr_t last  = reinterpret_cast<std::uintptr_t>(p) + bytes;

    for (std::uintptr_t a = first; a < last; a += line)
        _mm_clflush(reinterpret_cast<void const*>(a));

    _mm_mfence();   // 等冲完再往下走
}
#endif  // x86

}  // namespace bench
