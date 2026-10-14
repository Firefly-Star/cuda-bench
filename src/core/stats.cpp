// src/core/stats.cpp —— 样本分布的计算
#include "bench/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace bench {

stats compute_stats(std::vector<double> s)
{
    stats r;
    r.n = static_cast<int>(s.size());
    if (s.empty()) return r;

    std::sort(s.begin(), s.end());
    r.min = s.front();
    r.max = s.back();

    double sum = 0.0;
    for (double v : s) sum += v;
    r.mean = sum / static_cast<double>(s.size());

    const std::size_t m = s.size();
    r.median = (m % 2) ? s[m / 2] : 0.5 * (s[m / 2 - 1] + s[m / 2]);

    // 最近秩法，不做插值：样本只有几十个时插值没有意义，最近秩法至少保证
    // 取到的是一个【真实存在的样本】。样本太少时它会退化成 max，见 stats.hpp。
    std::size_t idx = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(m))) - 1;
    if (idx >= m) idx = m - 1;
    r.p95 = s[idx];

    return r;
}

}  // namespace bench
