// bench/bench.hpp —— 伞头（对外公开）
//
// 通常只需要这一个 include：
//
//     #include "bench/bench.hpp"
//
// 想少引一点东西就按需只 include 其中一个：
//     bench/measure.hpp    测量入口（options + stats + measure）
//     bench/flush.hpp      冲 cache（CPU / GPU）
//     bench/affinity.hpp   绑核
//
// 不在公共 API 里的东西（preheat 等假负载）藏在 include/bench/detail/ 下，
// 这个伞头【不会】把它们带出来。
#pragma once

#include "bench/affinity.hpp"
#include "bench/flush.hpp"
#include "bench/measure.hpp"
#include "bench/options.hpp"
#include "bench/stats.hpp"
