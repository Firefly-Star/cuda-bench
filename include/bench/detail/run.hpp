// bench/detail/run.hpp —— 内部：测量的主流程（非模板）
//
// 【内部头】实现放在 src/core/run.cpp。
//
// 这个函数是整个框架的核心：绑核 → 预热 → 预跑 → 计时 → 统计 → 打印。
// 它故意不是模板——只有 measure() 那一层薄壳需要是模板。
#pragma once

#include "bench/detail/config.hpp"
#include "bench/stats.hpp"

namespace bench::detail {

// 跑完一次完整测量，返回样本分布。
//
// launch / setup / cleanup 是【引用】调用者栈上的 std::function，
// run_measure 只在本次调用期间使用它们，不保存。
stats run_measure(config const& cfg,
                  callback const& launch,
                  callback const& setup,
                  callback const& cleanup);

}  // namespace bench::detail
