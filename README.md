# cuda-bench

**A CUDA / CPU benchmark harness that reports how much you can trust the number.**

> Measurements lie. The same binary, run four times, gave `cpu_matmul3`
> 19.6 / 15.6 / 12.1 / 7.19 ms — a 2.73× spread. If you can't even say which build is
> faster, every "optimization" after that is self-deception. This harness exists to
> press that 2.73× down.

It reports `min / median / p95` instead of just the mean, derives a jitter metric
`(p95-min)/min` that tells you **whether the run is trustworthy at all** (>20% → don't
draw conclusions), pins to physical cores to avoid SMT siblings, moves allocation and
cache-flushing outside the timing window, and documents the cases it *cannot* explain.

The body of this README is in Chinese; [Known limitations](#已知限制) is worth reading even
if you only skim the rest — it lists the measurements this environment fundamentally cannot make.

---

## 为什么做这个

一个给自己用的 CUDA / CPU 基准框架。

起因很简单：**测量本身会骗人。** 同一个二进制跑四次，`cpu_matmul3` 得到
19.6 / 15.6 / 12.1 / 7.19 ms —— 差 2.73 倍。如果连"哪个更快"都说不清，
后面所有的优化都是自欺欺人。这个框架就是为了把那 2.73 倍按下去。

## 它解决什么

| 措施 | 效果 |
|---|---|
| 报 `min / median / p95` 而不是只报均值 | 均值会被个别离群点单方面拉偏，`min` 才是"没有干扰时能做到多快" |
| 一个抖动指标 `(p95-min)/min` | 直接告诉你**这次测量可不可信**，> 20% 就别下结论 |
| 绑核到物理核（避开 SMT 兄弟） | 消除"这一轮有多少对线程撞在同一物理核上"的随机性 |
| 把分配 / 清零移出计时窗口 | 被测区间里只剩真正想测的计算 |
| `inner` 内循环 | 摊薄固定开销，拉长单个计时窗口 |

## 目录

```
bench/
├── include/bench/            ← 对外（用户只需要看这里）
│   ├── bench.hpp             伞头，通常只 include 这一个
│   ├── measure.hpp           测量入口：measure / measure_stats / noop
│   ├── options.hpp           测量参数：options / TimingSource
│   ├── stats.hpp             结果分布：stats / compute_stats
│   ├── flush.hpp             冲 cache：flush_cpu_cache / flush_cpu_lines / flush_gpu_cache
│   ├── affinity.hpp          绑核：physical_cpus / pin_to_cpus
│   └── detail/               ← 内部，公共头不引用（preheat 就藏在这儿）
└── src/                      ← 实现
    ├── core/                 与 CPU/GPU 都无关：主流程、统计、chrono 计时
    ├── cpu/                  绑核、冲 CPU cache
    └── gpu/                  preheat、cudaEvent 计时、冲 GPU cache
```

`src/cpu` 和 `src/gpu` 分开是为了内聚；公共头只是引用它们，不暴露实现。

## 快速开始

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/01-matmul-cpu
```

```cpp
#include "bench/bench.hpp"

bench::measure<bench::options{
    .clock         = bench::TimingSource::Host,   // 纯 CPU 用墙钟
    .preheat_on    = false,                       // CPU 侧不需要假负载
    .warmup_seconds = 1.0f,                       // 真实 workload 先热身 1 秒
    .reps          = 20,                          // ≥20：太少时 p95 会退化成 max
    .pin_cores     = 8,                           // 钉在 8 个物理核上
    .omp_threads   = 8}>(                         // 一核一线程
    [&] { my_kernel(A, B, C); },                  // launch：被测的就是它
    [&] { C.clear(); },                           // setup：计时窗口之前
    bench::noop{},                                // cleanup
    "my kernel");
```

输出：

```
my kernel: min 6.0614  median 6.82268  p95 7.46704  mean 6.85162  max 7.87261 ms
  [抖动(p95-min)/min 23.19%]  （绑核 8 物理核[0 2 4 6 8 10 12 14] + OpenMP 8 线程 + ...）
```

想拿结构化的结果就用 `bench::measure_stats(...)`，它返回一个 `bench::stats`。

## 三条硬性要求

| | 缺了会怎样 |
|---|---|
| `-std=c++20` | `options` 是 C++20 的 class NTTP，报 `a nontype template parameter may not have class type` |
| `-Iinclude` | 公共头用 `bench/...` 前缀引入 |
| `-fopenmp` | **不报错**，但 `#pragma omp parallel for` 被静默忽略，OpenMP 版本退化成单线程（实测慢 3 倍） |

第三条是最阴的：没有警告，你只会看到"OpenMP 好像没用"。

## 几个设计决定

**只有一层薄模板。** `options` 是非类型模板参数，所以 `measure()` 必须是模板——
模板的定义必须在使用点可见，这是语言规则。但**除此之外没有别的东西必须是模板**：
计时循环、预热、绑核、统计全部在 `.cpp` / `.cu` 里。做法是把 `options` 翻成运行时
`config`，把三个回调擦成 `std::function`。代价是每次调用多一次间接跳转（约 2 ns），
换来的是框架主体可以独立编译、独立测试。

**`preheat` 不在公共 API 里。** 它藏在 `include/bench/detail/`，公共头不引用它——
只 include `bench/bench.hpp` 的程序调用 `bench::detail::preheat` 会编译失败。

**绑核和线程数是两件事。** `pin_cores` 只决定"能用哪些核"，线程数用 `omp_threads`。
早期版本把它们绑在一起（绑 8 核 → 强制 8 线程），代价是连带丢掉 SMT，很难查。

**`options::out` 必须是最后一个成员。** 不是风格问题，是 nvcc 13.3 的 cudafe++ 的
一个 bug：类被当非类型模板参数、成员里有指针、而指针后面还有初始化值非零的成员时，
会报一句完全无关的 `invalid conversion from 'int' to 'std::ostream*'`，而且错误位置
指向每个 `template <options Opt = options{}>` 的声明行。详见 `options.hpp`。

## 已知限制

- **WSL2 里锁不了 CPU 频率。** `/sys/.../cpufreq/` 根本不存在（vCPU 是 Hyper-V 虚拟
  处理器，没有 ACPI CPPC），`/proc/cpuinfo` 的频率是编出来的常数。读不到真实频率，
  就意味着"锁定"无法被证明。绑核能减少方差，但消不掉频率漂移。
- **跨会话的绝对值不可比。** 同一个配置（绑 8 核 + 8 线程 + 20 reps）在一个会话里
  量到 8.6~9.2 ms、另一个会话里 5.0~5.3 ms，差 1.7 倍——宿主负载和频率状态在漂移。
  **只有同一次运行内的相对比较可信。**
- **拉长计时窗口只能靠加大问题规模。** `inner` 对累加型 kernel（`C[i][k] += ...`）
  无效，因为连跑 k 次会算错。那种情况要加大 N。经验：rep 从 6 ms 提到 380 ms，
  抖动从 15~44% 降到 7~9%。
- **`Host` 计时含一次 `cudaDeviceSynchronize()`。** 纯 CPU 基准也付这个钱
  （WSL2 里是半虚拟化调用，几十微秒）。占比很小，但它是噪声来源之一。
- **⚠️ 未解释：同一份源码、两种构建方式，`cpu_matmul3` 差 1.6 倍。**
  实测 CMake 构建 ≈ 7.5 ms，而直接 `nvcc` 一行命令构建 ≈ 4.4 ms（N=512，20 reps，
  绑 8 核 + 8 线程，交替运行 4 轮，差距稳定跟二进制走、不跟运行顺序走）。
  已经排除的：编译标志（`nvcc --dryrun` 里宿主编译命令逐字相同）、目标文件反汇编
  （相同）、链接行（相同）、加载的库与 `DT_NEEDED`（相同）、libgomp 的 ICV 配置
  （`OMP_DISPLAY_ENV` 逐行相同）。唯一找到的差异是 `.text` 差 32 字节，以及一个
  反相关：**`matmul0`（单线程）跑得快的那个构建，`matmul2`（8 线程）就慢。**
  另外确认：慢的那个**必须先跑过 `matmul0`/`matmul1`**——单独只跑 `matmul3` 时
  两种构建都快。

  这高度指向 CPU 功耗/频率状态的滞后效应，而这个环境**没有任何手段观测频率**
  （`/sys/.../cpufreq/` 不存在，`/proc/cpuinfo` 的频率是编出来的常数）——
  观测不到就无法证明，也就无法修。

  **实践结论：在当前环境里，绝对数值只在【同一个构建 + 同一个二进制】内可比。
  要做基线就固定一种构建方式，所有对比都在它里面做。** 真要解决需要原生 Linux。

## 实测记录

同一短窗口内，`cpu_matmul3`（N=512，20 reps）三种配置各 5 轮：

| 配置 | min 中位数 | 抖动 |
|---|---|---|
| 不绑核 + 16 线程 | 6.32 ms | 75% ~ 249% |
| 绑 8 核 + 16 线程 | 6.51 ms | 41% ~ 88% |
| **绑 8 核 + 8 线程** | **6.06 ms** | **15% ~ 44%** |

三者的**最好成绩几乎一样**（5.0~7.2 ms，落在噪声里），但 16 线程的中位数比 min
高 30~50%，8 线程只高 8%。原因是这个内核**访存受限**（B 是列访问、流式读），
瓶颈在共享带宽上，同物理核的第二个线程不增加吞吐，只增加争抢。

⚠️ 别把这个结论推广到别的内核：纯计算、依赖延迟受限的活儿（烧 FP 的微基准那种）
SMT 能接近 2 倍。**"SMT 有没有用"完全取决于瓶颈在哪。**

---

## 相关实验

这套抖动判据是在下面几个实验里磨出来的，它们共用同一套方法：

- [`cuda-managed-vs-malloc`](https://github.com/Firefly-Star/cuda-managed-vs-malloc) —— 用这套
  「排除法 + 比值而非绝对值」的方式定位到：Windows 系平台上 `cudaMallocManaged` 的数据
  从不进显存，访存全程跨 PCIe，带宽恒为 ~9.9 GB/s，对比 `cudaMalloc` 的 ~131 GB/s（13 倍）。
- `cuda-benchmark-warmup` —— 预热、`nsys` 计时不变性、ramp 步进的对照实验
- `cuda-execution-model` —— 执行模型相关基准

## License

[MIT](LICENSE)
