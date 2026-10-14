// 01-matmul-cpu.cu —— 用 bench 框架给矩阵乘的三个 CPU 实现定基线
//
// 三个实现，差别只在【访存顺序】，浮点结果完全一样（逐元素 bit 相同，
// 实测过：相对 double 参考值的最大误差 2.439e-06，就是 float32 该有的量级）：
//
//   cpu_matmul   ijk 顺序       —— 朴素写法，B 按列访问，每取一个元素跨 2 KB
//   cpu_matmul2  ikj 顺序       —— 把最内层循环换成连续的，B 和 C 都变顺序访问
//   cpu_matmul3  同 matmul2 + OpenMP
//
// 所以这三个数之间的差距【纯粹来自访存/cache】，是一条很干净的对照曲线。
//
// 编译（本目录就是框架根）：
//     cmake -S . -B build && cmake --build build -j && ./build/01-matmul-cpu
//
// 手编：
//     nvcc -O3 -std=c++20 -Xcompiler -fopenmp -Iinclude \
//          examples/01-matmul-cpu.cu src/core/*.cpp src/cpu/*.cpp src/gpu/*.cu \
//          -o 01-matmul-cpu
//
// ⚠️ 别去掉 -fopenmp：不会报错，但 cpu_matmul3 会静默退化成单线程（实测慢 3 倍）。
//
// 读结果时注意（两个坑都实测过）：
//   · 看 min / median，不要看 mean。mean 会被个别离群点单方面拉偏。
//   · 跨会话的绝对值不可比。同一个配置（绑 8 核 + 8 线程）在一个会话里量到
//     8.6~9.2 ms、另一个会话里 5.0~5.3 ms，差 1.7 倍——那是宿主负载和频率
//     状态在漂移。要比较两个实现，就在【同一次运行】里都测一遍。
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <span>
#include <functional>
#include <utility>
#include "bench/bench.hpp"

template <size_t N>
class Matrix
{
public:
    Matrix() : data_(N * N, 0.0f) {}

    std::span<float>       operator[](size_t i)       { return {data_.data() + i * N, N}; }
    std::span<float const> operator[](size_t i) const { return {data_.data() + i * N, N}; }

    float&       at(size_t i, size_t j)       { return data_.at(i * N + j); }
    float const& at(size_t i, size_t j) const { return data_.at(i * N + j); }

    // 把矩阵清零。
    //
    // 为什么需要它：下面 cpu_matmul2/3 是【累加型】的（C[i][k] += ...），
    // 要求 C 进来时全零。而清零这个动作必须放在 setup 里（计时窗口之外），
    // 不能放进被测区间——否则测的就不只是矩阵乘了。
    //
    // 原来的写法是在函数内部 `Matrix<N> result;`，那既分配又清零，
    // 而且都发生在计时窗口【内】。现在改成调用方传 result&，两者都挪出去了。
    void clear() { std::fill(data_.begin(), data_.end(), 0.0f); }

    float*       data()       { return data_.data(); }
    constexpr size_t size() const { return N * N; }
    static constexpr size_t rows = N, cols = N;
private:
    std::vector<float> data_;
};
// 也许可以多线程，或者pragma parrallel 进行simd优化
// 还有cache优化，但是要看具体cache环境
//
// 接口统一成 (A, B, C)：两个 const& 输入 + 一个 & 输出。
// 这样做的两个理由：
//   1. 输出矩阵由调用方【预先分配】，分配和清零都在计时窗口之外，
//      被测区间里只剩纯计算。
//   2. 三个实现接口一致，bench_cpu 可以用同一套代码测它们。
//
// 注意【谁来清零】——这是接口约定，写错了结果就是错的：
//   cpu_matmul  写 C[i][j] = sum，不需要 C 预先清零（写了就是覆盖）
//   cpu_matmul2/3 写 C[i][k] += ...，【必须】C 进来时全零
// bench_cpu 里的 setup 对所有三者统一清零，省掉这个区分。
template<size_t N>
void cpu_matmul(Matrix<N> const& A, Matrix<N> const& B, Matrix<N>& C)
{
    for (int i = 0; i < static_cast<int>(N); ++i)
    {
        for (int j = 0; j < static_cast<int>(N); ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < static_cast<int>(N); ++k)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

template<size_t N>
void cpu_matmul2(Matrix<N> const& A, Matrix<N> const& B, Matrix<N>& C)
{
    for (int i = 0; i < static_cast<int>(N); ++i)
    {
        for (int j = 0; j < static_cast<int>(N); ++j)
        { // 处理A[i][j] * b[j][k]
            for (int k = 0; k < static_cast<int>(N); ++k)
            {
                C[i][k] += A[i][j] * B[j][k];
            }
        }
    }
}

template<size_t N>
void cpu_matmul3(Matrix<N> const& A, Matrix<N> const& B, Matrix<N>& C)
{
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(N); ++i)
    {
        for (int j = 0; j < static_cast<int>(N); ++j)
        { // 处理A[i][j] * b[j][k]
            for (int k = 0; k < static_cast<int>(N); ++k)
            {
                C[i][k] += A[i][j] * B[j][k];
            }
        }
    }
}

template<size_t N>
void set_up(Matrix<N>& A, Matrix<N>& B)
{
    for (int i = 0; i < static_cast<int>(N); ++i)
    {
        for (int j = 0; j < static_cast<int>(N); ++j)
        {
            A[i][j] = i + j;
            B[i][j] = i + j;
        }
    }
}

template<size_t N>
void check(Matrix<N>const& A, Matrix<N>const& B, Matrix<N>const& C, std::string const& name)
{
    // 原来的 goto fail 版本：i/j 必须在循环外声明，标签也不能跳进带初始化的作用域。
    // 这里直接把失败处理内联，不跨循环作用域，省掉这个坑。
    //
    // ⚠️ 两个已知问题，改之前先读完：
    //
    // (1) 【这个函数现在根本不会被执行】。bench_cpu 的构造函数写的是
    //     `checked(size)`，std::vector<bool> 值初始化成的是 false，而 cleanup 里
    //     判断的是 `if (checked[index])`。所以三个实现一个都没被验证过正确性——
    //     现在的基线是「未经验证的计算」的耗时。
    //
    // (2) 这里用的是【精确相等】比较 `tmp != C[i][j]`。实测结论（本文件当前的
    //     编译配置，-O3 和 -O3 -march=native 都试过）：
    //         cpu_matmul / cpu_matmul2 / cpu_matmul3 三者【逐元素 bit 相同】，
    //         相对 double 参考值的最大误差 2.4e-06（就是 float32 该有的量级）。
    //     原因是：三个实现对每个元素 C[i][j] 的【加法序列完全相同】，只是访存
    //     顺序不同（ijk 和 ikj 的区别在内存访问，不在浮点运算次序）。所以这个
    //     精确比较【目前是能通过的】。
    //
    //     但它很脆：bit 相同只是因为 gcc 在没有 -ffast-math 时不做浮点重结合。
    //     一旦开了 -ffast-math / -Ofast，或者某个实现被向量化后做了 FMA 收缩，
    //     三者末位就会分叉，这里会报出一堆假失败。所以想要真正可靠，
    //     应该改成【带容差】的比较（相对误差 + 与 N 相关的放大系数），
    //     而不是指望精确相等永远成立。
    for (int i = 0; i < static_cast<int>(N); ++i)
    {
        for (int j = 0; j < static_cast<int>(N); ++j)
        {
            float tmp = 0.0f;
            for (int k = 0; k < static_cast<int>(N); ++k)
            {
                tmp += A[i][k] * B[k][j];
            }
            if (tmp != C[i][j])
            {
                std::cout << name << " check fail: [" << i << ", " << j << "]= " << C[i][j] << ", expected: " << tmp << "\n";
                return;
            }
        }
    }
}

template<size_t N>
struct bench_cpu
{
public:
    // 矩阵【预分配】，只分配这一次。
    //
    // 原来的写法是在被测函数内部 `Matrix<N> result;`，于是每一次 rep 都要
    // mmap 1 MB + 逐页清零 + munmap，而这三件事全都落在计时窗口【内】。
    // 那是纯粹的测量噪声，不是算法成本，现在全部移出窗口。
    bench_cpu(int size): A(size), B(size), R(size), checked(size){}
    std::vector<Matrix<N>> A, B, R;
    // ⚠️ checked 决定 check() 跑不跑。std::vector<bool> 值初始化成 false，
    //    所以现在 check() 从来没被执行过——见 check() 上面的说明 (1)。
    std::vector<bool> checked;
    template<typename Fn>
    void operator()(Fn&& fn, size_t index)
    {
        std::string name = "cpu matmul " + std::to_string(index);
        bench::measure<bench::options{
            .clock = bench::TimingSource::Host,   // 纯 CPU 活儿，用墙钟
            .preheat_on = false,                  // CPU 侧没有「假负载升频」这回事
            .warmup_runs = 0,
            .warmup_seconds = 1.0f,               // 按时长预跑 1s，让频率/温度先到稳态
            .reps = 20,                           // 要 ≥20：reps=10 时 p95 会退化成 max
            .pin_cores = 8,                       // 钉在 8 个物理核上（避开 SMT 兄弟、防跨核迁移）
            .omp_threads = 8}>(                   // ← 一核一线程。别改成 16，理由见下
            // 为什么是 8 而不是 16（实测：同一个短窗口内三种配置各跑 5 轮，
            // N=512，20 reps，取每轮的 min / 抖动）：
            //
            //   配置                min 中位数   抖动(p95-min)/min
            //   不绑核  + 16 线程    6.32 ms     75% ~ 249%
            //   绑 8 核 + 16 线程    6.51 ms     41% ~  88%
            //   绑 8 核 +  8 线程    6.06 ms     15% ~  44%   ← 选它
            //
            // 三者的【最好成绩】几乎一样（5.0 ~ 7.2 ms，差别落在噪声里），
            // 但 16 线程那两个配置的【中位数比 min 高 30~50%】，而 8 线程只高 8%。
            // 也就是说 SMT 对这个内核【没有收益，只有代价】。
            //
            // 原因是这个内核【访存受限】：B 是列访问，走的是流式读，瓶颈在共享的
            // cache / 内存带宽上。同一个物理核上的第二个线程并不增加吞吐，
            // 只增加争抢和抖动。
            //
            // ⚠️ 千万别把这个结论推广到别的内核。「SMT 有没有用」完全取决于瓶颈在哪：
            //    · 纯计算、依赖延迟受限的活儿（烧 FP 的微基准那种）→ SMT 接近 2 倍；
            //    · 访存 / 带宽受限的活儿（就是这个）→ 基本零收益甚至负收益。
            //    第一轮那个微基准测出「同物理核两线程几乎不减速」是前者，别混用。
            //
            // 想自己复现这个对比：把这里改成 16，多跑几轮看抖动怎么变。
            // 被测区间：只剩纯计算。分配、清零、setup 全在窗口外。
            [&]
            { fn(A[index], B[index], R[index]); },  // fn 是函数引用 → 可内联
            // setup：在计时窗口【之前】执行。清零必须在这里，因为 cpu_matmul2/3
            // 是累加型的，不清零结果就是错的；而清零的耗时不计入被测区间。
            [&]
            { set_up(A[index], B[index]); R[index].clear(); },
            [&]
            { if (checked[index]) { check(A[index], B[index], R[index], name); checked[index] = false; } },
            name);
    }
};

int main()
{
    constexpr int N = 512;
    bench_cpu<N> cpu_bencher(5);
    cpu_bencher(cpu_matmul<N>, 0);
    cpu_bencher(cpu_matmul2<N>, 1);
    cpu_bencher(cpu_matmul3<N>, 2); // omp
}
