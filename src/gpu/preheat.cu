// src/gpu/preheat.cu —— GPU 侧的假负载（内部，不对外暴露）
//
// 唯一的使用方是 src/core/run.cpp。公共头（bench/bench.hpp 那条线）里
// 看不到这个函数的存在。
#include "bench/detail/gpu_internal.hpp"

#include <cuda_runtime.h>

namespace {

// 烤机 kernel：纯计算，一次显存都不碰。
//
// 为什么不能用「读数组写数组」那种循环？那烤的是显存带宽，SM 反而没压满。
// 这里所有变量都在寄存器里，专门把 SM 频率拉起来。
//
// 两条独立的 FMA 链：只跑一条的话会被指令延迟卡住，ALU 喂不满。
__global__ void burn_kernel(float* sink)
{
    float a = threadIdx.x,        b = 1.000001f, c = 1e-7f;
    float d = threadIdx.x + 1.0f, e = 1.000002f, f = 2e-7f;

    #pragma unroll 4
    for (int i = 0; i < 50000; i++) {
        a = fmaf(a, b, c);
        d = fmaf(d, e, f);
    }

    // 防止整个循环被优化掉。
    // 注意必须真的写内存——空的内联汇编挡不住 nvcc 的死代码消除。
    if (a == -1.0f) sink[0] = a + d;
}

// 落点。用 __device__ 全局变量而不是 cudaMalloc：
// 它在模块加载时由驱动分配、跟着 context 一起销毁，
// 既不需要 cudaMalloc，也没有东西需要 cudaFree。
__device__ float g_sink;

}  // namespace

namespace bench::detail {

void preheat(float seconds)
{
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);

    // 按【时间】循环，不按次数：升频要多久是不确定的，按秒走最稳。
    float elapsed = 0.0f;
    while (elapsed < seconds * 1000.0f) {
        // 每 SM 一个 1024 线程的 block —— 一波跑满，没有波间空档。
        // 写成 <<<sm * 4, 1024>>> 就是 4 波（blockDim=1024 时每 SM 只装 1 个 block）。
        burn_kernel<<<sm, 1024>>>(&g_sink);

        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);   // 不能省：等 GPU 真的跑到 t1
        cudaEventElapsedTime(&elapsed, t0, t1);
    }

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

}  // namespace bench::detail
