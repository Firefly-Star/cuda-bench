// src/gpu/flush_gpu.cu —— 冲 GPU cache
#include "bench/flush.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace {

// 冲 cache 的 kernel：把一块比 L2 大的缓冲整个读一遍。
// 读操作会把 L2 里的旧行挤出去；grid 铺满所有 SM，各 SM 的 L1/TEX 也一并被冲掉。
//
// 按 uint4 读：同样的字节数，指令数少 4 倍。
// 累加用整数异或而不是浮点：读到的是未初始化的显存，可能全是 NaN / 非规格化数，
// 整数运算不受影响，结果也更确定。
__global__ void flush_kernel(const uint4* __restrict__ buf, long n4, unsigned* sink)
{
    unsigned acc = 0;
    const long stride = (long)gridDim.x * blockDim.x;

    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const uint4 v = buf[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }

    if (acc == 0xDEADBEEFu) sink[0] = acc;   // 防止整个循环被优化掉（同 burn_kernel）
}

__device__ unsigned g_flush_sink;

// 冲 cache 用的缓冲。第一次调用时按 L2 大小算出来分配，之后一直复用。
float* g_flush_buf = nullptr;
long   g_flush_n4  = 0;

}  // namespace

namespace bench {

void flush_gpu_cache()
{
    const int threads = 256;

    if (g_flush_buf == nullptr) {
        int l2 = 0;
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);

        // 2 倍 L2：每个 cache set 通常会被扫到两轮以上，够用——但这【不是定理】：
        // 它要求 (a) 扫荡的 miss 会 allocate，(b) 替换策略不是「保留最新」那类。
        // 现代 L2/L3 有流式检测，大块顺序访问可能被 no-allocate / evict-first
        // 处理，于是「冲干净」只是高概率。想确定性失效只能靠驱动的精确 flush
        // （ncu --cache-control all 就是它）；CPU 侧有 CLFLUSH，见 flush_cpu_lines。
        //
        // 另外扫荡本身是扰动源：CPU 侧实测它会让紧接着的 kernel 慢 0.1~0.3 ms，
        // 远大于「重取一遍工作集」的代价——详见 include/bench/flush.hpp。
        long bytes = (long)l2 * 2;
        if (bytes < 4L * 1024 * 1024) bytes = 4L * 1024 * 1024;   // L2 很小的卡给个下限

        g_flush_n4 = bytes / (long)sizeof(uint4);
        cudaMalloc(&g_flush_buf, (std::size_t)g_flush_n4 * sizeof(uint4));
    }

    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);

    // 铺满 SM，但不超过缓冲本身需要的 block 数
    const long want = (g_flush_n4 + threads - 1) / threads;
    const long cap  = (long)sm * 32;
    const int  blocks = (int)(want < cap ? want : cap);

    flush_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint4*>(g_flush_buf), g_flush_n4, &g_flush_sink);

    // 故意不同步：同一个流上顺序执行，调用者后面发射的 kernel 必然在它之后，
    // 不需要 CPU 在这里等——等待会白白打断流水。
}

}  // namespace bench
