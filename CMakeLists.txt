# bench —— 一个给自己用的 CUDA / CPU 基准框架
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/01-matmul-cpu
#
# 直接手编（不想用 CMake 时）：
#   nvcc -O3 -std=c++20 -Xcompiler -fopenmp -Iinclude \
#        examples/01-matmul-cpu.cu src/core/*.cpp src/cpu/*.cpp src/gpu/*.cu -o 01-matmul-cpu
#
# 三条硬性要求，缺一个就出问题（都踩过）：
#   -std=c++20        options 是 C++20 的 class NTTP，缺了会报
#                     "a nontype template parameter may not have class type"
#   -Iinclude         公共头是用 bench/... 这个前缀引的
#   -fopenmp          缺了【不报错】，但 #pragma omp parallel for 会被静默忽略，
#                     OpenMP 版本退化成单线程（实测慢 3 倍）
cmake_minimum_required(VERSION 3.22)   # 本机实测版本；CMAKE_CUDA_ARCHITECTURES 要 3.18+

project(bench LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# 补一格 CMake 的编译标志表：CUDA + C++20
#
# CMake 3.22 的 Modules/Compiler/NVIDIA-CUDA.cmake 只定义到
# CMAKE_CUDA17_STANDARD_COMPILE_OPTION，没有 CUDA20（3.25+ 才加）。
# 而 nvcc 13.3 早就支持 c++20 了。不补的话，一旦 C++ 标准设成 20，
# CMake 会把它继承给 CUDA 侧然后直接报：
#     Target "bench" requires the language dialect "CUDA20" ... but CMake does
#     not know the compile flags to use to enable it
# 这是个 generate 阶段的错，只在配 CMake 时出现，很容易误判成代码问题。
#
# 必须在 project() 之后、建 target 之前设。升到 CMake 3.25+ 后可以删掉这两行。
# ---------------------------------------------------------------------------
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION  "-std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")

# options 要当非类型模板参数用，必须 C++20。
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认按 Turing 编（GTX 1650 = sm_75）。
# 换卡时用 -DCMAKE_CUDA_ARCHITECTURES=native（那个值要 CMake ≥ 3.24），
# 或者直接写具体架构，例如 -DCMAKE_CUDA_ARCHITECTURES=89。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# 框架本体
#
# .cu 归 nvcc 编（有 kernel），.cpp 归宿主编译器编（纯宿主代码）。
# 两者【合成一个库】，这个库依赖 CUDA —— 也就是说不存在「不需要 CUDA 的
# CPU-only 目标」。想全都交给 nvcc，把这些 .cpp 改名成 .cu 即可。
# ---------------------------------------------------------------------------
add_library(bench STATIC
    src/core/run.cpp          # 绑核 → 预热 → 预跑 → 计时 → 统计 → 打印
    src/core/stats.cpp        # 样本分布的统计
    src/core/host_clock.cpp   # chrono 计时（Host）
    src/cpu/affinity.cpp      # physical_cpus / pin_to_cpus
    src/cpu/flush_cpu.cpp     # flush_cpu_cache / flush_cpu_lines
    src/gpu/preheat.cu        # 假负载（内部，不对外暴露）
    src/gpu/device_clock.cu   # cudaEvent 计时（Device）
    src/gpu/flush_gpu.cu      # flush_gpu_cache
)

# include 用 PUBLIC：用户程序需要 include/bench/*.hpp
target_include_directories(bench PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(bench PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)

# OpenMP 必须【同时】作用到 CUDA 侧 —— 这是最容易漏的一条。
#
# 只写 OpenMP::OpenMP_CXX 是不够的：它的 INTERFACE_COMPILE_OPTIONS 带
# $<$<COMPILE_LANGUAGE:CXX>:-fopenmp> 这个语言限定，而使用者的 #pragma omp
# 一般写在 .cu 里（本框架的示例就是）。结果那个 pragma 被【静默忽略】，
# OpenMP 版本退化成单线程，而且【没有任何警告】。
#
# 实测代价（就是本仓库的示例）：
#     只有 CXX 侧有 -fopenmp   → cpu_matmul3  24.8 ms
#     CUDA 侧也补上            → cpu_matmul3   7.5 ms
# 慢 3.3 倍，且只会表现成"OpenMP 好像没用"。
#
# nvcc 要用 -Xcompiler=-fopenmp 把标志转给宿主编译器（单 token 写法，
# 因为 genex 里不能塞带空格的参数）。链接侧也补上，避免 omp_* 符号未定义。
target_compile_options(bench PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
target_link_options(bench PUBLIC
    $<$<LINK_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

# 带命名空间的别名，和别的目标区分开
add_library(bench::bench ALIAS bench)

# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------
add_executable(matmul_cpu examples/01-matmul-cpu.cu)
set_target_properties(matmul_cpu PROPERTIES OUTPUT_NAME "01-matmul-cpu")
target_link_libraries(matmul_cpu PRIVATE bench::bench)
