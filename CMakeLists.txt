cmake_minimum_required(VERSION 3.10)

project(cuda-exp LANGUAGES C CXX CUDA)

find_package(spdlog REQUIRED) # apt-get install libspdlog-dev
find_package(fmt REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CUDA_ARCHS "") # nvidia-smi --query-gpu=compute_cap
list(APPEND CUDA_ARCHS 75-real 75-virtual) # Turing=75
list(APPEND CUDA_ARCHS 80-real 80-virtual 86-real 86-virtual) # Ampere=80/86
list(APPEND CUDA_ARCHS 89-real 89-virtual) # Ada Lovelace=89
list(APPEND CUDA_ARCHS 90-real 90-virtual) # Hopper=90
list(APPEND CUDA_ARCHS 100-real 100-virtual 120-real 120-virtual) # Blackwell=100/120
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHS}")

add_subdirectory(common)
add_subdirectory(0-elementwise)
add_subdirectory(1-transpose)
add_subdirectory(2-dot)
add_subdirectory(3-gemm)
add_subdirectory(4-softmax)
