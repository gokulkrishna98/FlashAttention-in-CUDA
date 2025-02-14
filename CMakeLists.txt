cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flashAttention-in-cuda LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_PREFIX_PATH "./libtorch")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(cuda_kernels STATIC transpose.cu)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_kernels PROPERTY LANGUAGE CUDA)

add_executable(fa main.cpp)
set_property(TARGET fa PROPERTY CXX_STANDARD 20)

target_link_libraries(fa "${TORCH_LIBRARIES}" cuda_kernels cuda cudart)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(TARGET fa PROPERTY CXX_STANDARD 20)
