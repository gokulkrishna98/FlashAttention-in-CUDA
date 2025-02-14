cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flashAttention-in-cuda)
set(CMAKE_PREFIX_PATH "./libtorch")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(fa main.cpp)
target_link_libraries(fa "${TORCH_LIBRARIES}" cuda cudart)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(TARGET fa PROPERTY CXX_STANDARD 20)
