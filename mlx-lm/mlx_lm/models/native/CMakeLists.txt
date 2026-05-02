# CMakeLists.txt — IsoQuant native KV decode kernels
#
# Build:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Produces:
#   libisoquant_kv.a          — static library (runtime + reference)
#   isoquant_kv_kernels.metallib — compiled Metal shader library
#   isoquant_test             — correctness test (Metal vs CPU reference)
#
# Requirements:
#   - macOS 13+ / Xcode 15+
#   - Metal-cpp headers (auto-downloaded if not found)
#   - Apple Silicon or compatible Metal GPU

cmake_minimum_required(VERSION 3.24)
project(isoquant_kv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Metal-cpp headers ----
# Metal-cpp is a C++ wrapper for Metal.  If not installed, download from Apple.
set(METAL_CPP_INCLUDE "" CACHE PATH "Path to metal-cpp headers")
if(NOT METAL_CPP_INCLUDE)
    file(GLOB MLX_METAL_CPP_HINTS
        "${CMAKE_SOURCE_DIR}/../../../../turboquant-mlx/.venv/lib/python*/site-packages/mlx/include/metal_cpp"
    )
    list(LENGTH MLX_METAL_CPP_HINTS MLX_METAL_CPP_HINT_COUNT)
    if(MLX_METAL_CPP_HINT_COUNT GREATER 0)
        list(GET MLX_METAL_CPP_HINTS 0 METAL_CPP_INCLUDE)
        set(METAL_CPP_INCLUDE "${METAL_CPP_INCLUDE}" CACHE PATH "Path to metal-cpp headers" FORCE)
    else()
        find_path(METAL_CPP_INCLUDE Metal/Metal.hpp
            HINTS
                ${CMAKE_SOURCE_DIR}/metal-cpp
                /usr/local/include
                $ENV{HOME}/metal-cpp
        )
    endif()
endif()

if(NOT METAL_CPP_INCLUDE)
    message(STATUS "Metal-cpp not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        metal-cpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip
    )
    FetchContent_MakeAvailable(metal-cpp)
    set(METAL_CPP_INCLUDE ${metal-cpp_SOURCE_DIR})
endif()

message(STATUS "Metal-cpp: ${METAL_CPP_INCLUDE}")

# ---- Compile Metal shaders ----
file(GLOB METAL_TOOLCHAIN_BINS
    "/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin"
)
list(LENGTH METAL_TOOLCHAIN_BINS METAL_TOOLCHAIN_BIN_COUNT)
if(METAL_TOOLCHAIN_BIN_COUNT GREATER 0)
    list(GET METAL_TOOLCHAIN_BINS 0 METAL_TOOLCHAIN_BIN_HINT)
endif()

find_program(METAL_BIN metal HINTS ${METAL_TOOLCHAIN_BIN_HINT})
find_program(METALLIB_BIN metallib HINTS ${METAL_TOOLCHAIN_BIN_HINT})

if(NOT METAL_BIN OR NOT METALLIB_BIN)
    message(FATAL_ERROR "Could not locate metal/metallib binaries. Install the Metal Toolchain with `xcodebuild -downloadComponent MetalToolchain`.")
endif()

message(STATUS "metal: ${METAL_BIN}")
message(STATUS "metallib: ${METALLIB_BIN}")

set(METAL_SOURCE ${CMAKE_SOURCE_DIR}/isoquant_kv_kernels.metal)
set(METAL_AIR    ${CMAKE_BINARY_DIR}/isoquant_kv_kernels.air)
set(METALLIB     ${CMAKE_BINARY_DIR}/isoquant_kv_kernels.metallib)

add_custom_command(
    OUTPUT ${METAL_AIR}
    COMMAND ${METAL_BIN}
        -std=metal3.0
        -O2
        -fmodules-cache-path=${CMAKE_BINARY_DIR}/metal-modcache
        -c ${METAL_SOURCE}
        -o ${METAL_AIR}
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal shaders"
)

add_custom_command(
    OUTPUT ${METALLIB}
    COMMAND ${METALLIB_BIN}
        ${METAL_AIR}
        -o ${METALLIB}
    DEPENDS ${METAL_AIR}
    COMMENT "Linking Metal library"
)

add_custom_target(metal_shaders ALL DEPENDS ${METALLIB})

# ---- Static library ----
add_library(isoquant_kv STATIC
    isoquant_kv_runtime.cpp
    isoquant_kv_reference.cpp
)

target_include_directories(isoquant_kv PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${METAL_CPP_INCLUDE}
)

target_link_libraries(isoquant_kv PUBLIC
    "-framework Metal"
    "-framework Foundation"
)

add_dependencies(isoquant_kv metal_shaders)

# ---- Correctness test ----
add_executable(isoquant_test
    isoquant_kv_test.cpp
)

target_link_libraries(isoquant_test PRIVATE isoquant_kv)

add_executable(isoquant_metal_probe
    metal_device_probe.cpp
)

target_include_directories(isoquant_metal_probe PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${METAL_CPP_INCLUDE}
)

target_link_libraries(isoquant_metal_probe PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

add_executable(isoquant_benchmark
    isoquant_kv_benchmark.cpp
)

target_link_libraries(isoquant_benchmark PRIVATE isoquant_kv)

# Copy metallib to test location
add_custom_command(TARGET isoquant_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${METALLIB}
        $<TARGET_FILE_DIR:isoquant_test>/isoquant_kv_kernels.metallib
)

add_custom_command(TARGET isoquant_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${METALLIB}
        $<TARGET_FILE_DIR:isoquant_benchmark>/isoquant_kv_kernels.metallib
)
