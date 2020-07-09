cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 14)
project(CrazyAra CXX)

option(USE_PROFILING             "Build with profiling"   OFF)
option(USE_RL                    "Build with reinforcement learning support"  OFF)
option(USE_TENSORRT              "Build with TensorRT support"  ON)
option(USE_MXNET                 "Build with MXNet backend (Blas/IntelMKL/CUDA/TensorRT) support"  OFF)
option(USE_960                   "Build with 960 variant support"  OFF)
option(BUILD_TESTS               "Build and run tests"  OFF)

add_definitions(-DIS_64BIT)
add_definitions(-DCRAZYHOUSE)
add_definitions(-DKOTH)
add_definitions(-DTHREECHECK)
add_definitions(-DANTI)
add_definitions(-DATOMIC)
add_definitions(-DHORDE)
add_definitions(-DRACE)

# enable a single mode for different model input / outputs
add_definitions(-DMODE_CRAZYHOUSE)
#add_definitions(-DMODE_LICHESS)
#add_definitions(-DMODE_CHESS)

if (BUILD_TESTS)
    add_definitions(-DBUILD_TESTS)
endif()

# -pg performance profiling flags
if (USE_PROFILING)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

if(DEFINED ENV{BLAZE_PATH})
    MESSAGE(STATUS "BLAZE_PATH set to: $ENV{BLAZE_PATH}")
    include_directories("$ENV{BLAZE_PATH}/include")
    include_directories("$ENV{BLAZE_PATH}")
else()
    MESSAGE(STATUS "BLAZE_PATH not set")
endif()

find_package( blaze )
if( blaze_FOUND )
     add_library( blaze_target INTERFACE )
     target_link_libraries( blaze_target INTERFACE blaze::blaze )
endif()

file(GLOB source_files
    "3rdparty/Stockfish/src/syzygy/*.h"
    "3rdparty/Stockfish/src/syzygy/*.cpp"
    "*.h"
    "*.cpp"
    "*.md"
    "3rdparty/Stockfish/src/*.h"
    "3rdparty/Stockfish/src/*.cpp"
    "3rdparty/catch-2.91/*.hpp"
    "src/*.h"
    "src/*.cpp"
    "src/tests/*.h"
    "src/tests/*.cpp"
    "src/util/*.h"
    "src/util/*.cpp"
    "src/domain/crazyhouse/*.h"
    "src/domain/crazyhouse/*.cpp"
    "src/agents/*.cpp"
    "src/agents/*.h"
    "src/agents/config/*.cpp"
    "src/agents/config/*.h"
    "src/agents/util/*.cpp"
    "src/agents/util/*.h"
    "src/nn/*.cpp"
    "src/nn/*.h"
    "src/nn/util/*.cpp"
    "src/nn/util/*.h"
    "src/manager/*.cpp"
    "src/manager/*.h"
    "src/rl/*.cpp"
    "src/rl/*.h"
)

set(CPP_PACKAGE_INCLUDE_DIR $ENV{MXNET_PATH}/cpp-package/include)
file(GLOB_RECURSE CPP_PACKAGE_HEADERS
    "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
    "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp"
    )

find_package (Threads)
include_directories("3rdparty/Stockfish/src")
include_directories("3rdparty/catch-2.9.1")

# incude dirent library seperately because it's missing in the stdlib
if(MSVC OR MSYS OR MINGW)
    include_directories("3rdparty/dirent")
endif()

include_directories("src/domain/crazyhouse")
include_directories("src/agents")
include_directories("src/agents/config")
include_directories("src/nn")


if (USE_MXNET)
    IF(DEFINED ENV{MXNET_PATH})
    MESSAGE(STATUS "MXNET_PATH set to: $ENV{MXNET_PATH}")
    ELSE()
    MESSAGE(STATUS "MXNET_PATH not set")
    ENDIF()

    include_directories("$ENV{MXNET_PATH}/cpp-package/include")
    include_directories("$ENV{MXNET_PATH}/include/")
    include_directories("$ENV{MXNET_PATH}/3rdparty/tvm/nnvm/include")
    include_directories("$ENV{MXNET_PATH}/3rdparty/dmlc-core/include")

    link_directories("$ENV{MXNET_PATH}/lib")
    link_directories("$ENV{MXNET_PATH}/Release/lib")
    link_directories("$ENV{MXNET_PATH}/build/lib")
    link_directories("$ENV{MXNET_PATH}/build/Release")
    link_directories("$ENV{MXNET_PATH}/build")

    set(CPP_EXAMPLE_LIBS
        ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
        ${BEGIN_WHOLE_ARCHIVE} dmlc ${END_WHOLE_ARCHIVE}
        ${mxnet_LINKER_LIBS}
        )

    add_definitions(-DMXNET)
endif()

if (USE_RL)
    message(STATUS "Enabled Reinforcement Learning functionality")
    if(DEFINED ENV{Z5_PATH})
        MESSAGE(STATUS "Z5_PATH set to: $ENV{Z5_PATH}")
    else()
        MESSAGE(STATUS "Z5_PATH not set")
    endif()
    include_directories($ENV{Z5_PATH}/include)
    include_directories("3rdparty/json-3.7.0")
    include_directories($ENV{XTL_PATH}/include)
    include_directories($ENV{XTENSOR_PATH}/include)
    add_definitions(-DUSE_RL)
endif()

if (USE_TENSORRT)
    # build CrazyAra with TensorRT support, requires a working TensorRT-MXNet library package
    message(STATUS "Enabled TensorRT support")
    message(STATUS "TensorRT path: $ENV{TENSORRT_PATH}")
    # add additional source file
    set(source_files
        ${source_files}
        $ENV{TENSORRT_PATH}/samples/common/logger.cpp
        )
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA path: $ENV{CUDA_PATH}")
    include_directories("$ENV{CUDA_PATH}/include")
    link_directories("$ENV{CUDA_PATH}/lib64")
    link_directories("$ENV{CUDA_PATH}/lib/x64")
    link_directories("$ENV{TENSORRT_PATH}/lib")
if(WIN32)
  find_library(TENSORRT_LIBRARY_MYELIN myelin64_1
    HINTS  ${TENSORRT_PATH}
    PATH_SUFFIXES lib lib64 lib/x64)
endif()
    include_directories("$ENV{TENSORRT_PATH}/include")
    include_directories("$ENV{TENSORRT_PATH}/samples/common/")
    add_definitions(-DTENSORRT)
endif()

if (USE_960)
    add_definitions(-DSUPPORT960)
endif()

add_executable(${PROJECT_NAME} ${source_files})

if (USE_TENSORRT)
    target_link_libraries(${PROJECT_NAME} nvonnxparser nvinfer cudart myelin ${CUDART_LIB} ${CUBLAS_LIB} ${CUDNN_LIB})
endif()

if (USE_RL)
    # include filesystem (needed for z5)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Threads REQUIRED)

if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(${PROJECT_NAME} PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()

# add target directory as library run path for unix systems
if(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,./")
endif()

if (USE_MXNET)
    if(UNIX)
        target_link_libraries(${PROJECT_NAME} mxnet)
    else()
        target_link_libraries(${PROJECT_NAME} libmxnet)
    endif()
endif()
