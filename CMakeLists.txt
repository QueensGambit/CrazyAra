cmake_minimum_required(VERSION 2.8)

project(CrazyAra CXX)

option(USE_PROFILING             "Build with profiling"   OFF)
option(USE_RL                    "Build with reinforcment learning support"  ON)

# -pg performance profiling flags
if (USE_PROFILING)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

if(DEFINED ENV{BLAZE_PATH})
    MESSAGE(STATUS "BLAZE_PATH set to: $ENV{BLAZE_PATH}")
else()
    MESSAGE(STATUS "BLAZE_PATH not set")
endif()

if(MSVC OR MSYS OR MINGW)
    include_directories("$ENV{BLAZE_PATH}include")
else()
    find_package( blaze )
    if( blaze_FOUND )
        add_library( blaze_target INTERFACE )
        target_link_libraries( blaze_target INTERFACE blaze::blaze )
    else()
        include_directories("$ENV{BLAZE_PATH}include")
    endif()
endif()

file(GLOB source_files
    "lib/Stockfish/src/syzygy/*.h"
    "lib/Stockfish/src/syzygy/*.cpp"
    "*.h"
    "*.cpp"
    "*.md"
    "lib/Stockfish/src/*.h"
    "lib/Stockfish/src/*.cpp"
    "lib/catch-2.91/*.hpp"
    "libs/Eigen/*h"
    "libs/Eigen/*.cpp"
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
    "src/nn/*.cpp"
    "src/nn/*.h"
    "src/manager/*.cpp"
    "src/manager/*.h"
    "src/rl/*.cpp"
    "src/rl/*.h"
    )

set(CPP_EXAMPLE_LIBS
    ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
    ${BEGIN_WHOLE_ARCHIVE} dmlc ${END_WHOLE_ARCHIVE}
    ${mxnet_LINKER_LIBS}
    )

set(CPP_PACKAGE_INCLUDE_DIR $ENV{MXNET_PATH}cpp-package/include)
file(GLOB_RECURSE CPP_PACKAGE_HEADERS
    "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
    "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp"
    )

find_package (Threads)
include_directories("lib/Stockfish/src")
include_directories("lib/catch-2.9.1")

# incude dirent library seperately because it's missing in the stdlib
if(MSVC OR MSYS OR MINGW)
    include_directories("lib/dirent")
endif()

include_directories("src/domain/crazyhouse")
include_directories("src/agents")
include_directories("src/agents/config")
include_directories("src/nn")

IF(DEFINED ENV{MXNET_PATH})
MESSAGE(STATUS "MXNET_PATH set to: $ENV{MXNET_PATH}")
ELSE()
MESSAGE(STATUS "MXNET_PATH not set")
ENDIF()

include_directories("$ENV{MXNET_PATH}cpp-package/include")
include_directories("$ENV{MXNET_PATH}include/")
include_directories("$ENV{MXNET_PATH}3rdparty/tvm/nnvm/include")
include_directories("$ENV{MXNET_PATH}3rdparty/dmlc-core/include")

link_directories("$ENV{MXNET_PATH}lib")
link_directories("$ENV{MXNET_PATH}Release/lib")
link_directories("$ENV{MXNET_PATH}build/lib")
link_directories("$ENV{MXNET_PATH}build/Release")
link_directories("$ENV{MXNET_PATH}build")

link_directories("/home/queensgambit/Downloads/driver/cuda_10.1/TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.1.cudnn7.5/TensorRT-5.1.5.0/lib")

add_executable(${PROJECT_NAME} ${source_files})

if(UNIX)
    target_link_libraries(${PROJECT_NAME} mxnet)
else()
    target_link_libraries(${PROJECT_NAME} libmxnet)
endif()

if (USE_RL)
    if(DEFINED ENV{Z5_PATH})
        MESSAGE(STATUS "Z5_PATH set to: $ENV{Z5_PATH}")
    else()
        MESSAGE(STATUS "Z5_PATH not set")
    endif()
    include_directories($ENV{Z5_PATH}include)
    # include filesystem (needed for z5)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
    include_directories("lib/json-3.7.0")
    include_directories($ENV{XTL_PATH}include)
    include_directories($ENV{XTENSOR_PATH}include)
    add_definitions(-DUSE_RL)
endif()

find_package(Threads REQUIRED)
add_definitions(-DIS_64BIT)
add_definitions(-DCRAZYHOUSE)

# build CrazyAra with TensorRT support, requires a working TensorRT-MXNet library package
# add_definitions(-DTENSORRT)

if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(${PROJECT_NAME} PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()
