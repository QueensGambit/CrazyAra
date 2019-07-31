cmake_minimum_required(VERSION 2.8)

project(CrazyAraMCTS CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -pg performance profiling flags
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# find_package(xtl REQUIRED)
# find_package(xtensor REQUIRED)
# if xtensor was built with xsimd support:
# find_package(xsimd REQUIRED)

find_package( blaze )
if( blaze_FOUND )
   add_library( blaze_target INTERFACE )
   target_link_libraries( blaze_target INTERFACE blaze::blaze )
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
include_directories("libs/Eigen")
include_directories("src/domain/crazyhouse")
include_directories("src/agents")
include_directories("src/agents/config")
include_directories("src/nn")

#set(MXNET_PATH
#    "/media/queensgambit/Volume/Deep_Learning/libraries/mxnet/"
#    # "/home/queensgambit/libs/mxnet/
#    )

IF(DEFINED ENV{MXNET_PATH})
    MESSAGE(STATUS "MXNET_PATH set to: $ENV{MXNET_PATH}")
ELSE()
    MESSAGE(STATUS "MXNET_PATH not set")
ENDIF()

include_directories("$ENV{MXNET_PATH}cpp-package/include")
include_directories("$ENV{MXNET_PATH}include/")
include_directories("$ENV{MXNET_PATH}3rdparty/tvm/nnvm/include")
include_directories("$ENV{MXNET_PATH}3rdparty/dmlc-core/include")

#include_directories("/home/queensgambit/libs/intelMKL/mklml_lnx_2019.0.5.20190502/include")
#link_directories(/home/queensgambit/libs/intelMKL/mklml_lnx_2019.0.5.20190502/lib)

find_package(Threads REQUIRED)
add_definitions(-DIS_64BIT)
add_definitions(-DCRAZYHOUSE)

# link_directories(${MXNET_PATH}lib)
link_directories($ENV{MXNET_PATH}lib)

add_executable(${PROJECT_NAME} ${source_files})
# add_executable(${PROJECT_NAME} "main.cpp")
# target_link_libraries (${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
# target_link_libraries( ${PROJECT_NAME} pthread)
target_link_libraries(${PROJECT_NAME} mxnet)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(${PROJECT_NAME} PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)
#target_link_libraries(${PROJECT_NAME} Threads::Threads)
