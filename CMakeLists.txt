cmake_minimum_required(VERSION 2.8)

project(CrazyAra CXX)

option(USE_PROFILING             "Build with profiling"   FALSE)

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

add_executable(${PROJECT_NAME} ${source_files})

if(UNIX)
    target_link_libraries(${PROJECT_NAME} mxnet)
else()
    target_link_libraries(${PROJECT_NAME} libmxnet)
endif()

find_package(Threads REQUIRED)
add_definitions(-DIS_64BIT)
add_definitions(-DCRAZYHOUSE)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(${PROJECT_NAME} PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()
