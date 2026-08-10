cmake_minimum_required(VERSION 3.16)
project(hedgehog_fgp)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

#find_package(TBB REQUIRED)
set(BUILD_SHARED_LIBS ON)
set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_STRICT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    onetbb
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG        v2023.1.0
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(onetbb)
if (TARGET tbb AND NOT TARGET TBB::tbb)
    add_library(TBB::tbb ALIAS tbb)
endif()
unset(BUILD_SHARED_LIBS)

FetchContent_Declare(
    hedgehog
    GIT_REPOSITORY https://github.com/usnistgov/hedgehog.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(hedgehog)

add_executable(hh-tbb main.cpp)
target_link_libraries(hh-tbb PRIVATE TBB::tbb hedgehog::hedgehog)
