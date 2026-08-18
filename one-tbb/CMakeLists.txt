cmake_minimum_required(VERSION 3.16)
project(hedgehog_fgp)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.6.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        5.1.1
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)
include(${kokkos_BINARY_DIR}/KokkosConfigCommon.cmake) #FIXME: https://github.com/kokkos/kokkos/issues/7359#event-14339616469

FetchContent_Declare(
    hedgehog
    GIT_REPOSITORY https://github.com/usnistgov/hedgehog.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(hedgehog)

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

add_executable(tbbb tbb_main.cpp)
target_link_libraries(tbbb PRIVATE CLI11::CLI11 Kokkos::kokkos TBB::tbb hedgehog::hedgehog)

add_executable(hh_tbb hh_main.cpp)
target_link_libraries(hh_tbb PRIVATE CLI11::CLI11 Kokkos::kokkos TBB::tbb hedgehog::hedgehog)

add_executable(kokkos_tbb kokkos_main.cpp)
target_include_directories(kokkos_tbb PUBLIC ../kokkos-tbb/)
target_link_libraries(kokkos_tbb PRIVATE CLI11::CLI11 Kokkos::kokkos TBB::tbb hedgehog::hedgehog)
