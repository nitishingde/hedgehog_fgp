cmake_minimum_required(VERSION 4.3)
project(hedgehog_fgp)

set(CMAKE_CXX_STANDARD 23)

# makes refactoring easier, can be removed to understand complex errors
add_compile_options(-ftemplate-backtrace-limit=1 -fmax-errors=5)

include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.6.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
    hedgehog
    GIT_REPOSITORY https://github.com/drfailer/hedgehog.git
    GIT_TAG        refactor
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(hedgehog)

FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG        v1.0.5
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(concurrentqueue)
include_directories(${CMAKE_BINARY_DIR}/_deps/concurrentqueue-src/)

FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        5.1.1
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)
include(${kokkos_BINARY_DIR}/KokkosConfigCommon.cmake) #FIXME: https://github.com/kokkos/kokkos/issues/7359#event-14339616469

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(HWLOC REQUIRED hwloc)
endif()
if(NOT hwloc_FOUND AND NOT HWLOC_FOUND)
    find_package(hwloc QUIET)
endif()

if(hwloc_FOUND OR HWLOC_FOUND)
    if(NOT TARGET hwloc::hwloc)
        add_library(hwloc::hwloc UNKNOWN IMPORTED)
        if(HWLOC_INCLUDE_DIRS)
            set_target_properties(hwloc::hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIRS}")
        endif()
        if(HWLOC_LINK_LIBRARIES)
            set_target_properties(hwloc::hwloc PROPERTIES IMPORTED_LOCATION "${HWLOC_LINK_LIBRARIES}")
        elseif(HWLOC_LIBRARIES)
            set_target_properties(hwloc::hwloc PROPERTIES IMPORTED_LOCATION "${HWLOC_LIBRARIES}")
        endif()
    endif()
else()
    include(ExternalProject)
    set(HWLOC_PREFIX      "${CMAKE_CURRENT_BINARY_DIR}/external/hwloc")
    set(HWLOC_INCLUDE_DIR "${HWLOC_PREFIX}/include")
    set(HWLOC_LIBRARY     "${HWLOC_PREFIX}/lib64/libhwloc.a")
    file(MAKE_DIRECTORY ${HWLOC_INCLUDE_DIR})
    ExternalProject_Add(
        hwloc_external
        GIT_REPOSITORY    https://github.com/open-mpi/hwloc.git
        GIT_TAG           hwloc-2.14.0
        CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh && <SOURCE_DIR>/configure --prefix=${HWLOC_PREFIX} --disable-shared --enable-static
        BUILD_COMMAND     make -j4
        INSTALL_COMMAND   make install
        BUILD_BYPRODUCTS  "${HWLOC_LIBRARY}"
    )
    add_library(hwloc::hwloc STATIC IMPORTED)
    set_target_properties(hwloc::hwloc PROPERTIES
        IMPORTED_LOCATION "${HWLOC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIR}"
    )
    add_dependencies(hwloc::hwloc hwloc_external)
endif(hwloc_FOUND OR HWLOC_FOUND)

add_executable(baseline baseline.cpp)
target_link_libraries(baseline PRIVATE CLI11::CLI11 hedgehog::hedgehog)

add_executable(kokkos_baseline kokkos_baseline.cpp)
target_link_libraries(kokkos_baseline PRIVATE CLI11::CLI11 Kokkos::kokkos hwloc::hwloc)

