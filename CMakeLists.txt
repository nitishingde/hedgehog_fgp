cmake_minimum_required(VERSION 4.3)
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
    hedgehog
    GIT_REPOSITORY https://github.com/usnistgov/hedgehog.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(hedgehog)

add_executable(baseline baseline.cpp)
target_link_libraries(baseline PRIVATE CLI11::CLI11 hedgehog::hedgehog)
