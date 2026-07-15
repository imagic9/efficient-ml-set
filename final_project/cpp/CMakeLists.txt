cmake_minimum_required(VERSION 3.22)
project(wildlife_trigger LANGUAGES CXX)

# DESIGN §11: C++17, Release with target-scoped -O3, never a global
# CMAKE_CXX_FLAGS_RELEASE, and never -march=native on gx10 because that would
# tune for the build host (Cortex-X925) instead of the Pi (Cortex-A76).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Empty means "generic ARMv8-A": runs anywhere, tuned for nothing. Set explicitly
# to cortex-a76 (Pi 5) or cortex-a72 (Pi 4) for a target build. `native` is only
# legitimate when compiling on that same Pi, and is rejected below otherwise.
set(WILDLIFE_CPU_TARGET "" CACHE STRING
    "Explicit -mcpu value, e.g. cortex-a76. Empty = generic.")

if(WILDLIFE_CPU_TARGET STREQUAL "native" AND NOT WILDLIFE_ALLOW_NATIVE)
  message(FATAL_ERROR
    "-mcpu=native selects the build host, not the target. On gx10 that is a "
    "Cortex-X925 with i8mm and SVE2, which the Pi does not have: the binary "
    "would fail or silently take a different kernel path. Pass an explicit "
    "target (cortex-a76 / cortex-a72), or set WILDLIFE_ALLOW_NATIVE=ON only when "
    "compiling on the Pi itself.")
endif()

add_library(wildlife_trigger_core
  src/cpu_features.cpp
)
target_include_directories(wildlife_trigger_core PUBLIC include)
target_compile_options(wildlife_trigger_core PRIVATE
  $<$<CONFIG:Release>:-O3>
  -Wall -Wextra -Wpedantic
)
if(WILDLIFE_CPU_TARGET)
  target_compile_options(wildlife_trigger_core PUBLIC -mcpu=${WILDLIFE_CPU_TARGET})
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

message(STATUS "wildlife_trigger: build=${CMAKE_BUILD_TYPE} "
               "cpu_target=${WILDLIFE_CPU_TARGET}")
