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
  src/hashing.cpp
)
target_include_directories(wildlife_trigger_core PUBLIC include)
target_compile_options(wildlife_trigger_core PRIVATE
  $<$<CONFIG:Release>:-O3>
  -Wall -Wextra -Wpedantic
)
if(WILDLIFE_CPU_TARGET)
  target_compile_options(wildlife_trigger_core PUBLIC -mcpu=${WILDLIFE_CPU_TARGET})
endif()

# Vendored nlohmann/json (DESIGN §11: no system JSON package, no network fetch).
# SYSTEM so its ~950 KB of templates do not drown our own -Wall -Wextra output; we
# do not maintain it and cannot act on its warnings.
add_library(wildlife_trigger_json INTERFACE)
target_include_directories(wildlife_trigger_json SYSTEM INTERFACE third_party)

# --- ONNX Runtime ------------------------------------------------------------
# The prebuilt aarch64 tarball unpacked in the target container, located by
# ORT_PREFIX (configs/env/pins.env sets it; the Dockerfile exports it). Found by
# explicit path rather than find_package: ORT ships no CMake config in this
# tarball, and a find_library search could silently bind to some other
# libonnxruntime on the host -- exactly the kind of drift the pinned container
# exists to prevent.
set(ORT_PREFIX "$ENV{ORT_PREFIX}" CACHE PATH "ONNX Runtime install prefix")

option(WILDLIFE_BUILD_ORT_TOOLS "Build the ONNX Runtime probe (needs ORT_PREFIX)" ON)

if(WILDLIFE_BUILD_ORT_TOOLS)
  if(NOT ORT_PREFIX OR NOT EXISTS "${ORT_PREFIX}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
      "WILDLIFE_BUILD_ORT_TOOLS is ON but ORT_PREFIX (\"${ORT_PREFIX}\") does not "
      "contain include/onnxruntime_cxx_api.h. Build inside "
      "wildlife-trigger-target:bookworm, where scripts/build_target_container.sh "
      "installs the pinned ORT, or pass -DWILDLIFE_BUILD_ORT_TOOLS=OFF.")
  endif()

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_PREFIX}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_PREFIX}/include")

  add_executable(ort_probe tools/ort_probe.cpp)
  target_link_libraries(ort_probe PRIVATE wildlife_trigger_core onnxruntime)
  target_compile_options(ort_probe PRIVATE
    $<$<CONFIG:Release>:-O3>
    -Wall -Wextra -Wpedantic
  )
  message(STATUS "wildlife_trigger: ort_probe against ${ORT_PREFIX}")

  # --- the application ---------------------------------------------------------
  # OpenCV: only the modules actually used. Linking the world would drag video I/O
  # and GUI backends into a bundle that must run headless on a Pi.
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  message(STATUS "wildlife_trigger: OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")

  add_library(wildlife_trigger_app
    src/preprocess.cpp
    src/session.cpp
    src/policy.cpp
    src/benchmark.cpp
  )
  target_link_libraries(wildlife_trigger_app
    PUBLIC wildlife_trigger_core wildlife_trigger_json onnxruntime
    PRIVATE ${OpenCV_LIBS}
  )
  target_include_directories(wildlife_trigger_app SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_compile_options(wildlife_trigger_app PRIVATE
    $<$<CONFIG:Release>:-O3>
    -Wall -Wextra -Wpedantic
  )

  add_executable(wildlife_trigger src/main.cpp)
  target_link_libraries(wildlife_trigger PRIVATE wildlife_trigger_app ${OpenCV_LIBS})
  target_compile_options(wildlife_trigger PRIVATE
    $<$<CONFIG:Release>:-O3>
    -Wall -Wextra -Wpedantic
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

message(STATUS "wildlife_trigger: build=${CMAKE_BUILD_TYPE} "
               "cpu_target=${WILDLIFE_CPU_TARGET}")
