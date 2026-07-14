cmake_minimum_required(VERSION 3.10)
project(ONNXInference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime /usr/include/onnxruntime)
find_library(ONNXRUNTIME_LIBRARIES onnxruntime HINTS /usr/local/lib /usr/lib)

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

add_executable(inference main.cpp)
target_link_libraries(inference ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
