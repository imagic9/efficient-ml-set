# Plain CTest with assert-based tests. A framework (doctest/Catch2) would be
# vendored under third_party/ per the DESIGN §11 no-network rule if E1's real
# test suite needs richer assertions; introducing one now, when plain asserts
# still express every case clearly, would be speculative.
add_executable(test_cpu_features test_cpu_features.cpp)
target_link_libraries(test_cpu_features PRIVATE wildlife_trigger_core)
add_test(NAME cpu_features COMMAND test_cpu_features)

# Logging is header-only, so its test needs no library beyond core's include path.
add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging PRIVATE wildlife_trigger_core)
add_test(NAME logging COMMAND test_logging)

# These need the app library, which only exists when the ORT tools are built.
if(WILDLIFE_BUILD_ORT_TOOLS)
  add_executable(test_policy test_policy.cpp)
  target_link_libraries(test_policy PRIVATE wildlife_trigger_app)
  add_test(NAME policy COMMAND test_policy)

  add_executable(test_preprocess test_preprocess.cpp)
  target_link_libraries(test_preprocess PRIVATE wildlife_trigger_app ${OpenCV_LIBS})
  add_test(NAME preprocess COMMAND test_preprocess)

  # summarise() (percentile math) lives in the app library; the test needs no ORT
  # symbols but links the library that carries it.
  add_executable(test_benchmark test_benchmark.cpp)
  target_link_libraries(test_benchmark PRIVATE wildlife_trigger_app)
  add_test(NAME benchmark COMMAND test_benchmark)
endif()
