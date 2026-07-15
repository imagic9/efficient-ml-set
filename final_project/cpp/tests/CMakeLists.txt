# Plain CTest with assert-based tests. A framework (doctest/Catch2) would be
# vendored under third_party/ per the DESIGN §11 no-network rule if E1's real
# test suite needs richer assertions; introducing one now, when plain asserts
# still express every case clearly, would be speculative.
add_executable(test_cpu_features test_cpu_features.cpp)
target_link_libraries(test_cpu_features PRIVATE wildlife_trigger_core)
add_test(NAME cpu_features COMMAND test_cpu_features)

# These need the app library, which only exists when the ORT tools are built.
if(WILDLIFE_BUILD_ORT_TOOLS)
  add_executable(test_policy test_policy.cpp)
  target_link_libraries(test_policy PRIVATE wildlife_trigger_app)
  add_test(NAME policy COMMAND test_policy)

  add_executable(test_preprocess test_preprocess.cpp)
  target_link_libraries(test_preprocess PRIVATE wildlife_trigger_app ${OpenCV_LIBS})
  add_test(NAME preprocess COMMAND test_preprocess)
endif()
