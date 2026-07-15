# Plain CTest with assert-based tests. A framework (doctest/Catch2) would be
# vendored under third_party/ per the DESIGN §11 no-network rule if E1's real
# test suite needs richer assertions; introducing one now, with no behaviour to
# assert against, would be speculative.
add_executable(test_cpu_features test_cpu_features.cpp)
target_link_libraries(test_cpu_features PRIVATE wildlife_trigger_core)
add_test(NAME cpu_features COMMAND test_cpu_features)
