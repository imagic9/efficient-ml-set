// Benchmark percentile tests (E5) -- DESIGN §11 component 5, §12.3.
//
// The percentile math is the one place a benchmark can produce plausible-looking but
// wrong numbers: an off-by-one in the order statistic, a mean-and-stddev shortcut, or
// a percentile convention that disagrees with numpy would all pass a human skim while
// making every latency table subtly incomparable with the Python evidence. These pin
// summarise() to numpy's default linear-interpolation quantile on known inputs.

#include "wildlife_trigger/benchmark.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using wildlife_trigger::Percentiles;
using wildlife_trigger::summarise;

namespace {

bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

void test_empty_is_zeros() {
    const Percentiles p = summarise({});
    assert(close(p.p50, 0.0) && close(p.p95, 0.0) && close(p.p99, 0.0));
    assert(close(p.min, 0.0) && close(p.max, 0.0) && close(p.mean, 0.0));
    std::puts("  PASS  empty sample yields zeros, not a divide-by-zero");
}

void test_single_value() {
    const Percentiles p = summarise({7.5});
    // Every statistic of a one-element sample is that element.
    assert(close(p.p50, 7.5) && close(p.p95, 7.5) && close(p.p99, 7.5));
    assert(close(p.min, 7.5) && close(p.max, 7.5) && close(p.mean, 7.5));
    std::puts("  PASS  single value: every percentile is that value");
}

void test_linear_interpolation_small() {
    // [1,2,3,4], q=0.5 -> position 0.5*3 = 1.5 -> 2*0.5 + 3*0.5 = 2.5.
    const Percentiles p = summarise({4.0, 1.0, 3.0, 2.0});  // unsorted on purpose
    assert(close(p.p50, 2.5));
    assert(close(p.min, 1.0) && close(p.max, 4.0));
    assert(close(p.mean, 2.5));
    std::puts("  PASS  [1,2,3,4] p50 interpolates to 2.5 (input need not be sorted)");
}

void test_matches_numpy_on_1_to_100() {
    // 1..100. numpy.percentile default (linear): p50=50.5, p95=95.05, p99=99.01.
    std::vector<double> values;
    for (int i = 1; i <= 100; ++i) values.push_back(static_cast<double>(i));
    const Percentiles p = summarise(values);
    assert(close(p.p50, 50.5));
    assert(close(p.p95, 95.05));
    assert(close(p.p99, 99.01));
    assert(close(p.min, 1.0) && close(p.max, 100.0));
    assert(close(p.mean, 50.5));
    std::puts("  PASS  1..100 matches numpy linear percentile (50.5 / 95.05 / 99.01)");
}

void test_ordering_holds() {
    // The property a4_gate checks on real output: p50 <= p95 <= p99, min <= p50 <= max.
    std::vector<double> values = {12.0, 9.0, 30.0, 11.0, 10.5, 200.0, 9.5, 10.0};
    const Percentiles p = summarise(values);
    assert(p.min <= p.p50 && p.p50 <= p.p95 && p.p95 <= p.p99 && p.p99 <= p.max);
    std::puts("  PASS  p50 <= p95 <= p99 and min <= p50 <= max");
}

}  // namespace

int main() {
    std::puts("benchmark percentiles (DESIGN §12.3):");
    test_empty_is_zeros();
    test_single_value();
    test_linear_interpolation_small();
    test_matches_numpy_on_1_to_100();
    test_ordering_holds();
    std::puts("all benchmark tests passed");
    return 0;
}
