// BenchmarkRunner and SystemMonitor -- DESIGN §11 components 5 and 6.
//
// DESIGN §12 is unusually strict about what a timing may claim, and these types exist
// to make the strictness structural rather than remembered:
//
//   - the target is p95 END-TO-END (decode + preprocess + inference + policy), not
//     inference alone. A 20 ms model behind a 60 ms decode misses a 50 ms budget, and
//     reporting only the model would hide that;
//   - percentiles come from a sorted sample of individual iterations, never from a
//     mean and a standard deviation, because the latency distribution of a
//     thermally-throttled Pi is not normal;
//   - warm-up iterations are discarded, not averaged in. The first inference pays
//     lazy allocation and page faults;
//   - a host that cannot report a sensor records `unavailable`, never 0 and never an
//     estimate. A fabricated 0 degrees C would silently pass a throttling check.
//
// No number produced here is a Pi result until it is produced ON a Pi (DESIGN §12.4).

#ifndef WILDLIFE_TRIGGER_BENCHMARK_HPP
#define WILDLIFE_TRIGGER_BENCHMARK_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wildlife_trigger {

// Milliseconds for one frame, split by stage. Kept per-iteration rather than
// accumulated so percentiles are computable per stage.
struct StageTimings {
    double decode_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double policy_ms = 0.0;
    double end_to_end_ms = 0.0;
};

struct Percentiles {
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
};

// Linear-interpolated percentile over a copy of `values`. Empty input yields zeros.
Percentiles summarise(std::vector<double> values);

// Peak resident set size and CPU time, plus whatever the host exposes about
// temperature and frequency. Every optional is genuinely optional: a container on
// gx10 exposes almost none of this, and the Pi exposes all of it.
struct SystemSnapshot {
    int64_t peak_rss_kib = 0;
    double user_cpu_seconds = 0.0;
    double system_cpu_seconds = 0.0;

    // std::nullopt means "this host does not expose it", which is recorded as
    // `unavailable` in JSON rather than being invented.
    std::optional<double> cpu_temperature_c;
    std::optional<int64_t> cpu_frequency_khz;
    std::optional<std::string> throttling;

    static SystemSnapshot capture();
};

struct BenchmarkResult {
    int warmup_iterations = 0;
    int measured_iterations = 0;
    int intra_op_threads = 0;

    Percentiles decode;
    Percentiles preprocess;
    Percentiles inference;
    Percentiles policy;
    Percentiles end_to_end;

    double inference_fps = 0.0;    // 1000 / inference p50
    double end_to_end_fps = 0.0;   // 1000 / end-to-end p50

    SystemSnapshot system;
    std::vector<StageTimings> samples;
};

BenchmarkResult summarise_benchmark(const std::vector<StageTimings> &samples,
                                    int warmup_iterations, int intra_op_threads);

}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_BENCHMARK_HPP
