#include "wildlife_trigger/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>

#include <sys/resource.h>

namespace wildlife_trigger {
namespace {

std::optional<std::string> read_first_line(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        return std::nullopt;
    }
    std::string line;
    if (!std::getline(file, line)) {
        return std::nullopt;
    }
    return line;
}

}  // namespace

Percentiles summarise(std::vector<double> values) {
    Percentiles result;
    if (values.empty()) {
        return result;
    }
    std::sort(values.begin(), values.end());

    // Linear interpolation between order statistics -- the same definition numpy
    // uses by default, so the Python and C++ evidence tables are comparable rather
    // than differing by a percentile convention.
    const auto quantile = [&values](double q) {
        if (values.size() == 1) {
            return values.front();
        }
        const double position = q * static_cast<double>(values.size() - 1);
        const auto lower = static_cast<size_t>(std::floor(position));
        const auto upper = static_cast<size_t>(std::ceil(position));
        const double weight = position - static_cast<double>(lower);
        return values[lower] * (1.0 - weight) + values[upper] * weight;
    };

    result.p50 = quantile(0.50);
    result.p95 = quantile(0.95);
    result.p99 = quantile(0.99);
    result.min = values.front();
    result.max = values.back();
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
    return result;
}

SystemSnapshot SystemSnapshot::capture() {
    SystemSnapshot snapshot;

    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is KiB on Linux (bytes on macOS -- this application only ever
        // runs on Linux, and the Pi and the container agree).
        snapshot.peak_rss_kib = usage.ru_maxrss;
        snapshot.user_cpu_seconds = static_cast<double>(usage.ru_utime.tv_sec) +
                                    usage.ru_utime.tv_usec / 1e6;
        snapshot.system_cpu_seconds = static_cast<double>(usage.ru_stime.tv_sec) +
                                      usage.ru_stime.tv_usec / 1e6;
    }

    // thermal_zone0 is the Pi's CPU sensor and is absent in a container on gx10.
    // Absent stays absent: DESIGN §11 requires recording an unavailable sensor
    // rather than inventing a value, and a fabricated 0 C would pass a throttle check
    // that a real 85 C would fail.
    if (const auto milli = read_first_line("/sys/class/thermal/thermal_zone0/temp")) {
        try {
            snapshot.cpu_temperature_c = std::stod(*milli) / 1000.0;
        } catch (const std::exception &) {
            // A sensor that exists but reads garbage is unavailable, not zero.
        }
    }

    if (const auto khz = read_first_line(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")) {
        try {
            snapshot.cpu_frequency_khz = std::stoll(*khz);
        } catch (const std::exception &) {
        }
    }

    // The Pi's throttled flag lives behind vcgencmd, which is not readable this way.
    // E5/F own that; leaving it explicitly unavailable here is the honest state.
    return snapshot;
}

BenchmarkResult summarise_benchmark(const std::vector<StageTimings> &samples,
                                    int warmup_iterations, int intra_op_threads) {
    BenchmarkResult result;
    result.warmup_iterations = warmup_iterations;
    result.measured_iterations = static_cast<int>(samples.size());
    result.intra_op_threads = intra_op_threads;
    result.samples = samples;

    const auto column = [&samples](double StageTimings::*field) {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto &sample : samples) {
            values.push_back(sample.*field);
        }
        return values;
    };

    result.decode = summarise(column(&StageTimings::decode_ms));
    result.preprocess = summarise(column(&StageTimings::preprocess_ms));
    result.inference = summarise(column(&StageTimings::inference_ms));
    result.policy = summarise(column(&StageTimings::policy_ms));
    result.end_to_end = summarise(column(&StageTimings::end_to_end_ms));

    // FPS from the median, not from the mean: a throttling spike should not make the
    // reported throughput look worse than the machine's typical behaviour, and a
    // fast outlier should not make it look better.
    result.inference_fps =
        result.inference.p50 > 0.0 ? 1000.0 / result.inference.p50 : 0.0;
    result.end_to_end_fps =
        result.end_to_end.p50 > 0.0 ? 1000.0 / result.end_to_end.p50 : 0.0;

    result.system = SystemSnapshot::capture();
    return result;
}

}  // namespace wildlife_trigger
