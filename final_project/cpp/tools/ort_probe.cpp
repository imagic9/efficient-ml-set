// P0 evidence: load a model with the real C++ ONNX Runtime and report what ran.
//
// PLAN A3 requires all three model forms (FP32 / PTQ / QAT) to execute under the
// exact C++ ORT build that ships to the Pi, inside the target-compatible bookworm
// container -- not merely under Python ORT on gx10. Python proves the graph is
// valid; only this proves the deployed call site can run it.
//
// What it deliberately does NOT do is decide anything. It writes the
// session-optimized graph and the ORT profile, and
// `wildlife_trigger.validate.ort_coverage` reads both and reaches the verdict. One
// implementation of "is this integer execution?" serves Python and C++, so the two
// call sites cannot answer the question differently -- which is the whole point of
// asking it twice.
//
// The CPU features are reported alongside because they are the reason this runs
// twice: natively gx10 is a Cortex-X925 with i8mm and SVE2, and under
// `qemu-aarch64 -cpu cortex-a76` it must answer exactly as a Pi 5 does. A
// quantized path that only works because gx10 has i8mm is a P0 failure, and
// `looks_like_pi5` is what makes the emulated run self-verifying rather than
// trusted.
//
// Timing is not printed. DESIGN §12.4: QEMU models no caches and no memory
// bandwidth, so no number produced here is ever a latency result.

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "wildlife_trigger/cpu_features.hpp"

namespace {

struct Options {
    std::string model;
    std::string optimized_out;
    std::string profile_prefix;
    std::string input_bin;
    std::string output_bin;
    int iterations = 1;
};

[[noreturn]] void usage(const char *program, const std::string &problem) {
    std::cerr << problem << "\n\n"
              << "usage: " << program << " --model M.onnx --optimized-out O.onnx\n"
              << "         --profile-prefix P [--input-bin X.bin] "
                 "[--output-bin Y.bin]\n"
              << "         [--iterations N]\n\n"
              << "  --input-bin   raw float32 blob matching the model's input "
                 "shape.\n"
              << "                Omitted, a deterministic ramp is used instead; "
                 "that\n"
              << "                exercises the kernels but cannot be compared "
                 "against\n"
              << "                Python, so pass one when parity matters.\n";
    std::exit(2);
}

Options parse_args(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0], "missing value for " + flag);
            }
            return argv[++i];
        };
        if (flag == "--model") {
            options.model = value();
        } else if (flag == "--optimized-out") {
            options.optimized_out = value();
        } else if (flag == "--profile-prefix") {
            options.profile_prefix = value();
        } else if (flag == "--input-bin") {
            options.input_bin = value();
        } else if (flag == "--output-bin") {
            options.output_bin = value();
        } else if (flag == "--iterations") {
            options.iterations = std::stoi(value());
        } else if (flag == "--help" || flag == "-h") {
            usage(argv[0], "wildlife_trigger ORT probe");
        } else {
            usage(argv[0], "unknown flag: " + flag);
        }
    }
    if (options.model.empty() || options.optimized_out.empty() ||
        options.profile_prefix.empty()) {
        usage(argv[0], "--model, --optimized-out and --profile-prefix are required");
    }
    return options;
}

std::vector<float> read_floats(const std::string &path, size_t expected) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open input fixture: " + path);
    }
    const auto bytes = static_cast<size_t>(file.tellg());
    if (bytes != expected * sizeof(float)) {
        // A size mismatch would otherwise be read as garbage and produce
        // confident, wrong numbers.
        throw std::runtime_error(
            "fixture " + path + " holds " + std::to_string(bytes) +
            " bytes but the model wants " + std::to_string(expected * sizeof(float)) +
            " (" + std::to_string(expected) + " float32)");
    }
    file.seekg(0);
    std::vector<float> data(expected);
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

// Deterministic stand-in when no fixture is supplied: values spread over roughly
// the range ImageNet-normalised pixels occupy, so activation ranges are plausible
// and every kernel is exercised.
std::vector<float> ramp(size_t count) {
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>((i % 255) / 255.0 * 4.0 - 2.0);
    }
    return data;
}

void write_floats(const std::string &path, const std::vector<float> &data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot write: " + path);
    }
    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

std::string json_escape(const std::string &text) {
    std::string out;
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    const Options options = parse_args(argc, argv);

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_probe");
        Ort::SessionOptions session_options;

        // DESIGN §8: ORT_ENABLE_ALL is the starting point. ORT_ENABLE_EXTENDED is
        // an explicitly named E6 candidate and is never substituted silently.
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetOptimizedModelFilePath(options.optimized_out.c_str());
        session_options.EnableProfiling(options.profile_prefix.c_str());

        // One thread: the profile should attribute work to kernels, not to thread
        // scheduling. Correctness evidence only.
        session_options.SetIntraOpNumThreads(1);

        Ort::Session session(env, options.model.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const auto input_shape =
            session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        const size_t element_count = std::accumulate(
            input_shape.begin(), input_shape.end(), static_cast<size_t>(1),
            [](size_t acc, int64_t dim) {
                // A dynamic dimension (-1) has no fixed size; the export contract
                // pins the batch to 1 precisely so this cannot happen.
                return acc * static_cast<size_t>(dim > 0 ? dim : 1);
            });

        std::vector<float> input_data = options.input_bin.empty()
                                            ? ramp(element_count)
                                            : read_floats(options.input_bin, element_count);

        const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), input_shape.data(),
            input_shape.size());

        const char *input_names[] = {input_name.get()};
        const char *output_names[] = {output_name.get()};

        std::vector<Ort::Value> outputs;
        for (int i = 0; i < options.iterations; ++i) {
            outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor,
                                  1, output_names, 1);
        }

        const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        const size_t output_count = output_info.GetElementCount();
        const float *output_values = outputs[0].GetTensorData<float>();
        const std::vector<float> output_data(output_values, output_values + output_count);

        if (!options.output_bin.empty()) {
            write_floats(options.output_bin, output_data);
        }

        const double sum =
            std::accumulate(output_data.begin(), output_data.end(), 0.0);
        const double mean = output_count ? sum / static_cast<double>(output_count) : 0.0;
        const auto argmax = static_cast<int64_t>(
            std::max_element(output_data.begin(), output_data.end()) -
            output_data.begin());

        // EndProfiling returns the real filename; ORT appends its own timestamp, so
        // reconstructing it from the prefix would break on the next ORT release.
        const auto profile_file = session.EndProfilingAllocated(allocator);
        const auto features = wildlife_trigger::detect_cpu_features();

        std::cout << "{\n"
                  << "  \"model\": \"" << json_escape(options.model) << "\",\n"
                  << "  \"onnxruntime_version\": \"" << Ort::GetVersionString()
                  << "\",\n"
                  << "  \"call_site\": \"c++\",\n"
                  << "  \"optimized_graph\": \"" << json_escape(options.optimized_out)
                  << "\",\n"
                  << "  \"profile\": \"" << json_escape(profile_file.get()) << "\",\n"
                  << "  \"iterations\": " << options.iterations << ",\n"
                  << "  \"input_fixture\": \""
                  << json_escape(options.input_bin.empty() ? "<deterministic ramp>"
                                                           : options.input_bin)
                  << "\",\n"
                  << "  \"output_elements\": " << output_count << ",\n"
                  << "  \"output_mean\": " << mean << ",\n"
                  << "  \"output_argmax\": " << argmax << ",\n"
                  << "  \"cpu_features\": \"" << wildlife_trigger::describe(features)
                  << "\",\n"
                  << "  \"looks_like_pi5\": "
                  << (features.looks_like_pi5() ? "true" : "false") << "\n"
                  << "}\n";
        return 0;
    } catch (const Ort::Exception &error) {
        std::cerr << "ONNX Runtime rejected the model: " << error.what() << "\n";
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "probe failed: " << error.what() << "\n";
        return 1;
    }
}
