// wildlife_trigger -- the C++ application (DESIGN §9.1, §11).
//
// A4 implements the thin vertical slice: saved JPEG -> decode/preprocess -> ORT ->
// policy -> SHUTTER_TRIGGER JSON, plus a benchmark and a self-test. E1-E5 harden
// this; the point of building it now, before any data or training exists, is that a
// vertical slice discovers integration problems while they are still cheap.
//
// Every subcommand is non-interactive, prints concise human output to stderr and
// complete machine-readable JSON to stdout, so evidence can be piped and the chatter
// cannot corrupt it.

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/version.hpp>

#include "wildlife_trigger/benchmark.hpp"
#include "wildlife_trigger/cpu_features.hpp"
#include "wildlife_trigger/hashing.hpp"
#include "wildlife_trigger/policy.hpp"
#include "wildlife_trigger/preprocess.hpp"
#include "wildlife_trigger/session.hpp"

namespace {

using nlohmann::json;
using namespace wildlife_trigger;
using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point &start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct CommonArgs {
    std::string model;
    std::string policy;
    std::string class_map;
    std::string image;
    std::string output;
    std::string output_bin;
    std::string preprocess_mode = "fused";
    std::string profile_prefix;
    std::string optimized_model;
    int threads = 1;
    int warmup = 10;
    int iterations = 100;
    int width = 256;
    int height = 192;
    std::string manifest;
    std::string images_root;
    std::string on_corrupt = "fail";
    int limit = 0;
};

[[noreturn]] void usage(int exit_code) {
    std::cerr
        << "wildlife_trigger -- wildlife shutter trigger (A4 vertical slice)\n\n"
        << "usage:\n"
        << "  wildlife_trigger infer --model M.onnx --class-map C.json \\\n"
        << "                         --policy P.json --image X.jpg [--output R.json]\n"
        << "  wildlife_trigger benchmark --model M.onnx --class-map C.json \\\n"
        << "                         --policy P.json --image X.jpg \\\n"
        << "                         [--warmup N] [--iterations N] [--output B.json]\n"
        << "  wildlife_trigger self-test --model M.onnx --class-map C.json \\\n"
        << "                         --policy P.json --image X.jpg\n"
        << "  wildlife_trigger dump-tensor --image X.jpg --output-bin T.bin \\\n"
        << "                         [--preprocess fused|reference] [--output T.json]\n"
        << "  wildlife_trigger run-dataset --model M.onnx --class-map C.json \\\n"
        << "                         --policy P.json --manifest VAL.jsonl \\\n"
        << "                         --images-root DIR --output PRED.jsonl\n"
        << "                         [--on-corrupt fail|skip] [--limit N]\n\n"
        << "options:\n"
        << "  --manifest P           JSONL manifest, consumed in file order (run-dataset)\n"
        << "  --images-root D        directory file_name entries resolve against\n"
        << "  --on-corrupt M         fail (default) or skip: skipped frames are\n"
        << "                         recorded as error lines, never silently dropped\n"
        << "  --limit N              only the first N manifest records (0 = all)\n"
        << "  --output-bin P         raw float32 tensor destination (dump-tensor)\n"
        << "  --preprocess M         fused (the shipping path) or reference\n"
        << "                         (unfused OpenCV primitives; P1's third column)\n"
        << "  --threads N            intra-op threads (default 1, stated not implied)\n"
        << "  --width/--height N     input geometry (default 256x192, DESIGN §5.5)\n"
        << "  --profile-prefix P     enable ORT profiling with this prefix\n"
        << "  --optimized-model P    persist the session-optimized graph (inspection\n"
        << "                         only -- never ship one to the Pi)\n";
    std::exit(exit_code);
}

CommonArgs parse(int argc, char **argv) {
    CommonArgs args;
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                usage(2);
            }
            return argv[++i];
        };
        if (flag == "--model") args.model = value();
        else if (flag == "--policy") args.policy = value();
        else if (flag == "--class-map") args.class_map = value();
        else if (flag == "--image") args.image = value();
        else if (flag == "--output") args.output = value();
        else if (flag == "--output-bin") args.output_bin = value();
        else if (flag == "--preprocess") args.preprocess_mode = value();
        else if (flag == "--profile-prefix") args.profile_prefix = value();
        else if (flag == "--optimized-model") args.optimized_model = value();
        else if (flag == "--threads") args.threads = std::stoi(value());
        else if (flag == "--warmup") args.warmup = std::stoi(value());
        else if (flag == "--iterations") args.iterations = std::stoi(value());
        else if (flag == "--manifest") args.manifest = value();
        else if (flag == "--images-root") args.images_root = value();
        else if (flag == "--on-corrupt") args.on_corrupt = value();
        else if (flag == "--limit") args.limit = std::stoi(value());
        else if (flag == "--width") args.width = std::stoi(value());
        else if (flag == "--height") args.height = std::stoi(value());
        else if (flag == "--help" || flag == "-h") usage(0);
        else {
            std::cerr << "unknown flag: " << flag << "\n";
            usage(2);
        }
    }
    return args;
}

void require(const std::string &value, const char *name) {
    if (value.empty()) {
        std::cerr << name << " is required\n";
        usage(2);
    }
}

void emit(const json &document, const std::string &path) {
    const std::string text = document.dump(2);
    std::cout << text << "\n";
    if (!path.empty()) {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("cannot write output: " + path);
        }
        file << text << "\n";
        std::cerr << "wrote " << path << "\n";
    }
}

json environment_json() {
    const auto features = detect_cpu_features();
    return json{
        {"onnxruntime_version", ModelSession::ort_version()},
        {"cpu_features", describe(features)},
        {"looks_like_pi5", features.looks_like_pi5()},
    };
}

json letterbox_json(const LetterboxInfo &info) {
    return json{
        {"source", {info.source_width, info.source_height}},
        {"resized", {info.resized_width, info.resized_height}},
        {"pad_left", info.pad_left},
        {"pad_top", info.pad_top},
        {"scale", info.scale},
        {"pixel_utilisation", info.pixel_utilisation()},
    };
}

json decision_json(const Decision &decision) {
    json targets = json::array();
    for (const auto &target : decision.targets) {
        targets.push_back(json{{"class", target.class_name},
                               {"index", target.class_index},
                               {"score", target.score},
                               {"threshold", target.threshold},
                               {"passed", target.passed}});
    }
    return json{
        {"SHUTTER_TRIGGER", decision.shutter_trigger ? 1 : 0},
        {"passing_targets", decision.passing},
        {"targets", targets},
        {"top1", {{"class", decision.top1_class},
                  {"index", decision.top1_index},
                  {"score", decision.top1_score}}},
    };
}

json percentiles_json(const Percentiles &p) {
    return json{{"p50", p.p50}, {"p95", p.p95}, {"p99", p.p99},
                {"min", p.min}, {"max", p.max}, {"mean", p.mean}};
}

json system_json(const SystemSnapshot &s) {
    // An absent sensor becomes the string "unavailable", never 0. DESIGN §11: record
    // what the host does not expose rather than inventing a value that would pass a
    // check the real reading might fail.
    const auto optional_or_unavailable = [](auto value) -> json {
        return value.has_value() ? json(*value) : json("unavailable");
    };
    return json{
        {"peak_rss_kib", s.peak_rss_kib},
        {"user_cpu_seconds", s.user_cpu_seconds},
        {"system_cpu_seconds", s.system_cpu_seconds},
        {"cpu_temperature_c", optional_or_unavailable(s.cpu_temperature_c)},
        {"cpu_frequency_khz", optional_or_unavailable(s.cpu_frequency_khz)},
        {"throttling", optional_or_unavailable(s.throttling)},
    };
}

struct Pipeline {
    ClassMap class_map;
    Policy policy;
    Preprocessor preprocessor;
    ModelSession session;
    std::string model_sha256;
};

Pipeline build_pipeline(const CommonArgs &args, bool with_profiling) {
    require(args.model, "--model");
    require(args.class_map, "--class-map");
    require(args.policy, "--policy");

    PreprocessConfig preprocess_config;
    preprocess_config.width = args.width;
    preprocess_config.height = args.height;

    SessionConfig session_config;
    session_config.model_path = args.model;
    session_config.intra_op_threads = args.threads;
    if (with_profiling) {
        session_config.profile_prefix = args.profile_prefix;
        session_config.optimized_model_path = args.optimized_model;
    }

    const std::string model_hash = sha256_file(args.model);
    ClassMap class_map = load_class_map(args.class_map);
    Policy policy = Policy::load(args.policy, class_map, model_hash);

    ModelSession session(session_config);

    // The model's class count and the class map must agree before any inference: a
    // mismatch means every threshold is bound to the wrong animal.
    if (session.contract().class_count !=
        static_cast<int64_t>(class_map.classes.size())) {
        throw std::runtime_error(
            "model emits " + std::to_string(session.contract().class_count) +
            " classes but the class map declares " +
            std::to_string(class_map.classes.size()) +
            ". Every class index would denote a different animal.");
    }

    // The tensor the preprocessor builds must be the tensor the model wants. Checked
    // here rather than discovered as a size error inside run().
    const auto &shape = session.contract().input_shape;
    if (shape[1] != 3 || shape[2] != args.height || shape[3] != args.width) {
        throw std::runtime_error(
            "model expects input [1, " + std::to_string(shape[1]) + ", " +
            std::to_string(shape[2]) + ", " + std::to_string(shape[3]) +
            "] but preprocessing is configured for [1, 3, " +
            std::to_string(args.height) + ", " + std::to_string(args.width) +
            "]. Pass --width/--height matching the model.");
    }

    return Pipeline{std::move(class_map), std::move(policy),
                    Preprocessor(preprocess_config), std::move(session), model_hash};
}

int command_infer(const CommonArgs &args) {
    require(args.image, "--image");
    Pipeline pipeline = build_pipeline(args, true);

    StageTimings timings;
    const auto start = Clock::now();

    const auto preprocess_start = Clock::now();
    PreprocessResult input = pipeline.preprocessor.from_file(args.image);
    timings.preprocess_ms = ms_since(preprocess_start);

    const auto inference_start = Clock::now();
    const std::vector<float> logits = pipeline.session.run(input.tensor);
    timings.inference_ms = ms_since(inference_start);

    const auto policy_start = Clock::now();
    const Decision decision = pipeline.policy.decide(logits, pipeline.class_map);
    timings.policy_ms = ms_since(policy_start);
    timings.end_to_end_ms = ms_since(start);

    const std::string profile = pipeline.session.end_profiling();

    json result{
        {"image", args.image},
        {"model", args.model},
        {"model_sha256", pipeline.model_sha256},
        {"policy_id", pipeline.policy.policy_id()},
        {"class_map_sha256", pipeline.class_map.sha256},
        {"decision", decision_json(decision)},
        // The raw logits, for parity: P2's ORT-Python half compares against
        // these, and P3/P4 will want them for the quantized candidates. The
        // decision block alone cannot support a numeric comparison.
        {"logits", logits},
        {"letterbox", letterbox_json(input.letterbox)},
        {"timings_ms", {{"preprocess", timings.preprocess_ms},
                        {"inference", timings.inference_ms},
                        {"policy", timings.policy_ms},
                        {"end_to_end", timings.end_to_end_ms}}},
        {"environment", environment_json()},
        {"note", "Single-shot timings are not a benchmark; use `benchmark`."},
    };
    if (!profile.empty()) {
        result["ort_profile"] = profile;
    }

    std::cerr << (decision.shutter_trigger ? "SHUTTER_TRIGGER=1" : "SHUTTER_TRIGGER=0")
              << "  top1=" << decision.top1_class << " (" << std::fixed
              << std::setprecision(4) << decision.top1_score << ")\n";
    emit(result, args.output);
    return 0;
}

int command_benchmark(const CommonArgs &args) {
    require(args.image, "--image");
    if (args.iterations <= 0) {
        std::cerr << "--iterations must be positive\n";
        return 2;
    }
    Pipeline pipeline = build_pipeline(args, false);

    // Warm-up is discarded, never averaged in: the first inference pays lazy
    // allocation, page faults and arena growth that no steady-state frame pays.
    for (int i = 0; i < args.warmup; ++i) {
        PreprocessResult input = pipeline.preprocessor.from_file(args.image);
        const auto logits = pipeline.session.run(input.tensor);
        (void)pipeline.policy.decide(logits, pipeline.class_map);
    }

    std::vector<StageTimings> samples;
    samples.reserve(static_cast<size_t>(args.iterations));
    for (int i = 0; i < args.iterations; ++i) {
        StageTimings timings;
        const auto start = Clock::now();

        const auto preprocess_start = Clock::now();
        PreprocessResult input = pipeline.preprocessor.from_file(args.image);
        timings.preprocess_ms = ms_since(preprocess_start);

        const auto inference_start = Clock::now();
        const std::vector<float> logits = pipeline.session.run(input.tensor);
        timings.inference_ms = ms_since(inference_start);

        const auto policy_start = Clock::now();
        (void)pipeline.policy.decide(logits, pipeline.class_map);
        timings.policy_ms = ms_since(policy_start);

        timings.end_to_end_ms = ms_since(start);
        samples.push_back(timings);
    }

    const BenchmarkResult result =
        summarise_benchmark(samples, args.warmup, args.threads);

    json document{
        {"schema_version", 1},
        {"model", args.model},
        {"model_sha256", pipeline.model_sha256},
        {"image", args.image},
        {"warmup_iterations", result.warmup_iterations},
        {"measured_iterations", result.measured_iterations},
        {"intra_op_threads", result.intra_op_threads},
        {"stages_ms", {{"preprocess", percentiles_json(result.preprocess)},
                       {"inference", percentiles_json(result.inference)},
                       {"policy", percentiles_json(result.policy)},
                       {"end_to_end", percentiles_json(result.end_to_end)}}},
        {"fps", {{"inference_from_p50", result.inference_fps},
                 {"end_to_end_from_p50", result.end_to_end_fps}}},
        {"system", system_json(result.system)},
        {"environment", environment_json()},
        {"provenance",
         "DESIGN §12.4: a latency number is a Pi result only when measured ON a Pi. "
         "Measured on any other host -- gx10 natively or under QEMU -- this is a "
         "smoke check of the timing path, never a performance claim."},
    };

    std::cerr << "end-to-end p50=" << std::fixed << std::setprecision(2)
              << result.end_to_end.p50 << "ms p95=" << result.end_to_end.p95
              << "ms  (" << result.end_to_end_fps << " FPS)\n";
    emit(document, args.output);
    return 0;
}

int command_self_test(const CommonArgs &args) {
    // Deliberately asserts behaviour that must hold anywhere, on any host, with no
    // reference values baked in -- a Pi in the field has no fixtures.
    Pipeline pipeline = build_pipeline(args, false);
    int failures = 0;

    const auto check = [&failures](bool ok, const std::string &what) {
        std::cerr << (ok ? "  PASS  " : "  FAIL  ") << what << "\n";
        if (!ok) ++failures;
    };

    check(pipeline.session.contract().class_count ==
              static_cast<int64_t>(pipeline.class_map.classes.size()),
          "model class count matches the class map");
    check(!pipeline.policy.targets().empty(), "policy has at least one target");

    for (const auto &target : pipeline.policy.targets()) {
        check(target.class_index >= 0 &&
                  target.class_index < static_cast<int>(pipeline.class_map.classes.size()),
              "target '" + target.class_name + "' maps to a valid class index");
    }

    if (!args.image.empty()) {
        PreprocessResult input = pipeline.preprocessor.from_file(args.image);
        check(input.tensor.size() ==
                  static_cast<size_t>(3) * args.height * args.width,
              "preprocessed tensor has the contracted element count");

        const std::vector<float> logits = pipeline.session.run(input.tensor);
        check(logits.size() == pipeline.class_map.classes.size(),
              "model returns one logit per class");

        const std::vector<float> scores = softmax(logits);
        double sum = 0.0;
        for (const float score : scores) sum += score;
        check(std::abs(sum - 1.0) < 1e-4, "softmax sums to 1");

        const Decision decision = pipeline.policy.decide(logits, pipeline.class_map);
        check(decision.targets.size() == pipeline.policy.targets().size(),
              "every configured target is scored and reported");
    }

    std::cerr << (failures == 0 ? "self-test PASSED\n" : "self-test FAILED\n");
    emit(json{{"self_test", failures == 0 ? "PASSED" : "FAILED"},
              {"failures", failures},
              {"environment", environment_json()}},
         args.output);
    return failures == 0 ? 0 : 1;
}

int command_run_dataset(const CommonArgs &args) {
    // DatasetRunner -- DESIGN §11 component 4, the C++ half of gate P4. Consumes
    // a manifest in file order (which is the order Python evaluation scored it
    // in), writes one JSONL prediction line per frame, and never reorders,
    // buffers, or drops silently: a skipped frame under --on-corrupt=skip is an
    // explicit error line the comparator must look at, not an absence it could
    // miss.
    require(args.manifest, "--manifest");
    require(args.images_root, "--images-root");
    require(args.output, "--output");
    if (args.on_corrupt != "fail" && args.on_corrupt != "skip") {
        std::cerr << "--on-corrupt must be 'fail' or 'skip', got '"
                  << args.on_corrupt << "'\n";
        return 2;
    }

    Pipeline pipeline = build_pipeline(args, false);

    std::ifstream manifest(args.manifest);
    if (!manifest) {
        throw std::runtime_error("cannot open manifest: " + args.manifest);
    }
    std::ofstream out(args.output);
    if (!out) {
        throw std::runtime_error("cannot write output: " + args.output);
    }

    // The header line binds the whole file to what produced it: the comparator
    // refuses a JSONL whose model, policy or class map is not the one under
    // test, exactly as the loaders refused them at startup.
    out << json{{"kind", "run_dataset_header"},
                {"model", args.model},
                {"model_sha256", pipeline.model_sha256},
                {"policy_id", pipeline.policy.policy_id()},
                {"class_map_sha256", pipeline.class_map.sha256},
                {"manifest", args.manifest},
                {"manifest_sha256", sha256_file(args.manifest)},
                {"threads", args.threads},
                {"onnxruntime_version", ModelSession::ort_version()},
                {"input", {args.width, args.height}}}
            .dump()
        << "\n";

    int processed = 0;
    int skipped = 0;
    int fired = 0;
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty()) continue;
        if (args.limit > 0 && processed + skipped >= args.limit) break;

        const json record = json::parse(line);
        const std::string file_name = record.at("file_name").get<std::string>();
        const std::string image_id = record.at("image_id").get<std::string>();
        const std::string path = args.images_root + "/" + file_name;

        StageTimings timings;
        const auto start = Clock::now();
        try {
            const auto preprocess_start = Clock::now();
            PreprocessResult input = pipeline.preprocessor.from_file(path);
            timings.preprocess_ms = ms_since(preprocess_start);

            const auto inference_start = Clock::now();
            const std::vector<float> logits = pipeline.session.run(input.tensor);
            timings.inference_ms = ms_since(inference_start);

            const auto policy_start = Clock::now();
            const Decision decision = pipeline.policy.decide(logits, pipeline.class_map);
            timings.policy_ms = ms_since(policy_start);
            timings.end_to_end_ms = ms_since(start);

            json targets = json::object();
            for (const auto &target : decision.targets) {
                targets[target.class_name] = target.score;
            }
            out << json{{"image_id", image_id},
                        {"seq_id", record.value("seq_id", "")},
                        {"labels", record.value("labels", json::array())},
                        {"target_scores", targets},
                        {"shutter_trigger", decision.shutter_trigger ? 1 : 0},
                        {"top1_index", decision.top1_index},
                        {"top1_class", decision.top1_class},
                        {"timings_ms", {{"preprocess", timings.preprocess_ms},
                                        {"inference", timings.inference_ms},
                                        {"policy", timings.policy_ms},
                                        {"end_to_end", timings.end_to_end_ms}}}}
                    .dump()
                << "\n";
            if (decision.shutter_trigger) ++fired;
            ++processed;
        } catch (const std::exception &error) {
            if (args.on_corrupt == "fail") {
                throw std::runtime_error("frame " + image_id + " (" + path +
                                         "): " + error.what());
            }
            out << json{{"image_id", image_id},
                        {"error", error.what()},
                        {"skipped", true}}
                    .dump()
                << "\n";
            ++skipped;
        }
        if ((processed + skipped) % 500 == 0) {
            std::cerr << "  " << (processed + skipped) << " frames...\n";
        }
    }
    out << json{{"kind", "run_dataset_footer"},
                {"processed", processed},
                {"skipped", skipped},
                {"fired", fired}}
            .dump()
        << "\n";
    out.close();
    if (!out) {
        throw std::runtime_error("short write: " + args.output);
    }

    std::cerr << "run-dataset: " << processed << " frames (" << skipped
              << " skipped), " << fired << " fired -> " << args.output << "\n";
    return 0;
}

int command_dump_tensor(const CommonArgs &args) {
    // P1's C++ half: preprocess one image and write the tensor where Python can
    // read it. No model, no policy -- the comparison must isolate preprocessing,
    // and loading ORT here would only add ways for this command to fail.
    //
    // The .bin is raw little-endian float32, C order -- numpy.tofile's format,
    // already the convention `validate.fixture` and ort_probe established. The
    // C++ tensor is plane-major (C,H,W) contiguous, which is exactly numpy's
    // contiguous (3,H,W): a straight np.fromfile(...).reshape(3,H,W).
    require(args.image, "--image");
    require(args.output_bin, "--output-bin");
    if (args.preprocess_mode != "fused" && args.preprocess_mode != "reference") {
        std::cerr << "--preprocess must be 'fused' or 'reference', got '"
                  << args.preprocess_mode << "'\n";
        return 2;
    }

    PreprocessConfig config;
    config.width = args.width;
    config.height = args.height;
    Preprocessor preprocessor(config);

    const PreprocessResult result = args.preprocess_mode == "reference"
                                        ? preprocessor.from_file_reference(args.image)
                                        : preprocessor.from_file(args.image);

    std::ofstream bin(args.output_bin, std::ios::binary);
    if (!bin) {
        throw std::runtime_error("cannot open for writing: " + args.output_bin);
    }
    bin.write(reinterpret_cast<const char *>(result.tensor.data()),
              static_cast<std::streamsize>(result.tensor.size() * sizeof(float)));
    bin.close();
    if (!bin) {
        throw std::runtime_error("short write: " + args.output_bin);
    }

    const std::string tensor_sha256 =
        sha256_bytes(reinterpret_cast<const uint8_t *>(result.tensor.data()),
                     result.tensor.size() * sizeof(float));

    json document{
        {"image", args.image},
        {"preprocess", args.preprocess_mode},
        {"shape", {1, 3, args.height, args.width}},
        {"dtype", "float32"},
        {"layout", "NCHW, C order, little-endian"},
        {"letterbox", letterbox_json(result.letterbox)},
        {"tensor_sha256", tensor_sha256},
        {"output_bin", args.output_bin},
        {"opencv_version", CV_VERSION},
    };
    std::cerr << args.preprocess_mode << " tensor " << tensor_sha256.substr(0, 16)
              << "... -> " << args.output_bin << "\n";
    emit(document, args.output);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) usage(2);
    const std::string command = argv[1];
    if (command == "--help" || command == "-h") usage(0);

    const CommonArgs args = parse(argc, argv);
    try {
        if (command == "infer") return command_infer(args);
        if (command == "benchmark") return command_benchmark(args);
        if (command == "self-test") return command_self_test(args);
        if (command == "dump-tensor") return command_dump_tensor(args);
        if (command == "run-dataset") return command_run_dataset(args);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    std::cerr << "unknown command: " << command << "\n";
    usage(2);
}
