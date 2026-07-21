// tiger_probe -- run M2_plus.onnx via the deployed C++ ONNX Runtime and report the
// 16-class logits AND the baked `tiger_score` side-head, for Python<->C++ parity and
// on-Pi latency. Adapted from cpp/tools/ort_probe.cpp (same ORT, same call site).
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static std::vector<float> read_floats(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    if (!f) { std::cerr << "short read " << path << "\n"; std::exit(2); }
    return v;
}

int main(int argc, char** argv) {
    std::string model, input_bin; int iterations = 1; bool time_it = false; bool base = false;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        auto val = [&]{ return std::string(argv[++i]); };
        if (f == "--model") model = val();
        else if (f == "--input-bin") input_bin = val();
        else if (f == "--iterations") iterations = std::stoi(val());
        else if (f == "--time") time_it = true;
        else if (f == "--base") base = true;
    }
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tiger_probe");
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(1);
        Ort::Session session(env, model.c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;

        auto in_name = session.GetInputNameAllocated(0, alloc);
        const auto in_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        size_t n = std::accumulate(in_shape.begin(), in_shape.end(), (size_t)1,
                   [](size_t a, int64_t d){ return a * (size_t)(d > 0 ? d : 1); });
        std::vector<float> in = read_floats(input_bin, n);

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(),
                                                          in_shape.data(), in_shape.size());
        const char* in_names[] = { in_name.get() };
        const char* out_names[] = { "logits", "tiger_score" };
        size_t n_out = base ? 1 : 2;

        std::vector<Ort::Value> out;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            out = session.Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1, out_names, n_out);
        auto t1 = std::chrono::steady_clock::now();

        const float* logits = out[0].GetTensorData<float>();
        size_t nl = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        float tiger = base ? 0.f : out[1].GetTensorData<float>()[0];
        auto argmax = std::max_element(logits, logits + nl) - logits;

        std::cout.precision(8);
        std::cout << "{\n  \"call_site\": \"c++\",\n  \"ort\": \"" << Ort::GetVersionString() << "\",\n";
        std::cout << "  \"logits\": [";
        for (size_t i = 0; i < nl; ++i) std::cout << (i? ", ":"") << logits[i];
        std::cout << "],\n  \"argmax\": " << argmax << ",\n";
        std::cout << "  \"tiger_score\": " << tiger << ",\n";
        if (time_it) {
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            std::cout << "  \"ms_per_infer\": " << ms << ",\n";
        }
        std::cout << "  \"iterations\": " << iterations << "\n}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tiger_probe failed: " << e.what() << "\n"; return 1;
    }
}
