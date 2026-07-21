// shutter_registry -- the GENERAL, registry-driven shutter engine.
//
// One frozen backbone produces a 1280-d embedding; a JSON registry lists any number of
// target animals, each a linear head {weight[1280], bias, threshold}. For every frame we
// L2-normalise the embedding once and score every target: fire the shutter for any target
// whose score clears its threshold. Adding a new animal is a new registry row -- no
// retrain, no recompile, no graph change. Bobcat, tiger, or a species added tomorrow are
// all just entries here.
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

static std::vector<float> read_floats(const std::string& p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(2); }
    std::vector<float> v(n); f.read(reinterpret_cast<char*>(v.data()), n*sizeof(float));
    if (!f) { std::cerr << "short read " << p << "\n"; std::exit(2); }
    return v;
}

int main(int argc, char** argv) {
    std::string model, registry, input_bin, emb_name = "/Flatten_output_0";
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i]; auto val = [&]{ return std::string(argv[++i]); };
        if (f == "--model") model = val();
        else if (f == "--registry") registry = val();
        else if (f == "--input-bin") input_bin = val();
        else if (f == "--emb-name") emb_name = val();
    }
    // load registry
    json reg; { std::ifstream r(registry); if (!r) { std::cerr << "no registry\n"; return 2; } r >> reg; }
    const bool l2 = reg.value("l2_normalise_before_dot", true);
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "shutter_registry");
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(1);
        Ort::Session session(env, model.c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = session.GetInputNameAllocated(0, alloc);
        auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        size_t n = std::accumulate(shape.begin(), shape.end(), (size_t)1,
                   [](size_t a, int64_t d){ return a * (size_t)(d>0?d:1); });
        std::vector<float> in = read_floats(input_bin, n);
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(),
                                                          shape.data(), shape.size());
        const char* in_names[] = { in_name.get() };
        const char* out_names[] = { emb_name.c_str() };
        auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1, out_names, 1);
        const float* emb = out[0].GetTensorData<float>();
        size_t dim = out[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // L2-normalise the embedding once
        std::vector<float> e(emb, emb + dim);
        if (l2) {
            double s = 0; for (float v : e) s += (double)v*v;
            float inv = 1.0f / (float)(std::sqrt(s) + 1e-12);
            for (float& v : e) v *= inv;
        }
        // score every registered target
        json fired = json::array(), all = json::array();
        for (const auto& t : reg["targets"]) {
            const auto& w = t["weight"];
            if (w.size() != dim) { std::cerr << "dim mismatch for " << t["name"] << "\n"; return 3; }
            double acc = t.value("bias", 0.0);
            for (size_t i = 0; i < dim; ++i) acc += (double)w[i].get<double>() * e[i];
            double thr = t["threshold"].get<double>();
            bool hit = acc > thr;
            json row = { {"name", t["name"]}, {"score", acc}, {"threshold", thr}, {"fired", hit} };
            all.push_back(row);
            if (hit) fired.push_back(t["name"]);
        }
        json result = { {"call_site","c++"}, {"n_targets", reg["targets"].size()},
                        {"fired", fired}, {"scores", all} };
        std::cout << result.dump(2) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "shutter_registry failed: " << e.what() << "\n"; return 1;
    }
}
