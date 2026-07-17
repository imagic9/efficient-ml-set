#include "wildlife_trigger/session.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>

namespace wildlife_trigger {
namespace {

std::string shape_to_string(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    out << "]";
    return out.str();
}

size_t element_count(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                           [](size_t acc, int64_t dim) {
                               return acc * static_cast<size_t>(dim > 0 ? dim : 1);
                           });
}

}  // namespace

std::string ModelSession::ort_version() { return Ort::GetVersionString(); }

ModelSession::ModelSession(SessionConfig config)
    : config_(std::move(config)),
      // Label the input buffer's memory to match the session's allocator: with the
      // arena on, ORT's CPU allocator is the arena; with it off, the plain device
      // allocator. It is only a tag on a user-owned buffer, but keeping it truthful
      // avoids a needless allocator-mismatch copy inside Run.
      memory_info_(Ort::MemoryInfo::CreateCpu(
          config_.enable_cpu_arena ? OrtArenaAllocator : OrtDeviceAllocator,
          OrtMemTypeDefault)) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "wildlife_trigger");

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(config_.enable_extended_only
                                          ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED
                                          : GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(config_.intra_op_threads);

    // DisableCpuMemArena is the only way to turn the arena off; there is no
    // SetCpuMemArena(bool). Enabled is ORT's default, so we only ever call the
    // disable, never a re-enable.
    if (!config_.enable_cpu_arena) {
        options.DisableCpuMemArena();
    }

    if (!config_.optimized_model_path.empty()) {
        options.SetOptimizedModelFilePath(config_.optimized_model_path.c_str());
    }
    if (!config_.profile_prefix.empty()) {
        options.EnableProfiling(config_.profile_prefix.c_str());
    }

    try {
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(),
                                                  options);
    } catch (const Ort::Exception &error) {
        throw std::runtime_error("ONNX Runtime could not load " + config_.model_path +
                                 ": " + error.what());
    }
    allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
        throw std::runtime_error(
            "model contract: expected exactly one input and one output, got " +
            std::to_string(session_->GetInputCount()) + " and " +
            std::to_string(session_->GetOutputCount()) +
            ". This application feeds one image and reads one logit vector.");
    }

    contract_.input_name = session_->GetInputNameAllocated(0, *allocator_).get();
    contract_.output_name = session_->GetOutputNameAllocated(0, *allocator_).get();
    contract_.input_shape =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    contract_.output_shape =
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    if (contract_.input_shape.size() != 4) {
        throw std::runtime_error("model contract: input must be NCHW, got " +
                                 shape_to_string(contract_.input_shape));
    }
    if (contract_.input_shape[0] != 1) {
        // A dynamic or batched input would let ORT choose a different kernel than the
        // one benchmarked, and the application infers one frame at a time.
        throw std::runtime_error(
            "model contract: batch dimension must be exactly 1, got " +
            shape_to_string(contract_.input_shape) +
            ". A dynamic batch lets ORT pick a kernel the benchmark never measured.");
    }
    if (contract_.output_shape.size() != 2) {
        throw std::runtime_error("model contract: output must be [1, classes], got " +
                                 shape_to_string(contract_.output_shape));
    }
    contract_.class_count = contract_.output_shape[1];
    if (contract_.class_count <= 0) {
        throw std::runtime_error(
            "model contract: class count must be static and positive, got " +
            shape_to_string(contract_.output_shape) +
            ". A policy binds thresholds to class indices; a dynamic class count "
            "means an index has no fixed meaning.");
    }
}

ModelSession::~ModelSession() {
    // ORT keeps the profile file open until EndProfiling. A destructor that skipped
    // it would leave a truncated profile precisely when the run crashed -- which is
    // when the profile matters most.
    if (!profiling_ended_ && !config_.profile_prefix.empty() && session_) {
        try {
            session_->EndProfilingAllocated(*allocator_);
        } catch (...) {
            // Nothing useful to do while unwinding.
        }
    }
}

std::vector<float> ModelSession::run(const std::vector<float> &input) {
    const size_t expected = element_count(contract_.input_shape);
    if (input.size() != expected) {
        throw std::runtime_error(
            "input tensor holds " + std::to_string(input.size()) +
            " floats but the model wants " + std::to_string(expected) + " " +
            shape_to_string(contract_.input_shape));
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(input.data()), input.size(),
        contract_.input_shape.data(), contract_.input_shape.size());

    const char *input_names[] = {contract_.input_name.c_str()};
    const char *output_names[] = {contract_.output_name.c_str()};

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor,
                                 1, output_names, 1);

    const size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float *values = outputs[0].GetTensorData<float>();
    return std::vector<float>(values, values + count);
}

std::string ModelSession::end_profiling() {
    if (config_.profile_prefix.empty() || profiling_ended_) {
        return {};
    }
    profiling_ended_ = true;
    return session_->EndProfilingAllocated(*allocator_).get();
}

}  // namespace wildlife_trigger
