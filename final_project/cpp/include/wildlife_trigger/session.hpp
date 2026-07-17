// ModelSession -- DESIGN §11 component 1.
//
// Owns the ORT environment and session through RAII, validates the model contract
// before the first inference rather than crashing inside it, and exposes the
// profiling and optimized-graph capture that P0 established.
//
// The contract validation is not ceremony. A policy binds a threshold to a class
// index; if a model with a different output count is loaded, index 2 silently means
// a different animal and the shutter fires on the wrong species. That failure has no
// symptom other than being wrong, so it is checked at load.

#ifndef WILDLIFE_TRIGGER_SESSION_HPP
#define WILDLIFE_TRIGGER_SESSION_HPP

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace wildlife_trigger {

struct SessionConfig {
    std::string model_path;

    // One thread by default. The Pi 5 has four cores, but DESIGN §12 benchmarks a
    // stated thread count rather than "whatever the machine did"; the default being
    // explicit keeps an unstated 4-thread run from being compared against a stated
    // 1-thread one.
    int intra_op_threads = 1;

    // DESIGN §8/§11: ORT_ENABLE_ALL is the default. ORT_ENABLE_EXTENDED is a
    // registered E6 comparison, reachable but never silently substituted.
    bool enable_extended_only = false;

    // The CPU memory arena is ORT's default and stays on for the shipping binary.
    // E6 measures it off as well (options.DisableCpuMemArena): the arena trades a
    // little steady-state memory for fewer allocations, and on a 4-frame-per-second
    // Pi workload the arena's benefit may not be worth its resident footprint. Off
    // is a measured alternative, never a silent default.
    bool enable_cpu_arena = true;

    // Empty disables. P0 proved these are the evidence that settles integer
    // execution, so the deployed binary keeps the capability.
    std::string profile_prefix;

    // Empty disables. NOTE (P0, 2026-07-15): ORT warns that a graph serialized above
    // ORT_ENABLE_EXTENDED "should only be used in the same environment the model was
    // optimized in". Persist it for inspection; never ship one as a deployable.
    std::string optimized_model_path;
};

struct ModelContract {
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t class_count = 0;
};

class ModelSession {
  public:
    explicit ModelSession(SessionConfig config);
    ~ModelSession();

    // Non-copyable: two objects must not own one ORT session. Movable, because a
    // factory returns one by value — and declaring the copy constructor deleted
    // suppresses the implicit move, so these must be spelled out or `return
    // Pipeline{...}` silently tries to copy.
    ModelSession(const ModelSession &) = delete;
    ModelSession &operator=(const ModelSession &) = delete;
    ModelSession(ModelSession &&) noexcept = default;
    ModelSession &operator=(ModelSession &&) noexcept = default;

    // Run one NCHW float32 tensor, returning raw logits. `input` must hold exactly
    // the element count the contract declares.
    std::vector<float> run(const std::vector<float> &input);

    const ModelContract &contract() const { return contract_; }
    const SessionConfig &config() const { return config_; }

    // Returns the profile's real filename, or empty when profiling was off. ORT
    // stamps its own timestamp onto the prefix, so the name is asked for, never
    // reconstructed.
    std::string end_profiling();

    static std::string ort_version();

  private:
    SessionConfig config_;
    ModelContract contract_;

    // Pimpl-free but header-order sensitive: Env must outlive Session.
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    Ort::MemoryInfo memory_info_;
    bool profiling_ended_ = false;
};

}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_SESSION_HPP
