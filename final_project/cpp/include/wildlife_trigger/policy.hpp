// Policy -- DESIGN §11 component 3, implementing DESIGN §4/§6.1.
//
//     fire(frame, T) = any(softmax(logits)[class] >= threshold[class] for class in T)
//
// A policy is configuration, not a model. The same network serves every target
// selection; choosing to photograph coyotes as well as bobcats is a JSON edit, not a
// retrain. `mode: any` is the only Core combination mode.
//
// This loader is deliberately hostile to its input. Every rejection below
// corresponds to a way a plausible-looking policy silently does the wrong thing:
//
//   - an unknown or duplicate class, or a non-animal one (`car`, `empty`), means the
//     operator believes they configured something they did not;
//   - a class with no calibrated threshold (`badger`, `deer`, `fox` -- DESIGN §4
//     gives them null for want of validation support) would otherwise get an
//     invented number, and an invented threshold is indistinguishable from a
//     measured one once it is in a file;
//   - a model or class-map hash mismatch means the thresholds were calibrated
//     against a *different* model, so index 2 no longer denotes the animal the
//     operator chose. The shutter would fire, confidently, on the wrong species.
//
// Parsing happens once at startup and never in the per-frame hot path.

#ifndef WILDLIFE_TRIGGER_POLICY_HPP
#define WILDLIFE_TRIGGER_POLICY_HPP

#include <map>
#include <string>
#include <vector>

namespace wildlife_trigger {

struct ClassMap {
    std::vector<std::string> classes;
    std::vector<std::string> animal_classes;
    std::vector<std::string> non_selectable_classes;
    std::string sha256;

    int index_of(const std::string &name) const;
    bool is_animal(const std::string &name) const;
};

struct PolicyTarget {
    std::string class_name;
    int class_index = -1;
    float threshold = 0.0F;
};

struct TargetScore {
    std::string class_name;
    int class_index = -1;
    float score = 0.0F;
    float threshold = 0.0F;
    bool passed = false;
};

struct Decision {
    bool shutter_trigger = false;
    std::vector<TargetScore> targets;   // every configured target, passed or not
    std::vector<std::string> passing;   // just those that fired
    std::string top1_class;
    int top1_index = -1;
    float top1_score = 0.0F;
};

// Loads and validates the class map. `expected_sha256` empty skips the binding check;
// the CLI always supplies it.
ClassMap load_class_map(const std::string &path);

class Policy {
  public:
    // Throws std::runtime_error with a specific reason on any invalid policy.
    // `model_sha256` binds the policy to the model actually loaded.
    static Policy load(const std::string &path, const ClassMap &class_map,
                       const std::string &model_sha256);

    // softmax over all logits, then the `any` rule over configured targets.
    Decision decide(const std::vector<float> &logits, const ClassMap &class_map) const;

    const std::string &policy_id() const { return policy_id_; }
    const std::vector<PolicyTarget> &targets() const { return targets_; }

  private:
    std::string policy_id_;
    std::string mode_;
    std::vector<PolicyTarget> targets_;
};

// Numerically stable softmax. Exposed for tests: the max-subtraction is the only
// reason a 16-way softmax over unbounded logits does not overflow.
std::vector<float> softmax(const std::vector<float> &logits);

}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_POLICY_HPP
