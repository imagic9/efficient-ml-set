#include "wildlife_trigger/policy.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "wildlife_trigger/hashing.hpp"

namespace wildlife_trigger {
namespace {

using nlohmann::json;

constexpr int kSupportedSchemaVersion = 1;

json read_json(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open JSON file: " + path);
    }
    try {
        return json::parse(file);
    } catch (const json::parse_error &error) {
        throw std::runtime_error("malformed JSON in " + path + ": " + error.what());
    }
}

void require_schema_version(const json &document, const std::string &path) {
    if (!document.contains("schema_version")) {
        throw std::runtime_error(path + ": missing schema_version");
    }
    const auto version = document.at("schema_version").get<int>();
    if (version != kSupportedSchemaVersion) {
        throw std::runtime_error(path + ": schema_version " + std::to_string(version) +
                                 " is not supported (this build reads " +
                                 std::to_string(kSupportedSchemaVersion) + ")");
    }
}

}  // namespace

int ClassMap::index_of(const std::string &name) const {
    const auto it = std::find(classes.begin(), classes.end(), name);
    return it == classes.end() ? -1 : static_cast<int>(it - classes.begin());
}

bool ClassMap::is_animal(const std::string &name) const {
    return std::find(animal_classes.begin(), animal_classes.end(), name) !=
           animal_classes.end();
}

ClassMap load_class_map(const std::string &path) {
    const json document = read_json(path);
    require_schema_version(document, path);

    ClassMap map;
    map.classes = document.at("classes").get<std::vector<std::string>>();
    map.animal_classes = document.at("animal_classes").get<std::vector<std::string>>();
    map.non_selectable_classes =
        document.at("non_selectable_classes").get<std::vector<std::string>>();

    if (map.classes.empty()) {
        throw std::runtime_error(path + ": class list is empty");
    }
    const std::set<std::string> unique(map.classes.begin(), map.classes.end());
    if (unique.size() != map.classes.size()) {
        // A duplicate name makes index_of() return the first match, so one of the two
        // classes becomes unreachable and its threshold silently never fires.
        throw std::runtime_error(path + ": class list contains duplicates");
    }

    // Hash the file's exact bytes: the policy binds to this, so it must be what is on
    // disk rather than a re-serialisation of the parsed object.
    map.sha256 = sha256_file(path);
    return map;
}

Policy Policy::load(const std::string &path, const ClassMap &class_map,
                    const std::string &model_sha256) {
    const json document = read_json(path);
    require_schema_version(document, path);

    Policy policy;
    policy.policy_id = document.value("policy_id", std::string("<unnamed>"));
    policy.mode_ = document.at("mode").get<std::string>();

    if (policy.mode_ != "any") {
        throw std::runtime_error(path + ": mode '" + policy.mode_ +
                                 "' is unsupported. DESIGN §4 defines 'any' as the "
                                 "only Core combination mode.");
    }

    // The hash bindings. These are the checks that stop a policy calibrated for one
    // model from being applied to another, where the same class index means a
    // different animal.
    const auto expected_model = document.value("model_sha256", std::string());
    if (!expected_model.empty() && !model_sha256.empty() &&
        expected_model != model_sha256) {
        throw std::runtime_error(
            path + ": policy was calibrated for model " + expected_model.substr(0, 16) +
            "... but the loaded model is " + model_sha256.substr(0, 16) +
            ".... Thresholds are model-specific (DESIGN §6.3: quantization changes "
            "score distributions), and class indices may not even agree.");
    }
    const auto expected_class_map = document.value("class_map_sha256", std::string());
    if (!expected_class_map.empty() && expected_class_map != class_map.sha256) {
        throw std::runtime_error(
            path + ": policy expects class map " + expected_class_map.substr(0, 16) +
            "... but the loaded map is " + class_map.sha256.substr(0, 16) +
            ".... A threshold would be applied to the wrong class index.");
    }

    if (!document.contains("targets") || !document.at("targets").is_array()) {
        throw std::runtime_error(path + ": missing targets array");
    }
    const auto &targets = document.at("targets");
    if (targets.empty()) {
        throw std::runtime_error(
            path + ": target list is empty. A policy that selects nothing can never "
            "fire; that is a configuration mistake, not a valid 'off' switch.");
    }

    std::set<std::string> seen;
    for (const auto &entry : targets) {
        PolicyTarget target;
        target.class_name = entry.at("class").get<std::string>();

        if (!entry.contains("threshold") || entry.at("threshold").is_null()) {
            // DESIGN §4: badger/deer/fox carry threshold: null because validation
            // support cannot define an operating point for them.
            throw std::runtime_error(
                path + ": target '" + target.class_name +
                "' has no threshold. Classes without a calibrated operating point "
                "must be rejected, never given an invented number.");
        }
        target.threshold = entry.at("threshold").get<float>();

        if (!seen.insert(target.class_name).second) {
            throw std::runtime_error(path + ": target '" + target.class_name +
                                     "' is listed twice");
        }
        target.class_index = class_map.index_of(target.class_name);
        if (target.class_index < 0) {
            throw std::runtime_error(path + ": target '" + target.class_name +
                                     "' is not a class this model knows");
        }
        if (!class_map.is_animal(target.class_name)) {
            throw std::runtime_error(
                path + ": target '" + target.class_name +
                "' is a model class but not a selectable wildlife target "
                "(DESIGN §4 excludes car and empty).");
        }
        if (!(target.threshold >= 0.0F && target.threshold <= 1.0F)) {
            // Written as >= && <= so NaN fails: NaN >= x is false, and every
            // comparison against a NaN threshold would return false forever, i.e. a
            // trigger that never fires and never errors.
            throw std::runtime_error(path + ": threshold for '" + target.class_name +
                                     "' is outside [0, 1]");
        }
        policy.targets_.push_back(target);
    }

    return policy;
}

std::vector<float> softmax(const std::vector<float> &logits) {
    std::vector<float> result(logits.size());
    if (logits.empty()) {
        return result;
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        // Subtract the max before exp: exp(88) already overflows float32, and an
        // unbounded logit is entirely legal output from an unnormalised head.
        result[i] = std::exp(logits[i] - max_logit);
        sum += result[i];
    }
    for (auto &value : result) {
        value = static_cast<float>(value / sum);
    }
    return result;
}

Decision Policy::decide(const std::vector<float> &logits,
                        const ClassMap &class_map) const {
    if (logits.size() != class_map.classes.size()) {
        throw std::runtime_error(
            "model produced " + std::to_string(logits.size()) +
            " logits but the class map declares " +
            std::to_string(class_map.classes.size()) +
            " classes. Every class index would be wrong.");
    }

    const std::vector<float> scores = softmax(logits);

    Decision decision;
    const auto top = std::max_element(scores.begin(), scores.end());
    decision.top1_index = static_cast<int>(top - scores.begin());
    decision.top1_score = *top;
    decision.top1_class = class_map.classes[decision.top1_index];

    for (const auto &target : targets_) {
        TargetScore score;
        score.class_name = target.class_name;
        score.class_index = target.class_index;
        score.score = scores[target.class_index];
        score.threshold = target.threshold;

        // >=, not >. DESIGN §6.1 writes the rule as `>= threshold`, and the
        // calibration searches unique observed scores, so a score exactly equal to
        // the chosen threshold is the common case, not an edge case.
        score.passed = score.score >= target.threshold;

        if (score.passed) {
            decision.shutter_trigger = true;  // mode: any
            decision.passing.push_back(target.class_name);
        }
        decision.targets.push_back(score);
    }

    return decision;
}

}  // namespace wildlife_trigger
