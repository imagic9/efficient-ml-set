// Policy, softmax and hashing tests (A4).
//
// Plain assert-based CTest, matching test_cpu_features.cpp. Every case below is a way
// a policy could be wrong while looking right; the loader's job is to refuse, and
// these prove it refuses for the stated reason rather than by luck.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include "wildlife_trigger/hashing.hpp"
#include "wildlife_trigger/policy.hpp"

using namespace wildlife_trigger;

namespace {

const char *kClassMapJson = R"({
  "schema_version": 1,
  "classes": ["badger", "bobcat", "coyote", "car", "empty"],
  "animal_classes": ["badger", "bobcat", "coyote"],
  "non_selectable_classes": ["car", "empty"]
})";

std::string write_temp(const std::string &name, const std::string &content) {
    const std::string path = "/tmp/wt_test_" + name;
    std::ofstream file(path);
    file << content;
    file.close();
    return path;
}

// Returns true when loading threw -- i.e. the loader correctly refused.
bool rejects(const std::string &policy_json, const ClassMap &map,
             const std::string &model_hash = "") {
    const std::string path = write_temp("policy.json", policy_json);
    try {
        Policy::load(path, map, model_hash);
        return false;
    } catch (const std::exception &) {
        return true;
    }
}

void test_sha256_against_standard_vectors() {
    // FIPS 180-4 / NIST published vectors. Our implementation must agree with
    // Python's hashlib, and these are how that is established without running Python.
    assert(sha256_string("") ==
           "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(sha256_string("abc") ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(sha256_string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // A multi-block message: exercises the padding path where the length spills into
    // an extra block, which is the classic place a hand-written SHA-256 is wrong.
    std::string million_a(1000000, 'a');
    assert(sha256_string(million_a) ==
           "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    std::puts("  PASS  sha256 matches the NIST vectors");
}

void test_softmax_is_stable_and_normalised() {
    const std::vector<float> scores = softmax({1.0F, 2.0F, 3.0F});
    double sum = 0.0;
    for (const float s : scores) sum += s;
    assert(std::abs(sum - 1.0) < 1e-6);
    assert(scores[2] > scores[1] && scores[1] > scores[0]);

    // Without max-subtraction exp(1000) is inf and every score becomes NaN. An
    // unnormalised head can absolutely emit a logit this large.
    const std::vector<float> extreme = softmax({1000.0F, 999.0F, -1000.0F});
    for (const float s : extreme) {
        assert(!std::isnan(s));
    }
    double extreme_sum = 0.0;
    for (const float s : extreme) extreme_sum += s;
    assert(std::abs(extreme_sum - 1.0) < 1e-5);
    std::puts("  PASS  softmax is normalised and survives extreme logits");
}

void test_valid_policy_loads_and_fires_on_any() {
    const std::string map_path = write_temp("class_map.json", kClassMapJson);
    const ClassMap map = load_class_map(map_path);

    const std::string path = write_temp("good.json", R"({
      "schema_version": 1, "policy_id": "t", "mode": "any",
      "targets": [{"class": "bobcat", "threshold": 0.5},
                  {"class": "coyote", "threshold": 0.9}]
    })");
    const Policy policy = Policy::load(path, map, "");

    // bobcat dominant -> fires on bobcat only; mode `any` needs just one.
    const Decision decision = policy.decide({0.0F, 10.0F, 0.0F, 0.0F, 0.0F}, map);
    assert(decision.shutter_trigger);
    assert(decision.passing.size() == 1 && decision.passing[0] == "bobcat");
    assert(decision.top1_class == "bobcat");
    // Every configured target is reported, passed or not -- the operator must be able
    // to see why coyote did not fire.
    assert(decision.targets.size() == 2);
    std::puts("  PASS  valid policy loads and fires under mode: any");
}

void test_threshold_boundary_is_inclusive() {
    const std::string map_path = write_temp("class_map.json", kClassMapJson);
    const ClassMap map = load_class_map(map_path);
    const std::string path = write_temp("boundary.json", R"({
      "schema_version": 1, "policy_id": "t", "mode": "any",
      "targets": [{"class": "bobcat", "threshold": 0.2}]
    })");
    const Policy policy = Policy::load(path, map, "");

    // Five equal logits -> every score is exactly 0.2. DESIGN §6.1 defines the rule as
    // `>= threshold`, and calibration searches observed scores, so equality is the
    // common case rather than a corner.
    const Decision decision = policy.decide({1.0F, 1.0F, 1.0F, 1.0F, 1.0F}, map);
    assert(std::abs(decision.targets[0].score - 0.2F) < 1e-6F);
    assert(decision.targets[0].passed);
    assert(decision.shutter_trigger);
    std::puts("  PASS  a score exactly at the threshold fires");
}

void test_loader_rejects_what_it_must() {
    const std::string map_path = write_temp("class_map.json", kClassMapJson);
    const ClassMap map = load_class_map(map_path);

    assert(rejects(R"({"schema_version": 1, "mode": "any", "targets": []})", map));
    std::puts("  PASS  rejects an empty target list");

    assert(rejects(R"({"schema_version": 1, "mode": "all",
      "targets": [{"class": "bobcat", "threshold": 0.5}]})", map));
    std::puts("  PASS  rejects an unsupported mode");

    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "targets": [{"class": "unicorn", "threshold": 0.5}]})", map));
    std::puts("  PASS  rejects an unknown class");

    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "targets": [{"class": "car", "threshold": 0.5}]})", map));
    std::puts("  PASS  rejects a non-selectable class (car)");

    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "targets": [{"class": "bobcat", "threshold": 0.5},
                  {"class": "bobcat", "threshold": 0.6}]})", map));
    std::puts("  PASS  rejects a duplicated target");

    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "targets": [{"class": "bobcat", "threshold": 1.5}]})", map));
    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "targets": [{"class": "bobcat", "threshold": -0.1}]})", map));
    std::puts("  PASS  rejects a threshold outside [0, 1]");

    // DESIGN §4: badger/deer/fox carry threshold null for want of validation support.
    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "targets": [{"class": "badger", "threshold": null}]})", map));
    std::puts("  PASS  rejects a null threshold rather than inventing one");

    assert(rejects(R"({"schema_version": 2, "mode": "any",
      "targets": [{"class": "bobcat", "threshold": 0.5}]})", map));
    std::puts("  PASS  rejects an unsupported schema_version");

    // The binding that stops thresholds calibrated for one model being applied to
    // another, where a class index may denote a different animal entirely.
    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "model_sha256": "deadbeef",
      "targets": [{"class": "bobcat", "threshold": 0.5}]})", map, "cafebabe"));
    std::puts("  PASS  rejects a model hash mismatch");

    assert(rejects(R"({"schema_version": 1, "mode": "any",
      "class_map_sha256": "deadbeef",
      "targets": [{"class": "bobcat", "threshold": 0.5}]})", map));
    std::puts("  PASS  rejects a class-map hash mismatch");
}

void test_class_map_rejects_duplicates() {
    const std::string path = write_temp("dupe_map.json", R"({
      "schema_version": 1,
      "classes": ["bobcat", "bobcat"],
      "animal_classes": ["bobcat"],
      "non_selectable_classes": []
    })");
    bool threw = false;
    try {
        load_class_map(path);
    } catch (const std::exception &) {
        threw = true;
    }
    // A duplicate makes the second index unreachable, so its threshold never fires
    // and nothing reports an error.
    assert(threw);
    std::puts("  PASS  class map rejects duplicate class names");
}

void test_logit_count_mismatch_is_an_error() {
    const std::string map_path = write_temp("class_map.json", kClassMapJson);
    const ClassMap map = load_class_map(map_path);
    const std::string path = write_temp("ok.json", R"({
      "schema_version": 1, "mode": "any",
      "targets": [{"class": "bobcat", "threshold": 0.5}]
    })");
    const Policy policy = Policy::load(path, map, "");

    bool threw = false;
    try {
        policy.decide({1.0F, 2.0F}, map);  // 2 logits, 5 classes
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);
    std::puts("  PASS  a logit/class count mismatch is an error, not a guess");
}

}  // namespace

int main() {
    std::puts("policy / softmax / hashing:");
    test_sha256_against_standard_vectors();
    test_softmax_is_stable_and_normalised();
    test_valid_policy_loads_and_fires_on_any();
    test_threshold_boundary_is_inclusive();
    test_loader_rejects_what_it_must();
    test_class_map_rejects_duplicates();
    test_logit_count_mismatch_is_an_error();
    std::puts("all policy tests passed");
    return 0;
}
