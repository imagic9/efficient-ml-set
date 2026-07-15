// The canonical preprocessing contract from DESIGN §5.5.
//
// This is the most safety-critical code in the application, and not because it is
// hard: because it is easy to get subtly wrong in a way that produces plausible
// numbers. The network must see the *complete frame* -- a centre crop can remove a
// small animal, which is precisely the event the product exists to catch -- so the
// resize preserves aspect ratio and pads.
//
// Steps, in order, all of which must match the Python implementation within a
// documented tolerance (P1):
//
//   1. decode JPEG as 8-bit BGR;
//   2. BGR -> RGB;
//   3. resize preserving aspect ratio to fit inside (width, height);
//   4. centre-pad the remainder with RGB (114, 114, 114);
//   5. float32, divide by 255;
//   6. normalise with ImageNet mean/std;
//   7. HWC RGB -> contiguous NCHW.
//
// Only step 3 can legitimately differ between implementations: everything else is
// exactly defined. That is why the OpenCV 4.6 (C++, bookworm apt) versus 4.13
// (Python wheel) gap is a named P1 risk rather than a curiosity -- INTER_LINEAR is
// the one place a version could disagree.

#ifndef WILDLIFE_TRIGGER_PREPROCESS_HPP
#define WILDLIFE_TRIGGER_PREPROCESS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace wildlife_trigger {

// DESIGN §5.5. The provisional Core input is 256x192 (width x height), pending the
// 224x224-versus-256x192 control that C1a resolves. Nothing here assumes either:
// the shape is configuration.
struct PreprocessConfig {
    int width = 256;
    int height = 192;

    // The letterbox fill. Not black: a black bar is a plausible night-time pixel
    // value, whereas mid-grey is not confusable with content.
    uint8_t pad_value = 114;

    // torchvision's ImageNet normalisation, repeated in DESIGN §5.5. The Python side
    // must carry the identical constants or P1 parity fails on a typo.
    float mean[3] = {0.485F, 0.456F, 0.406F};
    float stddev[3] = {0.229F, 0.224F, 0.225F};
};

// What the letterbox did, kept so a caller can map coordinates back to the original
// frame and so evidence records the geometry rather than re-deriving it.
struct LetterboxInfo {
    int source_width = 0;
    int source_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    double scale = 0.0;

    // Fraction of the tensor holding real pixels rather than grey bars. DESIGN §5.5
    // predicts 97.4% for 256x192 and 72.8% for 224x224 on the dominant CCT frame;
    // reporting it lets that claim be checked on real data instead of assumed.
    double pixel_utilisation() const;
};

struct PreprocessResult {
    // NCHW float32, size 1*3*height*width.
    std::vector<float> tensor;
    LetterboxInfo letterbox;
};

class Preprocessor {
  public:
    explicit Preprocessor(PreprocessConfig config);

    // Decode a JPEG from disk and run the full contract. Throws std::runtime_error
    // on a missing or undecodable file -- DESIGN §11 requires an explicit
    // corrupt-image policy, and silently returning a grey tensor would make a
    // corrupt frame indistinguishable from a legitimately empty one.
    PreprocessResult from_file(const std::string &path);

    // Steps 2-7 on an already-decoded BGR image, so the dataset runner can reuse a
    // decode and tests can supply a synthetic frame.
    PreprocessResult from_bgr(const cv::Mat &bgr);

    const PreprocessConfig &config() const { return config_; }

  private:
    PreprocessConfig config_;

    // Reused across calls. DESIGN §11 requires a reusable preallocated input buffer;
    // allocating 147 KB per frame inside the hot path is measurable on a Pi.
    cv::Mat resized_;
    cv::Mat canvas_;
};

}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_PREPROCESS_HPP
