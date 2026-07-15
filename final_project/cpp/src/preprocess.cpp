#include "wildlife_trigger/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace wildlife_trigger {

double LetterboxInfo::pixel_utilisation() const {
    const double tensor_px =
        static_cast<double>(resized_width + 2 * pad_left) *
        static_cast<double>(resized_height + 2 * pad_top);
    if (tensor_px <= 0.0) {
        return 0.0;
    }
    const double real_px = static_cast<double>(resized_width) *
                           static_cast<double>(resized_height);
    return real_px / tensor_px;
}

Preprocessor::Preprocessor(PreprocessConfig config) : config_(config) {
    if (config_.width <= 0 || config_.height <= 0) {
        throw std::invalid_argument("preprocess: width and height must be positive");
    }
    // Preallocate the canvas once; from_bgr only refills it.
    canvas_.create(config_.height, config_.width, CV_8UC3);
}

PreprocessResult Preprocessor::from_file(const std::string &path) {
    // IMREAD_COLOR gives 8-bit BGR regardless of the source's channel count, which
    // is step 1 exactly. A greyscale IR frame becomes 3 equal channels rather than
    // failing -- camera traps produce those at night, and they are ordinary input.
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error(
            "cannot decode image: " + path +
            " (missing, truncated, or not an image). A corrupt frame must be an "
            "explicit error, never a silently grey tensor.");
    }
    return from_bgr(bgr);
}

PreprocessResult Preprocessor::from_bgr(const cv::Mat &bgr) {
    if (bgr.empty()) {
        throw std::runtime_error("preprocess: empty image");
    }
    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error("preprocess: expected 8-bit 3-channel BGR");
    }

    LetterboxInfo info;
    info.source_width = bgr.cols;
    info.source_height = bgr.rows;

    // Step 3: the largest scale that fits the frame entirely inside the target.
    // min(), never max(): max() would fill the target and crop the overflow, which
    // is the centre-crop DESIGN §5.5 forbids.
    info.scale = std::min(static_cast<double>(config_.width) / bgr.cols,
                          static_cast<double>(config_.height) / bgr.rows);

    // round(), not truncate: a 1024x747 frame into 256x192 gives 186.75 rows, and
    // truncating loses a row of animal for no reason.
    info.resized_width =
        std::max(1, static_cast<int>(std::lround(bgr.cols * info.scale)));
    info.resized_height =
        std::max(1, static_cast<int>(std::lround(bgr.rows * info.scale)));

    // Clamp: rounding can exceed the target by one pixel at some aspect ratios, and
    // that would overflow the canvas.
    info.resized_width = std::min(info.resized_width, config_.width);
    info.resized_height = std::min(info.resized_height, config_.height);

    // INTER_LINEAR is the contract's interpolation and the one step whose
    // implementation could differ between OpenCV versions -- the named P1 risk.
    cv::resize(bgr, resized_, cv::Size(info.resized_width, info.resized_height), 0, 0,
               cv::INTER_LINEAR);

    // Step 4: centre-pad. Computed as a left/top offset into a prefilled canvas
    // rather than with copyMakeBorder, so the padded region is written once.
    info.pad_left = (config_.width - info.resized_width) / 2;
    info.pad_top = (config_.height - info.resized_height) / 2;

    canvas_.setTo(cv::Scalar(config_.pad_value, config_.pad_value, config_.pad_value));
    resized_.copyTo(canvas_(cv::Rect(info.pad_left, info.pad_top, info.resized_width,
                                     info.resized_height)));

    PreprocessResult result;
    result.letterbox = info;
    result.tensor.resize(static_cast<size_t>(3) * config_.height * config_.width);

    // Steps 2, 5, 6 and 7 fused into one pass. Done as a single traversal because
    // the alternative -- cvtColor, convertTo, subtract, divide, split -- allocates
    // and re-reads the frame five times, which is measurable on a Pi and buys
    // nothing in clarity.
    //
    // The channel index inversion (2 - c) is step 2: OpenCV holds BGR, the model
    // wants RGB. Getting this backwards is the "old BGR-as-RGB behaviour" PLAN E2
    // explicitly rejects, and it is invisible in every metric except accuracy.
    const size_t plane = static_cast<size_t>(config_.height) * config_.width;
    for (int y = 0; y < config_.height; ++y) {
        const uint8_t *row = canvas_.ptr<uint8_t>(y);
        for (int x = 0; x < config_.width; ++x) {
            const uint8_t *pixel = row + static_cast<size_t>(x) * 3;
            for (int c = 0; c < 3; ++c) {
                const float value = static_cast<float>(pixel[2 - c]) / 255.0F;
                const size_t index =
                    static_cast<size_t>(c) * plane +
                    static_cast<size_t>(y) * config_.width + static_cast<size_t>(x);
                result.tensor[index] =
                    (value - config_.mean[c]) / config_.stddev[c];
            }
        }
    }

    return result;
}

}  // namespace wildlife_trigger
