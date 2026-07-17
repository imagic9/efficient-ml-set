#include "wildlife_trigger/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace wildlife_trigger {

double LetterboxInfo::pixel_utilisation() const {
    // Denominator is the real canvas, not `resized + 2 * pad`: the pads are
    // floor-divided, so an odd difference leaves that product one pixel short and
    // inflates the result. Mirrors data/preprocess.py.
    const double tensor_px = static_cast<double>(target_width) *
                             static_cast<double>(target_height);
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
    if (config_.decode_reduction != 1 && config_.decode_reduction != 2 &&
        config_.decode_reduction != 4) {
        throw std::invalid_argument(
            "preprocess: decode_reduction must be 1, 2, or 4 (OpenCV only exposes "
            "IMREAD_REDUCED_COLOR_2/4/8; 8 is out of scope here)");
    }
    // Preallocate the canvas once; from_bgr only refills it.
    canvas_.create(config_.height, config_.width, CV_8UC3);
}

cv::Mat Preprocessor::decode(const std::string &path) const {
    // IMREAD_COLOR gives 8-bit BGR regardless of the source's channel count, which
    // is step 1 exactly. A greyscale IR frame becomes 3 equal channels rather than
    // failing -- camera traps produce those at night, and they are ordinary input.
    //
    // IMREAD_REDUCED_COLOR_2/4 ask libjpeg to emit the image at 1/2 or 1/4 each side
    // straight from the DCT coefficients, which is far cheaper than a full decode
    // followed by a resize. The result is still 8-bit BGR, so the rest of the
    // contract is unchanged -- only the source resolution the letterbox sees shrinks.
    int flag = cv::IMREAD_COLOR;
    if (config_.decode_reduction == 2) flag = cv::IMREAD_REDUCED_COLOR_2;
    else if (config_.decode_reduction == 4) flag = cv::IMREAD_REDUCED_COLOR_4;

    cv::Mat bgr = cv::imread(path, flag);
    if (bgr.empty()) {
        throw std::runtime_error(
            "cannot decode image: " + path +
            " (missing, truncated, or not an image). A corrupt frame must be an "
            "explicit error, never a silently grey tensor.");
    }
    return bgr;
}

PreprocessResult Preprocessor::from_file(const std::string &path) {
    return from_bgr(decode(path));
}

LetterboxInfo Preprocessor::compute_letterbox(const cv::Mat &bgr) const {
    LetterboxInfo info;
    info.source_width = bgr.cols;
    info.source_height = bgr.rows;
    info.target_width = config_.width;
    info.target_height = config_.height;

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

    // Step 4's offsets: centre, floor-divided, extra pixel on the far side.
    info.pad_left = (config_.width - info.resized_width) / 2;
    info.pad_top = (config_.height - info.resized_height) / 2;
    return info;
}

PreprocessResult Preprocessor::from_bgr(const cv::Mat &bgr) {
    if (bgr.empty()) {
        throw std::runtime_error("preprocess: empty image");
    }
    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error("preprocess: expected 8-bit 3-channel BGR");
    }

    LetterboxInfo info = compute_letterbox(bgr);

    // INTER_LINEAR is the contract's interpolation and the one step whose
    // implementation could differ between OpenCV versions -- the named P1 risk.
    cv::resize(bgr, resized_, cv::Size(info.resized_width, info.resized_height), 0, 0,
               cv::INTER_LINEAR);

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

PreprocessResult Preprocessor::from_file_reference(const std::string &path) const {
    return from_bgr_reference(decode(path));
}

PreprocessResult Preprocessor::from_bgr_reference(const cv::Mat &bgr) const {
    // DESIGN §11's "correct reference implementation": every step is its own named
    // OpenCV primitive, in the contract's order, with nothing shared with the fused
    // path except the geometry arithmetic. Slow on purpose -- five allocations and
    // five traversals -- because its job is to be *obviously* the spec, so that
    // when P1 compares Python, reference and fused, a disagreement localises: all
    // three agree = contract holds; fused alone differs = fusion bug; both C++
    // differ from Python = cross-version resize gap (the named P1 risk).
    if (bgr.empty()) {
        throw std::runtime_error("preprocess: empty image");
    }
    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error("preprocess: expected 8-bit 3-channel BGR");
    }

    const LetterboxInfo info = compute_letterbox(bgr);

    // Step 3: aspect-preserving INTER_LINEAR resize (still BGR).
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(info.resized_width, info.resized_height), 0, 0,
               cv::INTER_LINEAR);

    // Step 4: centre-pad with copyMakeBorder, the primitive the fused path avoids.
    // The far side carries the extra pixel when the difference is odd, exactly as
    // the offset arithmetic produces it.
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, info.pad_top,
                       config_.height - info.resized_height - info.pad_top,
                       info.pad_left,
                       config_.width - info.resized_width - info.pad_left,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(config_.pad_value, config_.pad_value,
                                  config_.pad_value));

    // Step 2: BGR -> RGB, as its own pass over the padded canvas.
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // Step 5: float32, divide by 255.
    cv::Mat scaled;
    rgb.convertTo(scaled, CV_32FC3, 1.0 / 255.0);

    // Step 6: ImageNet normalisation, channel by channel.
    cv::subtract(scaled,
                 cv::Scalar(config_.mean[0], config_.mean[1], config_.mean[2]),
                 scaled);
    cv::divide(scaled,
               cv::Scalar(config_.stddev[0], config_.stddev[1], config_.stddev[2]),
               scaled);

    // Step 7: HWC -> planar NCHW via split, the layout primitive.
    std::vector<cv::Mat> planes(3);
    cv::split(scaled, planes);

    PreprocessResult result;
    result.letterbox = info;
    result.tensor.resize(static_cast<size_t>(3) * config_.height * config_.width);
    const size_t plane = static_cast<size_t>(config_.height) * config_.width;
    for (int c = 0; c < 3; ++c) {
        // Each split plane is a freshly allocated CV_32FC1, hence continuous; the
        // check is cheap and turns a silent layout assumption into an error.
        if (!planes[c].isContinuous()) {
            throw std::runtime_error("preprocess: split plane is not continuous");
        }
        std::copy(planes[c].ptr<float>(0), planes[c].ptr<float>(0) + plane,
                  result.tensor.begin() + static_cast<long>(c) * plane);
    }
    return result;
}

}  // namespace wildlife_trigger
