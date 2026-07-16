// Preprocessing tests (A4) -- DESIGN §5.5's contract, asserted.
//
// These check the properties that a wrong implementation would still "work" without:
// the whole frame survives, the padding is where it should be, the channels are RGB
// and not BGR, and the normalisation is the one torchvision applied.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include <opencv2/core.hpp>

#include "wildlife_trigger/preprocess.hpp"

using namespace wildlife_trigger;

namespace {

PreprocessConfig config_256x192() {
    PreprocessConfig config;
    config.width = 256;
    config.height = 192;
    return config;
}

float tensor_at(const PreprocessResult &result, const PreprocessConfig &config, int c,
                int y, int x) {
    const size_t plane = static_cast<size_t>(config.height) * config.width;
    return result.tensor[static_cast<size_t>(c) * plane +
                         static_cast<size_t>(y) * config.width + x];
}

// The value a constant uint8 channel becomes after /255 and ImageNet normalisation.
float normalised(uint8_t raw, int channel, const PreprocessConfig &config) {
    return (static_cast<float>(raw) / 255.0F - config.mean[channel]) /
           config.stddev[channel];
}

void test_tensor_shape_and_letterbox_geometry() {
    const auto config = config_256x192();
    Preprocessor preprocessor(config);

    // The dominant CCT-20 `_sm` frame (DESIGN §5.5): 1024x747.
    cv::Mat frame(747, 1024, CV_8UC3, cv::Scalar(10, 20, 30));
    const PreprocessResult result = preprocessor.from_bgr(frame);

    assert(result.tensor.size() == static_cast<size_t>(3) * 192 * 256);

    // 1024 -> 256 exactly (scale 0.25); 747 * 0.25 = 186.75 -> 187 rows.
    assert(result.letterbox.resized_width == 256);
    assert(result.letterbox.resized_height == 187);
    assert(result.letterbox.pad_left == 0);
    assert(result.letterbox.pad_top == 2);  // (192 - 187) / 2

    // DESIGN §5.5 predicts 97.4% utilisation for this frame at 256x192: 256*187
    // real pixels in the full 256*192 canvas.
    //
    // Asserted exactly. The 0.96-0.99 bound this replaces was wide enough to pass a
    // denominator that read `resized + 2 * pad` — one pixel short of the canvas here,
    // because the 5 padding rows split 2 top and 3 bottom — and so returned 97.9%
    // while DESIGN said 97.4%. A tolerance wider than the error hides the error.
    const double utilisation = result.letterbox.pixel_utilisation();
    const double expected = (256.0 * 187.0) / (256.0 * 192.0);
    assert(std::fabs(utilisation - expected) < 1e-12);
    assert(utilisation > 0.9739 && utilisation < 0.9741);
    std::printf("  PASS  1024x747 -> 256x187 + pad, utilisation %.1f%%\n",
                utilisation * 100.0);
}

void test_square_input_wastes_a_quarter_of_the_tensor() {
    // The measured claim behind choosing 256x192 over 224x224. If this ever stops
    // holding, the input-shape decision needs revisiting rather than assuming.
    PreprocessConfig config;
    config.width = 224;
    config.height = 224;
    Preprocessor preprocessor(config);

    cv::Mat frame(747, 1024, CV_8UC3, cv::Scalar(10, 20, 30));
    const PreprocessResult result = preprocessor.from_bgr(frame);

    assert(result.letterbox.resized_width == 224);
    assert(result.letterbox.resized_height == 163);  // 747 * (224/1024) = 163.4
    const double utilisation = result.letterbox.pixel_utilisation();
    const double expected = (224.0 * 163.0) / (224.0 * 224.0);  // DESIGN says 72.8%
    assert(std::fabs(utilisation - expected) < 1e-12);
    assert(utilisation > 0.7276 && utilisation < 0.7278);
    std::printf("  PASS  224x224 wastes %.1f%% of the tensor on grey bars\n",
                (1.0 - utilisation) * 100.0);
}

void test_channels_are_rgb_not_bgr() {
    const auto config = config_256x192();
    Preprocessor preprocessor(config);

    // A pure-blue frame in OpenCV's BGR order.
    cv::Mat frame(192, 256, CV_8UC3, cv::Scalar(255, 0, 0));
    const PreprocessResult result = preprocessor.from_bgr(frame);

    // The model wants RGB, so channel 2 must carry the 255 and channel 0 the 0.
    // Getting this backwards is the "old BGR-as-RGB behaviour" PLAN E2 rejects: it
    // is invisible in every metric except accuracy, and only on colour images.
    const float r = tensor_at(result, config, 0, 96, 128);
    const float b = tensor_at(result, config, 2, 96, 128);
    assert(std::abs(r - normalised(0, 0, config)) < 1e-5F);
    assert(std::abs(b - normalised(255, 2, config)) < 1e-5F);
    std::puts("  PASS  BGR is converted to RGB, not passed through");
}

void test_padding_uses_the_configured_grey() {
    PreprocessConfig config;
    config.width = 256;
    config.height = 192;
    Preprocessor preprocessor(config);

    // A tall frame: padding lands left and right.
    cv::Mat frame(400, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    const PreprocessResult result = preprocessor.from_bgr(frame);
    assert(result.letterbox.pad_left > 0);

    // The leftmost column is pad. 114 normalised, per channel.
    for (int c = 0; c < 3; ++c) {
        const float value = tensor_at(result, config, c, 96, 0);
        assert(std::abs(value - normalised(114, c, config)) < 1e-5F);
    }
    std::puts("  PASS  letterbox padding is the configured grey, normalised");
}

void test_whole_frame_survives_no_crop() {
    const auto config = config_256x192();
    Preprocessor preprocessor(config);

    // A white frame with black corners. A centre crop would discard them -- and a
    // small animal at the frame edge with it, which is the failure DESIGN §5.5
    // forbids by name.
    cv::Mat frame(747, 1024, CV_8UC3, cv::Scalar(255, 255, 255));
    frame.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 0);
    frame.at<cv::Vec3b>(746, 1023) = cv::Vec3b(0, 0, 0);

    const PreprocessResult result = preprocessor.from_bgr(frame);
    const int top = result.letterbox.pad_top;
    const int bottom = top + result.letterbox.resized_height - 1;

    // The corners map to the first and last real rows; interpolation blends them with
    // white neighbours, so assert they are merely darker than mid-grey rather than
    // exactly black.
    const float top_left = tensor_at(result, config, 0, top, 0);
    const float bottom_right =
        tensor_at(result, config, 0, bottom, result.letterbox.resized_width - 1);
    const float white = normalised(255, 0, config);
    assert(top_left < white);
    assert(bottom_right < white);
    std::puts("  PASS  frame corners survive: aspect-preserving fit, never a crop");
}

void test_corrupt_input_is_an_error() {
    const auto config = config_256x192();
    Preprocessor preprocessor(config);

    bool threw = false;
    try {
        preprocessor.from_file("/tmp/wt_definitely_not_an_image_4a3b.jpg");
    } catch (const std::exception &) {
        threw = true;
    }
    // A corrupt frame must not silently become a grey tensor: that would be
    // indistinguishable from a legitimately empty night frame.
    assert(threw);
    std::puts("  PASS  an undecodable image raises rather than returning grey");
}

// A structured BGR frame with gradients for the resize to interpolate. Flat
// colour would make reference-vs-fused trivially equal even if one of them
// mangled the interpolation.
cv::Mat structured_frame(int width, int height) {
    cv::Mat bgr(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto &pixel = bgr.at<cv::Vec3b>(y, x);
            pixel[0] = static_cast<uint8_t>((x * 255) / std::max(1, width - 1));
            pixel[1] = static_cast<uint8_t>((y * 255) / std::max(1, height - 1));
            pixel[2] = static_cast<uint8_t>(((x + y) * 127) / std::max(1, width + height - 2));
        }
    }
    return bgr;
}

void test_reference_and_fused_agree() {
    // DESIGN §11's reference implementation exists so that a P1 disagreement can be
    // localised: same OpenCV, same host, so the two C++ paths must agree to float
    // rounding -- anything beyond ~1 ULP of the arithmetic is a fusion bug, not a
    // version gap. Geometries chosen to exercise every pad case, including the odd
    // difference that puts the extra pixel on the far side and left-pads (portrait).
    const auto config = config_256x192();
    Preprocessor preprocessor(config);

    const int geometries[][2] = {
        {1024, 747},  // dominant CCT landscape: pad top/bottom, odd difference
        {747, 1024},  // portrait: pad_left > 0, which no real CCT fixture exercises
        {512, 512},   // square
        {1023, 767},  // odd dimensions
        {100, 80},    // tiny: upscale path
        {256, 192},   // exact fit: no resize, no pad
    };
    for (const auto &geometry : geometries) {
        const cv::Mat bgr = structured_frame(geometry[0], geometry[1]);
        const PreprocessResult fused = preprocessor.from_bgr(bgr);
        const PreprocessResult reference = preprocessor.from_bgr_reference(bgr);

        assert(fused.tensor.size() == reference.tensor.size());
        assert(fused.letterbox.resized_width == reference.letterbox.resized_width);
        assert(fused.letterbox.resized_height == reference.letterbox.resized_height);
        assert(fused.letterbox.pad_left == reference.letterbox.pad_left);
        assert(fused.letterbox.pad_top == reference.letterbox.pad_top);

        float max_abs = 0.0F;
        for (size_t i = 0; i < fused.tensor.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(fused.tensor[i] - reference.tensor[i]));
        }
        // 1e-6 is the pre-registered same-host gate (DESIGN §10): both paths do
        // /255 then (x-mean)/std in float32 on identical uint8 letterboxes.
        assert(max_abs <= 1e-6F);
    }
    std::puts("  PASS  reference and fused paths agree to 1e-6 on every geometry");
}

void test_reference_rejects_what_fused_rejects() {
    const auto config = config_256x192();
    Preprocessor preprocessor(config);
    bool threw = false;
    try {
        preprocessor.from_bgr_reference(cv::Mat());
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);
    std::puts("  PASS  the reference path refuses an empty image too");
}

}  // namespace

int main() {
    std::puts("preprocessing (DESIGN §5.5):");
    test_tensor_shape_and_letterbox_geometry();
    test_square_input_wastes_a_quarter_of_the_tensor();
    test_channels_are_rgb_not_bgr();
    test_padding_uses_the_configured_grey();
    test_whole_frame_survives_no_crop();
    test_corrupt_input_is_an_error();
    test_reference_and_fused_agree();
    test_reference_rejects_what_fused_rejects();
    std::puts("all preprocessing tests passed");
    return 0;
}
