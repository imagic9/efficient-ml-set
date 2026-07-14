#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

std::string resolve_path(const std::string& filename) {
    namespace fs = std::filesystem;

    if (fs::exists(filename)) {
        return filename;
    }

    fs::path parent_candidate = fs::path("..") / filename;
    if (fs::exists(parent_candidate)) {
        return parent_candidate.string();
    }

    return filename;
}

std::vector<std::string> load_class_labels(const std::string& filename) {
    std::vector<std::string> labels;
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    }
    return labels;
}

std::vector<float> softmax(const float* logits, int size) {
    std::vector<float> probabilities(size);
    float max_logit = *std::max_element(logits, logits + size);
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    
    for (int i = 0; i < size; ++i) {
        probabilities[i] /= sum;
    }
    
    return probabilities;
}

int main() {
    try {
        std::string model_path = resolve_path("mobilenet_v2.onnx");
        std::string image_path = resolve_path("test_image.jpg");
        std::string labels_path = resolve_path("imagenet_classes.txt");
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        Ort::Session session(env, model_path.c_str(), session_options);
        
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << image_path << std::endl;
            return 1;
        }
        
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(224, 224));
        
        cv::Mat float_image;
        resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
        
        cv::Mat normalized_image;
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        cv::subtract(float_image, mean, normalized_image);
        cv::divide(normalized_image, std, normalized_image);
        
        std::vector<cv::Mat> channels(3);
        cv::split(normalized_image, channels);
        
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(1 * 3 * 224 * 224);
        
        for (int c = 0; c < 3; ++c) {
            input_tensor_values.insert(input_tensor_values.end(), 
                                      (float*)channels[c].data, 
                                      (float*)channels[c].data + 224 * 224);
        }
        
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );
        
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int num_classes = output_shape[1];
        
        std::vector<float> probabilities = softmax(output_data, num_classes);
        
        int max_index = 0;
        float max_probability = probabilities[0];
        for (int i = 1; i < num_classes; ++i) {
            if (probabilities[i] > max_probability) {
                max_probability = probabilities[i];
                max_index = i;
            }
        }
        
        std::vector<std::string> class_labels = load_class_labels(labels_path);
        
        std::cout << "Prediction complete!" << std::endl;
        std::cout << "Predicted class index: " << max_index << std::endl;
        if (!class_labels.empty() && max_index < class_labels.size()) {
            std::cout << "Predicted class: " << class_labels[max_index] << std::endl;
        }
        std::cout << "Confidence: " << (max_probability * 100.0f) << "%" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
