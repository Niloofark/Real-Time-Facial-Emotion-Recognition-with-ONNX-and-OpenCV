/**
 * emotion_classifier.hpp
 * Author: Niloofar Karimi
 * Description: Header file for the EmotionClassifier class.
 *              This class handles emotion prediction using a pre-trained ONNX model.
 *              Includes support for standard classification and test-time augmentation (TTA).
 */

#pragma once
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

class EmotionClassifier {
public:
    // Constructor: load ONNX model
    EmotionClassifier(const std::string& modelPath);

    // Predict emotion from a single face image
    std::string classify(const cv::Mat& faceROI);

    // Predict emotion with confidence output
    std::string classify(const cv::Mat& faceROI, float* confidence);

    // Predict emotion using test-time augmentation (flip + rotate)
    std::string classifyWithTTA(const cv::Mat& faceROI);
    std::string classifyWithTTA(const cv::Mat& faceROI, float* confidence);

private:
    cv::dnn::Net net;                    // Loaded ONNX model
    cv::Size inputSize;                  // Expected input size (width, height)
    std::vector<std::string> labels;     // Emotion labels
};
