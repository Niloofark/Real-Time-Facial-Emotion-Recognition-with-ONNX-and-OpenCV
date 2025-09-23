/**
 * emotion_classifier.cpp
 * Author: Niloofar Karimi
 * Description: Implements the EmotionClassifier class for performing emotion prediction
 *              using a CNN model in ONNX format. Includes preprocessing and
 *              test-time augmentation (TTA) support.
 */

#include "emotion_classifier.hpp"
#include "config.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

// Constructor: load the ONNX model and store input shape and emotion labels
EmotionClassifier::EmotionClassifier(const std::string& modelPath)
    : inputSize(config::INPUT_WIDTH, config::INPUT_HEIGHT), labels(config::EMOTION_LABELS) {

    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        throw std::runtime_error("Failed to load ONNX model from path: " + modelPath);
    }
}

// Preprocess a single grayscale face image: center-crop, resize, normalize to [-1, 1]
cv::Mat preprocess(const cv::Mat& faceROI, const cv::Size& targetSize) {
    int cropSize = std::min(faceROI.rows, faceROI.cols);
    int offsetX = (faceROI.cols - cropSize) / 2;
    int offsetY = (faceROI.rows - cropSize) / 2;
    cv::Rect roi(offsetX, offsetY, cropSize, cropSize);
    cv::Mat cropped = faceROI(roi);

    cv::Mat resized;
    cv::resize(cropped, resized, targetSize);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0); // Normalize to [0,1]
    resized = (resized - 0.5f) / 0.5f; // Normalize to [-1,1]

    return resized;
}

// Predict the emotion from a single face image (no confidence returned)
std::string EmotionClassifier::classify(const cv::Mat& faceROI) {
    return classify(faceROI, nullptr);
}

// Predict the emotion and output the confidence of the top prediction
std::string EmotionClassifier::classify(const cv::Mat& faceROI, float* confidence) {
    cv::Mat processed = preprocess(faceROI, inputSize);

    std::vector<int> shape = {1, inputSize.height, inputSize.width, 1};
    cv::Mat blob(4, &shape[0], CV_32F);
    std::memcpy(blob.data, processed.data, processed.total() * sizeof(float));

    net.setInput(blob);
    cv::Mat output = net.forward();

    // Apply softmax to get class probabilities
    cv::Mat scores = output.reshape(1, 1);
    cv::Mat expScores;
    cv::exp(scores, expScores);
    expScores /= cv::sum(expScores)[0];

    // Find the index of the class with highest score
    cv::Point classIdPoint;
    double maxVal;
    cv::minMaxLoc(expScores, nullptr, &maxVal, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    if (confidence) {
        *confidence = static_cast<float>(maxVal);
    }

    std::string label = (classId >= 0 && classId < labels.size()) ? labels[classId] : "Unknown";
    std::cout << label << " (" << maxVal << ")" << std::endl;

    // Filter low-confidence predictions
    return (maxVal < 0.2) ? "Uncertain" : label;
}





// Classify with Test-Time Augmentation (TTA): predict using original, flipped, and rotated variants
std::string EmotionClassifier::classifyWithTTA(const cv::Mat& faceROI, float* confidence) {
    std::vector<cv::Mat> variants = {faceROI};  // Start with original face

    // Add horizontally flipped version of the face
    cv::Mat flipped;
    cv::flip(faceROI, flipped, 1);
    variants.push_back(flipped);

    // Add rotated versions: -10 degrees and +10 degrees
    for (int angle : {-10, 10}) {
        cv::Mat rotated;
        cv::Point2f center(faceROI.cols / 2.0f, faceROI.rows / 2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(faceROI, rotated, rot_mat, faceROI.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        variants.push_back(rotated);
    }

    // Initialize a zeroed score vector to accumulate softmax scores across all variants
    cv::Mat avgScores = cv::Mat::zeros(1, static_cast<int>(labels.size()), CV_32F);

    // Run inference on each variant and accumulate the softmax outputs
    for (const auto& img : variants) {
        // Preprocess (crop, resize, normalize) for input to the model
        cv::Mat processed = preprocess(img, inputSize);

        // Create 4D blob with shape [1, height, width, 1] for grayscale image
        std::vector<int> shape = {1, inputSize.height, inputSize.width, 1};
        cv::Mat blob(4, &shape[0], CV_32F);
        std::memcpy(blob.data, processed.data, processed.total() * sizeof(float));

        // Feed into ONNX model and get raw output
        net.setInput(blob);
        cv::Mat output = net.forward();

        // Reshape and apply softmax to convert scores into class probabilities
        cv::Mat scores = output.reshape(1, 1);
        cv::Mat expScores;
        cv::exp(scores, expScores);
        expScores /= cv::sum(expScores)[0];  // Softmax normalization

        // Accumulate the class probability vector
        avgScores += expScores;
    }

    // Average the scores over the number of variants
    avgScores /= static_cast<float>(variants.size());

    // Find the index of the class with the highest averaged score
    cv::Point classIdPoint;
    double maxVal;
    cv::minMaxLoc(avgScores, nullptr, &maxVal, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    // Store max confidence score if requested
    if (confidence) {
        *confidence = static_cast<float>(maxVal);
    }

    // Return the label corresponding to the predicted class, or "Uncertain" if low confidence
    std::string label = (classId >= 0 && classId < labels.size()) ? labels[classId] : "Unknown";
    std::cout << "TTA " << label << " (" << maxVal << ")" << std::endl;

    return (maxVal < 0.2) ? "Uncertain" : label;
}
