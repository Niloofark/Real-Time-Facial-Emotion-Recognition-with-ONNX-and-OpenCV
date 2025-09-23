/**
 * config.hpp
 * Author: Niloofar Karimi
 * Description: Configuration constants used across the facial emotion recognition system.
 *              Includes paths, image size, and emotion label definitions.
 */

#pragma once
#include <string>
#include <vector>

namespace config {

    // Path to the ONNX emotion classification model
    const std::string MODEL_PATH = "/Users/niloofarkarimi/CV_Final/models/mini_xception.onnx";

    // Path to Haar cascade XML for face detection
    const std::string FACE_CASCADE_PATH = "/Users/niloofarkarimi/CV_Final/resources/haarcascade_frontalface_default.xml";

    // Input dimensions expected by the ONNX model
    const int INPUT_WIDTH = 64;
    const int INPUT_HEIGHT = 64;

    // Emotion label list in the order expected by the model
    const std::vector<std::string> EMOTION_LABELS = {
        "Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
    };

    // Minimum confidence threshold for valid detection (if DNN face detection is used)
    const float CONFIDENCE_THRESHOLD = 0.5;

}
