/**
 * batch_test.cpp
 * Author: Niloofar Karimi
 * Description: Batch evaluation script to test emotion classification on a folder of grayscale images
 *              using ONNX model with test-time augmentation (TTA). Outputs predictions to a CSV
 *              and computes overall accuracy.
 */

#ifdef RUN_BATCH

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>

#include "config.hpp"
#include "emotion_classifier.hpp"

namespace fs = std::filesystem;

// Normalize various label forms to a consistent format (e.g., "angry" → "Anger")
std::string normalize_label(const std::string& raw_label) {
    static const std::map<std::string, std::string> label_map = {
        {"angry",    "Anger"},
        {"anger",    "Anger"},
        {"disgust",  "Disgust"},
        {"fear",     "Fear"},
        {"happy",    "Happiness"},
        {"happiness","Happiness"},
        {"neutral",  "Neutral"},
        {"sad",      "Sadness"},
        {"sadness",  "Sadness"},
        {"surprise", "Surprise"}
    };

    std::string lower = raw_label;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    auto it = label_map.find(lower);
    return it != label_map.end() ? it->second : raw_label;
}

int main() {
    try {
        // Initialize classifier with ONNX model path
        EmotionClassifier classifier(config::MODEL_PATH);

        std::ofstream log("results1.csv");
        log << "Image,TrueLabel,Predicted\n";

        // Directory with test images (subfolders as class labels)
        std::string testDir = "/Users/niloofarkarimi/CV_Final/test_images";
        if (!fs::exists(testDir)) {
            std::cerr << "Directory '" << testDir << "' does not exist.\n";
            return 1;
        }

        // Loop through all images recursively in test folder
        for (const auto& entry : fs::recursive_directory_iterator(testDir)) {
            if (!entry.is_regular_file()) continue;

            std::string filepath = entry.path().string();
            cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Failed to read image: " << filepath << std::endl;
                continue;
            }

            // Improve contrast for better detection
            cv::equalizeHist(img, img);

            // Predict label using test-time augmentation
            std::string predicted_label = classifier.classifyWithTTA(img);
            std::string true_label = normalize_label(entry.path().parent_path().filename().string());

            // Log predictions to CSV and print to console
            log << entry.path().filename().string() << "," << true_label << "," << predicted_label << "\n";
            std::cout << entry.path().filename().string()
                      << " | True: " << true_label
                      << " | Predicted: " << predicted_label << std::endl;
        }

        log.close();
        std::cout << "Results written to results1.csv\n";

        // =============================
        // Compute overall classification accuracy
        // =============================
        std::ifstream infile("results1.csv");
        std::string line;
        int correct = 0, total = 0;
        std::getline(infile, line); // Skip header

        while (std::getline(infile, line)) {
            std::istringstream ss(line);
            std::string image, true_label, predicted_label;

            std::getline(ss, image, ',');
            std::getline(ss, true_label, ',');
            std::getline(ss, predicted_label, ',');

            if (normalize_label(true_label) == normalize_label(predicted_label))
                ++correct;
            ++total;
        }

        infile.close();
        double accuracy = total > 0 ? (double)correct / total * 100.0 : 0.0;
        std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << " correct predictions)\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#endif // RUN_BATCH
