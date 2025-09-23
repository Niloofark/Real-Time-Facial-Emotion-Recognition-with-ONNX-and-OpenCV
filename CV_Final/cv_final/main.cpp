/**
 * main.cpp
 * Author: Niloofar Karimi
 * Description: Real-time facial emotion recognition pipeline using OpenCV and ONNX.
 *              Captures webcam input, performs face detection and alignment,
 *              runs emotion classification with optional test-time augmentation (TTA),
 *              applies smoothing and confidence filtering, and logs results to CSV.
 */

#ifdef RUN_REALTIME

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <deque>
#include <unordered_map>
#include <fstream>  // For saving results to CSV

#include "config.hpp"
#include "face_detector.hpp"
#include "emotion_classifier.hpp"
#include "video_overlay.hpp"
#include "utils.hpp"

const int SMOOTHING_WINDOW = 5;

// Get the most frequent prediction label from a rolling buffer
std::string getSmoothedPrediction(const std::deque<std::string>& buffer) {
    std::unordered_map<std::string, int> freq;
    for (const auto& label : buffer) freq[label]++;

    std::string majorityLabel;
    int maxCount = 0;
    for (const auto& [label, count] : freq) {
        if (count > maxCount) {
            maxCount = count;
            majorityLabel = label;
        }
    }
    return majorityLabel;
}

// Align face using detected eyes (only applies if exactly 2 eyes found)
cv::Mat alignFace(const cv::Mat& faceROI, cv::CascadeClassifier& eye_cascade) {
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, cv::Size(20, 20));
    if (eyes.size() != 2) return faceROI;

    // Calculate center points of both eyes
    cv::Point2f eye1(eyes[0].x + eyes[0].width / 2.0f, eyes[0].y + eyes[0].height / 2.0f);
    cv::Point2f eye2(eyes[1].x + eyes[1].width / 2.0f, eyes[1].y + eyes[1].height / 2.0f);
    if (eye2.x < eye1.x) std::swap(eye1, eye2);

    // Compute rotation angle between the eyes
    double dx = eye2.x - eye1.x;
    double dy = eye2.y - eye1.y;
    double angle = atan2(dy, dx) * 180.0 / CV_PI;

    // Rotate face to horizontally align eyes
    cv::Point2f center(faceROI.cols / 2.0f, faceROI.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

    cv::Mat aligned;
    cv::warpAffine(faceROI, aligned, rot_mat, faceROI.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return aligned;
}

int main() {
    try {
        // Open webcam
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open webcam." << std::endl;
            return -1;
        }

        // Initialize classifier and detectors
        EmotionClassifier classifier(config::MODEL_PATH);
        FaceDetector detector(config::FACE_CASCADE_PATH);

        // Load eye detector for face alignment
        cv::CascadeClassifier eye_cascade;
        if (!eye_cascade.load("/Users/niloofarkarimi/CV_Final/resources/haarcascade_eye.xml")) {
            std::cerr << "Error: Could not load eye cascade." << std::endl;
            return -1;
        }

        std::deque<std::string> predictionBuffer;
        bool useTTA = false;
        cv::Mat frame;

        // Open CSV file to save frame-by-frame results
        std::ofstream csvFile("results.csv");
        csvFile << "Frame,Emotion,Confidence,TTA\n";
        int frameCount = 0;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            // Convert to grayscale for detection
            cv::Mat gray = Utils::toGrayscale(frame);
            std::vector<cv::Rect> faces = detector.detect(gray);

            std::vector<std::string> smoothedLabels;
            std::vector<float> confidences;

            for (const auto& face : faces) {
                // Align the detected face using eyes
                cv::Mat faceROI = gray(face).clone();
                cv::Mat aligned = alignFace(faceROI, eye_cascade);

                float confidence = 0.0f;
                std::string emotion;

                // Run emotion classification (with or without TTA)
                if (useTTA) {
                    emotion = classifier.classifyWithTTA(aligned, &confidence);
                } else {
                    emotion = classifier.classify(aligned, &confidence);
                }

                // Filter low-confidence predictions
                if (confidence < 0.2f) {
                    emotion = "Uncertain";
                }

                // Log to console and to CSV
                std::string prefix = useTTA ? "TTA " : "";
                std::cout << prefix << emotion << " (" << confidence << ")" << std::endl;
                csvFile << frameCount << "," << emotion << "," << confidence << "," << (useTTA ? "Yes" : "No") << "\n";

                // Add to smoothing buffer
                predictionBuffer.push_back(emotion);
                if (predictionBuffer.size() > SMOOTHING_WINDOW)
                    predictionBuffer.pop_front();

                std::string smoothed = getSmoothedPrediction(predictionBuffer);
                smoothedLabels.push_back(smoothed);
                confidences.push_back(confidence);
            }

            // Draw predictions and show video
            VideoOverlay::drawDetections(frame, faces, smoothedLabels, confidences);
            cv::imshow("Emotion Recognition", frame);

            int key = cv::waitKey(1);
            if (key == 27) break; // ESC to quit
            if (key == 't' || key == 'T') {
                useTTA = !useTTA;
                std::cout << "TTA toggled " << (useTTA ? "ON" : "OFF") << std::endl;
            }

            frameCount++;
        }

        // Clean up resources
        csvFile.close();
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

#endif // RUN_REALTIME
