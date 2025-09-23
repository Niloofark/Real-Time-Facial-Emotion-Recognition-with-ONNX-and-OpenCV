/**
 * face_detector.hpp
 * Author: Niloofar Karimi
 * Description: Header file for the FaceDetector class.
 *              Provides an interface to detect faces in grayscale images using
 *              OpenCV's Haar cascade classifier.
 */

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>

// Simple face detector using OpenCV's Haar cascades
class FaceDetector {
public:
    // Constructor: loads the Haar cascade from the given file path
    FaceDetector(const std::string& cascadePath);

    // Detect faces in a grayscale image and return bounding boxes
    std::vector<cv::Rect> detect(const cv::Mat& frameGray);

private:
    cv::CascadeClassifier faceCascade; // OpenCV face detector
};
