# Real-Time Facial Emotion Recognition with ONNX and OpenCV

**Author:** Niloofar Karimi  
**Email:** karimi.ni@northeastern.edu  
**Institution:** Northeastern University  
**Course:** Computer Vision Final Project – Spring 2025



## Project Description

This project implements a real-time facial emotion recognition system using a lightweight convolutional neural network (CNN) model exported to ONNX and deployed with OpenCV in C++. The system classifies live webcam input into one of seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

To enhance real-world performance and prediction reliability, the system integrates:

- **Face alignment** using Haar cascade eye detection  
- **Test-Time Augmentation (TTA)** via flipping and rotation  
- **Confidence filtering**: predictions with confidence below 0.2 are labeled “Uncertain”  
- **Temporal smoothing** with a majority vote buffer  
- **CSV logging** of frame-wise predictions for analysis


## Presentation & Demo

- **Presentation Video**: https://drive.google.com/drive/u/0/folders/1hBH7lQVuWiKAvAubBI_cEfOlCUd6kZWx
- **Sample Output CSV**: `results.csv` (auto-generated during live testing)  



## Project Structure

cv_final/
main.cpp # Real-time application entry point
- Captures webcam input
- Performs face detection
- Applies emotion classification (with TTA)
- Draws overlays and logs to CSV

batch_test.cpp # (Optional) Script to test emotion recognition on static images
config.hpp # Global paths, constants, and emotion label definitions
emotion_classifier.hpp # Class declaration for EmotionClassifier
emotion_classifier.cpp # ONNX model loading, inference, and TTA logic
face_detector.hpp # Class declaration for FaceDetector (Haar cascade)
face_detector.cpp # Face detection logic using OpenCV's Haar cascades


video_overlay.hpp # Overlay utility for drawing boxes, labels, and confidence
video_overlay.cpp # Draws real-time detection results on the video frame


utils.hpp # General-purpose utility functions (e.g., grayscale conversion)
utils.cpp # Preprocessing helpers for image handling


models/
mini_xception.onnx # Pretrained Mini-Xception model in ONNX format


resources/
haarcascade_frontalface_default.xml # OpenCV face detector
haarcascade_eye.xml # OpenCV eye detector for face alignment

test_images/ # (Optional) Folder containing sample grayscale face images for offline evaluation
results.csv # Auto-generated CSV containing frame-by-frame:
- Frame number
- Emotion label
- Confidence score
- TTA mode used (Yes/No)

README.md # Project documentation (this file)
Niloofar_Karimi_CV_Final_Project_Report.pdf # Final IEEE-style written report

## Build & Run Instructions

### Requirements

- C++17 compatible compiler (e.g., `g++`, Clang, or Xcode)
- OpenCV 4.x installed and properly linked
- ONNX model file `mini_xception.onnx` placed in `/models/`

### Build (Linux/macOS example with g++)

```bash
g++ -std=c++17 -o emotion_app \
    main.cpp emotion_classifier.cpp face_detector.cpp video_overlay.cpp utils.cpp \
    `pkg-config --cflags --libs opencv4`
```
### Run Main.cpp
- Press T to toggle Test-Time Augmentation (TTA) on/off
- Press ESC to exit
- Frame-by-frame predictions and confidence scores are saved in results.csv




###  Evaluation Summary

The system was tested under multiple configurations, measuring accuracy improvements from different preprocessing and enhancement techniques:

| Configuration                             | Accuracy (%) |
|------------------------------------------|--------------|
| Baseline (Grayscale + Resize)            | 18.7         |
| + Normalization + Center Crop            | 25.2         |
| + Face Alignment                         | 50.4         |
| + TTA + Smoothing + Confidence Filtering | 61.6         |

Visual analysis of confidence scores showed improved consistency and stability with TTA enabled.




## Acknowledgments

This project was completed as part of the **Computer Vision Final Project** at Northeastern University (Spring 2025). Special thanks to the course instructors and peers for their support and feedback throughout development.
