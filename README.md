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
## **Project Structure**
1. cv_final/

- main.cpp – Entry point for the real-time emotion recognition app
Captures webcam input, performs face detection and emotion classification (with optional TTA), and logs results to CSV.
- batch_test.cpp – (Optional) Tests emotion recognition on static images.
- config.hpp – Global paths, constants, and emotion label definitions.
- emotion_classifier.hpp / .cpp – Loads and runs the ONNX model, performs inference, and implements Test-Time Augmentation (TTA).
- face_detector.hpp / .cpp – Detects faces and eyes using OpenCV Haar cascades; handles alignment.
- video_overlay.hpp / .cpp – Draws bounding boxes, labels, and confidence scores on video frames in real time.
- utils.hpp / .cpp – Contains helper functions for preprocessing (e.g., grayscale conversion, normalization).

2. models/

- mini_xception.onnx – Pretrained Mini-Xception model used for emotion classification.
  
3. resources/

- haarcascade_frontalface_default.xml – OpenCV frontal face detector.
- haarcascade_eye.xml – Eye detector used for face alignment.
- test_images/ – (Optional) Sample grayscale faces for offline evaluation.
- results.csv – Auto-generated log with:
    - Frame number
    - Predicted emotion label
    - Confidence score
    - TTA mode (Yes/No)

4. README.md – Project documentation (this file)

5. Niloofar_Karimi_CV_Final_Project_Report.pdf – Final report for the project



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
