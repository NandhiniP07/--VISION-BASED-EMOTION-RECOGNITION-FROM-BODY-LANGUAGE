# Vision-Based Emotion Recognition from Body Language

## Overview
Vision-Based Emotion Recognition from Body Language is a computer vision and deep learning project that automatically identifies human emotions from visual cues, primarily facial expressions. The system uses a Convolutional Neural Network (CNN) to classify emotions from grayscale facial images and performs real-time emotion detection using a webcam.

This project enhances human–computer interaction and has applications in mental health monitoring, assistive technologies, security, and marketing.

---

## Key Features
- CNN-based emotion classification
- Real-time emotion detection using webcam
- Face detection using Haar Cascade
- Emotion display with text and emojis
- FPS calculation for performance monitoring
- Screenshot freeze mode with countdown

---

## Dataset Preparation
- Training and testing images are organized into folders by emotion class  
  (angry, happy, sad, neutral, surprised, fearful, disgusted)
- Images are converted to **grayscale**
- Input image size: **48 × 48**

---

## Model Architecture (CNN)

### Layers Used
- Conv2D
- Batch Normalization
- Max Pooling
- Dropout (to prevent overfitting)
- Fully Connected (Dense) layers
- Softmax output layer for multi-class classification

### Model Details
- **Framework:** TensorFlow / Keras
- **Number of Emotion Classes:** 7
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Output:** Probability distribution over emotion classes

---

## Training Configuration
- **Batch Size:** 64
- **Epochs:** Up to 50
- **Early Stopping:** Enabled (patience = 10)
- **Model Checkpoint:** Saves best model as `emotion_model.h5`
- **Normalization:** Pixel values scaled to [0, 1]

---

## Real-Time Emotion Detection
- Webcam input captured using OpenCV
- Faces detected using Haar Cascade classifier
- Each detected face is:
  - Converted to grayscale
  - Resized to 48×48
  - Normalized and passed to CNN
- Predicted emotion is displayed with:
  - Text label
  - Corresponding emoji
- FPS displayed for performance monitoring

### Controls
- **`s`** → Freeze frame and take screenshot (10-second countdown)
- **`q`** → Quit application

---

## Output
- Live webcam feed with:
  - Face bounding boxes
  - Predicted emotion label
  - Emoji representation
  - FPS counter

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

---

## Files Description
- `trainer_classifier.py` – Trains the CNN emotion classification model
- `emotion_model.h5` – Saved trained model
- `live_emotion.py` – Real-time emotion detection using webcam

---

## Applications
- Human–computer interaction
- Mental health monitoring
- Assistive systems
- Surveillance and security
- User behavior analysis

---

## Conclusion
This project successfully demonstrates real-time vision-based emotion recognition using deep learning and computer vision techniques. While performance depends on dataset quality and lighting conditions, the system provides a strong foundation for future enhancements such as advanced CNN architectures, attention mechanisms, and multimodal emotion analysis.

