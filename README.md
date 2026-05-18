# AI Face Mask Detection With Live Alert System

## 📌 Project Overview

This project is an AI-powered real-time face mask detection and monitoring system built using Python, OpenCV, TensorFlow, and Deep Learning.

The system detects whether a person is wearing a face mask through a live webcam feed and automatically performs alert actions whenever a person without a mask is detected.

---

## 🚀 Features

- Real-time face detection using Haar Cascade
- Face mask classification using CNN model
- Voice alert system for no-mask detection
- Automatic image capture of detected person
- Automated email alert with evidence image
- Smart cooldown logic to avoid alert spamming
- Real-time webcam monitoring

---

## 🛠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Haar Cascade Classifier
- SMTP Email Automation
- NumPy

---

## ⚙️ System Workflow

1. Webcam captures live video stream
2. Haar Cascade detects faces
3. CNN model predicts:
   - Mask
   - No Mask
4. If no mask detected:
   - Voice alert triggered
   - Screenshot captured
   - Email sent to admin with evidence

---

## 📂 Project Structure

```bash
Face Mask Detection with Live Alert System/
│
├── captures/
├── dataset/
├── haarcascade/
├── models/
│
├── detect_mask_video.py
├── train_mask_detector.py
└── README.md


Install Dependencies
pip install tensorflow opencv-python numpy pyttsx3

Run Detection System
python detect_mask_video.py