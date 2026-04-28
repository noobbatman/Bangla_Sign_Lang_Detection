# Bangla Sign Language Detection

This project is a real-time Bangla Sign Language Detection application using deep learning and computer vision. It utilizes MediaPipe for holistic body tracking (pose, face, and hands) to extract complex spatial and motion features, and a sophisticated sequence model implemented using TensorFlow/Keras to perform the classification.

## Features

- **Real-Time Detection**: Processes live webcam feeds using OpenCV.
- **Universal Motion Extractor**: A custom physics-engine-like feature extractor that generates 607 distinct features per frame, including:
  - Body-relative normalized landmarks
  - Motion derivatives (Velocity & Acceleration)
  - Spatial relationships and interactions
  - Directional patterns and hand configurations
- **Deep Sequence Model**: Utilizes an advanced architecture combining Conv1D, Bidirectional LSTMs, and Multi-Head Attention mechanisms to process sequences of 30 frames.
- **Robustness**: Advanced preprocessing and interpolation layers ensure consistent temporal scaling and prediction stability.

## Supported Signs (Actions)

The model is currently trained to detect the following Bangla signs:
1. valobasha (ভালোবাসা - Love)
2. valo (ভালো - Good)
3. kharap (খারাপ - Bad)
4. olosh (অলস - Lazy)
5. boka (বোকা - Fool/Stupid)
6. chad (চাঁদ - Moon)
7. akash (আকাশ - Sky)
8. dhonnobad (ধন্যবাদ - Thank you)
9. shikkhok (শিক্ষক - Teacher)
10. football (ফুটবল - Football)
11. stree (স্ত্রী - Wife)
12. durbol (দুর্বল - Weak)

## Files Included

- `app.py`: The main entry point for running the real-time inference loop using a webcam.
- `action_universal_final.h5`: The compiled weights for the trained neural network model.
- `Untitled3 (1).ipynb`: A Jupyter Notebook likely used for training, experiments, and exploratory analysis.

## Requirements

Ensure you have a Python environment set up (virtual environments are recommended). The key dependencies are:

- `opencv-python`
- `numpy`
- `mediapipe`
- `tensorflow`
- `scipy`

Install dependencies generally using:
```bash
pip install opencv-python numpy mediapipe tensorflow scipy
```

## Usage

Run the main application:
```bash
python app.py
```
Press `q` within the live feed window to exit the application.
