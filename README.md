# Bangla Sign Language Detector

Real-time recognition of Bangladeshi Sign Language (BdSL) gestures from
a live webcam feed. Unlike static hand-pose classifiers, this system
recognises **dynamic, motion-based signs** — analysing sequences of 30
frames with a deep learning model that understands velocity, acceleration,
and spatial body relationships.

## Demo


---

## Why dynamic gesture recognition is hard

Most sign language projects classify a single static hand image. BdSL
signs are motions — they evolve over time. A static snapshot of "ভালোবাসা"
(Love) is meaningless without seeing the full movement arc.

This system solves that by:
1. Capturing 30 consecutive frames per prediction window
2. Extracting 607 rich features per frame (not just raw landmarks)
3. Running a sequence model that understands how features **change** over time

---

## Architecture

```
Webcam (30 frames)
       │
       ▼
MediaPipe Holistic
  ├── 33 pose landmarks
  ├── 468 face landmarks
  ├── 21 left hand landmarks
  └── 21 right hand landmarks
       │
       ▼
Universal Motion Extractor  ──► 607 features/frame
  ├── Body-relative normalised landmarks
  ├── Velocity (frame-to-frame delta)
  ├── Acceleration (second-order delta)
  ├── Spatial inter-landmark relationships
  └── Directional patterns & hand configurations
       │
       ▼
Sequence Model  (input: 30 × 607)
  ├── Conv1D          ← local temporal pattern extraction
  ├── Bidirectional LSTM  ← forward + backward temporal context
  ├── Multi-Head Attention  ← focus on most relevant frames
  └── Dense + Softmax  ← 12-class prediction
       │
       ▼
Predicted BdSL Sign + Confidence
```

---

## About this project

**Sole developer** — I designed and implemented the full pipeline:
feature engineering, model architecture, training, and real-time inference.

**What I implemented:**

- **Custom feature extractor** — built a "universal motion extractor"
  that generates 607 features per frame. Instead of feeding raw (x, y, z)
  coordinates, I engineered body-relative normalised landmarks so that
  predictions are scale- and position-invariant (works regardless of
  how close you stand to the camera). Added first and second-order motion
  derivatives (velocity and acceleration) to capture the physics of signs.

- **Deep sequence model** — designed the architecture from scratch:
  Conv1D layers to capture local temporal patterns, Bidirectional LSTMs
  to model both forward and backward context in the 30-frame window, and
  Multi-Head Attention to let the model focus on the most informative
  frames in a sequence. Added preprocessing and interpolation layers for
  temporal scaling robustness.

- **MediaPipe Holistic integration** — used the full holistic model
  (pose + face + both hands) rather than hands-only, because some BdSL
  signs involve upper body posture and head orientation.

- **Real-time inference loop** — built `app.py` as a low-latency OpenCV
  webcam loop with a rolling 30-frame buffer, live prediction overlay,
  and a confidence-gated display so the label only shows when the model
  is sufficiently certain.

**What I learnt:**

- **Sequence modelling for gestures** — why static image classification
  fails for dynamic signs and how to frame gesture recognition as a
  time-series problem
- **Feature engineering over raw landmarks** — body-relative
  normalisation, why scale-invariance matters, and how motion derivatives
  (velocity, acceleration) dramatically improve temporal model performance
- **Bidirectional LSTM + Attention architecture** — how BiLSTMs capture
  context in both directions and why attention helps the model ignore
  irrelevant frames at the start/end of a gesture sequence
- **MediaPipe Holistic API** — extracting and structuring landmarks from
  pose, face, and hand models simultaneously and handling missing
  detections gracefully
- **Real-time deep learning inference with OpenCV** — managing frame
  buffers, prediction latency, and display overlays in a live loop

---

## Supported signs (12 classes)

| Bangla | Transliteration | Meaning |
|---|---|---|
| ভালোবাসা | valobasha | Love |
| ভালো | valo | Good |
| খারাপ | kharap | Bad |
| অলস | olosh | Lazy |
| বোকা | boka | Fool / Stupid |
| চাঁদ | chad | Moon |
| আকাশ | akash | Sky |
| ধন্যবাদ | dhonnobad | Thank you |
| শিক্ষক | shikkhok | Teacher |
| ফুটবল | football | Football |
| স্ত্রী | stree | Wife |
| দুর্বল | durbol | Weak |

---

## Tech stack
Python · TensorFlow / Keras · MediaPipe Holistic · OpenCV · NumPy · SciPy

---

## Run locally

```bash
git clone https://github.com/noobbatman/bangla-sign-language-detector
cd bangla-sign-language-detector

pip install opencv-python numpy mediapipe tensorflow scipy

python app.py
```

Press `q` to exit the webcam window.

---

## Why this matters

Bangladesh has an estimated 1.5 million deaf and hard-of-hearing people.
BdSL recognition is severely underrepresented in computer vision research
compared to ASL or BSL — most papers and open-source tools focus on
western sign languages. This project is a step toward building accessible
technology specifically for the Bangladeshi deaf community.
