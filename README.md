# Multimodal Emotion Recognition System

## Project Overview

This project is a Deep Learning based Multimodal Emotion Recognition System that predicts human emotions using both speech and text inputs.

The system combines:
- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion Learning

The project uses CNN, LSTM, MFCC audio features, and Streamlit deployment for real-time prediction.

---

# Technologies Used

- Python
- TensorFlow / Keras
- CNN
- LSTM
- MFCC Feature Extraction
- Librosa
- NumPy
- Scikit-learn
- Streamlit

---

# Dataset Used

TESS (Toronto Emotional Speech Set)

Emotions Used:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise
- Sad

---

# Project Architecture

Speech Input (.wav)
        │
        ▼
MFCC Feature Extraction
        │
        ▼
CNN + LSTM Speech Model
        │
        ▼
Speech Embeddings
        │
        ├──────────────┐
        │              │
        ▼              │
Text Input             │
        │              │
        ▼              │
Tokenizer + Embedding  │
        │              │
        ▼              │
LSTM Text Model        │
        │              │
        └──────┬───────┘
               ▼
          Fusion Layer
               ▼
         Dense Layers
               ▼
       Emotion Prediction

---

# Project Structure

project/
│
├── Dataset/
├── models/
├── Results/
├── deployment/
├── notebooks/
├── README.md
└── requirements.txt

---

# Features

- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion
- Audio Upload
- Microphone Recording
- Emotion Emojis
- Real-time Prediction
- Streamlit Web Interface

---

# Model Performance

| Model | Accuracy |
|------|-----------|
| Speech Model | ~98% |
| Text Model | ~96% |
| Fusion Model | ~96-98% |

---

# How To Run

## Step 1

Install dependencies:

pip install -r requirements.txt

## Step 2

Go to deployment folder:

cd deployment

## Step 3

Run Streamlit app:

streamlit run app.py

---

# Supported Emotions

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😊 Pleasant
- 😢 Sad
