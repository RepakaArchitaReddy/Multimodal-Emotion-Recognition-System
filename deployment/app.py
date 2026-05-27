import streamlit as st
import numpy as np
import librosa

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load fusion model
fusion_model = load_model(
    "../models/fusion_pipeline/fusion_model.h5",
    compile=False
)

# emotion labels
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant",
    "sad"
]

# emotion emojis
emotion_emojis = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "pleasant": "😊",
    "sad": "😢"
}

# preprocess audio
def preprocess_audio(file):

    audio, sr = librosa.load(file, sr=16000)

    audio, _ = librosa.effects.trim(audio)

    audio = audio / np.max(np.abs(audio))

    max_len = 16000 * 3

    if len(audio) < max_len:

        audio = np.pad(
            audio,
            (0, max_len - len(audio))
        )

    else:
        audio = audio[:max_len]

    return audio


# extract mfcc
def extract_mfcc(file):

    audio = preprocess_audio(file)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40
    )

    mfcc = mfcc.T

    if mfcc.shape[0] < 94:

        pad = 94 - mfcc.shape[0]

        mfcc = np.pad(
            mfcc,
            ((0, pad), (0, 0))
        )

    else:
        mfcc = mfcc[:94, :]

    return mfcc


# tokenizer dictionary
word_index = {
    'back': 1,
    'bar': 2,
    'base': 3,
    'bath': 4,
    'bean': 5,
    'beg': 6,
    'bite': 7,
    'boat': 8,
    'bone': 9,
    'book': 10
}

# streamlit title
st.title("Multimodal Emotion Recognition")

st.write("Upload speech audio and enter spoken word to predict emotion.")

# audio upload
audio_file = st.file_uploader(
    "Upload Audio File",
    type=['wav']
)

# text input
text_input = st.text_input(
    "Enter Spoken Word"
)

# prediction button
if st.button("Predict Emotion"):

    if audio_file is not None and text_input != "":

        # extract speech feature
        speech_feat = extract_mfcc(audio_file)

        speech_feat = np.expand_dims(
            speech_feat,
            axis=0
        )

        # process text
        text_word = text_input.lower()

        text_seq = [
            word_index.get(text_word, 0)
        ]

        text_seq = pad_sequences(
            [text_seq],
            maxlen=5
        )

        # prediction
        pred = fusion_model.predict(
            [speech_feat, text_seq]
        )

        pred_class = np.argmax(pred)

        emotion = emotion_labels[pred_class]

        emoji = emotion_emojis[emotion]

        # display result
        st.success(
            f"Predicted Emotion: {emotion} {emoji}"
        )

    else:

        st.warning(
            "Please upload audio and enter spoken word"
        )