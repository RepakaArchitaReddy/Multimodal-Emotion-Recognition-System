import librosa
import numpy as np

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