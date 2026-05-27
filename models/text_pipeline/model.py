from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

def build_text_model(vocab_size):

    model = Sequential()

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=64,
            input_length=5
        )
    )

    model.add(LSTM(64))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(7, activation='softmax'))

    return model