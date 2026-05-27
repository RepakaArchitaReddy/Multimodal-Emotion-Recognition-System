from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def build_speech_model():

    model = Sequential()

    model.add(
        Conv1D(
            64,
            3,
            activation='relu',
            input_shape=(94,40)
        )
    )

    model.add(MaxPooling1D(2))

    model.add(LSTM(64))

    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(7, activation='softmax'))

    return model