from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model

def build_fusion_model(vocab_size):

    speech_input = Input(shape=(94,40))

    x1 = Conv1D(
        64,
        3,
        activation='relu'
    )(speech_input)

    x1 = MaxPooling1D(2)(x1)

    x1 = LSTM(64)(x1)

    text_input = Input(shape=(5,))

    x2 = Embedding(
        input_dim=vocab_size,
        output_dim=64
    )(text_input)

    x2 = LSTM(32)(x2)

    combined = Concatenate()([x1, x2])

    z = Dense(64, activation='relu')(combined)

    z = Dropout(0.3)(z)

    output = Dense(
        7,
        activation='softmax'
    )(z)

    fusion_model = Model(
        inputs=[speech_input, text_input],
        outputs=output
    )

    return fusion_model