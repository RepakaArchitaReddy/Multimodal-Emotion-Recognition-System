from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

X_text = tokenizer.texts_to_sequences(texts)

X_text = pad_sequences(
    X_text,
    maxlen=5
)

vocab_size = len(tokenizer.word_index) + 1