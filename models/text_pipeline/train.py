from model import build_text_model

model_text = build_text_model(vocab_size)

model_text.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_text = model_text.fit(
    X_t_train,
    y_t_train,
    epochs=15,
    batch_size=16,
    validation_split=0.2
)

model_text.save("text_model.h5")