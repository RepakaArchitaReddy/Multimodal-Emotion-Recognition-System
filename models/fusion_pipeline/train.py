from fusion import build_fusion_model

fusion_model = build_fusion_model(vocab_size)

fusion_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fusion = fusion_model.fit(
    [X_train, X_t_train],
    y_train,
    epochs=15,
    batch_size=16,
    validation_split=0.2
)

fusion_model.save("fusion_model.h5")