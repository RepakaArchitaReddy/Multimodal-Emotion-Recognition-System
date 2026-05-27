loss, acc = model_text.evaluate(
    X_t_test,
    y_t_test
)

print("Text Accuracy:", acc)