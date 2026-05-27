loss, acc = model.evaluate(
    X_test,
    y_test
)

print("Speech Accuracy:", acc)