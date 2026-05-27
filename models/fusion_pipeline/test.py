from tensorflow.keras.models import load_model
import numpy as np

# load model
fusion_model = load_model(
    "fusion_model.h5",
    compile=False
)

# dummy placeholders
# replace with actual test data during execution

# X_test = ...
# X_t_test = ...
# y_test = ...

# evaluation
loss, acc = fusion_model.evaluate(
    [X_test, X_t_test],
    y_test
)

print("Fusion Accuracy:", acc)