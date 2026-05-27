from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

pred_fusion = fusion_model.predict(
    [X_test, X_t_test]
)

pred_fusion = pred_fusion.argmax(axis=1)

cm = confusion_matrix(
    y_test,
    pred_fusion
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Fusion Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print(
    classification_report(
        y_test,
        pred_fusion
    )
)