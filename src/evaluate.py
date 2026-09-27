"""
evaluate.py — Loads the trained model and processed test set, computes
test loss/accuracy, generates a confusion matrix image, and writes
metrics.json at the project root.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
METRICS_PATH = "metrics.json"
CM_IMAGE_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def main():
    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    model = load_model(os.path.join(MODELS_DIR, "model.h5"))

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH)
    plt.close()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    print(f"Confusion matrix saved to {CM_IMAGE_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()