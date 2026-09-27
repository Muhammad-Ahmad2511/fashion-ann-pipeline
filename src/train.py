"""
train.py — Builds and trains a Sequential ANN on Fashion-MNIST.
Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax).
Saves the trained model and training history.
"""
import os
import numpy as np
import yaml
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    params = load_params()["train"]

    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(params["dense_units"], activation="relu"),
        Dropout(params["dropout_rate"]),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )

    model.save(os.path.join(MODELS_DIR, "model.h5"))
    pd.DataFrame(history.history).to_csv(os.path.join(MODELS_DIR, "history.csv"), index=False)

    print(f"Model saved to {MODELS_DIR}/model.h5")
    print(f"History saved to {MODELS_DIR}/history.csv")

if __name__ == "__main__":
    main()