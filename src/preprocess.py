"""
preprocess.py — Normalizes raw Fashion-MNIST arrays to [0,1] and splits
a validation set out of the training data. Saves results to data/processed/.
"""
import os
import argparse
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    params = load_params()["preprocess"]
    test_size = params["test_size"]
    seed = params["seed"]

    x_train = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # mean-std standardization (resolved: chosen over [-1,1] scaling for better convergence properties)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Split validation set out of training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=test_size, random_state=seed
    )

    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"), x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"Saved processed data to {PROCESSED_DIR}/")
    print(f"x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")

if __name__ == "__main__":
    main()