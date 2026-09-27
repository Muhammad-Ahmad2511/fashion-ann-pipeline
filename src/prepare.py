"""
prepare.py — Downloads Fashion-MNIST and saves raw arrays to data/raw/.
No hyperparameters needed at this stage.
"""
import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

RAW_DIR = os.path.join("data", "raw")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save(os.path.join(RAW_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(RAW_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RAW_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(RAW_DIR, "y_test.npy"), y_test)

    print(f"Saved raw data to {RAW_DIR}/")
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

if __name__ == "__main__":
    main()