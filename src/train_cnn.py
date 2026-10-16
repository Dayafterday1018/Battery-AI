"""Step 2.2 · a small Keras CNN (Conv2D 64 + Conv2D 128) that sorts laptop vs normal.

Run: python src/train_cnn.py [--epochs 30]
Saves models/cnn.keras and reports/cnn_curves.png. Never loads the test set.
"""
import argparse

import keras
from keras import layers

from config import IMG, MODELS, REPORTS, SEED
from training import load_train_val, plot_history


def build() -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(IMG, IMG, 3)),
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),  # augmentation: fix for overfitting (2.2 validate)
        layers.RandomRotation(0.1, seed=SEED),                    # only active during training
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),                   # 1 = laptop
    ])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    X_tr, X_val, y_tr, y_val = load_train_val()
    model = build()
    stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=args.epochs, callbacks=[stop], verbose=2)
    model.save(MODELS / "cnn.keras")
    plot_history(history, REPORTS / "cnn_curves.png", "CNN 64 + 128")


if __name__ == "__main__":
    main()
