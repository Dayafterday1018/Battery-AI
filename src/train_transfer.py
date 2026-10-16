"""Step 2.3 · transfer learning: MobileNetV2 pre-trained on ImageNet, frozen, new sigmoid head.

Run: python src/train_transfer.py [--epochs 10]
Saves models/mobilenet.keras and reports/mobilenet_curves.png. Never loads the test set.
The first run downloads the ImageNet weights (~9 MB) once.
"""
import argparse

import keras
from keras import layers

from config import IMG, MODELS, REPORTS, SEED
from training import load_train_val, plot_history


def build(weights: str | None = "imagenet") -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights=weights)
    base.trainable = False                       # keep ImageNet features
    model = keras.Sequential([
        keras.Input(shape=(IMG, IMG, 3)),
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.Rescaling(2.0, offset=-1.0),      # 0..1 -> -1..1, what MobileNetV2 expects
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),   # 1 = laptop
    ])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (offline smoke test only)")
    args = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    X_tr, X_val, y_tr, y_val = load_train_val()
    model = build(None if args.no_pretrained else "imagenet")
    stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=args.epochs, callbacks=[stop], verbose=2)
    model.save(MODELS / "mobilenet.keras")
    plot_history(history, REPORTS / "mobilenet_curves.png", "MobileNetV2 (frozen)")


if __name__ == "__main__":
    main()
