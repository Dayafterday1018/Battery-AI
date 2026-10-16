"""Step 2.2 · a small Keras CNN (Conv2D 64 + Conv2D 128) that sorts laptop vs normal.

Run: python src/train_cnn.py [--epochs 30] [--no-augment]
Saves models/cnn.keras (+ models/cnn.json) and reports/cnn_curves.png. Never loads the test set.
"""
import argparse

import keras
from keras import layers

from config import IMG, MODELS, REPORTS, SEED
from training import load_train_val, plot_history, save_meta, tune_threshold


def build(augment: bool = True) -> keras.Model:
    aug = [layers.RandomFlip("horizontal_and_vertical", seed=SEED),
           layers.RandomRotation(0.1, seed=SEED)] if augment else []
    model = keras.Sequential([
        keras.Input(shape=(IMG, IMG, 3)),
        *aug,                                    # only active during training
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),   # 1 = laptop
    ])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-augment", action="store_true", help="last week's baseline without augmentation")
    ap.add_argument("--name", default="cnn")
    args = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    X_tr, X_val, y_tr, y_val = load_train_val(IMG)
    model = build(not args.no_augment)
    stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=args.epochs, callbacks=[stop], verbose=2)
    path = MODELS / f"{args.name}.keras"
    model.save(path)
    curves = plot_history(history, REPORTS / f"{args.name}_curves.png", "CNN 64 + 128")
    threshold = tune_threshold(model.predict(X_val, verbose=0).ravel(), y_val)
    save_meta(path, img=IMG, threshold=threshold, family="CNN 64 + 128",
              changes=["Augmentation: random flips and rotations"] if not args.no_augment else [],
              settings={"epochs_max": args.epochs, "augment": not args.no_augment}, **curves)
    print(f"threshold tuned on validation: {threshold}")


if __name__ == "__main__":
    main()
