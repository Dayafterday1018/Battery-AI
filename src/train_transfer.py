"""Step 2.3 · transfer learning with MobileNetV2 (ImageNet weights).

Phase 1  frozen base, train the new sigmoid head          (last week's model)
Phase 2  unfreeze the top N base layers, fine-tune at a low learning rate   (improvement)

Run:
  python src/train_transfer.py --name mobilenet_baseline --img 128 --fine-tune-layers 0 --no-augment \
      --no-class-weight --no-tune
  python src/train_transfer.py                         # improved: 224 px, augmentation, fine-tune 40 layers
Saves models/<name>.keras + models/<name>.json and reports/<name>_curves.png. Never loads the test set.
The first run downloads the ImageNet weights (~9 MB) once.
"""
import argparse

import keras
from keras import layers

from config import IMG_TRANSFER, MODELS, REPORTS, SEED
from training import class_weights, load_train_val, plot_history, save_meta, tune_threshold


def build(img: int, augment: bool, weights: str | None = "imagenet"):
    base = keras.applications.MobileNetV2(input_shape=(img, img, 3), include_top=False, weights=weights)
    base.trainable = False                                   # phase 1: keep ImageNet features
    aug = [layers.RandomFlip("horizontal_and_vertical", seed=SEED),
           layers.RandomRotation(0.15, seed=SEED),
           layers.RandomZoom(0.15, seed=SEED),
           layers.RandomContrast(0.2, seed=SEED),
           layers.RandomBrightness(0.15, value_range=(0, 1), seed=SEED)] if augment else []
    inputs = keras.Input(shape=(img, img, 3))
    x = inputs
    for layer in aug:
        x = layer(x)
    x = layers.Rescaling(2.0, offset=-1.0)(x)                # 0..1 -> -1..1, what MobileNetV2 expects
    x = base(x, training=False)                              # keep BatchNorm statistics frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)       # 1 = laptop
    return keras.Model(inputs, outputs), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mobilenet")
    ap.add_argument("--img", type=int, default=IMG_TRANSFER)
    ap.add_argument("--epochs-head", type=int, default=10)
    ap.add_argument("--epochs-ft", type=int, default=15)
    ap.add_argument("--fine-tune-layers", type=int, default=40, help="0 = frozen base only (baseline)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--no-tune", action="store_true", help="keep threshold 0.5 (baseline)")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (offline smoke test only)")
    args = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    X_tr, X_val, y_tr, y_val = load_train_val(args.img)
    model, base = build(args.img, not args.no_augment, None if args.no_pretrained else "imagenet")
    cw = None if args.no_class_weight else class_weights(y_tr)
    callbacks = lambda: [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2)]

    model.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy"])
    histories = [model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs_head,
                           class_weight=cw, callbacks=callbacks(), verbose=2)]

    if args.fine_tune_layers > 0:                            # phase 2: fine-tune the top of the base
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:                            # BatchNorm stays frozen while fine-tuning
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(keras.optimizers.Adam(1e-5), "binary_crossentropy", metrics=["accuracy"])
        histories.append(model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs_ft,
                                   class_weight=cw, callbacks=callbacks(), verbose=2))

    path = MODELS / f"{args.name}.keras"
    model.save(path)
    curves = plot_history(histories, REPORTS / f"{args.name}_curves.png", f"MobileNetV2 · {args.name}")
    threshold = 0.5 if args.no_tune else tune_threshold(model.predict(X_val, verbose=0).ravel(), y_val)

    changes = []
    if args.img != 128:
        changes.append(f"Input size 128 → {args.img} px (MobileNetV2's native resolution)")
    if not args.no_augment:
        changes.append("Augmentation: flips, rotation, zoom, contrast, brightness")
    if args.fine_tune_layers:
        changes.append(f"Fine-tuned the top {args.fine_tune_layers} MobileNetV2 layers at learning rate 1e-5")
    if cw:
        changes.append("Class weights to balance laptop vs normal")
    if not args.no_tune:
        changes.append(f"Decision threshold {threshold} tuned on validation (normal recall ≥ target)")
    save_meta(path, img=args.img, threshold=threshold, family="MobileNetV2", changes=changes,
              settings={k: v for k, v in vars(args).items() if k != "name"}, **curves)
    print(f"threshold tuned on validation: {threshold}")


if __name__ == "__main__":
    main()
