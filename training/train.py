"""Train and export the casting-defect classifier for the ESP32-CAM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (96, 96)
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("archive/casting_data/casting_data"),
        help="Directory containing train/ and test/ class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/export"),
        help="Directory where the Keras model, TFLite model and labels are saved.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_datasets(
    data_dir: Path, batch_size: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}'. "
            "Use --data-dir to point to the dataset root."
        )

    common = {
        "labels": "inferred",
        "label_mode": "categorical",
        "class_names": None,
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "seed": SEED,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, **common
    )
    if train_ds.class_names != test_ds.class_names:
        raise ValueError(
            f"Train/test classes differ: {train_ds.class_names} vs "
            f"{test_ds.class_names}"
        )
    return train_ds, test_ds, train_ds.class_names


def class_weights(train_ds: tf.data.Dataset, class_count: int) -> dict[int, float]:
    counts = np.zeros(class_count, dtype=np.int64)
    for _, labels in train_ds.unbatch():
        counts[int(tf.argmax(labels).numpy())] += 1
    if np.any(counts == 0):
        raise ValueError(f"Every class needs at least one image; counts={counts.tolist()}")
    total = int(counts.sum())
    return {
        index: total / (class_count * int(count))
        for index, count in enumerate(counts)
    }


def build_model(class_count: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="probabilities"
    )(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_tflite(
    model: tf.keras.Model, representative_ds: tf.data.Dataset, output_path: Path
) -> None:
    def representative_data():
        for images, _ in representative_ds.take(100):
            for image in images:
                yield [tf.cast(image[None, ...], tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    tf.keras.utils.set_random_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    weights = class_weights(train_ds, len(class_names))
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=callbacks,
    )
    metrics = model.evaluate(test_ds, return_dict=True)
    model.save(args.output_dir / "model.keras")
    export_tflite(model, train_ds, args.output_dir / "model_int8.tflite")
    (args.output_dir / "labels.txt").write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({key: float(value) for key, value in metrics.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Classes: {class_names}")
    print(f"Test metrics: {metrics}")
    print(f"Exported: {args.output_dir / 'model_int8.tflite'}")


if __name__ == "__main__":
    main()
