"""
LeafGuard AI – Transfer Learning Training Script
Uses EfficientNetB0 pretrained on ImageNet, with optional fine-tuning.

Usage:
    python ml/train_transfer.py

Requires GPU for reasonable training speed.
"""

import argparse
import os
import matplotlib.pyplot as plt
import subprocess
import sys

# Check CPU flags before importing TensorFlow to avoid "Illegal instruction" crashes
def cpu_supports_avx():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            info = f.read()
    except Exception:
        return True
    return ('avx' in info) or ('avx2' in info)

if not cpu_supports_avx():
    sys.stderr.write("\nERROR: CPU does not report AVX/AVX2 support.\n")
    sys.stderr.write("This system may crash when importing prebuilt TensorFlow wheels.\n")
    sys.stderr.write("Recommended options:\n")
    sys.stderr.write("  1) Install TensorFlow via conda-forge (Miniconda) which has broader CPU support.\n")
    sys.stderr.write("     Example: conda create -n leafguard python=3.10 -c conda-forge tensorflow -y\n")
    sys.stderr.write("  2) Run training on a GPU (Colab / remote machine).\n")
    sys.stderr.write("  3) Use a Docker image built for your platform.\n\n")
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
FINE_TUNE_EPOCHS = 8
LEARNING_RATE = 1e-4
DATA_DIR = "data/PlantVillage"
MODEL_SAVE_PATH = "ml/leafguard_model.h5"

CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
]


def build_generators(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = train_gen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode="sparse",
        subset="training",
        seed=42,
    )
    val_data = val_gen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode="sparse",
        subset="validation",
        seed=42,
    )
    return train_data, val_data


def build_model(num_classes: int = 4, img_size=IMG_SIZE, backbone="efficientnetb0"):
    """Build model using selected backbone. Supported: efficientnetb0, efficientnetb4, resnet50."""
    backbone = backbone.lower()
    if backbone == "efficientnetb4":
        base = tf.keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    elif backbone == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    else:
        # default
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    base.trainable = False

    inputs = layers.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"LeafGuard_{backbone}")
    return model, base


def plot_history(history, outpath="ml/training_history_transfer.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    print("  Plot saved →", outpath)


def train(args):
    print("LeafGuard — transfer learning training")
    train_data, val_data = build_generators(args.data_dir, img_size=tuple(args.img_size), batch_size=args.batch_size)
    print(f"Training samples: {train_data.samples}, Validation samples: {val_data.samples}")

    model, base = build_model(num_classes=len(CLASS_NAMES), img_size=tuple(args.img_size), backbone=args.backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cb = [
        callbacks.ModelCheckpoint(args.model_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("\n[1] Training head (base frozen) ...")
    history = model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=cb)
    plot_history(history)

    if args.fine_tune_epochs > 0:
        print("\n[2] Fine-tuning — unfreezing base model ...")
        base.trainable = True
        # Optionally freeze first N layers
        if args.freeze_until_layer > 0:
            for layer in base.layers[: args.freeze_until_layer]:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ft_history = model.fit(train_data, validation_data=val_data, epochs=args.fine_tune_epochs, callbacks=cb)
        # append/plot fine-tune history
        plot_history(ft_history, outpath="ml/training_history_finetune.png")

    print(f"Saving model to {args.model_path} ...")
    model.save(args.model_path)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--img-size", nargs=2, type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--fine-tune-epochs", dest="fine_tune_epochs", type=int, default=FINE_TUNE_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--fine-tune-lr", dest="fine_tune_lr", type=float, default=1e-5)
    p.add_argument("--freeze-until-layer", dest="freeze_until_layer", type=int, default=0,
                   help="If >0, keep first N base layers frozen during fine-tuning")
    p.add_argument("--backbone", choices=["efficientnetb0", "efficientnetb4", "resnet50"], default="efficientnetb0",
                   help="Backbone model to use for transfer learning (stronger models are larger and slower)")
    p.add_argument("--model-path", dest="model_path", default=MODEL_SAVE_PATH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
