"""
LeafGuard AI – CNN Training Script
====================================
Trains a custom CNN on the PlantVillage tomato subset.

Usage (Google Colab / local GPU):
    python ml/train.py

Requirements:
    pip install tensorflow pillow numpy matplotlib kaggle

Dataset:
    Kaggle: "emmarex/plantdisease" or download from plantvillage.psu.edu
    Expected structure:
        data/
          PlantVillage/
            Tomato_healthy/
            Tomato_Early_blight/
            Tomato_Leaf_Mold/
            Tomato_Tomato_Yellow_Leaf_Curl_Virus/

Output:
    ml/leafguard_model.h5
    ml/training_history.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DATA_DIR = "data/PlantVillage"
MODEL_SAVE_PATH = "ml/leafguard_model.h5"

# The 4 classes we care about (subfolder names from PlantVillage)
CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
]

# ── Data generators ─────────────────────────────────────────────────────────
def build_generators(data_dir: str):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_data = train_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="sparse",
        subset="training",
        seed=42,
    )
    val_data = val_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="sparse",
        subset="validation",
        seed=42,
    )
    return train_data, val_data


# ── Custom CNN architecture ─────────────────────────────────────────────────
def build_cnn(num_classes: int = 4) -> models.Model:
    """
    Custom CNN — no transfer learning (per SRS constraint).
    Architecture: 4 Conv blocks → Global Average Pool → Dense → Softmax
    """
    model = models.Sequential(
        [
            # Input
            layers.Input(shape=(*IMG_SIZE, 3)),

            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Block 4
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="LeafGuard_CNN",
    )
    return model


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  LeafGuard AI — CNN Training")
    print("=" * 60)

    # Build data pipelines
    print(f"\n[1/4] Loading data from '{DATA_DIR}' ...")
    train_data, val_data = build_generators(DATA_DIR)
    print(f"      Training samples : {train_data.samples}")
    print(f"      Validation samples: {val_data.samples}")
    print(f"      Classes           : {train_data.class_indices}")

    # Build model
    print("\n[2/4] Building CNN ...")
    model = build_cnn(num_classes=4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    cb = [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print("\n[3/4] Training ...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=cb,
    )

    # Save + plot
    print(f"\n[4/4] Saving model to '{MODEL_SAVE_PATH}' ...")
    model.save(MODEL_SAVE_PATH)

    plot_history(history)

    # Final eval
    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    print(f"\n  ✓ Final validation accuracy: {val_acc * 100:.2f}%")
    print(f"  ✓ Final validation loss    : {val_loss:.4f}")
    print(f"\n  Model saved → {MODEL_SAVE_PATH}")
    print("=" * 60)


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LeafGuard AI — Training History", fontsize=14, fontweight="bold")

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
    plt.savefig("ml/training_history.png", dpi=150)
    print("  Plot saved → ml/training_history.png")


if __name__ == "__main__":
    train()
