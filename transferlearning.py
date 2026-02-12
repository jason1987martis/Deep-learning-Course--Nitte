"""
Transfer Learning with VGG16 and VGG19 (Keras / TensorFlow)

What this script does:
1) Loads images from a directory structure using image_dataset_from_directory
2) Builds TWO transfer-learning models:
   - VGG16 backbone + custom classification head
   - VGG19 backbone + custom classification head
3) Trains each model in two stages:
   - Stage A: Train only the new head (backbone frozen)
   - Stage B: Fine-tune last few convolution blocks (optional, controlled by settings)
4) Evaluates and prints results

Expected dataset folder structure:
data_dir/
  train/
    class_a/
      img1.jpg
      ...
    class_b/
      ...
  val/
    class_a/
    class_b/
  test/            (optional but recommended)
    class_a/
    class_b/

If you don't have a test folder, you can skip test_ds parts.

Install:
pip install tensorflow
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

# -----------------------------
# 0) Basic configuration
# -----------------------------

# Set your dataset root folder here (must contain train/ and val/ subfolders at minimum)
DATA_DIR = "data_dir"

# Image settings for VGG models (VGG expects 224x224 inputs)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS_HEAD = 5          # epochs for training only the classification head
EPOCHS_FINE = 5          # epochs for fine-tuning (unfreeze some backbone layers)

# Fine-tuning controls:
# - Set to True to run a second stage of training with some VGG layers unfrozen.
# - Set to False to only train the new head.
DO_FINE_TUNE = True

# How many layers from the END of the backbone to unfreeze (common approach)
# Example: unfreeze last 4-8 conv layers; you can tune this.
UNFREEZE_LAST_N_LAYERS = 8

# -----------------------------
# 1) Load datasets
# -----------------------------

# We use image_dataset_from_directory which creates a tf.data.Dataset from folders.
# "label_mode='categorical'" gives one-hot labels (good for multi-class softmax).
train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "val")
test_path = os.path.join(DATA_DIR, "test")  # optional

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Test dataset is optional; create it only if folder exists.
test_ds = None
if os.path.isdir(test_path):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

# Number of classes is derived from the subfolder names in train/.
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Improve input pipeline performance:
# - cache(): keeps batches in memory after first epoch (useful if dataset fits RAM)
# - prefetch(): overlaps preprocessing and model execution
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
if test_ds is not None:
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 2) Data augmentation (optional but recommended)
# -----------------------------

# Data augmentation increases robustness by applying random transformations.
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)

# -----------------------------
# 3) Build a transfer-learning model (generic builder)
# -----------------------------

def build_transfer_model(backbone_name: str, num_classes: int):
    """
    Build a transfer-learning model using either VGG16 or VGG19 backbone.

    backbone_name: "vgg16" or "vgg19"
    num_classes: number of target classes in your dataset
    """

    # Define an input layer consistent with VGG requirements.
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")

    # Apply data augmentation only during training (Keras handles this automatically).
    x = data_augmentation(inputs)

    # Apply the correct preprocess_input for the chosen backbone.
    # VGG preprocess_input (Caffe style) converts RGB -> BGR and subtracts channel means.
    if backbone_name.lower() == "vgg16":
        x = layers.Lambda(vgg16_preprocess, name="vgg16_preprocess")(x)
        backbone = VGG16(weights="imagenet", include_top=False, input_tensor=x)
    elif backbone_name.lower() == "vgg19":
        x = layers.Lambda(vgg19_preprocess, name="vgg19_preprocess")(x)
        backbone = VGG19(weights="imagenet", include_top=False, input_tensor=x)
    else:
        raise ValueError("backbone_name must be 'vgg16' or 'vgg19'")

    # Freeze the backbone initially (we train only the new classification head first).
    backbone.trainable = False

    # Add a custom classification head on top of the frozen backbone.
    # - GlobalAveragePooling2D reduces feature maps to a single vector.
    # - Dense + Dropout provide a small trainable head for your dataset.
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout")(x)

    # Final classification layer:
    # - softmax for multi-class classification (num_classes >= 2)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    # Assemble the final model.
    model = models.Model(inputs=inputs, outputs=outputs, name=f"{backbone_name}_transfer")

    return model, backbone

# -----------------------------
# 4) Compile helper
# -----------------------------

def compile_model(model: tf.keras.Model, lr: float):
    """
    Compile model with standard settings for multi-class classification.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

# -----------------------------
# 5) Fine-tuning helper
# -----------------------------

def unfreeze_last_n_layers(backbone: tf.keras.Model, n: int):
    """
    Unfreeze the last n layers of the backbone for fine-tuning.

    Notes:
    - Fine-tuning too many layers can overfit or destabilize training.
    - Start small (e.g., 4-12 layers) and use a low learning rate.
    """
    # Keep the backbone trainable overall, but selectively unfreeze only last n layers.
    backbone.trainable = True

    # Freeze everything except the last n layers.
    for layer in backbone.layers[:-n]:
        layer.trainable = False

# -----------------------------
# 6) Train and evaluate routine (used for both VGG16 and VGG19)
# -----------------------------

def train_model(backbone_name: str):
    """
    Build, train (head + optional fine-tune), and evaluate a model for given backbone.
    """

    print(f"\n==============================")
    print(f"Training transfer model: {backbone_name.upper()}")
    print(f"==============================")

    # Build model and keep reference to backbone for later fine-tuning.
    model, backbone = build_transfer_model(backbone_name, num_classes)

    # ---- Stage A: Train only the new head (backbone frozen) ----
    compile_model(model, lr=1e-3)

    # EarlyStopping stops if val_loss doesn't improve for a few epochs.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    print("\n[Stage A] Training head (backbone frozen)...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks
    )

    # ---- Stage B: Fine-tune (optional) ----
    history_fine = None
    if DO_FINE_TUNE:
        print(f"\n[Stage B] Fine-tuning last {UNFREEZE_LAST_N_LAYERS} backbone layers...")

        # Unfreeze last N layers of the backbone.
        unfreeze_last_n_layers(backbone, UNFREEZE_LAST_N_LAYERS)

        # IMPORTANT: re-compile after changing layer.trainable values.
        # Use a smaller LR for fine-tuning to avoid destroying pretrained features.
        compile_model(model, lr=1e-5)

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINE,
            callbacks=callbacks
        )

    # ---- Evaluate on validation and (optional) test ----
    print("\nEvaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")

    if test_ds is not None:
        print("\nEvaluating on test set...")
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"Test accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

    return model, history_head, history_fine

# -----------------------------
# 7) Run training for both models
# -----------------------------

if __name__ == "__main__":
    # Train VGG16 transfer-learning model
    model16, hist16_head, hist16_fine = train_model("vgg16")

    # Train VGG19 transfer-learning model
    model19, hist19_head, hist19_fine = train_model("vgg19")

    # Save models (optional)
    # model16.save("vgg16_transfer.h5")
    # model19.save("vgg19_transfer.h5")

    print("\nDone.")
