"""
Educational GAN (Keras / TensorFlow)

Goal:
- Show how a Generator creates fake samples from noise
- Show how a Discriminator learns to tell real vs fake
- Train both in an adversarial loop on a tiny 2D toy dataset

This keeps the math visible and avoids dataset downloads.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# If needed:
# pip install tensorflow matplotlib

# -----------------------------
# 0) Reproducibility + settings
# -----------------------------

SEED = 7
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

LATENT_DIM = 8
BATCH_SIZE = 128
STEPS = 3000
PRINT_EVERY = 300

# -----------------------------
# 1) Toy "real" data generator
# -----------------------------

def sample_real(batch_size: int) -> np.ndarray:
    """
    Real data distribution: a simple 2D mixture of Gaussians.
    This makes it easy to visualize what the Generator should learn.
    """
    half = batch_size // 2
    a = np.random.randn(half, 2) * 0.4 + np.array([-2.0, 0.0])
    b = np.random.randn(batch_size - half, 2) * 0.4 + np.array([2.0, 0.0])
    x = np.vstack([a, b]).astype("float32")
    np.random.shuffle(x)
    return x

# -----------------------------
# 2) Build the Generator
# -----------------------------

def build_generator(latent_dim: int) -> keras.Model:
    """
    Generator maps noise z -> fake sample (x, y).
    Output is linear because the real data is unbounded in 2D.
    """
    inputs = layers.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, name="fake_xy")(x)
    return keras.Model(inputs, outputs, name="generator")

# -----------------------------
# 3) Build the Discriminator
# -----------------------------

def build_discriminator() -> keras.Model:
    """
    Discriminator maps a 2D point -> logit (real vs fake).
    We return logits (no sigmoid) for numerical stability.
    """
    inputs = layers.Input(shape=(2,), name="xy")
    x = layers.Dense(32)(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Dense(1, name="logit")(x)
    return keras.Model(inputs, outputs, name="discriminator")

# -----------------------------
# 4) Training utilities
# -----------------------------

def plot_samples(real_xy: np.ndarray, fake_xy: np.ndarray, title: str) -> None:
    """Quick scatter plot to compare real vs generated samples."""
    plt.figure(figsize=(5, 4))
    plt.scatter(real_xy[:, 0], real_xy[:, 1], s=10, alpha=0.5, label="Real")
    plt.scatter(fake_xy[:, 0], fake_xy[:, 1], s=10, alpha=0.5, label="Fake")
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def demo_forward_pass(generator: keras.Model, discriminator: keras.Model) -> None:
    """
    Show how data flows through the models before training.
    Prints shapes and a few example values.
    """
    z = tf.random.normal([5, LATENT_DIM])
    fake = generator(z, training=False)
    logits = discriminator(fake, training=False)
    probs = tf.sigmoid(logits)

    print("Generator input z shape:", z.shape)
    print("Generator output (fake samples) shape:", fake.shape)
    print("Discriminator logits shape:", logits.shape)
    print("Discriminator probs (first 3):", probs[:3, 0].numpy())

# -----------------------------
# 5) Main training loop
# -----------------------------

def train_gan() -> None:
    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator()

    # Binary cross-entropy with logits (stable)
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    # Separate optimizers for each network
    g_opt = keras.optimizers.Adam(learning_rate=1e-3)
    d_opt = keras.optimizers.Adam(learning_rate=1e-3)

    # Show initial forward pass (before training)
    demo_forward_pass(generator, discriminator)

    for step in range(1, STEPS + 1):
        # -----------------------------
        # A) Train Discriminator
        # -----------------------------
        real_xy = sample_real(BATCH_SIZE)
        z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        fake_xy = generator(z, training=True)

        with tf.GradientTape() as d_tape:
            real_logits = discriminator(real_xy, training=True)
            fake_logits = discriminator(fake_xy, training=True)
            d_loss_real = bce(tf.ones_like(real_logits), real_logits)
            d_loss_fake = bce(tf.zeros_like(fake_logits), fake_logits)
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_weights)
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_weights))

        # -----------------------------
        # B) Train Generator
        # -----------------------------
        z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        with tf.GradientTape() as g_tape:
            fake_xy = generator(z, training=True)
            # Discriminator is used, but we do NOT update its weights here
            fake_logits = discriminator(fake_xy, training=False)
            g_loss = bce(tf.ones_like(fake_logits), fake_logits)

        g_grads = g_tape.gradient(g_loss, generator.trainable_weights)
        g_opt.apply_gradients(zip(g_grads, generator.trainable_weights))

        # -----------------------------
        # C) Simple logging
        # -----------------------------
        if step % PRINT_EVERY == 0 or step == 1:
            real_prob = tf.sigmoid(real_logits).numpy().mean()
            fake_prob = tf.sigmoid(fake_logits).numpy().mean()
            print(
                f"Step {step:04d} | D_loss={d_loss.numpy():.3f} | "
                f"G_loss={g_loss.numpy():.3f} | D(real)={real_prob:.2f} | "
                f"D(fake)={fake_prob:.2f}"
            )

    # Visualize final samples
    real_xy = sample_real(500)
    z = tf.random.normal([500, LATENT_DIM])
    fake_xy = generator(z, training=False).numpy()
    plot_samples(real_xy, fake_xy, title="GAN: Real vs Generated (2D Toy Data)")

# -----------------------------
# 6) Run
# -----------------------------

if __name__ == "__main__":
    train_gan()
