import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19

# -----------------------------
# 1) Load pretrained models
# -----------------------------
model_vgg16 = tf.keras.applications.VGG16(weights="imagenet")
model_vgg19 = tf.keras.applications.VGG19(weights="imagenet")

# -----------------------------
# 2) Preprocess helper
# -----------------------------
def load_and_preprocess(img_path, target_size=(224, 224), which="vgg16"):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)            # (224, 224, 3)
    x = np.expand_dims(x, axis=0)          # (1, 224, 224, 3)

    if which.lower() == "vgg16":
        x = vgg16.preprocess_input(x)      # BGR + mean subtraction (Caffe style)
    elif which.lower() == "vgg19":
        x = vgg19.preprocess_input(x)
    else:
        raise ValueError("which must be 'vgg16' or 'vgg19'")
    return x

# -----------------------------
# 3) Predict helper
# -----------------------------
def predict_topk(model, x, which="vgg16", topk=5):
    preds = model.predict(x, verbose=0)
    if which.lower() == "vgg16":
        decoded = vgg16.decode_predictions(preds, top=topk)[0]
    else:
        decoded = vgg19.decode_predictions(preds, top=topk)[0]
    return decoded

# -----------------------------
# 4) Run on an image
# -----------------------------
img_path = "your_image.jpg"  # <-- change this

x16 = load_and_preprocess(img_path, which="vgg16")
x19 = load_and_preprocess(img_path, which="vgg19")

preds16 = predict_topk(model_vgg16, x16, which="vgg16", topk=5)
preds19 = predict_topk(model_vgg19, x19, which="vgg19", topk=5)

print("\nVGG16 Top-5:")
for _, label, prob in preds16:
    print(f"{label:25s} {prob:.4f}")

print("\nVGG19 Top-5:")
for _, label, prob in preds19:
    print(f"{label:25s} {prob:.4f}")
