import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import urllib.request

# --- Load labels for ImageNet
@st.cache_resource
def load_labels():
    labels_path = "imagenet_labels.txt"
    if not os.path.exists(labels_path):
        url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        urllib.request.urlretrieve(url, labels_path)
    with open(labels_path, "r") as f:
        return f.read().splitlines()

# --- Load the model
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5")
    return model

# --- Preprocess image
def preprocess(image):
    image = image.convert("RGB")
    img = image.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# --- Classify image
def classify(model, image, labels):
    try:
        processed_image = preprocess(image)
        predictions = model(processed_image).numpy()
        top_k = predictions[0].argsort()[-9:][::-1]  # Changed from 3 to 5
        results = [(labels[i], predictions[0][i]) for i in top_k]
        return results
    except Exception as e:
        st.error(f"Failed to classify image. Reason: {e}")
        return None


# --- Streamlit app
def main():
    st.set_page_config(page_title="Classify an image!", page_icon="🤖", layout="centered")
    st.title("Classifier")
    st.write("Upload an image — AI will tell you what it sees!")

    model = load_model()
    labels = load_labels()

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="The uploaded file is:", use_container_width=True)
        btn = st.button("AI Classify!")

        if btn:
            with st.spinner("Good things are coming along your way."):
                image = Image.open(uploaded_file)
                results = classify(model, image, labels)

                if results:
                    st.subheader("Top Predictions:")
                    for label, score in results:
                        st.write(f"**{label}** — {score:.2%}")

if __name__ == "__main__":
    import os
    main()
