import os
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Model Download (Direct requests)
# -----------------------------
model_path = "brain_tumor_model.keras"
file_id = "1F7AfBngiMXLosK0iXxZNU4LSBjLipg5Z"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    with st.spinner("Downloading AI model..."):
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Brain Tumor MRI Detection")
st.write("Upload an MRI image and the AI model will predict the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(img)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("Analyzing MRI..."):
        prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # Probabilities Table
    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability (%)": (prediction[0]*100).round(2)
    }).sort_values(by="Probability (%)", ascending=False)
    st.subheader("Prediction Probabilities")
    st.table(prob_df)

    # Confidence progress bar
    st.progress(int(confidence*100))
