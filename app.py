import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps
import streamlit as st
import numpy as np
import pandas as pd
import os

# ---------------------------
# Google Drive model download
# ---------------------------
drive_url = "https://drive.google.com/uc?id=1F7AfBngiMXLosK0iXxZNU4LSBjLipg5Z"  # Direct download link
output_model = "brain_tumor_model.keras"

if not os.path.exists(output_model):
    with st.spinner("Downloading EfficientNetB0 model..."):
        gdown.download(drive_url, output_model, quiet=False)

# Load model
model = tf.keras.models.load_model(output_model)

classes = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧠 Brain Tumor MRI Detection")
st.write("Upload an MRI image to detect tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)  # fix orientation
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image for EfficientNetB0
        img = img.resize((224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # EfficientNet preprocessing

        # Prediction
        with st.spinner("Predicting..."):
            prediction = model.predict(img_array)

        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display prediction
        st.success(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # Display probabilities table
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability (%)": (prediction[0]*100).round(2)
        }).sort_values(by="Probability (%)", ascending=False)
        st.subheader("Class Probabilities")
        st.table(prob_df)

        # Confidence progress bar
        st.progress(int(confidence*100))

    except Exception as e:
        st.error(f"Error processing image: {e}")
