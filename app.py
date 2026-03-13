import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import streamlit as st
import numpy as np
import os
import pandas as pd

# ---------------------------
# Download model from Drive if not exists
# ---------------------------
drive_url = "https://drive.google.com/file/d/1F7AfBngiMXLosK0iXxZNU4LSBjLipg5Z/view?usp=drive_link"
output_model = "brain_tumor_model.keras"

if not os.path.exists(output_model):
    with st.spinner("Downloading model..."):
        gdown.download(drive_url, output_model, quiet=False)

# Load model
model = tf.keras.models.load_model(output_model)

classes = ["glioma","meningioma","notumor","pituitary"]

# ---------------------------
# App UI
# ---------------------------
st.title("🧠 Brain Tumor MRI Detection")
st.write("Upload an MRI image to detect tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)  # fix orientation if needed
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = img.resize((224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction with loader
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
            "Probability": (prediction[0]*100).round(2)
        }).sort_values(by="Probability", ascending=False)
        st.subheader("Class Probabilities")
        st.table(prob_df)

        # Optional: progress bar for confidence
        st.progress(int(confidence*100))

    except Exception as e:
        st.error(f"Error processing image: {e}")
