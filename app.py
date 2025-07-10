import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Google Drive model link
DRIVE_FILE_ID = "1GspKyldumkNt3O3a8ZIl0hJpyqBvtLf7"
MODEL_PATH = "Modelenv.v1.h5"

# Download and load model
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = download_and_load_model()

# Class labels
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Streamlit UI
st.title("🌍 Satellite Image Classifier")
st.markdown("Upload a satellite image to classify it as **Cloudy, Desert, Green Area, or Water**.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
