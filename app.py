import streamlit as st
import tensorflow as tf
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/file/d/1GspKyldumkNt3O3a8ZIl0hJpyqBvtLf7/view?usp=drive_link"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return load_model(MODEL_PATH)

model = download_and_load_model()

st.title("Deep Learning Model Deployment")
st.write("Upload an image to get prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))  # Adjust based on your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    st.write("Prediction:", prediction)
