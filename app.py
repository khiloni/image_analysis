import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = "model.h5"
DRIVE_ID = "1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = download_and_load_model()

st.title("Image Classification App")
st.write("Upload an image for prediction:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))  # Adjust if your model needs different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    st.write("Prediction:", prediction)
