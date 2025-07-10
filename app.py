import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Constants
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Download and load the model only once
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

model = download_and_load_model()

# Set up UI config
st.set_page_config(
    page_title="Environmental Monitoring & Land Cover Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header UI
st.markdown("""
    <style>
    .main-title {text-align: center; font-size: 2.5rem; font-weight: bold; color: #ffffff;}
    .subtitle {text-align: center; font-size: 1.2rem; color: #ffffffcc; margin-bottom: 2rem;}
    .header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px;}
    </style>
    <div class='header'>
        <h1 class='main-title'>🛰️ Environmental Monitoring & Land Cover Classification</h1>
        <p class='subtitle'>Classify satellite images into Cloudy, Desert, Green Area, or Water using Deep Learning</p>
    </div>
""", unsafe_allow_html=True)

# Upload Section
st.markdown("""
    ### 🌍 Satellite Image Classification
    Upload a satellite image in **JPG, JPEG, or PNG** format.
""")

uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((256, 256))

    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ fixed deprecated parameter

    # Preprocess
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Making prediction..."):
        prediction = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

    # Display Results
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; color: white; border-radius: 15px;'>
            <h3>🎯 Predicted Class: <b>{predicted_class}</b></h3>
            <p style='font-size: 1.2rem;'>Confidence: <b>{confidence * 100:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence * 100))  # ✅ fixed type error (expects 0–100 int)

    st.markdown("#### 🔍 Class Probabilities")
    for class_name, prob in zip(CLASS_NAMES, prediction):
        st.write(f"{class_name}: {prob * 100:.2f}%")
        st.progress(int(prob * 100))  # ✅ fixed type error here too
