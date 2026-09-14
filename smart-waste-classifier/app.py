
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Project paths and constants
# -----------------------------

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "waste_classifier_frozen.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------
# Load model and class names
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)

model = load_model()
class_names = load_class_names()

# -----------------------------
# Waste information
# -----------------------------

waste_information = {
    "metal": {
        "color": "#7f8c8d",
        "advice": "Place clean metal items in the recyclable-waste bin."
    },
    "paper": {
        "color": "#f1c40f",
        "advice": "Keep paper dry and place it in the paper-recycling bin."
    },
    "plastic": {
        "color": "#3498db",
        "advice": "Clean the plastic item and place it in the recyclable-waste bin."
    }
}

# -----------------------------
# App interface
# -----------------------------

st.title("Smart Waste Classification Assistant")

st.write(
    "Upload an image of waste. The machine-learning model will classify it "
    "as metal, paper, or plastic."
)

st.info(
    "This application uses a MobileNetV2 transfer-learning model trained on "
    "labelled waste images."
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    # Prepare image exactly as during model validation
    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.array(resized_image).astype("float32") / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Predict
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction")

    st.success(
        f"Detected category: {predicted_class.title()}"
    )

    st.metric(
        label="Confidence",
        value=f"{confidence:.2%}"
    )

    advice = waste_information.get(
        predicted_class,
        {"advice": "Dispose of this item according to local guidelines."}
    )["advice"]

    st.info(f"Recommended action: {advice}")

    # Probability chart
    st.subheader("Model probability distribution")

    probability_table = pd.DataFrame({
        "Category": [name.title() for name in class_names],
        "Probability": probabilities
    })

    probability_table["Probability"] = (
        probability_table["Probability"] * 100
    ).round(2)

    st.bar_chart(
        probability_table.set_index("Category")
    )

    st.dataframe(
        probability_table,
        hide_index=True,
        use_container_width=True
    )

st.divider()

st.caption(
    "Educational prototype. Predictions may be affected by lighting, "
    "background, image quality, and objects that were not represented in "
    "the training dataset."
)
