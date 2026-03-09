import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model("accident_model.keras")

IMG_SIZE = 224

st.title("Accident Detection System")
st.write("Upload an image to detect accident or normal traffic.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write(f"Prediction Probability: {prediction:.4f}")

    if prediction < 0.5:
        st.error(" Accident Detected")
    else:
        st.success("Normal Traffic")