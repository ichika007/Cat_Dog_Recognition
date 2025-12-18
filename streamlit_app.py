# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Load the trained model ---
model = tf.keras.models.load_model("cat_dog_model.h5")
class_names = ['Cat', 'Dog']  # Make sure this matches your training

# --- Streamlit UI ---
st.title("Cat vs Dog Image Classifier")
st.write("Upload an image and the AI will predict whether it's a cat or a dog.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    st.write(f"Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: {pred[0][class_idx]*100:.2f}%")
