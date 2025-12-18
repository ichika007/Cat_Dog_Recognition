import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="cat_dog_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['Cat', 'Dog']

st.title("Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    pred_class = np.argmax(output)
    st.write(f"Prediction: **{class_names[pred_class]}**")
    st.write(f"Confidence: {output[0][pred_class]*100:.2f}%")
