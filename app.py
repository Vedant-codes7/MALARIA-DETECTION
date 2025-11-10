# -*- coding: utf-8 -*-
"""Malaria Detection App - Corrected Version"""

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# ✅ Load your trained model
model = tf.keras.models.load_model('model.h5', compile=False)

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    """Resize and normalize image"""
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Prediction Logic (Fixed)
# -------------------------------
def predict(image):
    """Run malaria detection prediction"""
    if image is None:
        return "⚠️ Please upload an image."

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]

    # ✅ Fixed logic: flip conditions because model outputs 1 = Uninfected, 0 = Parasitized
    if prediction > 0.5:
        result = "✅ **UNINFECTED** (No Malaria)"
        confidence = prediction * 100
        color = "green"
    else:
        result = "🦠 **PARASITIZED** (Malaria Detected)"
        confidence = (1 - prediction) * 100
        color = "red"

    # Return formatted result with color & confidence
    html_result = f"""
    <div style='text-align:center; font-size:22px; color:{color}; font-weight:bold;'>
        {result}<br>
        <span style='font-size:18px; color:black;'>Confidence: {confidence:.2f}%</span>
    </div>
    """
    return html_result

# -------------------------------
# Gradio UI
# -------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="🩸 Upload Blood Cell Image"),
    outputs=gr.HTML(label="Diagnosis Result"),
    title="🔬 Malaria Detection AI System",
    description=(
        "Upload a microscopic image of a blood cell, and this AI model will detect "
        "whether it is **Parasitized** (Malaria detected) or **Uninfected** (Healthy)."
    ),
    theme="soft"
)

# -------------------------------
# Launch the App
# -------------------------------
if __name__ == "__main__":
    interface.launch(enable_queue=False)
