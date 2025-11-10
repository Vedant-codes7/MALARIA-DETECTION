import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Load model
model = tf.keras.models.load_model("model.h5", compile=False)

# Stats file
STATS_FILE = "stats.json"

# Load or create stats
if os.path.exists(STATS_FILE):
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
else:
    stats = {"total": 0, "infected": 0, "uninfected": 0, "avg_conf": 0.0}

def update_stats(result, conf):
    stats["total"] += 1
    if "PARASITIZED" in result:
        stats["infected"] += 1
    else:
        stats["uninfected"] += 1
    # Update average confidence
    stats["avg_conf"] = round((stats["avg_conf"] * (stats["total"] - 1) + conf) / stats["total"], 2)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def predict(image):
    if image is None:
        return "Please upload an image", None, None, None, None

    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    
    if pred > 0.5:
        result = " PARASITIZED (Malaria Detected)"
        conf = round(pred * 100, 2)
    else:
        result = " Uninfected (Malaria not Detected)"
        conf = round((1 - pred) * 100, 2)
    
    update_stats(result, conf)
    
    return result, f"{conf}%", stats["total"], stats["infected"], stats["uninfected"]

def dashboard_view():
    return f"""
    <h3>📊 Dashboard Overview</h3>
    <p><b>Total Analyses:</b> {stats['total']}</p>
    <p><b>Uninfected:</b> {stats['uninfected']}</p>
    <p><b>Infected:</b> {stats['infected']}</p>
    <p><b>Average Confidence:</b> {stats['avg_conf']}%</p>
    """

# Build Gradio app
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🔬 Malaria Detection AI System")
    gr.Markdown("Upload a microscopic image to analyze and view live dashboard results.")
    
    with gr.Tabs():
        with gr.TabItem("🩸 Detector"):
            image = gr.Image(type="pil", label="Upload Image")
            analyze_btn = gr.Button("Analyze")
            result = gr.Textbox(label="Result")
            confidence = gr.Textbox(label="Confidence")
            total = gr.Number(label="Total Analyses", interactive=False)
            infected = gr.Number(label="Infected Count", interactive=False)
            uninfected = gr.Number(label="Uninfected Count", interactive=False)
            
            analyze_btn.click(
                fn=predict,
                inputs=image,
                outputs=[result, confidence, total, infected, uninfected]
            )
        
        with gr.TabItem("📊 Dashboard"):
            refresh_btn = gr.Button("🔄 Refresh Dashboard")
            dash = gr.HTML(value=dashboard_view())
            refresh_btn.click(fn=lambda: dashboard_view(), outputs=dash)

demo.launch()

