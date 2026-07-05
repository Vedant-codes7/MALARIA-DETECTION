# Malaria Detection Using Deep Learning

A deep learning-based malaria detection system that classifies cell images as **Parasitized** or **Uninfected** using a VGG19-based Convolutional Neural Network with Grad-CAM explainability.

The system assists in automated malaria screening by analyzing microscopic blood cell images and highlighting important regions influencing model predictions.

---

## Live Demo

Hugging Face Space: https://vedant07777-malaria-detection.hf.space/

---

## Features

- Malaria infected cell classification
- Deep learning-based image analysis
- Transfer learning using VGG19
- Grad-CAM visualization for explainable AI
- Image preprocessing pipeline
- Real-time prediction interface
- Web deployment using Hugging Face Spaces

---

## Tech Stack

### Deep Learning

- TensorFlow
- Keras
- VGG19
- Convolutional Neural Networks (CNN)

### Computer Vision

- OpenCV
- Image Processing
- Grad-CAM Visualization

### Backend / Deployment

- Python
- Gradio
- Hugging Face Spaces

---

## Dataset

Dataset used:

Malaria Cell Images Dataset

Classes:

```
Parasitized
Uninfected
```

The dataset contains microscopic images of blood cells used to train the classification model.

---

## Model Architecture

```
Input Cell Image
        |
        ↓
Image Preprocessing
        |
        ↓
VGG19 Feature Extractor
        |
        ↓
Fully Connected Layers
        |
        ↓
Classification Layer
        |
        ↓
Prediction

Parasitized / Uninfected
```

---

## Explainable AI

Grad-CAM is integrated to visualize important image regions used by the model while making predictions.

This helps provide transparency instead of treating the neural network as a black box.

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPO_LINK
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

---

## Project Structure

```
MALARIA-DETECTION

├── app.py
├── model
│   └── malaria_vgg19_model.h5
├── requirements.txt
├── utils
│   ├── preprocessing.py
│   └── gradcam.py
└── README.md
```

---

## AI Workflow

```
Upload Cell Image
        |
        ↓
Resize and Normalize Image
        |
        ↓
VGG19 CNN Model
        |
        ↓
Prediction Generation
        |
        ↓
Grad-CAM Heatmap
        |
        ↓
Final Diagnosis Result
```

---

## Future Improvements

- Improve model accuracy with larger datasets
- Add more explainability techniques
- Deploy REST API backend
- Mobile application integration
- Multi-disease cell classification

---

## Author

**Vedant Ahuja**

AI | Machine Learning | Deep Learning | Computer Vision
