# 🚧 Helmet Detection using YOLOv10 🚧

Welcome to the Helmet Detection project! This application leverages the power of YOLOv10 to detect safety helmets in images, ensuring safety compliance in various environments such as construction sites, factories, and traffic.

## 📜 Overview

This project aims to provide real-time helmet detection using the YOLOv10 model. The model is trained on a custom dataset and deployed using Streamlit to create an interactive web application.

## 🚀 Getting Started

### 1. Train the Model

First, you need to train the model using your custom dataset. Follow these steps:

1. Open and run the `train.ipynb` notebook.
2. Once the training is complete, download the trained model from the path: `runs/detect/train/weights/best.pt`.

### 2. Run the Streamlit Application

After training the model, you can run the Streamlit application to detect helmets in images. Use the following command:

```sh
streamlit run app.py
