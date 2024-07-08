import streamlit as st
from ultralytics import YOLOv10
import numpy as np
from PIL import Image
import cv2

# Constants and Configuration
MODEL_PATH = 'weights/best.pt'  # Path to your model weights

COLORS = {
    'head': (0, 255, 0),    # Green
    'helmet': (255, 0, 0),  # Blue
    'person': (0, 0, 255)   # Red
}

# Load the model
model = YOLOv10(MODEL_PATH)

# Function to detect helmets in the image


def helmet_detection(image):
    # Convert the image to numpy array
    image_np = np.array(image.convert("RGB"))

    # Use the model to predict bounding boxes and labels
    results = model.predict(image_np)

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2 = map(int, box[:4])
            label = result.names[int(box[5])]
            score = box[4]

            color = COLORS.get(label, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            # Draw label and score
            cv2.putText(image_np, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_np


# Set up the page layout and background
st.set_page_config(
    page_title="Helmet Safety Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for background and footer
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        background-color: #f5f5f5;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
        color: #333333;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Application Logic
st.title("Helmet Safety Detection 🚧")
st.markdown("## Ensure Safety Compliance with Detection")

# Columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"])

with col2:
    st.header("Detection Result")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        result_image = helmet_detection(image)
        if result_image is not None:
            st.image(result_image, caption="Detected Helmets", width=500)

# Footer
st.markdown(
    """
    <div class="footer">
        Made with ❤️ by TRUONGDAT
    </div>
    """,
    unsafe_allow_html=True
)
