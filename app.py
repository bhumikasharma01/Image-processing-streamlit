import streamlit as st
import cv2
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

st.title("🧠 Image Processing App with Streamlit")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Wavelet Transform
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    cA, (cH, cV, cD) = coeffs2

    def normalize(img):
        return ((np.abs(img) / np.max(np.abs(img))) * 255).astype(np.uint8)

    st.subheader("Wavelet Components")
    col1, col2 = st.columns(2)
    col1.image(normalize(cA), caption="Approximation", clamp=True)
    col2.image(normalize(cH), caption="Horizontal Detail", clamp=True)
