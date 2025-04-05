import streamlit as st
import cv2
import numpy as np
import pywt
from PIL import Image
import io

st.title("🧠 Image Processing App with Streamlit")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    # Sidebar for options
    option = st.sidebar.selectbox(
        "Choose Image Processing Operation",
        ("None", "Grayscale", "Blur", "Edge Detection", "Wavelet Transform")
    )

    if option == "Grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray, caption="Grayscale Image", use_container_width=True, channels="GRAY")

    elif option == "Blur":
        blur = cv2.GaussianBlur(image, (15, 15), 0)
        st.image(blur, caption="Blurred Image", use_container_width=True)

    elif option == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        st.image(edges, caption="Edge Detection", use_container_width=True, channels="GRAY")

    elif option == "Wavelet Transform":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coeffs = pywt.dwt2(gray, 'haar')
        cA, (cH, cV, cD) = coeffs

        st.subheader("Wavelet Components")
        st.image(cA, caption="Approximation", use_container_width=True, channels="GRAY")
        st.image(cH, caption="Horizontal Detail", use_container_width=True, channels="GRAY")
        st.image(cV, caption="Vertical Detail", use_container_width=True, channels="GRAY")
        st.image(cD, caption="Diagonal Detail", use_container_width=True, channels="GRAY")
