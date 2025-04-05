
       import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="🧠 Image Processing App", layout="centered")
st.title("🧠 Image Processing App with Streamlit")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def load_image(img_file):
    image = Image.open(img_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if uploaded_file is not None:
    img = load_image(uploaded_file)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_container_width=True)

    st.sidebar.title("Choose Image Operation")

    option = st.sidebar.selectbox("Select a transformation", [
        "Negative/Inverted Colors",
        "Edge Detection - Canny",
        "Edge Detection - Sobel",
        "Edge Detection - Laplacian",
        "Image Blurring - Gaussian",
        "Image Blurring - Median",
        "Image Blurring - Bilateral",
        "Image Sharpening",
        "Rotate Image",
        "Scale/Resize",
        "Segment using Watershed",
        "Flip Horizontally",
        "Flip Vertically"
    ])

    processed_img = None

    if option == "Negative/Inverted Colors":
        processed_img = cv2.bitwise_not(img)

    elif option == "Edge Detection - Canny":
        threshold1 = st.sidebar.slider("Threshold1", 0, 255, 100)
        threshold2 = st.sidebar.slider("Threshold2", 0, 255, 200)
        processed_img = cv2.Canny(img, threshold1, threshold2)

    elif option == "Edge Detection - Sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        processed_img = cv2.magnitude(sobelx, sobely)
        processed_img = np.uint8(processed_img)

    elif option == "Edge Detection - Laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        processed_img = np.uint8(np.absolute(laplacian))

    elif option == "Image Blurring - Gaussian":
        k = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
        processed_img = cv2.GaussianBlur(img, (k, k), 0)

    elif option == "Image Blurring - Median":
        k = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
        processed_img = cv2.medianBlur(img, k)

    elif option == "Image Blurring - Bilateral":
        d = st.sidebar.slider("Diameter", 1, 20, 9)
        sigma_color = st.sidebar.slider("Sigma Color", 1, 150, 75)
        sigma_space = st.sidebar.slider("Sigma Space", 1, 150, 75)
        processed_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    elif option == "Image Sharpening":
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        processed_img = cv2.filter2D(img, -1, kernel)

    elif option == "Rotate Image":
        angle = st.sidebar.slider("Angle", -180, 180, 45)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed_img = cv2.warpAffine(img, M, (w, h))

    elif option == "Scale/Resize":
        fx = st.sidebar.slider("Scale fx", 0.1, 3.0, 1.0)
        fy = st.sidebar.slider("Scale fy", 0.1, 3.0, 1.0)
        processed_img = cv2.resize(img, None, fx=fx, fy=fy)

    elif option == "Segment using Watershed":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        segmented_img = img.copy()
        segmented_img[markers == -1] = [255, 0, 0]
        processed_img = segmented_img

    elif option == "Flip Horizontally":
        processed_img = cv2.flip(img, 1)

    elif option == "Flip Vertically":
        processed_img = cv2.flip(img, 0)

    if processed_img is not None:
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB) if len(processed_img.shape) == 3 else processed_img,
                 caption='Processed Image',
                 use_container_width=True)
