import streamlit as st
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pca_image_compression import load_grayscale_image, pca_compression, evaluate_metrics, calculate_compression_ratio

st.set_page_config(page_title="PCA Image Compression", layout="centered")
st.title("📉 PCA Image Compression (from Scratch)")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE).astype(float)
    st.image(image.astype(np.uint8), caption="Original Image", use_column_width=True)

    max_components = min(image.shape)
    k = st.slider("Number of principal components", min_value=1, max_value=max_components, value=50)

    with st.spinner("Compressing image with PCA..."):
        reconstructed, _, _ = pca_compression(image, k)
        psnr_val, ssim_val = evaluate_metrics(image, reconstructed)
        reduction = calculate_compression_ratio(image.shape, k)

    st.image(reconstructed.astype(np.uint8), caption=f"Reconstructed Image (k={k})", use_column_width=True)

    st.markdown("---")
    st.subheader("📊 Compression Metrics")
    st.metric("PSNR", f"{psnr_val:.2f} dB")
    st.metric("SSIM", f"{ssim_val:.4f}")
    st.metric("Size Reduction", f"{reduction:.2f}%")

    download_btn = cv2.imencode('.jpg', reconstructed.astype(np.uint8))[1].tobytes()
    st.download_button("📥 Download Reconstructed Image", download_btn, file_name="reconstructed.jpg", mime="image/jpeg")
else:
    st.info("Please upload a grayscale image to begin.")
