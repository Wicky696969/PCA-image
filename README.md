# 🧠 PCA Image Compression (From Scratch)

This project demonstrates image compression using **Principal Component Analysis (PCA)** — implemented manually using NumPy, with an interactive interface built in **Streamlit**.

---

## 🚀 Why This Project?

✅ **Resume-Ready Skills**
- Manual PCA using eigen decomposition and covariance matrices
- Real-world application: grayscale image compression
- Evaluation metrics: PSNR, SSIM, Compression Ratio
- Deployed with **Streamlit Cloud** – public app link for demo

✅ **Community Impact**
- Helps learners understand PCA without black-box libraries
- Interactive and visual – great for students and educators
- 100% open-source and reusable

---

## 📦 Tech Stack
- Python, NumPy (manual PCA)
- OpenCV (image handling)
- Scikit-image (metrics)
- Streamlit (UI, interactivity)

---

## 📊 Features
- Upload grayscale image and compress using PCA
- Choose number of components (`k`)
- View original vs reconstructed image
- View PSNR, SSIM, and compression ratio
- Download compressed image

---

## 📎 How to Use

```bash
pip install -r requirements.txt
streamlit run app.py
