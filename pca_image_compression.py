import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def load_grayscale_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image.astype(float)

def pca_compression(image, k):
    mean = np.mean(image, axis=0)
    centered = image - mean

    # Covariance matrix
    cov = np.cov(centered, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx[:k]]

    # Projection
    reduced = np.dot(centered, eigenvectors)
    reconstructed = np.dot(reduced, eigenvectors.T) + mean
    return np.clip(reconstructed, 0, 255), reduced, eigenvectors

def evaluate_metrics(original, reconstructed):
    original = original.astype(np.uint8)
    reconstructed = reconstructed.astype(np.uint8)
    return psnr(original, reconstructed), ssim(original, reconstructed)

def calculate_compression_ratio(original_shape, k):
    original_size = original_shape[0] * original_shape[1]
    compressed_size = (original_shape[0] + original_shape[1]) * k
    return 100 * (1 - (compressed_size / original_size))

def save_image(path, image):
    cv2.imwrite(path, image.astype(np.uint8))
