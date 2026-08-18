"""
Phase 6.1 — پایپ‌لاین corruption روی تصاویر test، طبق ۴ دسته‌ی سند اصلی:
Lighting (darker/brighter), Noise (Gaussian/Salt&Pepper), Blur (Gaussian/Motion),
Compression (JPEG 20%/50%).

هر تابع یک PIL.Image می‌گیرد و یک PIL.Image برمی‌گرداند (RGB) -- مستقل از فریمورک تشخیص،
هم برای Faster R-CNN هم YOLO11 قابل استفاده است.
"""
import io

import cv2
import numpy as np
from PIL import Image


def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------- Lighting ----------
def darker(img: Image.Image, factor: float = 0.5) -> Image.Image:
    arr = _to_np(img).astype(np.float32) * factor
    return _to_pil(arr)


def brighter(img: Image.Image, factor: float = 1.6) -> Image.Image:
    arr = _to_np(img).astype(np.float32) * factor
    return _to_pil(arr)


# ---------- Noise ----------
def gaussian_noise(img: Image.Image, sigma: float = 25.0, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = _to_np(img).astype(np.float32)
    noise = rng.normal(0, sigma, arr.shape)
    return _to_pil(arr + noise)


def salt_pepper_noise(img: Image.Image, amount: float = 0.03, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = _to_np(img).copy()
    h, w, _ = arr.shape
    n_salt = int(amount * h * w / 2)
    n_pepper = int(amount * h * w / 2)

    ys = rng.integers(0, h, n_salt)
    xs = rng.integers(0, w, n_salt)
    arr[ys, xs] = 255

    ys = rng.integers(0, h, n_pepper)
    xs = rng.integers(0, w, n_pepper)
    arr[ys, xs] = 0

    return _to_pil(arr)


# ---------- Blur ----------
def gaussian_blur(img: Image.Image, ksize: int = 9) -> Image.Image:
    arr = _to_np(img)
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return _to_pil(blurred)


def motion_blur(img: Image.Image, ksize: int = 15) -> Image.Image:
    arr = _to_np(img)
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = 1.0 / ksize  # حرکت افقی
    blurred = cv2.filter2D(arr, -1, kernel)
    return _to_pil(blurred)


# ---------- Compression ----------
def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {
    "darker": darker,
    "brighter": brighter,
    "gaussian_noise": gaussian_noise,
    "salt_pepper_noise": salt_pepper_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg_20": lambda img: jpeg_compress(img, 20),
    "jpeg_50": lambda img: jpeg_compress(img, 50),
}
