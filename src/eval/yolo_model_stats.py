"""
Phase 4.4 — محاسبه‌ی FPS/Params/GFLOPs برای YOLO11، با همان روش‌شناسی دقیق Faster R-CNN
(src/eval/model_stats.py) تا مقایسه‌ی Phase 5 apples-to-apples باشد.

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/eval/yolo_model_stats.py \\
        --checkpoint /kaggle/working/brain-tumor-detector/outputs/yolo11/run1/weights/best.pt \\
        --image-size 640
"""
import time
from pathlib import Path

import torch


def compute_yolo_stats(checkpoint_path: str, image_size: int = 640, n_fps_runs: int = 20):
    from ultralytics import YOLO

    model = YOLO(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- Params / GFLOPs: از متد بومی Ultralytics ---
    # نکته: verbose=False در ultralytics==8.4.117 مقدار None برمی‌گرداند (رفتار این نسخه)؛
    # verbose=True هم مقدار صحیح را برمی‌گرداند هم یک خط خلاصه چاپ می‌کند (بی‌ضرر).
    n_layers, n_params, n_gradients, gflops = model.info(verbose=True)

    # --- FPS + GPU Memory: دقیقاً همان روش model_stats.py، برای قابل‌مقایسه‌بودن Phase 5 ---
    underlying_model = model.model
    underlying_model.eval()
    dummy_input = torch.rand(1, 3, image_size, image_size).to(device)

    # --- GPU Memory (نیاز صریح سند: Computational metrics -> GPU memory) ---
    peak_memory_mb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            underlying_model(dummy_input)
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

    with torch.no_grad():
        for _ in range(3):  # warm-up
            underlying_model(dummy_input)
        t0 = time.time()
        for _ in range(n_fps_runs):
            underlying_model(dummy_input)
        elapsed = time.time() - t0
    fps = n_fps_runs / elapsed

    # --- Model size روی دیسک ---
    size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)

    return {
        "params": n_params,
        "params_millions": round(n_params / 1e6, 2),
        "gflops": round(gflops, 2),
        "fps": round(fps, 2),
        "inference_time_ms": round(1000 / fps, 3),
        "gpu_peak_memory_mb": peak_memory_mb,
        "model_size_mb": round(size_mb, 2),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    stats = compute_yolo_stats(args.checkpoint, args.image_size)
    print("== نتایج ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")
