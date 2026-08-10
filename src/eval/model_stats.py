"""
Phase 3.5 — محاسبه‌ی FPS, Parameters, GFLOPs, Model size (مستقل از فریمورک، با fvcore).
"""
import time
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


def compute_model_stats(model, device, image_size: int = 640, n_fps_runs: int = 20):
    model.eval()
    model.to(device)
    dummy_input = [torch.rand(3, image_size, image_size).to(device)]

    # --- Parameters ---
    n_params = parameter_count(model)[""]

    # --- GFLOPs (fvcore با یک ورودی نمونه) ---
    try:
        flops = FlopCountAnalysis(model, (dummy_input,))
        flops.unsupported_ops_warnings(False)
        gflops = flops.total() / 1e9
    except Exception as e:
        gflops = None
        print(f"⚠️ محاسبه‌ی GFLOPs با خطا مواجه شد (برای مدل‌های detection رایج است): {e}")

    # --- FPS (میانگین n_fps_runs بار inference) ---
    with torch.no_grad():
        for _ in range(3):  # warm-up
            model(dummy_input)
        t0 = time.time()
        for _ in range(n_fps_runs):
            model(dummy_input)
        elapsed = time.time() - t0
    fps = n_fps_runs / elapsed

    # --- Model size روی دیسک ---
    tmp_path = Path("/tmp/_model_size_check.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()

    return {
        "params": n_params,
        "params_millions": round(n_params / 1e6, 2),
        "gflops": round(gflops, 2) if gflops else None,
        "fps": round(fps, 2),
        "model_size_mb": round(size_mb, 2),
    }


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.faster_rcnn import build_faster_rcnn
    from models.coco_dataset import NUM_CLASSES_WITH_BACKGROUND

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="مسیر وزن آموزش‌دیده (اختیاری)")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=(args.checkpoint is None))
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    stats = compute_model_stats(model, device, image_size=args.image_size)
    print("== نتایج ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")
