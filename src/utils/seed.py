"""
سیاست بازتولیدپذیری مشترک پروژه (Phase 0.5).
این تابع باید در ابتدای هر اسکریپت آموزش/ارزیابی (Phase 3, 4, 6, 7, 8) صدا زده شود
تا نتایج هر دو دیتکتور (Faster R-CNN و YOLO11) با شرایط تصادفی‌سازی یکسان مقایسه شوند.
"""
import os
import random

import numpy as np

PROJECT_SEED = 42


def set_seed(seed: int = PROJECT_SEED, deterministic: bool = True) -> None:
    """
    تنظیم seed برای random / numpy / torch (در صورت نصب بودن torch).
    deterministic=True عملکرد cuDNN را کندتر اما کاملاً قابل بازتولید می‌کند؛
    برای اجرای نهایی گزارش‌شده در مقاله True بماند، برای توسعه‌ی سریع می‌توان False گذاشت.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch ممکن است در این مرحله (Phase 0) هنوز نصب نباشد؛ مشکلی نیست.
        pass


if __name__ == "__main__":
    set_seed()
    print(f"Seed تنظیم شد روی {PROJECT_SEED}")
