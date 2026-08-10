"""
دانلود دیتاست از Kaggle (برای اجرای Phase 2.1).
این اسکریپت در sandbox من اجرا نشده چون دسترسی شبکه من به kaggle.com محدود است —
باید در محیط شما (Kaggle Notebook / Colab / لوکال با kaggle.json) اجرا شود.

پیش‌نیاز:
  1. از https://www.kaggle.com/settings -> API -> Create New Token فایل kaggle.json را بگیرید.
  2. آن را در ~/.kaggle/kaggle.json قرار دهید (chmod 600).
  (در Kaggle Notebook این مرحله لازم نیست؛ دیتاست را مستقیم از پنل Add Data اضافه کنید.)

اجرا:
  python src/data/download_kaggle.py
"""
import zipfile
from pathlib import Path

DATASET_SLUG = "pkdarabi/medical-image-dataset-brain-tumor-detection"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"در حال دانلود {DATASET_SLUG} در {RAW_DIR} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), quiet=False)

    zip_path = RAW_DIR / f"{DATASET_SLUG.split('/')[-1]}.zip"
    if zip_path.exists():
        print("استخراج فایل zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    print("دانلود کامل شد. محتویات:")
    for p in sorted(RAW_DIR.rglob("*"))[:30]:
        print(" -", p.relative_to(RAW_DIR))


if __name__ == "__main__":
    download()
