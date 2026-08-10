# Comparative Analysis of Object Detectors for Brain Tumor Detection

درس یادگیری عمیق — استاد: دکتر باحقیقت
مقایسه‌ی Faster R-CNN (torchvision) در برابر YOLO11 روی دیتاست تصاویر مغزی.

## وضعیت: Phase 0 — تکمیل ⬜→✅ (در انتظار تایید نهایی شما)

## ساختار پروژه
```
brain-tumor-detector/
├── configs/
│   └── project_config.yaml   # منبع واحد حقیقت: کلاس‌ها، seed، config منجمد آموزش (Phase 3.3)
├── data/
│   ├── raw/                  # خروجی Phase 2.1 (دانلود خام کگل)
│   └── processed/            # خروجی Phase 2.7 (COCO JSON + YOLO txt)
├── notebooks/                # نوت‌بوک‌های اکتشافی/EDA
├── src/
│   ├── data/download_kaggle.py
│   ├── models/                # Phase 3, 4
│   ├── eval/                  # Phase 5
│   ├── robustness/            # Phase 6
│   ├── explainability/        # Phase 7
│   ├── stats/                 # Phase 8
│   └── utils/seed.py          # ✅ تست شد
├── reports/{figures,tables}
├── outputs/{checkpoints,logs}
├── requirements.txt           # ✅ نسخه‌ها روی PyPI وریفای شدند (2026-08-10)
└── README.md
```

## تصمیمات فنی این فاز
| مورد | تصمیم | وضعیت تایید |
|---|---|---|
| Faster R-CNN | ~~Detectron2~~ → **torchvision.models.detection** | ⚠️ تغییر یافت — نیازمند تایید شما (ریسک build با CUDA/torch جدید) |
| YOLO | **YOLO11** (ultralytics==8.4.117) | ✅ |
| کلاس‌ها | glioma, meningioma, pituitary (nc=3) | ✅ از data.yaml واقعی |
| Seed پروژه | 42 | ✅ تست شد در `src/utils/seed.py` |

## آنچه واقعاً در sandbox من اجرا و تایید شد (نه فقط ادعا)
- [x] ساخت کامل ساختار پوشه‌بندی
- [x] نصب و تست `numpy`, `pyyaml` و اجرای واقعی `seed.py` (خروجی: "Seed تنظیم شد روی 42")
- [x] parse واقعی `project_config.yaml` + assert صحت کلاس‌ها
- [x] بررسی سینتکسی (`py_compile`) هر دو اسکریپت پایتون
- [x] تایید وجود و صحت نسخه‌ی هر پکیج در `requirements.txt` روی PyPI (`pip index versions`)
- [x] نصب واقعی و بدون خطای پکیج‌های سبک: fvcore, pycocotools, grad-cam, statsmodels, scikit-learn, kaggle

## آنچه اجرا **نشد** (و باید در محیط GPU واقعی شما انجام شود)
- نصب کامل `torch`/`torchvision` (حجم بالا + بدون GPU در sandbox من — بی‌فایده برای تست واقعی)
- دانلود دیتاست از Kaggle (شبکه‌ی sandbox من به kaggle.com دسترسی ندارد)
- Build واقعی هر فریمورک روی GPU (این دقیقاً همان چیزی است که در Phase 3/4 با گزارش کامل خطا/موفقیت انجام می‌دهیم)

## قدم بعدی شما
۱. تصمیم Detectron2→torchvision را تایید یا رد کنید.
۲. فایل `kaggle.json` را طبق راهنمای داخل `src/data/download_kaggle.py` آماده کنید (برای Phase 2).
