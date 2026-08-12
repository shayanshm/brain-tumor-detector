# Comparative Analysis of Object Detectors for Brain Tumor Detection

درس یادگیری عمیق — استاد: دکتر باحقیقت
مقایسه‌ی Faster R-CNN (torchvision) در برابر YOLO11 روی دیتاست تصاویر مغزی (glioma/meningioma/pituitary).

## 📍 وضعیت فعلی
**همیشه `ROADMAP.md` را برای وضعیت زنده و دقیق پروژه ببینید** — این README فقط یک نمای کلی ثابت است و
جزئیات لحظه‌ای (کدام فاز باز/بسته است) را تکرار نمی‌کند تا هرگز با ROADMAP.md ناسازگار نشود.

خلاصه‌ی خیلی کلی: فازهای ۰ تا ۳ بسته شده‌اند (زیرساخت، مرور ادبیات، دیتاست، Faster R-CNN baseline).
فاز فعلی: مطابق `ROADMAP.md`.

## ساختار پروژه
```
brain-tumor-detector/
├── ROADMAP.md                 # ← وضعیت زنده و دقیق پروژه (همیشه اینجا را چک کنید)
├── CONTINUITY.md              # راهنمای مدیریت خروجی Kaggle + مقاومت در برابر قطع سشن
├── configs/project_config.yaml # منبع واحد حقیقت: کلاس‌ها، seed، config منجمد، نتایج نهایی هر مدل
├── data/processed/             # خروجی Phase 2.7 (COCO JSON + YOLO detection format)
├── src/
│   ├── data/                   # اسکریپت‌های آماده‌سازی دیتاست (Phase 2)
│   ├── models/                 # Dataset class + معماری‌ها + اسکریپت‌های آموزش (Phase 3, 4)
│   ├── eval/                   # ارزیابی mAP/FPS/Params مستقل از فریمورک (Phase 3.5, 5)
│   ├── robustness/             # Phase 6 (هنوز خالی)
│   ├── explainability/         # Phase 7 (هنوز خالی)
│   ├── stats/                  # Phase 8 (هنوز خالی)
│   └── utils/seed.py
├── outputs/{checkpoints,logs}  # چک‌پوینت‌ها (گیت‌هاب نمی‌شوند، طبق .gitignore) و لاگ‌ها (می‌شوند)
├── reports/{figures,tables}    # گزارش‌های هر فاز + نمودارها
└── requirements.txt
```

## تصمیمات فنی کلیدی (جزئیات کامل + دلیل هرکدام در ROADMAP.md)
- Faster R-CNN: `torchvision.models.detection` (نه Detectron2 — ریسک build)
- YOLO: **YOLO11** (بیشترین شواهد دقت روی brain-tumor MRI در ادبیات)
- دیتاست: همان دیتاست کلاسیک Cheng et al. (۳۰۶۴ تصویر)، annotation اصلی از نوع polygon (نه bbox ساده)
- محیط آموزش: Kaggle Notebook (GPU)، همیشه از طریق **Save & Run All (Commit)**، نه اجرای تعاملی

## قبل از شروع کار روی این ریپو
۱. `ROADMAP.md` را بخوانید تا ببینید دقیقاً کجای پروژه هستیم.
۲. `CONTINUITY.md` را بخوانید — نکات مهم درباره‌ی چه‌چیزی از Kaggle باید نگه‌داشت و چطور از قطع سشن جان سالم به‌در برد.
۳. `configs/project_config.yaml` را ببینید — همه‌ی تصمیمات/نتایج قطعی‌شده آنجاست.
