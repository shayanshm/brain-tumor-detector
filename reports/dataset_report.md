# گزارش دیتاست — Phase 2 Deliverable
تاریخ: 2026-08-10

## منبع
Kaggle: `pkdarabi/medical-image-dataset-brain-tumor-detection` (فرمت YOLOv11 از Roboflow)
همان دیتاست کلاسیک Cheng et al. (3064 تصویر T1-weighted CE-MRI)، صادرشده به فرمت YOLO Segmentation.

## آمار کلی
| Split | تصاویر | annotation | درصد از کل |
|---|---|---|---|
| train | 2144 | 2144 | 70.0% |
| valid | 612  | 612  | 20.0% |
| test  | 308  | 308  | 10.0% |
| **کل** | **3064** | **3064** | 100% |

## توزیع کلاس‌ها (تعداد bbox، هر تصویر دقیقاً ۱ annotation دارد)
| کلاس | train | valid | test | کل | درصد |
|---|---|---|---|---|---|
| glioma | 983 | 285 | 159 | 1427 | 46.6% |
| meningioma | 503 | 142 | 62 | 707 | 23.1% |
| pituitary | 658 | 185 | 87 | 930 | 30.3% |

⚠️ **Class imbalance**: نسبت glioma به meningioma تقریباً ۲ به ۱ است. توصیه برای Phase 3/4: در گزارش نهایی ذکر شود؛
در صورت نیاز از class-weighted loss یا stratified sampling استفاده شود (نسبت‌های 3 اسپلیت خودشان تقریباً یکسان‌اند، پس حداقل split عادلانه است).

## فرمت annotation — یافته‌ی کلیدی Phase 2.6
داده‌ی خام **YOLO Segmentation Polygon** است (نه bbox ساده): هر خط `class x1 y1 x2 y2 ... xn yn`
(اکثراً ۹ نقطه = ۱۹ ستون). برای Object Detection، bbox محاطی (min/max) استخراج شد.
کیفیت: **۰ annotation خراب از ۳۰۶۴** (بدون class_id نامعتبر، بدون سرریز از مرز، بدون annotation گمشده).

## پایپ‌لاین تبدیل (Phase 2.7)
- `src/data/convert_annotations.py` → دو خروجی موازی تولید کرد:
  1. `data/processed/coco_format/{train,valid,test}.json` — برای Faster R-CNN (torchvision + pycocotools)
  2. `data/processed/yolo_detection/{split}/labels/*.txt` — فرمت bbox خالص برای YOLO11
- نتیجه‌ی اجرای واقعی روی کگل: 3064/3064 annotation با موفقیت تبدیل شد، ۰ خط رد شده.

## نمونه‌های بصری
نمونه‌ی ویژوالایز شده (۹ تصویر تصادفی train با overlay چندضلعی) — annotation ها دقیقاً روی نواحی
غیرطبیعی MRI منطبق‌اند. تصویر واقعی: `reports/figures/dataset_sample_visualization.png`
اسکریپت مولد: `src/data/visualize_samples.py`

## نتیجه‌گیری Phase 2
دیتاست تمیز، کامل، و برای هر دو معماری (Faster R-CNN و YOLO11) آماده است. تنها نکته‌ی قابل‌ذکر برای فصل
گزارش نهایی: imbalance متوسط بین کلاس‌ها و اینکه annotation اصلی از نوع segmentation بوده و به bbox تبدیل شده
(این خودش می‌تواند در بحث محدودیت‌های روش‌شناسی گزارش نهایی ذکر شود — دقت لوکالیزیشن bbox محاطی نسبت به
annotation دستی اصلی ممکن است کمی متفاوت باشد).
