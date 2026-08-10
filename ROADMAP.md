# نقشه راه پروژه — تحلیل مقایسه‌ای دیتکتورهای اشیاء برای تصاویر پزشکی
درس: یادگیری عمیق | استاد: دکتر باحقیقت
دیتاست: Medical Image Dataset: Brain Tumor Detection (Kaggle — pkdarabi)
کلاس‌ها (تاییدشده از data.yaml): glioma, meningioma, pituitary (nc=3)
اسپلیت: train / valid / test (هر سه از قبل در دیتاست موجودند)
نسخه سند: v0.3 | آخرین بروزرسانی: 2026-08-10 | زبان گزارش نهایی: فارسی

## راهنمای وضعیت
⬜ شروع‌نشده   🔄 در حال انجام   ✅ تکمیل و تاییدشده   ⚠️ نیازمند تصمیم/ورودی شما

## قانون بازی (Zero Backtracking)
هر فاز → یک «گیت راستی‌آزمایی» → عبور فقط با تایید صریح شما ممکن است.

## تصمیمات فنی تثبیت‌شده
| مورد | تصمیم | دلیل |
|---|---|---|
| Faster R-CNN | **torchvision.models.detection** (نه Detectron2) | Detectron2 ریسک بالای شکست build با CUDA/torch جدید دارد (تایید شده در Phase 0)؛ torchvision بدون build جداگانه و سازگار با هر نسخه torch است |
| ارزیابی mAP | **pycocotools.COCOeval** | مستقل از فریمورک، استاندارد COCO-style |
| نسخه YOLO | **YOLO11** (ultralytics==8.4.117) | بالاترین/پایدارترین دقت گزارش‌شده در چند مطالعه‌ی مستقل روی brain-tumor MRI |
| کلاس‌های دیتاست | glioma / meningioma / pituitary (۳ کلاس) | تاییدشده از data.yaml واقعی |
| زبان گزارش نهایی | فارسی (XePersian) | طبق دستور شما |
| Seed پروژه | 42 | در `src/utils/seed.py` تست و تایید شد |

---

## Phase 0 — آماده‌سازی محیط و زیرساخت ✅ **[بسته شد]**
- [x] 0.1 ساختار ریپازیتوری پروژه
- [x] 0.2 requirements.txt نسخه‌بندی‌شده و وریفای‌شده روی PyPI
- [x] 0.3 seed.py تست‌شده و کارکرد آن تایید شد
- [x] 0.4 اسکریپت دانلود Kaggle آماده (اجرای واقعی در Phase 2 روی محیط شما)
- [x] 0.5 project_config.yaml با کلاس‌های واقعی
**🚪 گیت ۰: ✅ تایید شد توسط کاربر (تصمیم torchvision تایید شد)**

## Phase 1 — مرور ادبیات (Literature Review) 🔄 **[در انتظار تایید گیت]**
- [x] 1.1 جمع‌آوری ۵ مقاله درباره معماری YOLO/YOLO11
- [x] 1.2 جمع‌آوری ۴ مقاله درباره Explainable AI / Grad-CAM (سازگار با object detection)
- [x] 1.3 جمع‌آوری ۶ مقاله دقیقاً روی همین دامنه (glioma/meningioma/pituitary) — از جمله benchmark مستقیم YOLOv11 vs YOLOv8
- [x] 1.4 جدول خلاصه‌ی ۱۵ مرجع در `reports/literature_review.md`
**🚪 گیت ۱: در انتظار تایید شما**

## Phase 2 — آماده‌سازی دیتاست ✅ **[در انتظار تایید گیت]**
- [x] 2.1 دسترسی به دیتاست روی Kaggle
- [x] 2.2 ساختار خام تایید شد (train=2144, valid=612, test=308)
- [x] 2.3 کلاس‌ها: glioma, meningioma, pituitary
- [x] 2.4 توزیع کلاس‌ها: glioma=1427(47%), meningioma=707(23%), pituitary=930(30%)
- [x] 2.5 ویژوالایز نمونه‌ها — تایید بصری شد
- [x] 2.6 کیفیت annotation — ۰ مشکل از ۳۰۶۴؛ کشف فرمت polygon
- [x] 2.7 تبدیل به COCO JSON + YOLO detection — تایید شد با اجرای واقعی (3064/3064 موفق)
- [x] 2.8 گزارش دیتاست نهایی → `reports/dataset_report.md`
**🚪 گیت ۲: در انتظار تایید شما**

راهنمای مدیریت خروجی/مقاومت در برابر قطع سشن: `CONTINUITY.md`

## Phase 3 — پایه: Faster R-CNN (torchvision) 🔄
- [x] 3.1 مدل `fasterrcnn_resnet50_fpn_v2` با وزن pretrained COCO — تست شد (`faster_rcnn.py`)
- [x] 3.2 Dataset/DataLoader روی COCO JSON — تست شد با forward+loss واقعی (`coco_dataset.py`)
- [x] 3.3 config منجمد: batch=8, epochs=50, SGD(lr=0.005), StepLR — در project_config.yaml
- [x] کد آموزش (`train_faster_rcnn.py`) — قطعات (dataloader/model/optimizer/checkpoint) تست شدند
- [x] کد ارزیابی (`eval/model_stats.py`, `eval/eval_coco.py`) — با assert صحت تست شدند
- [ ] 3.4 اجرای آموزش واقعی روی GPU کگل — **در انتظار اجرای شما** (این قسمت زمان‌بر است، چند ساعت)
- [ ] 3.5 محاسبه‌ی نهایی FPS/Params/GFLOPs/mAP روی نتیجه‌ی واقعی
- [ ] 3.6 گزارش عملکرد پایه
**🚪 گیت ۳: هنوز باز — نیازمند اجرای آموزش واقعی توسط شما**

## Phase 4 — YOLO11 ⬜
- [ ] 4.1-4.5 (طبق الگوی Phase 3، با تحمیل دقیق config منجمدشده)
**🚪 گیت ۴**

## Phase 5 — مقایسه کمّی ⬜
## Phase 6 — استحکام (Robustness) ⬜
## Phase 7 — تبیین‌پذیری (Grad-CAM) ⬜
## Phase 8 — تحلیل آماری ⬜
## Phase 9 — تدوین گزارش نهایی ⬜
## Phase 10 — بحث علمی و نتیجه‌گیری ⬜

(جزئیات کامل فازهای ۵ تا ۱۰ در نسخه v0.2 چت حفظ شده و بدون تغییر باقی مانده‌اند.)
