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

## Phase 3 — پایه: Faster R-CNN (torchvision) ✅ **[در انتظار تایید گیت]**
- [x] 3.1-3.4 مدل/آموزش کامل — 50/50 epoch، ۱۰.۰۴ ساعت، loss کاهش ۹۶.۲٪
- [x] 3.5 مAP/Precision/Recall/FPS/Params/GFLOPs — همه محاسبه و تایید شد:
      mAP@0.5(test)=0.818 | mAP@0.5:0.95(test)=0.593 | AR@100(test)=0.670 |
      Params=43.27M | GFLOPs=280.38 | FPS=7.53 | Size=165.41MB
      بدون نشانه overfitting (منحنی val پایدار با وجود کاهش شدید train loss)
- [x] 3.6 گزارش نهایی → `reports/faster_rcnn_performance_report.md`
      + فایل‌های خام: `outputs/logs/faster_rcnn_val_curve.csv`, `faster_rcnn_test_eval.txt`, `faster_rcnn_model_stats.txt`
**🚪 گیت ۳: در انتظار تایید شما**

⚠️ **درس گرفته‌شده (اعمال‌شده برای Phase 4):** train_faster_rcnn.py فقط train loss لاگ می‌کرد؛ mAP/Recall
باید عقب‌گرد از چک‌پوینت‌ها بازسازی می‌شد. **برای Phase 4 این مشکل رخ نخواهد داد** چون Ultralytics YOLO
به‌صورت built-in و خودکار در هر epoch یک `results.csv` با ستون‌های
precision/recall/mAP50/mAP50-95 (هم train هم val) می‌نویسد — بدون نیاز به کد اضافه.

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
