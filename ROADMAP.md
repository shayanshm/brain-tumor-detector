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

## Phase 1 — مرور ادبیات (Literature Review) ✅ **[بسته شد]**
- [x] 1.1 جمع‌آوری ۵ مقاله درباره معماری YOLO/YOLO11
- [x] 1.2 جمع‌آوری ۴ مقاله درباره Explainable AI / Grad-CAM (سازگار با object detection)
- [x] 1.3 جمع‌آوری ۶ مقاله دقیقاً روی همین دامنه (glioma/meningioma/pituitary) — از جمله benchmark مستقیم YOLOv11 vs YOLOv8
- [x] 1.4 جدول خلاصه‌ی ۱۵ مرجع در `reports/literature_review.md`
**🚪 گیت ۱: ✅ تایید شد**

## Phase 2 — آماده‌سازی دیتاست ✅ **[بسته شد]**
- [x] 2.1 دسترسی به دیتاست روی Kaggle
- [x] 2.2 ساختار خام تایید شد (train=2144, valid=612, test=308)
- [x] 2.3 کلاس‌ها: glioma, meningioma, pituitary
- [x] 2.4 توزیع کلاس‌ها: glioma=1427(47%), meningioma=707(23%), pituitary=930(30%)
- [x] 2.5 ویژوالایز نمونه‌ها — تایید بصری شد + تصویر واقعی ذخیره شد در `reports/figures/dataset_sample_visualization.png`
- [x] 2.6 کیفیت annotation — ۰ مشکل از ۳۰۶۴؛ کشف فرمت polygon
- [x] 2.7 تبدیل به COCO JSON + YOLO detection — تایید شد با اجرای واقعی (3064/3064 موفق)
- [x] 2.8 گزارش دیتاست نهایی → `reports/dataset_report.md`
**🚪 گیت ۲: ✅ تایید شد**

راهنمای مدیریت خروجی/مقاومت در برابر قطع سشن: `CONTINUITY.md`

## Phase 3 — پایه: Faster R-CNN (torchvision) ✅ **[بسته شد]**
- [x] مدل، Dataset، augmentation (Horizontal Flip p=0.5 + Brightness/Contrast ±20%)، و آموزش کامل
      — ۵۰ epoch، ۱۰.۳۷ ساعت روی GPU کگل
- [x] 3.5 نتایج نهایی روی TEST set:
      mAP@0.5=0.853 | mAP@0.5:0.95=0.630 | mAP@0.75=0.707 | AR@100=0.702 |
      Params=43.27M | GFLOPs=280.38 | FPS=8.12 | Size=165.41MB
      بدون نشانه‌ی overfitting (منحنی val از epoch ۲۰ به بعد پایدار و بالا ماند)
- [x] تحلیل جانبی Ablation (اثر augmentation): بهبود یکنواخت +3.2 تا +3.8 واحد درصد در همه‌ی معیارها
      نسبت به یک اجرای کنترل بدون augmentation با شرایط کاملاً یکسان
- [x] 3.6 گزارش نهایی → `reports/faster_rcnn_performance_report.md`
      + نمودارها: `faster_rcnn_loss_curve.png`, `faster_rcnn_map_curve.png`, `faster_rcnn_augmentation_ablation.png`
      + فایل‌های خام: `outputs_v3/logs/*` (نتیجه‌ی نهایی) و `outputs_ablation_baseline/logs/*` (کنترل بدون aug)
      + `configs/project_config.yaml -> results.faster_rcnn` و `results.faster_rcnn_ablation_no_augmentation`
**🚪 گیت ۳: ✅ تایید شد (نهایی)**

⚠️ **درس گرفته‌شده (اعمال‌شده برای Phase 4):** train_faster_rcnn.py فقط train loss لاگ می‌کرد؛ mAP/Recall
باید عقب‌گرد از چک‌پوینت‌ها بازسازی می‌شد. **برای Phase 4 این مشکل رخ نخواهد داد** چون Ultralytics YOLO
به‌صورت built-in و خودکار در هر epoch یک `results.csv` با ستون‌های
precision/recall/mAP50/mAP50-95 (هم train هم val) می‌نویسد — بدون نیاز به کد اضافه.

## Phase 4 — YOLO11 ✅ **[بسته شد]**
- [x] آموزش کامل — ۵۰ epoch، فقط ۰.۴۱ ساعت (۲۵ برابر سریع‌تر از Faster R-CNN)
- [x] 4.4 نتایج نهایی روی TEST set: mAP50=0.886 | mAP50-95=0.639 | P=0.873 | R=0.829 |
      Params=2.59M | GFLOPs=6.5 | FPS=102.01 | Size=5.22MB
      بدون نشانه‌ی overfitting؛ چک‌پوینت best.pt در epoch~۴۳
- [x] یافته‌ی کلیدی per-class: glioma (بزرگ‌ترین کلاس داده) ضعیف‌ترین عملکرد را در هر دو مدل دارد
      (احتمالاً به‌خاطر مرزهای پخش‌تر تومور در MRI، نه کمبود داده)
- [x] 4.5 گزارش نهایی → `reports/yolo11_performance_report.md`
      + نمودارها: `yolo11_training_curves.png`, `yolo11_confusion_matrix.png`
      + فایل‌های خام: `outputs/yolo11/run1/{args.yaml, results.csv, model_stats.txt, test_eval.txt}`
      + `configs/project_config.yaml -> results.yolo11`
**🚪 گیت ۴: ✅ تایید شد**

## Phase 5 — مقایسه کمّی عملکرد ✅ **[بسته شد]**
- [x] پایپ‌لاین یکپارچه‌ی Precision/Recall/F1/IoU/Dice ساخته، تست، و روی داده‌ی واقعی اجرا شد
- [x] 5.1 نتایج (conf>=0.5): F1 -- FR-CNN=0.799 در برابر YOLO11=0.830؛ IoU/Dice برعکس (FR-CNN کمی بالاتر)
- [x] یافته‌ی per-class (سه‌بار مستقل تایید شده): glioma ضعیف‌ترین F1 در هر دو مدل (پشتوانه‌ی علمی کامل)
- [x] 5.2 معیارهای محاسباتی تجمیع شد: YOLO11 در Params(۱۷x)، GFLOPs(۴۳x)، FPS(۱۳x)، Train time(۲۵x) برتر
- [x] 5.3 گزارش نهایی → `reports/phase5_comparison_report.md`
      + ۴ نمودار: `phase5_detection_metrics.png`, `phase5_map_official.png`,
        `phase5_computational.png`, `phase5_per_class_f1.png`
      + `configs/project_config.yaml -> phase5_unified_comparison`
      + داده‌های خام: `outputs_v3/logs/faster_rcnn_test_predictions.json`,
        `outputs/yolo11/run1/yolo_test_predictions.json`, `data/processed/coco_format/test.json`
**🚪 گیت ۵: ✅ تایید شد (نهایی — شامل GPU memory)**

## Phase 6 — ارزیابی استحکام (Robustness) 🔄 **[فاز فعلی]**
- [x] 6.1 پایپ‌لاین ۸ نوع corruption — تست کمّی و بصری شد
      ⚠️ **اصلاح 2026-08-19:** تصویر پیش‌نمایش قبلی (`corruption_types_preview.png`) روی یک الگوی
      **مصنوعی** بود (نه MRI واقعی) — فقط برای تست کد کافی بود، برای گزارش نهایی نیست.
      **باید تکرار شود:** یک تصویر واقعی test از کاربر گرفته و پیش‌نمایش واقعی جایگزین شود.
- [x] کد اجرای هر دو مدل روی همه‌ی ۹ حالت — تست End-to-End کامل
- [x] کد تحلیل افت عملکرد — تست با سناریوی کنترل‌شده
- [ ] 6.2 اجرای واقعی روی Kaggle — در انتظار اجرای شما
- [ ] 6.3/6.4 تحلیل و گزارش نهایی
**🚪 گیت ۶: هنوز باز**

## ⚠️ نقص باز از Phase 5 (اصلاح 2026-08-19)
GPU memory آموزش YOLO11 هرگز واقعاً اندازه‌گیری نشد (فقط توضیح داده شد چرا سخت است — که کافی نبود).
**باید تکرار شود:** یا لاگ کنسول آموزش ۵۰-epoch اصلی (اگر هنوز دارید) چک شود، یا یک آموزش
۱-epoch‌ای کوتاه (~۳۰ ثانیه) فقط برای خواندن ستون `GPU_mem` از خروجی خود Ultralytics اجرا شود.
جزئیات دستور در ادامه‌ی چت.

## Phase 7 — تحلیل تبیین‌پذیری (Grad-CAM) ⬜
- [ ] 7.1 انتخاب کتابخانه سازگار با هر دو معماری (pytorch-grad-cam: EigenCAM/AblationCAM برای YOLO،
      Grad-CAM استاندارد روی backbone برای Faster R-CNN)
- [ ] 7.2 تولید heatmap برای: پیش‌بینی‌های خوب، پیش‌بینی‌های بد، failure cases — هر دو مدل
- [ ] 7.3 تحلیل کیفی (تمرکز روی ضایعه یا پس‌زمینه؟)
- [ ] 7.4 تولید «گزارش تبیین‌پذیری»
**🚪 گیت ۷**

## Phase 8 — تحلیل آماری ⬜
- [ ] 8.1 تعریف آرایه‌های زوجی per-image (مثلاً IoU/AP هر تصویر) برای هر دو دیتکتور
- [ ] 8.2 Bootstrap Confidence Intervals روی معیارهای کلیدی
- [ ] 8.3 Paired Permutation Test / Wilcoxon Signed-Rank Test بین دو دیتکتور
- [ ] 8.4 McNemar's Test (در صورت وجود تصمیم‌های matched detection)
- [ ] 8.5 گزارش p-values, CI ها, effect sizes
**🚪 گیت ۸**

## Phase 9 — تدوین گزارش نهایی ⬜ *(افزوده‌شده؛ در سند اصلی شماره‌گذاری این فاز جا افتاده بود)*
- [ ] 9.1 تدوین گزارش طبق ساختار ۱۲-فصلی سند اصلی (Introduction → Literature Review → Materials
      → Methodology → Faster R-CNN → YOLO → Quantitative Comparison → Robustness → Explainability
      → Statistical Analysis → Discussion → Conclusion)
- [ ] 9.2 تایپ‌ست فارسی با XePersian طبق تمپلیت قبلی شما (B Nazanin، RTL)
- [ ] 9.3 پروف‌خوانی و تطبیق نهایی جدول‌ها/نمودارها با نتایج واقعی
**🚪 گیت ۹**

## Phase 10 — بحث علمی و نتیجه‌گیری ⬜
- [ ] 10.1 بحث استقرار (Deployment) — کدام مدل برای کاربرد بالینی/edge مناسب‌تر است
      (نکته از Phase 3: Faster R-CNN با ۷.۵ FPS برای real-time نامناسب است؛ مقایسه با FPS واقعی YOLO11 در Phase 4)
- [ ] 10.2 بحث انتقادی trade-off (دقت / سرعت / استحکام / تبیین‌پذیری) — نه صرفاً «کدام mAP بالاتری دارد»
- [ ] 10.3 نتیجه‌گیری و کارهای آینده
**🚪 گیت ۱۰ (تحویل نهایی)**

---

## چک‌لیست نهایی تحویل
- [x] پیاده‌سازی بازتولیدپذیر دو دیتکتور (Faster R-CNN/torchvision و YOLO11) با شرایط آموزشی هم‌راستا
- [x] بنچمارک استاندارد دقت و محاسبات (هر دو مدل — Phase 3/4/5)
- [ ] بنچمارک استحکام تحت corruption های رایج (⬜ Phase 6)
- [ ] تحلیل تبیین‌پذیری با Grad-CAM (⬜ Phase 7)
- [ ] مقایسه‌های آماری معنادار (⬜ Phase 8)
- [ ] بحث انتقادی نهایی درباره مناسب‌ترین پارادایم تشخیص (⬜ Phase 10)
