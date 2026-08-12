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
- [x] 3.1-3.4 مدل/آموزش کامل — 50/50 epoch، ۱۰.۰۴ ساعت، loss کاهش ۹۶.۲٪
- [x] 3.5 مAP/Precision/Recall/FPS/Params/GFLOPs — همه محاسبه و تایید شد:
      mAP@0.5(test)=0.818 | mAP@0.5:0.95(test)=0.593 | AR@100(test)=0.670 |
      Params=43.27M | GFLOPs=280.38 | FPS=7.53 | Size=165.41MB
      بدون نشانه overfitting (منحنی val پایدار با وجود کاهش شدید train loss)
- [x] 3.6 گزارش نهایی → `reports/faster_rcnn_performance_report.md`
      + فایل‌های خام: `outputs/logs/faster_rcnn_val_curve.csv`, `faster_rcnn_test_eval.txt`, `faster_rcnn_model_stats.txt`
**🚪 گیت ۳: ✅ تایید شد**

⚠️ **درس گرفته‌شده (اعمال‌شده برای Phase 4):** train_faster_rcnn.py فقط train loss لاگ می‌کرد؛ mAP/Recall
باید عقب‌گرد از چک‌پوینت‌ها بازسازی می‌شد. **برای Phase 4 این مشکل رخ نخواهد داد** چون Ultralytics YOLO
به‌صورت built-in و خودکار در هر epoch یک `results.csv` با ستون‌های
precision/recall/mAP50/mAP50-95 (هم train هم val) می‌نویسد — بدون نیاز به کد اضافه.

## Phase 4 — YOLO11 ⬜ **[فاز فعلی]**
- [ ] 4.1 بارگذاری دیتاست با فرمت YOLO detection بومی (`data/processed/yolo_detection/`)
- [ ] 4.2 تحمیل شرایط آموزشی منجمدشده (epochs=50, imgsz=640, batch=8, optimizer≈SGD lr=0.005)
      روی hyp-config Ultralytics؛ هر تفاوت اجباری (augmentation پیش‌فرض YOLO مثل mosaic/hsv) صریحاً مستند شود
- [ ] 4.3 اجرای آموزش (`yolo detect train ...`) — طبق CONTINUITY.md حتماً با Save & Run All (Commit)
- [ ] 4.4 محاسبه معیارهای محاسباتی (FPS/Params/GFLOPs، با fvcore یا خروجی بومی Ultralytics)
- [ ] 4.5 گزارش عملکرد YOLO11 (مشابه Phase 3.6) — این‌بار منحنی precision/recall/mAP از همان
      `results.csv` بومی Ultralytics می‌آید، نیازی به بازسازی از چک‌پوینت نیست
**🚪 گیت ۴**

## Phase 5 — مقایسه کمّی عملکرد ⬜
- [ ] 5.1 تجمیع معیارهای Detection (Precision/Recall/F1/IoU/Dice/mAP@0.5/mAP@0.5:0.95) هر دو مدل
      (اعداد Faster R-CNN از قبل در `configs/project_config.yaml -> results.faster_rcnn` موجود است)
- [ ] 5.2 تجمیع معیارهای Computational (FPS/Params/GFLOPs/GPU mem/Train time/Inference time)
- [ ] 5.3 ساخت جدول‌ها و نمودارهای مقایسه‌ای نهایی + sanity-check با مرجع #۱۰ فاز ۱ (Taha et al.)
**🚪 گیت ۵**

## Phase 6 — ارزیابی استحکام (Robustness) ⬜
- [ ] 6.1 پایپ‌لاین corruption (Albumentations): Brightness±, Gaussian Noise, Salt & Pepper,
      Gaussian Blur, Motion Blur, JPEG@20%, JPEG@50% — فقط روی اسپلیت test
- [ ] 6.2 اجرای inference هر دو مدل روی هر نوع corruption
- [ ] 6.3 تحلیل افت عملکرد (Original→Blur→Noise→Brightness→Compression) + ویژوالایز زنجیره
- [ ] 6.4 تولید «گزارش بنچمارک استحکام»
**🚪 گیت ۶**

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
- [ ] پیاده‌سازی بازتولیدپذیر دو دیتکتور (Faster R-CNN/torchvision ✅ و YOLO11 ⬜) با شرایط آموزشی هم‌راستا
- [ ] بنچمارک استاندارد دقت و محاسبات (Faster R-CNN ✅، YOLO11 ⬜)
- [ ] بنچمارک استحکام تحت corruption های رایج (⬜)
- [ ] تحلیل تبیین‌پذیری با Grad-CAM (⬜)
- [ ] مقایسه‌های آماری معنادار (⬜)
- [ ] بحث انتقادی نهایی درباره مناسب‌ترین پارادایم تشخیص (⬜)
