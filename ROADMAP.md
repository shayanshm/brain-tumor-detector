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

## Phase 3 — پایه: Faster R-CNN (torchvision) 🔄 **[بازگشایی‌شده -- کشف مهم]**
- [x] اجرای ۱ (v1) و اجرای ۲ (اشتباهاً «v2» نامیده شد) — **هر دو بدون augmentation بودند** (کشف در
      2026-08-13 با بررسی مستقیم Output نوت‌بوک Kaggle: فایل واقعاً اجراشده فاقد `transforms=get_transform` بود)
- [x] علت: احتمالاً عدم `git push`/`git pull` صحیح قبل از اجرای دوم
- [x] **اصلاح دائمی:** `train_faster_rcnn.py` و `train_yolo11.py` حالا با `assert` صریح شروع می‌شوند —
      اگر augmentation فعال نباشد، در همان ثانیه‌ی اول crash می‌کنند (نه بعد از ۱۰ ساعت بی‌خبری)
- [ ] **اجرای سوم (واقعی، با augmentation تاییدشده) — در انتظار شما**

### 🛡️ پروتکل ضدخطا قبل از این اجرا (اجباری، به‌ترتیب)
۱. در **خود صفحه‌ی وب گیت‌هاب** (نه لوکال) فایل `src/models/train_faster_rcnn.py` را باز کنید و با چشم
   خودتان ببینید خط `transforms=get_transform(train=True)` آنجا هست — این تنها منبع قابل‌اعتماد است.
۲. در نوت‌بوک Kaggle، **در یک پوشه‌ی کاملاً تازه** (نه پوشه‌ی قبلی که ممکن است کهنه باشد) `git clone` بزنید.
۳. بلافاصله بعد از clone و **قبل از شروع آموزش**، این چک سریع (چند ثانیه، رایگان) را بزنید:
   ```
   !grep -n "get_transform" /kaggle/working/brain-tumor-detector/src/models/train_faster_rcnn.py
   ```
   باید ۳ خط ببینید (import + دو بار transforms=...). اگر چیزی چاپ نشد، متوقف شوید و git clone را دوباره بزنید.
۴. مسیر `--out` را **جدید** بدهید (`outputs_v3`، نه v2/outputs قبلی) تا resume logic اشتباه نکند.
۵. کد خودش هم در ثانیه‌ی اول با `[AUGMENTATION CHECK] hflip_prob=0.5` تایید نهایی را چاپ می‌کند.

```
!python .../train_faster_rcnn.py --data-root .../data/processed \
    --epochs 50 --batch-size 8 --out /kaggle/working/brain-tumor-detector/outputs_v3
```
**🚪 گیت ۳: بازگشایی‌شده — نیازمند اجرای سوم و تایید مجدد**

⚠️ **درس گرفته‌شده (اعمال‌شده برای Phase 4):** train_faster_rcnn.py فقط train loss لاگ می‌کرد؛ mAP/Recall
باید عقب‌گرد از چک‌پوینت‌ها بازسازی می‌شد. **برای Phase 4 این مشکل رخ نخواهد داد** چون Ultralytics YOLO
به‌صورت built-in و خودکار در هر epoch یک `results.csv` با ستون‌های
precision/recall/mAP50/mAP50-95 (هم train هم val) می‌نویسد — بدون نیاز به کد اضافه.

## Phase 4 — YOLO11 🔄 **[کد آماده، منتظر بازآموزی Phase 3]**
- [x] 4.1 دیتاست YOLO detection آماده از Phase 2.7 (`data/processed/yolo_detection/data.yaml`)
- [x] 4.2 config منجمد با augmentation صحیح (fliplr=0.5, hsv_v≈0.2) تحمیل شد -- منطبق با Faster R-CNN اصلاح‌شده
- [x] کد آموزش با resume خودکار — تست کامل End-to-End با اجرای واقعی ۵۰ epoch روی داده‌ی کوچک
- [ ] 4.3 اجرای آموزش واقعی روی GPU کگل — بعد از بازآموزی Faster R-CNN اجرا شود (برای مقایسه‌ی منصفانه)
- [ ] 4.4 محاسبه FPS/Params/GFLOPs
- [ ] 4.5 گزارش عملکرد YOLO11
**🚪 گیت ۴: هنوز باز**

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
