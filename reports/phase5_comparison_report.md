# مقایسه‌ی کمّی نهایی — Faster R-CNN در برابر YOLO11
تاریخ: 2026-08-17 | Phase 5 Deliverable

## روش‌شناسی
دو دسته معیار گزارش می‌شود:
1. **mAP رسمی** (`mAP@0.5`, `mAP@0.5:0.95`) — از ابزارهای استاندارد خود هر فریمورک (COCOeval برای
   Faster R-CNN، Ultralytics validator برای YOLO11)؛ این‌ها معتبرترین اعداد صنعتی‌اند.
2. **Precision/Recall/F1/IoU/Dice** — چون COCOeval/Ultralytics این‌ها را با روش‌های *ناهمسان* گزارش
   می‌دهند (مثلاً Precision/Recall Ultralytics در یک نقطه‌ی عملیاتی داخلی متفاوت با هر چیزی که از
   Faster R-CNN می‌گرفتیم)، یک پایپ‌لاین سفارشی (`src/eval/compare_models.py`) ساختیم که با
   **دقیقاً یک الگوریتم** (greedy IoU matching @ threshold=0.5) هر دو مدل را می‌سنجد. این پایپ‌لاین
   با داده‌ی کنترل‌شده تست و تایید شد (TP/FP/FN/P/R/F1/IoU/Dice). پیش‌بینی‌ها در آستانه‌ی
   **confidence >= 0.5** فیلتر شدند (نقطه‌ی عملیاتی واقع‌بینانه، نه شامل تشخیص‌های بسیار کم‌اطمینان).

## ۵.۱ معیارهای Detection

### mAP رسمی (نمودار: `reports/figures/phase5_map_official.png`)
| معیار | Faster R-CNN | YOLO11 | برنده |
|---|---|---|---|
| mAP@0.5 | 0.853 | **0.886** | YOLO11 (+3.3pp) |
| mAP@0.5:0.95 | 0.630 | **0.639** | YOLO11 (+0.9pp) |

### Precision/Recall/F1/IoU/Dice — پایپ‌لاین یکپارچه (نمودار: `reports/figures/phase5_detection_metrics.png`)
| معیار | Faster R-CNN | YOLO11 | برنده |
|---|---|---|---|
| Precision | 0.810 | **0.863** | YOLO11 (+5.3pp) |
| Recall | 0.789 | **0.799** | YOLO11 (+1.0pp، تقریباً برابر) |
| F1 | 0.799 | **0.830** | YOLO11 (+3.0pp) |
| mean IoU (روی تشخیص‌های درست) | **0.857** | 0.852 | Faster R-CNN (+0.5pp، ناچیز) |
| mean Dice (روی تشخیص‌های درست) | **0.919** | 0.915 | Faster R-CNN (+0.4pp، ناچیز) |

**نکته‌ی مهم:** YOLO11 در Precision/Recall/F1 برتر است (تشخیص‌های بیشتر و دقیق‌تر)، اما وقتی
Faster R-CNN تشخیص درستی می‌دهد، آن bbox کمی (به‌طور ناچیز) دقیق‌تر از YOLO11 است (IoU/Dice بالاتر).
یعنی: YOLO11 بهتر «چیزی را پیدا می‌کند»، Faster R-CNN وقتی پیدا می‌کند، کمی «دقیق‌تر مرزبندی می‌کند».

### تفکیک per-class F1 (یافته‌ی تکرارشونده و قطعی) — نمودار: `reports/figures/phase5_per_class_f1.png`
| کلاس | Faster R-CNN F1 | YOLO11 F1 |
|---|---|---|
| **glioma** | **0.682** ⚠️ | **0.742** ⚠️ |
| meningioma | 0.915 | 0.927 |
| pituitary | 0.915 | 0.909 |

این الگو **چهار بار مستقل** تایید شده: (۱) ماتریس درهم‌ریختگی YOLO در Phase 4، (۲) ارزیابی رسمی
test-split YOLO (mAP per-class)، (۳) پایپ‌لاین سفارشی برای هر دو مدل، و (۴) ماتریس درهم‌ریختگی
Faster R-CNN (`reports/figures/faster_rcnn_confusion_matrix.png`، افزوده‌شده در بازبینی
2026-08-17 برای parity کامل با YOLO). پشتوانه‌ی علمی کامل در `reports/literature_review.md`
(رفرنس‌های ۱۶-۲۱).

> **توجه روش‌شناسی:** اعداد TP در confusion matrix (۲۴۳) با TP جدول Precision/Recall بالا یکی است،
> اما FP/FN کمی متفاوت‌اند (۵۴/۶۲ در برابر ۵۷/۶۵) چون confusion matrix اجازه‌ی «تطبیق بین‌کلاسی»
> می‌دهد (مثلاً یک تشخیص که واقعاً glioma بوده ولی meningioma پیش‌بینی شده، در ماتریس یک سلول
> غیرقطری می‌شود، ولی در محاسبه‌ی سخت‌گیرانه‌ی per-class Precision/Recall به‌عنوان یک FP + یک FN
> جداگانه شمرده می‌شود). هر دو روش صحیح‌اند، فقط برای سوالات متفاوت طراحی شده‌اند.

## ۵.۲ معیارهای محاسباتی (نمودار: `reports/figures/phase5_computational.png`)
| معیار | Faster R-CNN | YOLO11 | نسبت |
|---|---|---|---|
| Parameters | 43.27M | 2.59M | YOLO11 **۱۷ برابر** سبک‌تر |
| GFLOPs | 280.38 | 6.5 | YOLO11 **۴۳ برابر** کمتر |
| Model Size | 165.41 MB | 5.22 MB | YOLO11 **۳۲ برابر** کوچک‌تر |
| FPS (GPU یکسان، روش یکسان) | 8.75 | 101.66 | YOLO11 **۱۲ برابر** سریع‌تر (inference) |
| Inference time | 114.25 ms | 9.84 ms | YOLO11 **۱۲ برابر** سریع‌تر |
| GPU Memory (inference) | 681.08 MB | 71.96 MB | YOLO11 **۹.۵ برابر** کم‌مصرف‌تر |
| GPU Memory (training, batch=8) | 13,444.64 MB (~13.1 GB) | 1,218.56 MB (~1.19 GB) | YOLO11 **۱۱ برابر** کم‌مصرف‌تر |
| زمان آموزش (۵۰ epoch، GPU یکسان) | 10.37h | 0.41h | YOLO11 **۲۵ برابر** سریع‌تر (train) |

هر دو عدد GPU memory (inference و training) واقعی‌اند — Faster R-CNN از اندازه‌گیری مستقیم
(forward/backward دستی)، YOLO11 از لاگ واقعی Ultralytics در یک آموزش ۱-epoch (چون شبیه‌سازی
دستی loss داخلی YOLO قابل‌اعتماد نبود).

⚠️ **یافته‌ی مهم برای Phase 10:** Faster R-CNN با batch=8 در آستانه‌ی سقف حافظه‌ی GPU های معمول
(T4/P100 با ۱۵-۱۶GB) قرار دارد — روی GPU های ضعیف‌تر یا batch size بزرگ‌تر، ممکن است اصلاً قابل
آموزش نباشد. این یک محدودیت عملی مهم برای بحث «قابلیت استقرار» است.

## ۵.۳ جمع‌بندی
روی این دیتاست خاص (تشخیص تومور مغزی، ۳ کلاس، ۳۰۶۴ تصویر)، **YOLO11n در همه‌ی ابعاد** — دقت
(mAP/F1)، سرعت inference، سرعت آموزش، و سبکی مدل — از Faster R-CNN جلوتر است، با یک استثنای جزئی
و ناچیز (IoU/Dice میانگین کمی بالاتر برای Faster R-CNN روی تشخیص‌های درست). این نتیجه با ادبیات
Phase 1 (مرجع #۱۰: Taha et al., که YOLOv11/v8 را روی همین سه کلاس مقایسه کرده) هم‌راستا است.

⚠️ سوال باز برای Phase 8 (تحلیل آماری): آیا این تفاوت‌های دقتی (مثلاً +۳.۳pp در mAP@0.5) از نظر
آماری معنادارند یا در بازه‌ی نویز طبیعی run-to-run هستند؟ Phase 8 با bootstrap CI و permutation
test به این پاسخ می‌دهد.
