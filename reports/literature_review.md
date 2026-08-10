# مرور ادبیات — Phase 1 Deliverable
تاریخ تدوین: 2026-08-10 | برای فصل ۲ گزارش نهایی

## ۱. معماری و تکامل YOLO (پشتوانه‌ی انتخاب YOLO11 در Phase 0)
| # | مرجع | نکته‌ی کلیدی مرتبط با پروژه |
|---|---|---|
| 1 | Khanam & Hussain, *YOLOv11: An Overview of the Key Architectural Enhancements*, arXiv:2410.17725 (2024) | معرفی C3k2 / SPPF / C2PSA؛ مرجع اصلی معماری YOLO11 |
| 2 | *Ultralytics YOLO Evolution: YOLO26, YOLO11, YOLOv8, YOLOv5*, arXiv:2510.09653 (2026) | زمینه‌ی تکاملی کامل خانواده YOLO تا جدیدترین نسخه |
| 3 | *YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review* | مقایسه‌ی مستقیم معماری YOLOv8 در برابر YOLO11 |
| 4 | Redmon et al., *You Only Look Once*, CVPR 2016 | مرجع بنیادین YOLO (فصل مقدمه) |
| 5 | Ren et al., *Faster R-CNN*, NeurIPS 2015 | مرجع بنیادین Faster R-CNN (فصل مقدمه/Baseline) |

## ۲. تبیین‌پذیری / Grad-CAM (پشتوانه‌ی Phase 7)
| # | مرجع | نکته‌ی کلیدی |
|---|---|---|
| 6 | Selvaraju et al., *Grad-CAM*, ICCV 2017 | مقاله‌ی بنیادین Grad-CAM |
| 7 | *Explaining YOLO: Leveraging Grad-CAM to Explain Object Detections* | چک صحت (sanity check) و معیار کیفی برای تبیین‌پذیری در object detection؛ روی Faster R-CNN/SSD/EfficientDet |
| 8 | GitHub: jacobgil/pytorch-grad-cam | ابزار عملی Phase 7؛ نوت‌بوک اختصاصی برای Faster-RCNN و YOLO |
| 9 | Yang et al. (2019), Grad-CAM++ روی YOLOv3 | نمونه‌ی تطبیق Grad-CAM برای دیتکتور تک‌مرحله‌ای |

## ۳. مطالعات دقیقاً روی دامنه‌ی پروژه (glioma / meningioma / pituitary)
| # | مرجع | یافته‌ی کلیدی |
|---|---|---|
| 10 | **Taha, Aly & Darwish (2025)**, arXiv:2504.00189 | مقایسه‌ی مستقیم YOLOv11 vs YOLOv8 روی همین ۳ کلاس — **مرجع اصلی sanity-check نتایج Phase 5** |
| 11 | Wahidin et al. (به‌نقل از PMC12425724) | YOLOv11m: mAP50=0.934 (بالاترین دقت)؛ YOLOv8m: 80.47 FPS (سریع‌ترین) — تنظیم با BOHB |
| 12 | *Application and improvement of YOLO11 for brain tumor detection*, PMC12425724 | بهبود YOLO11 با attention: کاهش ۲.۷٪ پارامتر، افزایش ۱.۰٪ mAP50 |
| 13 | *Brain Tumor Detection Using YOLOv5 and Faster R-CNN*, IEEE 2023 | خط مبنای تاریخی YOLO در برابر Faster R-CNN روی همین تومورها |
| 14 | ISMRM Proceedings 2022 | YOLOv5 mAP=89.5%؛ Faster R-CNN در کلاس‌بندی قوی اما در لوکالیزیشن ضعیف — نکته‌ی کلیدی برای بحث Phase 10 |
| 15 | *Detection and Localization of Brain Tumors Using YOLO*, JAIC | YOLOv12: recall=97.32%, mAP@0.5=92.2% — بستر مقایسه با نسخه‌های خیلی جدیدتر |

## جمع‌بندی برای Phase 5 (Sanity Check اعداد)
بر اساس مرجع #۱۰ و #۱۱ (دقیقاً همین سه کلاس)، انتظار می‌رود:
- mAP50 نهایی YOLO11 ما در بازه‌ی تقریبی **0.85–0.95** باشد (در صورت انحراف زیاد، باید در Phase 5 بررسی علت شود).
- Faster R-CNN طبق الگوی مرجع #۱۴، ممکن است در Precision/کلاس‌بندی رقابتی باشد اما در IoU/لوکالیزیشن ضعیف‌تر ظاهر شود — این دقیقاً همان trade-off است که سند اصلی در Phase 10 خواستار بحث انتقادی درباره‌ی آن است.
