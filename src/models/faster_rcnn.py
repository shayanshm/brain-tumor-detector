"""
Phase 3.1 — ساخت مدل Faster R-CNN با backbone ResNet50-FPN از‌پیش‌آموزش‌دیده روی COCO،
و جایگزینی سر (head) طبقه‌بندی برای ۳ کلاس تومور + ۱ background.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int, pretrained: bool = True):
    """
    num_classes باید شامل background هم باشد (یعنی 3 کلاس تومور + 1 = 4).
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    m = build_faster_rcnn(num_classes=4, pretrained=False)
    print(m.roi_heads.box_predictor)
