"""
Phase 7.1 — پیاده‌سازی دستی EigenCAM (Muhammad & Yeasin, 2020).
به‌جای وابستگی به API کتابخانه‌ی pytorch-grad-cam برای مدل‌های detection (که برای هر دو معماری
Faster R-CNN و YOLO یکسان کار نمی‌کند)، این پیاده‌سازی ساده و مستقل از معماری است: فقط به یک
forward hook روی یک لایه‌ی کانولوشنی نیاز دارد، هیچ نیازی به gradient/backward یا "target class"
خاص ندارد (برخلاف Grad-CAM کلاسیک) -- برای مدل‌های two-stage/one-stage به یک شکل کار می‌کند.

الگوریتم: فعال‌سازی‌های لایه [C,H,W] را به [C, H*W] صاف می‌کنیم، SVD می‌گیریم، اولین
principal component (تصویری از مهم‌ترین الگوی فعال‌سازی مکانی) را به‌عنوان saliency map برمی‌گردانیم.
"""
import numpy as np
import torch


class EigenCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.hook = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activations = output.detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        input_tensor: [1, C, H, W]
        خروجی: saliency map نرمال‌شده در بازه‌ی [0,1]، اندازه‌ی H_feat x W_feat (قبل از resize)
        """
        self.model.eval()
        with torch.no_grad():
            self.model(input_tensor)

        if self.activations is None:
            raise RuntimeError("هیچ activation ای ثبت نشد -- target_layer درست است؟")

        act = self.activations[0]  # [C, H, W] (اولین نمونه‌ی batch)
        act = torch.relu(act)  # اطمینان از غیرمنفی بودن (رفتار استاندارد فعال‌سازی‌های کانولوشنی)
        C, H, W = act.shape
        act_flat = act.reshape(C, H * W).cpu().numpy()

        # SVD روی فعال‌سازی‌های خام (بدون centering که سیگنال دامنه را از بین می‌برد)
        U, S, Vt = np.linalg.svd(act_flat, full_matrices=False)
        cam = Vt[0].reshape(H, W)

        # رفع ابهام علامت SVD: بردار تکین می‌تواند در جهت + یا - باشد؛ باید هم‌جهت با
        # میانگین فعال‌سازی واقعی باشد (وگرنه CAM دقیقاً برعکس نواحی مهم را نشان می‌دهد)
        mean_act_map = act.mean(dim=0).cpu().numpy()
        if np.corrcoef(cam.flatten(), mean_act_map.flatten())[0, 1] < 0:
            cam = -cam

        # نرمال‌سازی به [0,1]
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam

    def remove_hook(self):
        self.hook.remove()


def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """image_np: [H,W,3] uint8 RGB. cam: هر اندازه (resize خودکار می‌شود). خروجی: [H,W,3] uint8"""
    import cv2
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (image_np.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
