"""
Phase 3.4 — آموزش Faster R-CNN روی دیتاست تومور مغزی.

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/models/train_faster_rcnn.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --epochs 50 --batch-size 8 --out /kaggle/working/brain-tumor-detector/outputs
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.coco_dataset import CocoDetectionDataset, collate_fn, NUM_CLASSES_WITH_BACKGROUND  # noqa: E402
from models.faster_rcnn import build_faster_rcnn  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def get_dataloaders(data_root: Path, batch_size: int):
    from models.augmentation import get_transform

    # توجه: images_dir به پوشه‌ی خودِ symlink شده در yolo_detection اشاره می‌کند چون تصاویر آنجا
    # از قبل symlink شده‌اند؛ می‌توانستیم مستقیم از /kaggle/input هم بخوانیم، اما این مسیر یکپارچه‌تر است.
    train_ds = CocoDetectionDataset(
        images_dir=str(data_root / "yolo_detection" / "train" / "images"),
        coco_json_path=str(data_root / "coco_format" / "train.json"),
        transforms=get_transform(train=True),   # اصلاح باگ Phase 3: قبلاً None بود (بدون augmentation)
    )
    val_ds = CocoDetectionDataset(
        images_dir=str(data_root / "yolo_detection" / "valid" / "images"),
        coco_json_path=str(data_root / "coco_format" / "valid.json"),
        transforms=get_transform(train=False),  # eval/val همیشه بدون augmentation
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    return train_loader, val_loader


def train_one_epoch(model, optimizer, loader, device, epoch, log_path):
    model.train()
    epoch_losses = {}
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
        epoch_losses["total"] = epoch_losses.get("total", 0.0) + total_loss.item()

        if batch_idx % 20 == 0:
            print(f"  epoch {epoch} | batch {batch_idx}/{len(loader)} | loss={total_loss.item():.4f}")

    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
    elapsed = time.time() - t0

    log_line = f"epoch={epoch} | " + " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()) + f" | time={elapsed:.1f}s\n"
    with open(log_path, "a") as f:
        f.write(log_line)
    print(log_line.strip())

    return avg_losses


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"استفاده از device: {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / "faster_rcnn_train_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "training_state.pt"

    # --- خودتاییدی صریح: تا دیگر هرگز ابهام «آیا augmentation واقعا فعال بود؟» پیش نیاید ---
    # این پیام هم در stdout و هم در همان فایل لاگ آموزش نوشته می‌شود (قابل رجوع دائمی).
    from models.augmentation import get_transform
    _probe = get_transform(train=True)
    verification_msg = (
        "\n" + "=" * 70 +
        f"\n[AUGMENTATION CHECK] hflip_prob={_probe.hflip_prob} | "
        f"brightness={_probe.brightness} | contrast={_probe.contrast}\n"
        f"[AUGMENTATION CHECK] این پیام باید hflip_prob=0.5 نشان دهد؛ اگر 0.0 است یعنی\n"
        f"کد قدیمی (بدون augmentation) در حال اجراست -- سریعاً متوقف و git pull کنید!\n"
        + "=" * 70 + "\n"
    )
    print(verification_msg)
    with open(log_path, "a") as f:
        f.write(verification_msg)
    assert _probe.hflip_prob == 0.5, "خطای بحرانی: augmentation فعال نیست! کد قدیمی در حال اجراست."

    train_loader, val_loader = get_dataloaders(data_root, args.batch_size)
    print(f"تعداد batch های train: {len(train_loader)} | val: {len(val_loader)}")

    model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    start_epoch = 1
    # --- Resume خودکار: اگر training_state.pt از اجرای قبلی وجود داشته باشد، دقیقاً از همان‌جا ادامه می‌دهیم ---
    if resume_path.exists():
        print(f"⏯️  training_state.pt پیدا شد -- ادامه‌ی آموزش از همان‌جا (نه از صفر)")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        print(f"   آخرین epoch کامل‌شده: {state['epoch']} -> شروع از epoch {start_epoch}")
    else:
        print("هیچ training_state.pt ای پیدا نشد -- شروع از epoch 1 (اجرای تازه)")

    if start_epoch > args.epochs:
        print(f"آموزش قبلاً تا epoch {start_epoch - 1} تمام شده (>= هدف {args.epochs}). کاری نمانده.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, log_path)
        lr_scheduler.step()

        # ذخیره‌ی state کامل (نه فقط وزن مدل) تا resume دقیق ممکن باشد
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        }, resume_path)
        # یک نسخه‌ی سبک‌تر فقط-وزن هم برای استفاده در Phase 3.5/5 (inference/eval)
        torch.save(model.state_dict(), ckpt_dir / "faster_rcnn_last.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), ckpt_dir / f"faster_rcnn_epoch{epoch}.pt")
            print(f"چک‌پوینت epoch {epoch} ذخیره شد.")

    print("آموزش تمام شد.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
