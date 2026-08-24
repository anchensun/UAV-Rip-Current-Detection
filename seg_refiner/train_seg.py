"""Train a 6-class scene segmenter on the manuscript's TrainDataset_RGB
(215 train + 22 val frames at 1920x1080, 6-class BGR-bit-encoded labels).

Classes (BGR-bit code -> name, inferred from class-frequency + image content):
    0 (000)  void
    1 (100)  sky          (B-bit only)
    2 (010)  sea          (G-bit only — dominant class)
    3 (110)  unused
    4 (001)  land/sand    (R-bit only — bright pixels)
    5 (101)  foam         (B+R bits)
    6 (011)  structure    (G+R bits — vegetation/jetty)
    7 (111)  unused

We train a U-Net (resnet18 encoder, ImageNet init) with cross-entropy
over 6 classes, then expose .softmax() outputs that downstream code can
use as soft per-pixel probabilities for sea / foam / etc.

Usage:
    python seg_refiner/train_seg.py --epochs 30 --batch-size 4

Output:
    seg_refiner/checkpoints/best.pt
    seg_refiner/_train_log.txt
"""
from __future__ import annotations
import argparse, time, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "data" / "extracted" / "Rip Current Video" / "TrainDataset_RGB"


def encode_label(gt_bgr: np.ndarray) -> np.ndarray:
    """Convert 3-channel 0/128 label image to single-channel class-id mask
    using the BGR-bit encoding."""
    bits = (gt_bgr > 0).astype(np.uint8)
    return bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]


class SceneDataset(Dataset):
    def __init__(self, split: str, crop: int = 512, augment: bool = True,
                 drop: set[str] | None = None):
        self.img_dir = TR / split / "img" / "0"
        self.gt_dir  = TR / split / "gt"  / "0"
        drop = drop or set()
        self.files = sorted(p.name for p in self.img_dir.glob("*.png") if p.name not in drop)
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.imread(str(self.img_dir / name))  # BGR uint8
        gt  = cv2.imread(str(self.gt_dir / name))   # BGR 0/128
        if img is None or gt is None:
            raise RuntimeError(f"Missing {name}")
        lbl = encode_label(gt)  # H, W uint8 in {0..7}
        H, W = img.shape[:2]
        # Random crop
        if self.augment:
            ch, cw = self.crop, self.crop
            if H > ch and W > cw:
                y = np.random.randint(0, H - ch)
                x = np.random.randint(0, W - cw)
                img = img[y:y+ch, x:x+cw]
                lbl = lbl[y:y+ch, x:x+cw]
            # H-flip
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            # Center crop 512 for val
            ch, cw = self.crop, self.crop
            y = (H - ch) // 2
            x = (W - cw) // 2
            img = img[y:y+ch, x:x+cw]
            lbl = lbl[y:y+ch, x:x+cw]
        # ImageNet normalization
        img_t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        lbl_t = torch.from_numpy(lbl.astype(np.int64))
        return img_t, lbl_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--num-classes", type=int, default=8,
                    help="Use 8 to cover all BGR-bit codes including unused 3,7.")
    ap.add_argument("--out-name", default="best.pt",
                    help="Checkpoint filename under seg_refiner/checkpoints/.")
    ap.add_argument("--log-name", default="_train_log.txt")
    ap.add_argument("--exclude-csv", type=Path, default=None,
                    help="CSV with columns frame,best_test used to drop frames "
                         "resembling a held-out evaluation scene (leakage-free retrain).")
    ap.add_argument("--exclude-scene", default=None,
                    help="Value of best_test to drop (e.g. PalmBeach or Haulover).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    drop: set[str] = set()
    if args.exclude_csv and args.exclude_scene:
        import csv as _csv
        with args.exclude_csv.open() as fh:
            for r in _csv.DictReader(fh):
                if r.get("best_test") == args.exclude_scene:
                    drop.add(r["frame"])

    out_dir = ROOT / "seg_refiner" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = ROOT / "seg_refiner" / args.log_name
    log = log_file.open("w")

    def L(s):
        print(s); log.write(s + "\n"); log.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L(f"device: {device}, torch {torch.__version__}, smp {smp.__version__}")
    if drop:
        L(f"EXCLUDING {len(drop)} frames resembling '{args.exclude_scene}' "
          f"(from {args.exclude_csv.name})")

    train_ds = SceneDataset("train", crop=args.crop, augment=True, drop=drop)
    val_ds   = SceneDataset("val",   crop=args.crop, augment=False, drop=drop)
    L(f"train {len(train_ds)} frames, val {len(val_ds)} frames, crop={args.crop}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=args.num_classes,
    ).to(device)
    L(f"model: U-Net({args.encoder}), classes={args.num_classes}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Class weights: code 0 (void), 3 and 7 (unused) are very rare; set lower weight on void
    class_w = torch.tensor([0.2, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0], device=device)
    crit = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_iou_sea = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0; n = 0
        for img, lbl in train_loader:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            logits = model(img)
            loss = crit(logits, lbl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0); n += img.size(0)
        sched.step()
        train_loss /= n

        # Validation: compute per-class IoU
        model.eval()
        n_classes = args.num_classes
        inter = torch.zeros(n_classes, device=device)
        union = torch.zeros(n_classes, device=device)
        val_loss = 0.0; vn = 0
        with torch.no_grad():
            for img, lbl in val_loader:
                img = img.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                logits = model(img)
                loss = crit(logits, lbl)
                val_loss += loss.item() * img.size(0); vn += img.size(0)
                pred = logits.argmax(dim=1)
                for c in range(n_classes):
                    p = (pred == c)
                    g = (lbl == c)
                    inter[c] += (p & g).sum().float()
                    union[c] += (p | g).sum().float()
        val_loss /= vn
        iou = (inter / torch.clamp(union, min=1.0)).cpu().numpy()
        L(f"ep {epoch+1:>2}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"IoU void={iou[0]:.3f} sky={iou[1]:.3f} sea={iou[2]:.3f} land={iou[4]:.3f} "
          f"foam={iou[5]:.3f} struct={iou[6]:.3f}  ({time.time()-t0:.1f}s)")

        # Save best by sea IoU (the class we care about for gating)
        if iou[2] > best_val_iou_sea:
            best_val_iou_sea = float(iou[2])
            torch.save({"model_state_dict": model.state_dict(),
                        "encoder": args.encoder,
                        "num_classes": args.num_classes,
                        "epoch": epoch + 1,
                        "iou": iou.tolist(),
                        "excluded_scene": args.exclude_scene,
                        "n_excluded": len(drop)},
                       out_dir / args.out_name)
            L(f"  -> saved checkpoints/{args.out_name} (val sea IoU = {best_val_iou_sea:.4f})")

    L(f"DONE. best val sea IoU = {best_val_iou_sea:.4f}; saved {args.out_name}")
    log.close()


if __name__ == "__main__":
    main()
