"""Stage G: bigger 6-class scene segmenter (resnet34 encoder, 60 epochs,
color/scale augmentation). Same labels as train_seg.py but more capacity."""
from __future__ import annotations
import argparse, time, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "data" / "extracted" / "Rip Current Video" / "TrainDataset_RGB"


def encode_label(gt_bgr):
    bits = (gt_bgr > 0).astype(np.uint8)
    return bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]


class SceneDatasetV2(Dataset):
    def __init__(self, split: str, crop: int = 512, augment: bool = True):
        self.img_dir = TR / split / "img" / "0"
        self.gt_dir  = TR / split / "gt"  / "0"
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.crop = crop
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.imread(str(self.img_dir / name))
        gt  = cv2.imread(str(self.gt_dir / name))
        lbl = encode_label(gt)
        H, W = img.shape[:2]
        c = self.crop
        if self.augment:
            # Random scale 0.9-1.2
            scale = random.uniform(0.9, 1.2)
            new_h = max(c, int(H * scale)); new_w = max(c, int(W * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            lbl = cv2.resize(lbl, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            H, W = img.shape[:2]
            x = random.randint(0, W - c); y = random.randint(0, H - c)
            img = img[y:y+c, x:x+c]; lbl = lbl[y:y+c, x:x+c]
            if random.random() < 0.5:
                img = img[:, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            # Color jitter
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = np.clip(hsv[..., 0] + random.uniform(-15, 15), 0, 179)
            hsv[..., 1] *= random.uniform(0.8, 1.2)
            hsv[..., 2] *= random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            y = (H - c) // 2; x = (W - c) // 2
            img = img[y:y+c, x:x+c]; lbl = lbl[y:y+c, x:x+c]
        img_t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        lbl_t = torch.from_numpy(lbl.astype(np.int64))
        return img_t, lbl_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--num-classes", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "seg_refiner" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (ROOT / "seg_refiner" / "_train_v2_log.txt").open("w")
    def L(s): print(s); log.write(s + "\n"); log.flush()

    L(f"device: {device}, encoder: {args.encoder}, epochs: {args.epochs}")
    train_ds = SceneDatasetV2("train", crop=args.crop, augment=True)
    val_ds   = SceneDatasetV2("val",   crop=args.crop, augment=False)
    L(f"train {len(train_ds)} val {len(val_ds)} crop={args.crop}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                      in_channels=3, classes=args.num_classes).to(device)
    L(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    class_w = torch.tensor([0.2, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0], device=device)
    crit = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best = -1.0
    for ep in range(args.epochs):
        t0 = time.time(); model.train()
        tloss = 0.0; n = 0
        for img, lbl in train_loader:
            img = img.to(device, non_blocking=True); lbl = lbl.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(img); loss = crit(logits, lbl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tloss += loss.item() * img.size(0); n += img.size(0)
        sched.step(); tloss /= n
        model.eval()
        inter = torch.zeros(8, device=device); union = torch.zeros(8, device=device)
        with torch.no_grad():
            for img, lbl in val_loader:
                img = img.to(device, non_blocking=True); lbl = lbl.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    pred = model(img).argmax(dim=1)
                for c in range(8):
                    p = (pred == c); g = (lbl == c)
                    inter[c] += (p & g).sum().float(); union[c] += (p | g).sum().float()
        iou = (inter / torch.clamp(union, min=1.0)).cpu().numpy()
        L(f"ep {ep+1:>2}: loss={tloss:.4f} sea={iou[2]:.3f} land={iou[4]:.3f} foam={iou[5]:.3f} struct={iou[6]:.3f} ({time.time()-t0:.1f}s)")
        if iou[2] > best:
            best = float(iou[2])
            torch.save({"model_state_dict": model.state_dict(),
                        "encoder": args.encoder, "num_classes": args.num_classes,
                        "iou": iou.tolist(), "epoch": ep + 1},
                       out_dir / "best_v2.pt")
            L(f"  -> saved best_v2.pt (sea IoU={best:.4f})")
    L(f"DONE best sea IoU={best:.4f}")
    log.close()


if __name__ == "__main__":
    main()
