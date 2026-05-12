"""Train a BINARY U-Net (rip vs no-rip per pixel) on RipDetSeg's
polygon-format segmentation annotations.

The RipDetSeg labels are YOLO-segmentation format:
    class_id  x1 y1 x2 y2 ... xn yn   (normalized [0,1] coordinates)
where every line is one polygon instance. Most images have ZERO instances
(negative class — no rip current visible). Roughly 21,734 of 32,562 training
images carry positive instances.

We rasterize polygons on-the-fly into a binary {0,1} mask, then train a
binary U-Net (resnet34, ImageNet init) for ~15 epochs.

Output: seg_refiner/checkpoints/ripdetseg_binary.pt (used downstream by
run_pipeline_with_seg.py-equivalent variants for ensemble experiments).
"""
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
RDS_TRAIN = Path("/home/anchensun/Anchen/Research/Rip_Current/RipDetSeg_v1.1.6_train")
TRAIN_IMG = RDS_TRAIN / "train_images"
TRAIN_LBL = RDS_TRAIN / "train_labels_segmentation"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--max-train-imgs", type=int, default=12000,
                   help="Cap training set size for time budget; 0 = use all.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_polygons(txt_path: Path):
    polys = []
    if not txt_path.exists(): return polys
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7: continue   # need at least class + 3 (x,y) pairs
        coords = list(map(float, parts[1:]))
        n = len(coords) // 2
        if n < 3: continue
        polys.append(np.asarray(coords, dtype=np.float32).reshape(n, 2))
    return polys


def rasterize(polys, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    if not polys: return m
    pts_list = []
    for p in polys:
        p_pix = np.column_stack([p[:, 0] * W, p[:, 1] * H]).astype(np.int32)
        pts_list.append(p_pix)
    cv2.fillPoly(m, pts_list, 1)
    return m


class RipDetSegBinary(Dataset):
    def __init__(self, files: list[Path], crop: int, augment: bool):
        self.files = files
        self.crop = crop
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = cv2.imread(str(p))
        if img is None:
            return self.__getitem__((i + 1) % len(self))
        H, W = img.shape[:2]
        polys = parse_polygons(TRAIN_LBL / (p.stem + ".txt"))
        mask = rasterize(polys, H, W)
        c = self.crop
        # Resize to ensure shorter side >= crop
        scale = c / min(H, W) * (1.1 if self.augment else 1.0)
        if scale > 1.0:
            new_h = max(c, int(H * scale)); new_w = max(c, int(W * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            H, W = img.shape[:2]
        if self.augment:
            x = random.randint(0, W - c); y = random.randint(0, H - c)
        else:
            x = (W - c) // 2; y = (H - c) // 2
        img = img[y:y+c, x:x+c]
        mask = mask[y:y+c, x:x+c]
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy()
        img_t = torch.from_numpy(((img[..., ::-1].astype(np.float32) - self.mean) / self.std).copy()).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask.astype(np.float32))
        return img_t, mask_t


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    num = 2.0 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((num + eps) / (den + eps)).mean()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(TRAIN_IMG.glob("*.jpg"))
    random.shuffle(files)
    if args.max_train_imgs > 0 and len(files) > args.max_train_imgs:
        files = files[:args.max_train_imgs]
    n_val = max(200, len(files) // 30)
    val_files = files[:n_val]
    train_files = files[n_val:]
    print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")

    train_ds = RipDetSegBinary(train_files, args.crop, augment=True)
    val_ds   = RipDetSegBinary(val_files,   args.crop, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                      in_channels=3, classes=1).to(device)
    print(f"Model U-Net({args.encoder}), params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    out_dir = ROOT / "seg_refiner" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "seg_refiner" / "_ripdetseg_train_log.txt", "w")

    def L(s):
        print(s); log.write(s + "\n"); log.flush()

    L(f"device: {device}, encoder: {args.encoder}, epochs: {args.epochs}, batch: {args.batch_size}")
    best_iou = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tloss = 0.0; n = 0
        for img, mask in train_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).unsqueeze(1)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(img)
                loss = bce(logits, mask) + dice_loss(logits, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tloss += loss.item() * img.size(0); n += img.size(0)
        sched.step()
        tloss /= n

        model.eval()
        inter = 0.0; union = 0.0; vloss = 0.0; vn = 0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == "cuda")):
            for img, mask in val_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True).unsqueeze(1)
                logits = model(img)
                vloss += (bce(logits, mask) + dice_loss(logits, mask)).item() * img.size(0); vn += img.size(0)
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter += float((pred * mask).sum())
                union += float(((pred + mask) > 0).float().sum())
        vloss /= vn
        iou = inter / max(union, 1.0)
        L(f"ep {epoch+1:>2}: train_loss={tloss:.4f}  val_loss={vloss:.4f}  val_IoU={iou:.4f}  ({time.time()-t0:.1f}s)")
        if iou > best_iou:
            best_iou = iou
            torch.save({"model_state_dict": model.state_dict(),
                        "encoder": args.encoder,
                        "iou": iou, "epoch": epoch + 1},
                       out_dir / "ripdetseg_binary_best.pt")
            L(f"  -> saved checkpoints/ripdetseg_binary_best.pt (val IoU={iou:.4f})")

    torch.save({"model_state_dict": model.state_dict(),
                "encoder": args.encoder, "iou": iou, "epoch": args.epochs},
               out_dir / "ripdetseg_binary_final.pt")
    L(f"DONE. best val IoU = {best_iou:.4f}")
    log.close()


if __name__ == "__main__":
    main()
