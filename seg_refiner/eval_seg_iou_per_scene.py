"""Test-time per-scene sea-segmentation IoU/F1 of the leakage-free segmenter.

The six-class segmenter is trained only on the training-only captures (no frame
from either evaluation video, verified by detect_test_leakage.py). To report its
per-scene sea-segmentation quality on the two EVALUATION scenes honestly, we
evaluate it at test time: run the trained segmenter on sampled frames of each
evaluation video, take argmax==sea, and compare to that scene's static sea mask
as ground truth. These are out-of-training (transfer) numbers, not validation
numbers, so they are consistent with the no-leakage finding.

Usage: python seg_refiner/eval_seg_iou_per_scene.py --ckpt seg_refiner/checkpoints/best.pt
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENES = {
    "Palm Beach": ("data/DJI_0011.MOV", "script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"),
    "Haulover":   ("data/DJI_0009.MOV", "script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"),
}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m = smp.Unet(encoder_name=ck.get("encoder", "resnet18"), encoder_weights=None,
                 in_channels=3, classes=ck["num_classes"]).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def pred_sea(model, bgr, H, W):
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    t = (t - _MEAN.to(DEVICE)) / _STD.to(DEVICE)
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph))
    logits = model(t)[..., :H, :W]
    return (logits.argmax(1)[0] == 2).cpu().numpy()   # sea class == 2


def sample_frames(video, n=40, max_scan=12000):
    cap = cv2.VideoCapture(str(ROOT / video))
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_scan)
    stride = max(1, total // n)
    frames, idx, fail = [], 0, 0
    while idx < total:
        ret, fr = cap.read()
        if not ret:
            fail += 1
            if fail > 40: break
            idx += 1; continue
        fail = 0
        if idx % stride == 0: frames.append(fr)
        idx += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seg_refiner/checkpoints/best.pt")
    args = ap.parse_args()
    model = load_model(ROOT / args.ckpt)
    print(f"ckpt={args.ckpt} device={DEVICE}\n")
    print(f"{'Scene':12s} {'IoU(sea)':>9s} {'F1(sea)':>9s}  (test-time transfer, static-mask GT)")
    for scene, (video, mask) in SCENES.items():
        m = cv2.imread(str(ROOT / mask), cv2.IMREAD_GRAYSCALE)
        H, W = m.shape
        gt = (m == 255)
        inter = union = tp = fp = fn = 0
        frames = sample_frames(video)
        for fr in frames:
            p = pred_sea(model, fr, H, W)
            inter += np.logical_and(p, gt).sum()
            union += np.logical_or(p, gt).sum()
            tp += np.logical_and(p, gt).sum()
            fp += np.logical_and(p, ~gt).sum()
            fn += np.logical_and(~p, gt).sum()
        iou = inter / max(union, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"{scene:12s} {iou:>9.3f} {f1:>9.3f}  (n={len(frames)} frames)")


if __name__ == "__main__":
    main()
