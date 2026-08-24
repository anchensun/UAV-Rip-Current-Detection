"""Attribute each TrainDataset_RGB frame to its source UAV scene, to resolve the
"data leakage" question rigorously.

Background. The scene-parsing corpus is documented as having been assembled from
the two *training-only* captures, Ocean Reef Park and South Beach; the two
rip-current *evaluation* captures are Palm Beach (DJI_0011) and Haulover
(DJI_0009). An earlier "transparency" paragraph reported an image-MSE
nearest-neighbour match of 123 frames to Haulover and 61 to Palm Beach, but that
matcher was run against only the two evaluation videos, so every frame was forced
into one of those two buckets regardless of true origin. It therefore cannot
establish provenance and is not evidence of test-scene leakage. (That paragraph
is corrected in the current manuscript.)

This script re-does the attribution with robust deep (ResNet-18 ImageNet)
embeddings rather than brittle whole-image MSE.

IMPORTANT LIMITATION -- read before using the output. The candidate POOLS below
do NOT include Ocean Reef Park: no source video for that capture was recoverable.
Because Ocean Reef Park is visually near-identical to the Palm Beach capture
(the two sites are roughly 131 m apart), its frames are necessarily forced onto
the Palm Beach pool. That is why train_frame_scenes.csv labels 88 of the 215
distinct frames with an evaluation scene as nearest neighbour. Those labels are
an artifact of the missing pool, NOT evidence that the frames were extracted
from the evaluation videos -- which detect_test_leakage.py rules out directly
(minimum pixel-MSE 674 against both evaluation videos, none below 200).

Outputs:
  seg_refiner/results/train_frame_scenes.csv
      frame, split, scene, is_test, top_video, top_sim,
      sim_PalmBeach, sim_Haulover, sim_SouthBeach1, sim_SouthBeach2,
      sim_DJI0008, sim_DJI0047, test_margin
  seg_refiner/results/attribution_audit_PalmBeach.png   (frames assigned to PB + NN match)
  seg_refiner/results/attribution_audit_Haulover.png    (frames assigned to HV + NN match)
"""
from __future__ import annotations
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "data" / "extracted" / "Rip Current Video" / "TrainDataset_RGB"
NDV = ROOT / "data" / "extracted" / "Rip Current Video" / "New Drone Videos"
RES = ROOT / "seg_refiner" / "results"
OUT_CSV = RES / "train_frame_scenes.csv"

# Source pools grouped by scene. Test scenes first. Each entry: (scene, is_test,
# [video paths], n_samples_for_scene).  South Beach scenes use the readable
# sub-clips; the others use the raw captures with a bounded sequential scan
# (DJI_0011's tail is h264-corrupt beyond ~15k frames).
POOLS = [
    ("PalmBeach",   True,  [ROOT / "data/DJI_0011.MOV"], 220),
    ("Haulover",    True,  [ROOT / "data/DJI_0009.MOV"], 220),
    ("SouthBeach1", False, sorted(NDV.glob("DJI_0006_*.mp4")), 220),
    ("SouthBeach2", False, sorted(NDV.glob("DJI_0007_*.mp4")), 220),
    ("DJI0008",     False, [ROOT / "data/DJI_0008.MOV"], 150),
    ("DJI0047",     False, [ROOT / "data/DJI_0047-002.MP4"], 150),
]
MAX_SCAN = 20000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_encoder():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()          # 512-d global-avg-pool feature
    m.eval().to(DEVICE)
    return m


@torch.no_grad()
def embed(model, bgr_list):
    """Batch BGR frames -> L2-normalized 512-d ResNet features."""
    ts = []
    for bgr in bgr_list:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ts.append(torch.from_numpy(rgb).permute(2, 0, 1))
    x = torch.stack(ts).to(DEVICE)
    x = (x - _IMAGENET_MEAN.to(DEVICE)) / _IMAGENET_STD.to(DEVICE)
    feats = []
    for i in range(0, len(x), 64):
        f = model(x[i:i + 64])
        feats.append(torch.nn.functional.normalize(f, dim=1))
    return torch.cat(feats)


def sample_video(path: Path, n_target: int):
    cap = cv2.VideoCapture(str(path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scan = min(n_total, MAX_SCAN)
    stride = max(1, scan // max(n_target, 1))
    frames, tsecs = [], []
    idx = 0
    consec_fail = 0
    while idx < scan:
        ret, fr = cap.read()
        if not ret or fr is None:
            consec_fail += 1
            if consec_fail > 40:
                break
            idx += 1
            continue
        consec_fail = 0
        if idx % stride == 0:
            frames.append(fr)
            tsecs.append(idx / fps)
        idx += 1
    cap.release()
    return frames, tsecs


def load_train_frames():
    out = []
    for split in ("train", "val"):
        img_dir = TR / split / "img" / "0"
        for p in sorted(img_dir.glob("*.png"), key=lambda x: int(x.stem.replace("img", ""))):
            out.append((p.name, split, p))
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}, torchvision={torchvision.__version__}")
    model = build_encoder()

    # Build per-scene feature pools (keep thumbnails + provenance for audit montage)
    scene_feats, scene_thumbs, scene_prov = {}, {}, {}
    for scene, is_test, vids, n_target in POOLS:
        per_vid = max(1, n_target // max(len(vids), 1))
        allf, allt, allp = [], [], []
        for v in vids:
            if not Path(v).exists():
                print(f"  [skip missing] {v}")
                continue
            frames, tsecs = sample_video(Path(v), per_vid)
            allf.extend(frames)
            allt.extend([f"{Path(v).stem}@{t:.0f}s" for t in tsecs])
        if not allf:
            print(f"  [WARN] no frames for {scene}")
            scene_feats[scene] = torch.empty((0, 512), device=DEVICE)
            scene_thumbs[scene] = []
            scene_prov[scene] = []
            continue
        feats = embed(model, allf)
        scene_feats[scene] = feats
        scene_thumbs[scene] = [cv2.resize(f, (320, 180)) for f in allf]
        scene_prov[scene] = allt
        print(f"  {scene:12s} test={is_test} -> {len(allf)} samples from {len(vids)} video(s)")

    frames = load_train_frames()
    tf_feats = embed(model, [cv2.imread(str(p)) for _, _, p in frames])
    print(f"embedded {len(frames)} training frames")

    scene_names = [p[0] for p in POOLS]
    test_scenes = {p[0] for p in POOLS if p[1]}
    rows = []
    audit = defaultdict(list)   # scene -> (frame_name, path, best_pool_idx)
    for i, (name, split, path) in enumerate(frames):
        q = tf_feats[i:i + 1]                          # (1,512)
        sims, best_idx = {}, {}
        for scene in scene_names:
            fp = scene_feats[scene]
            if fp.shape[0] == 0:
                sims[scene] = -1.0; best_idx[scene] = -1; continue
            s = (fp @ q.T).squeeze(1)
            k = int(torch.argmax(s))
            sims[scene] = float(s[k]); best_idx[scene] = k
        top_scene = max(scene_names, key=lambda s: sims[s])
        best_test = max((sims[s] for s in test_scenes), default=-1.0)
        best_train = max((sims[s] for s in scene_names if s not in test_scenes), default=-1.0)
        rows.append({
            "frame": name, "split": split, "scene": top_scene,
            "is_test": int(top_scene in test_scenes),
            "top_video": scene_prov[top_scene][best_idx[top_scene]] if best_idx[top_scene] >= 0 else "",
            "top_sim": round(sims[top_scene], 4),
            "sim_PalmBeach": round(sims["PalmBeach"], 4),
            "sim_Haulover": round(sims["Haulover"], 4),
            "sim_SouthBeach1": round(sims["SouthBeach1"], 4),
            "sim_SouthBeach2": round(sims["SouthBeach2"], 4),
            "sim_DJI0008": round(sims["DJI0008"], 4),
            "sim_DJI0047": round(sims["DJI0047"], 4),
            "test_margin": round(best_test - best_train, 4),
        })
        if top_scene in test_scenes:
            audit[top_scene].append((name, path, best_idx[top_scene]))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # summary
    cnt = Counter(r["scene"] for r in rows)
    cnt_split = {s: Counter(r["scene"] for r in rows if r["split"] == s) for s in ("train", "val")}
    n_test = sum(1 for r in rows if r["is_test"])
    print("\n=== Attribution summary (ResNet-18 NN over all source scenes) ===")
    for s in scene_names:
        print(f"  {s:12s}: {cnt.get(s,0):3d}   (train {cnt_split['train'][s]}, val {cnt_split['val'][s]})"
              f"   {'[TEST scene]' if s in test_scenes else ''}")
    print(f"  --> {n_test}/{len(rows)} frames attributed to a TEST scene (potential leakage)")
    print(f"  --> {len(rows)-n_test}/{len(rows)} frames from training-only scenes")
    tm = np.array([r["top_sim"] for r in rows])
    print(f"  top-sim: min={tm.min():.3f} med={np.median(tm):.3f} max={tm.max():.3f}")
    print(f"  (manuscript L858 claimed 123 Haulover / 61 Palm Beach by 2-video image-MSE)")
    print(f"  wrote {OUT_CSV}")

    # audit montages: frames assigned to each test scene, next to their NN video frame
    for scene in test_scenes:
        items = audit[scene][:16]
        if not items:
            continue
        tiles = []
        for name, path, k in items:
            left = cv2.resize(cv2.imread(str(path)), (320, 180))
            right = scene_thumbs[scene][k].copy()
            cv2.putText(left, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(right, scene_prov[scene][k], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            tiles.append(np.hstack([left, np.full((180, 3, 3), 255, np.uint8), right]))
        per_row = 2
        rws = [np.hstack(tiles[r * per_row:(r + 1) * per_row]) for r in range((len(tiles) + per_row - 1) // per_row)]
        wmax = max(im.shape[1] for im in rws)
        rws = [np.pad(im, ((0, 0), (0, wmax - im.shape[1]), (0, 0)), constant_values=255) for im in rws]
        outp = RES / f"attribution_audit_{scene}.png"
        cv2.imwrite(str(outp), np.vstack(rws))
        print(f"  wrote {outp} ({len(items)} frames)")


if __name__ == "__main__":
    main()
