"""All 5 Haulover algos with smoothing + segmentation-gate combinations."""
from pathlib import Path
import cv2, numpy as np, sys, time, torch
from PIL import Image
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]


def load_gt(p, shape):
    arr = np.asarray(Image.open(p))
    g = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = g > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def auc(heat, gt, n_bins=2048):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


def recover(npz_path):
    z = np.load(str(npz_path))
    a = z["arr_0"].T  # (H, W) int8
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    return a_uint.astype(np.float32)


# Time-averaged seg gate
@torch.no_grad()
def time_avg_p_class(video_path, num_frames, skip, ckpt_path, classes, class_idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                      encoder_weights=None, in_channels=3, classes=classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    cap = cv2.VideoCapture(str(video_path))
    accum = None; n = 0; idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            kept = idx // skip
            if kept >= num_frames: break
            H, W = frame.shape[:2]
            rgb = frame[..., ::-1].astype(np.float32)/255.0
            t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
            t = (t - mean) / std
            ph = (32-H%32)%32; pw = (32-W%32)%32
            if ph or pw: t = torch.nn.functional.pad(t, (0,pw,0,ph))
            sm = torch.softmax(model(t), dim=1)[0, class_idx, :H, :W].cpu().numpy()
            if accum is None: accum = np.zeros_like(sm)
            accum += sm; n += 1
        idx += 1
    cap.release()
    return accum / max(n, 1)


def main():
    haul = ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22"
    mask_path = ROOT / "script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
    gt_path = ROOT / "script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    sea_valid = (mask == 255).astype(np.float32)

    print("Computing time-averaged P(land) for use as inverse gate...")
    t0 = time.time()
    p_land_avg = time_avg_p_class(ROOT / "data/DJI_0009.MOV", 600, 2,
                                    ROOT / "seg_refiner/checkpoints/best.pt",
                                    classes=8, class_idx=4)
    print(f"  P(land) computed in {time.time()-t0:.1f}s; mean={p_land_avg.mean():.3f}")

    files = {
        "HS_highorder":  haul / "heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz",
        "LK_highorder":  haul / "heatmap_Haulover_Miami_FL_LK_highorder_result_basic_mode.npz",
        "HS_firstorder": haul / "heatmap_Haulover_Miami_FL_HS_firstorder_result_basic_mode.npz",
        "lucaskanade":   haul / "heatmap_Haulover_Miami_FL_lucaskanade_result_basic_mode.npz",
        "typhoon":       haul / "heatmap_Haulover_Miami_FL_typhoon_result_typhoon_mode.npz",
    }
    gt = load_gt(gt_path, (mask.shape[0], mask.shape[1]))

    print(f"\n{'algo':<14s}{'baseline':>10s}{'+sea':>9s}{'σ=15':>9s}{'σ=25':>9s}{'σ=35':>9s}{'σ=25*(1-Pland)':>16s}{'σ=25 ensemble':>16s}")
    sigma_25_heats = []
    for nm, fp in files.items():
        canonical = recover(fp)
        # Only use the recovered version; for HS_firstorder/LK_highorder/etc.
        # uint8 reinterp may not differ from signed (small values), but doesn't hurt.
        b = auc(canonical, gt)
        s_only = auc(canonical * sea_valid, gt)
        s15 = auc(cv2.GaussianBlur(canonical, (0,0), 15), gt)
        s25 = auc(cv2.GaussianBlur(canonical, (0,0), 25), gt)
        s35 = auc(cv2.GaussianBlur(canonical, (0,0), 35), gt)
        h25 = cv2.GaussianBlur(canonical, (0,0), 25)
        s25_pland = auc(h25 * (1.0 - p_land_avg), gt)
        sigma_25_heats.append(h25)
        print(f"{nm:<14s}{b:>10.4f}{s_only:>9.4f}{s15:>9.4f}{s25:>9.4f}{s35:>9.4f}{s25_pland:>16.4f}")

    # Cross-algo ensemble of smoothed canonicals
    print("\n--- Ensembles of σ=25 smoothed canonicals ---")
    arrs = sigma_25_heats
    nm_arrs = list(files.keys())
    # Sum
    h_sum = sum(arrs) / len(arrs)
    print(f"  mean (all 5):                      AUC = {auc(h_sum, gt):.4f}")
    # Mean of HS_high + LK_high (manuscript's two best)
    h_mean = (arrs[0] + arrs[1]) / 2
    print(f"  mean (HS_high + LK_high):          AUC = {auc(h_mean, gt):.4f}")
    # Mean of HS_high + HS_first + LK_high
    h_mean = (arrs[0] + arrs[1] + arrs[2]) / 3
    print(f"  mean (HS_high+HS_first+LK_high):   AUC = {auc(h_mean, gt):.4f}")
    # Geometric mean (normalized)
    n = [a / max(a.max(), 1e-6) for a in arrs]
    h_gm = np.exp(np.log(np.maximum(np.stack(n), 1e-6)).mean(axis=0))
    print(f"  geomean (all 5):                   AUC = {auc(h_gm, gt):.4f}")
    # Max
    h_max = np.maximum.reduce(n)
    print(f"  max (all 5):                       AUC = {auc(h_max, gt):.4f}")


if __name__ == "__main__":
    main()
