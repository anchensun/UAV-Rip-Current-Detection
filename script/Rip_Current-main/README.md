# Rip Current Detection from UAV Video

This repo contains research code for detecting rip currents in near-shore UAV video using optical flow and offshore direction priors. The pipeline computes dense motion, compares it to estimated offshore flow direction, and accumulates a heatmap highlighting likely rip current regions.

**Requirements**
- Python 3.7+.
- Core deps: `numpy`, `opencv-python`, `scipy`, `pywavelets`, `matplotlib`.
- Utilities: `sympy` (offshore direction), `Pillow` and `pyyaml` (evaluation).

**Repository Layout**
- `Typhoonmode.py`, `Typhoonmode_thread.py`: Typhoon optical flow implementation.
- `OFlib.py`: helper routines (optical flow utilities, heatmap, dye visualization).
- `Optical_Flow_video.py`: batch optical flow on videos; outputs per-frame flow archives.
- `OptOffShoreDirection.py`: estimate offshore direction field from coastline masks.
- `GenDemo.py`, `GenDemo_MultiCore.py`: render flow visualizations / rip current heatmaps.
- `rip_curent_evaluation/`: evaluation utilities (precision-recall curves) + workspace configs.
- `New_Rip_Mask/`: per-video masks named `{video_name}_mask.png`.

**Data Layout and Naming**
- Video base name `video_name` is the filename without extension (e.g., `DJI_0006`).
- Masks live in `New_Rip_Mask/{video_name}_mask.png`.
- Optical flow outputs are `.npz` files with `arr_0` (u) and `arr_1` (v).

**Pipeline (Typical)**
1) **Prepare masks and videos**
   - Place input videos (e.g., `.MOV` or `.mp4`) under a folder such as `./DJI_Video/`.
   - Ensure each video has a matching mask in `New_Rip_Mask/`.

2) **Compute offshore direction**
   - Edit `video_lib` in `OptOffShoreDirection.py` or call `OffshoreDirLib(video_dir)`.
   - Outputs: `*_OffShoreDir_1by1.npz` and `*_OffShore_1by1.jpg` in the video folder.

3) **Compute optical flow**
   - Edit `vediolib`, `mode`, `core`, `frame_num` in `Optical_Flow_video.py`.
   - Modes: `typhoon`, `lucas_kanade`, `higher_order`, `HS`.
   - Outputs: `*_typhoon.zip`, `*_LK_firstorder.zip`, `*_LK_highorder.zip`, `*_HS_firstorder.zip`.

4) **Generate heatmaps / demos**
   - `GenDemo.py` (simple) or `GenDemo_MultiCore.py` (more advanced).
   - `mode='heatmap'` accumulates a rip current heatmap over time.
   - Outputs:
     - `GenDemo.py`: `*_HeatSave.npz` and per-frame images under `{video_name}/`.
     - `GenDemo_MultiCore.py`: `FinalPlot_{sec}s/` with `RipDetection_*.npz` and `*_finalplot.jpg`.

5) **Evaluate (optional)**
   - Unzip `rip_curent_evaluation/workspace/conf.yaml.gz`.
   - Run `rip_curent_evaluation/src/evaluation.py` with a local config path.
   - Evaluation expects heatmaps under `result_path` from the config.

**Notes**
- Many scripts use hard-coded paths in their `__main__` blocks; update those paths before running.
- Masks are used to suppress sky/land regions and define the coastline; missing masks will cause failures.
- The code is research-oriented and assumes consistent naming across videos, masks, and result files.
