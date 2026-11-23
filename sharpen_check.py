"""Stand-alone test: does stronger sharpening help the detector on night frames?

Does NOT change the pipeline. Each variant replaces fishmeasure's
apply_filters (CLAHE on L + unsharp mask) with its own version, then runs the
same YOLO (detect_boxes) and counts whole / partial fish and confidence.

Note on scale: YOLO sees the frame shrunk ~8x (5312 -> 640 px wide), so
sharpening only matters at a radius of several full-resolution pixels. The
original unsharp mask (Gaussian 55x55, sigma ~8.6 px) works at about 1 px of
YOLO's image; the variants change its strength and radius around that.

Output: table in the terminal, montage per frame in output/sharpen_check/.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import config as C
from count_frame import COLORS, brighten, read_frame

sys.path.insert(0, str(C.FISH_IMAGES_DIR))
from fishmeasure.detector_yolo import detect_boxes  # noqa: E402

FRAMES = [("videos/40/GX011099.MP4", f) for f in (173, 401, 635, 891, 1085)] + \
         [("videos/40/GX011094.MP4", 300)]


def clahe_L(img, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharp(img, amount, ksize=55, sigma=0):
    """img + amount * (img - blur); amount 0.8 = the original pipeline."""
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def denoise(f):
    return cv2.bilateralFilter(f, 9, 40, 9)


# Set of variants to run (ALL_VARIANTS has every one tried so far).
# "+ gN" = gamma N on L after the sharpening, exactly as BRIGHTEN_GAMMA does.
ALL_VARIANTS = {
    # name: function(frame) -> preprocessed image
    "original (0.8)":        lambda f: unsharp(clahe_L(f), 0.8),
    "unsharp x2 (1.6)":      lambda f: unsharp(clahe_L(f), 1.6),
    "unsharp x3 (2.4)":      lambda f: unsharp(clahe_L(f), 2.4),
    "wide radius (0.8,s16)": lambda f: unsharp(clahe_L(f), 0.8, 99, 16),
    "two scales":            lambda f: unsharp(unsharp(clahe_L(f), 0.8), 0.8, 99, 16),
    "CLAHE 4 + 0.8":         lambda f: unsharp(clahe_L(f, 4.0), 0.8),
    "denoise + x2":          lambda f: unsharp(clahe_L(denoise(f)), 1.6),
    # brightening + sharpening
    "orig + g0.7":           lambda f: brighten(unsharp(clahe_L(f), 0.8), 0.7),
    "denoise x2 + g0.85":    lambda f: brighten(unsharp(clahe_L(denoise(f)), 1.6), 0.85),
    "denoise x2 + g0.7":     lambda f: brighten(unsharp(clahe_L(denoise(f)), 1.6), 0.7),
    "wide + g0.85":          lambda f: brighten(unsharp(clahe_L(f), 0.8, 99, 16), 0.85),
    "wide + g0.7":           lambda f: brighten(unsharp(clahe_L(f), 0.8, 99, 16), 0.7),
    "denoise + wide":        lambda f: unsharp(clahe_L(denoise(f)), 0.8, 99, 16),
    "denoise wide + g0.85":  lambda f: brighten(unsharp(clahe_L(denoise(f)), 0.8, 99, 16), 0.85),
    "denoise wide + g0.7":   lambda f: brighten(unsharp(clahe_L(denoise(f)), 0.8, 99, 16), 0.7),
    "denoise wide + g0.55":  lambda f: brighten(unsharp(clahe_L(denoise(f)), 0.8, 99, 16), 0.55),
}
RUN = ["original (0.8)", "denoise + wide", "denoise wide + g0.85", "denoise wide + g0.7",
       "denoise wide + g0.55"]
VARIANTS = {k: ALL_VARIANTS[k] for k in RUN}


def run(proc):
    rows = detect_boxes(proc, weights=str(C.YOLO_WEIGHTS), conf=C.CONF_THRESHOLD)
    return [(("whole" if n == C.WHOLE_CLASS else "partial"), c, b)
            for n, c, b in rows if n in (C.WHOLE_CLASS, C.PARTIAL_CLASS)]


def tile(proc, dets, title):
    img = proc.copy()
    for cls, c, b in dets:
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[cls], 10)
        cv2.putText(img, f"{cls} {c:.2f}", (x0, max(60, y0 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, COLORS[cls], 6)
    for col, th in (((0, 0, 0), 20), ((255, 255, 255), 8)):
        cv2.putText(img, title, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 4.0, col, th)
    return cv2.resize(img, None, fx=0.2, fy=0.2)


def main():
    out_dir = C.OUTPUT_DIR / "sharpen_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    tot = {v: [0, 0, []] for v in VARIANTS}
    print(f"{'frame':<20}" + "".join(f"{v:>23}" for v in VARIANTS))
    for src, idx in FRAMES:
        frame = read_frame(C.PROJECT_DIR / src, idx)
        cells, tiles = [], []
        for name, fn in VARIANTS.items():
            proc = fn(frame)
            dets = run(proc)
            w = sum(d[0] == "whole" for d in dets)
            p = len(dets) - w
            confs = [d[1] for d in dets]
            tot[name][0] += w
            tot[name][1] += p
            tot[name][2] += confs
            cells.append(f"{w}/{p} c{np.mean(confs) if confs else 0:.2f}")
            tiles.append(tile(proc, dets, f"{name}: whole {w} partial {p}"))
        label = f"{src.split('/')[-1][:8]} f{idx}"
        print(f"{label:<20}" + "".join(f"{c:>23}" for c in cells))
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        mont = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(out_dir / f"{label.replace(' ', '_')}_montage.jpg"), mont)
    print("-" * (20 + 23 * len(VARIANTS)))
    print(f"{'TOTAL whole/partial':<20}" + "".join(f"{f'{t[0]}/{t[1]}':>23}" for t in tot.values()))
    print(f"{'mean conf':<20}" + "".join(f"{np.mean(t[2]):>23.3f}" for t in tot.values()))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
