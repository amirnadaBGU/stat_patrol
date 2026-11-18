"""Stage 2: measure every fish in a single frame and write a CSV.

For each detection, whole or partial, the existing chain from fish_images:

    SAM 2 (box prompt) -> clean mask -> midline -> 9 keypoints
    -> unit length K (the clip's own POLY lens + Snell, loo_fit_poly.k_of)

Whole fish then go on to a length and a weight:

    L = closed-form height correction of K * H   (loo_fit_poly.predict)
    W = 10 ** (A + B * log10 L)

Partial fish stop at K. Their fraction needs L_mean, which only exists once
the whole experiment has been measured (a later stage).

A detection that cannot be measured keeps its row, with status saying why,
and empty measurement columns. Nothing is invented for it.

All settings are global variables in config.py; edit them and run.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import config as C
from count_frame import COLORS, count_fish, detect_frame, read_frame

sys.path.insert(0, str(C.FISH_IMAGES_DIR))
from fishmeasure.masks import clean                    # noqa: E402
from fishmeasure.measurement import measure            # noqa: E402
from fishmeasure.quality import assess                 # noqa: E402
from fishmeasure.segmenter_sam import load_sam, segment_with_box  # noqa: E402
from fishmeasure.skeleton import midline_path          # noqa: E402
import loo_fit_poly as LFP                             # noqa: E402

LFP.H_MM = C.CAMERA_HEIGHT_MM   # predict() reads the camera height from here

SAVE_OVERLAY = True             # writes <frame>.jpg next to the CSV
SKELETON_THREADS = min(8, os.cpu_count() or 1)   # parallel per-fish measurement

CSV_COLUMNS = [
    "frame_id", "det_id", "cls", "conf", "x0", "y0", "x1", "y1",
    "status", "flags", "length_px", "unit_length_K", "rho",
    "length_mm", "weight_g",
]


def lens_of(video: Path):
    """The POLY lens model recorded in the clip's own GPMF metadata."""
    LFP.CLIP_ROOT = video.parent
    return LFP.lens_for(video.name)


def length_mm(K: float) -> float:
    """Whole-fish length from K, with the height correction d = K_HEIGHT * L."""
    return LFP.predict(C.K_HEIGHT, K)


def weight_g(L_mm: float) -> float:
    return 10 ** (C.LW_LOG10_A + C.LW_B * math.log10(L_mm))


def segment_all(proc, dets):
    """SAM masks for every detection of the frame, in ONE model call.

    SAM encodes the image once and decodes each box prompt separately, so this
    gives the same result as calling segment_with_box per box at a fraction of
    the cost. Not bit-identical: on 49 fish in dense frames, 6 masks differed
    by 1-5 pixels out of ~200k-400k (GPU float rounding with a different batch
    size); keypoints, K, length, weight and flags were identical in all 49.
    Each path on its own is deterministic.

    The post-processing below mirrors fishmeasure/segmenter_sam.py
    segment_with_box (threshold 0.5 -> 0/255, resize to frame). If that
    function changes, change this too.

    Returns a list aligned with ``dets``: a 0/255 mask, or an Exception.
    Falls back to one segment_with_box call per box if the batch fails or does
    not return one mask per box.
    """
    if not dets:
        return []
    H, W = proc.shape[:2]
    try:
        model = load_sam(str(C.SAM_WEIGHTS))
        res = model(proc, bboxes=[list(map(float, d["bbox"])) for d in dets], verbose=False)[0]
        data = res.masks.data
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        if arr.shape[0] != len(dets):
            raise ValueError(f"{arr.shape[0]} masks for {len(dets)} boxes")
        masks = []
        for a in arr:
            m = (a > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(m)
        return masks
    except Exception as exc:  # noqa: BLE001 - fall back to the original per-box call
        print(f"  [batched SAM failed ({exc!r}) - one call per box]")
        out = []
        for d in dets:
            try:
                out.append(segment_with_box(proc, d["bbox"], weights=str(C.SAM_WEIGHTS)))
            except Exception as e:  # noqa: BLE001
                out.append(e)
        return out


def measure_detection(proc_shape, det, raw_mask, lens):
    """Measure one detection from its SAM mask. Adds the results to ``det``.

    Runs in a worker thread (see measure_frame); touches only ``det``.
    """
    det.update(status="ok", flags="", length_px=None, unit_length_K=None,
               rho=None, length_mm=None, weight_g=None,
               mask=None, keypoints=None)
    if isinstance(raw_mask, Exception):
        det["status"] = f"sam_failed: {raw_mask!r}"
        return det
    mask = clean(raw_mask)
    det["mask"] = mask

    _, flags, measurable = assess(mask, proc_shape)
    det["flags"] = ";".join(flags)
    if not measurable:
        det["status"] = "mask_unmeasurable"
        return det

    try:
        keys, _, m = measure(midline_path(mask), mask, k=C.NUM_KEYPOINTS)
    except Exception as exc:  # noqa: BLE001
        det["status"] = f"skeleton_failed: {exc!r}"
        return det

    pts = [(float(x), float(y)) for x, y in keys]
    det["keypoints"] = pts
    det["length_px"] = m["curved_length_px"]
    det["unit_length_K"] = LFP.k_of(lens, pts)
    det["rho"] = LFP.rho_of(lens, pts)

    if det["cls"] == "whole":
        det["length_mm"] = length_mm(det["unit_length_K"])
        det["weight_g"] = weight_g(det["length_mm"])
    return det


_POOL = None


def _pool():
    """Thread pool for the per-fish skeleton step (CPU; results identical to serial)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=SKELETON_THREADS)
    return _POOL


def measure_frame(video: Path, frame_idx: int, frame=None):
    """Detect and measure every fish in one frame. Returns (proc, dets).

    ``frame`` - the already decoded BGR frame, if the caller has it
    (run_experiment reads clips sequentially); otherwise it is read here.
    """
    lens = lens_of(video)
    if frame is None:
        frame = read_frame(video, frame_idx)
    proc, dets = detect_frame(frame, C.CONF_THRESHOLD)
    masks = segment_all(proc, dets)                    # one SAM call per frame
    jobs = [_pool().submit(measure_detection, proc.shape, d, m, lens)
            for d, m in zip(dets, masks)]              # skeletons in parallel
    for j in jobs:
        j.result()
    return proc, dets


def frame_id(video: Path, frame_idx: int) -> str:
    return f"{video.stem}_f{frame_idx:06d}"


def write_csv(path: Path, fid: str, dets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for d in dets:
            x0, y0, x1, y1 = d["bbox"]
            row = {k: d.get(k) for k in CSV_COLUMNS}
            row.update(frame_id=fid, x0=round(x0, 1), y0=round(y0, 1),
                       x1=round(x1, 1), y1=round(y1, 1), conf=round(d["conf"], 4))
            for k, nd in (("length_px", 1), ("unit_length_K", 6), ("rho", 4),
                          ("length_mm", 1), ("weight_g", 1)):
                if row[k] is not None:
                    row[k] = round(row[k], nd)
            w.writerow(row)


def draw(proc, dets):
    """Boxes, translucent masks, midlines and the per-fish numbers."""
    out = proc.copy()
    tint = out.copy()
    for d in dets:
        if d.get("mask") is not None:
            tint[d["mask"] > 0] = COLORS[d["cls"]]
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    for d in dets:
        col = COLORS[d["cls"]]
        x0, y0, x1, y1 = map(int, d["bbox"])
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 6)
        if d.get("keypoints"):
            pts = np.array(d["keypoints"], np.int32)
            cv2.polylines(out, [pts], False, (0, 0, 255), 6)
            for p in pts:
                cv2.circle(out, tuple(int(v) for v in p), 12, (0, 255, 255), -1)
        if d["status"] != "ok":
            label = f'{d["det_id"]} {d["cls"]} FAILED'
        elif d["cls"] == "whole":
            label = f'{d["det_id"]} K={d["unit_length_K"]:.3f} {d["length_mm"]:.0f}mm {d["weight_g"]:.0f}g'
        else:
            label = f'{d["det_id"]} partial K={d["unit_length_K"]:.3f}'
        cv2.putText(out, label, (x0, max(50, y0 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 5)
    return out


def main():
    video = C.PROJECT_DIR / C.SOURCE
    fid = frame_id(video, C.FRAME_INDEX)
    proc, dets = measure_frame(video, C.FRAME_INDEX)

    out_dir = C.OUTPUT_DIR / "frames"
    write_csv(out_dir / f"{fid}.csv", fid, dets)

    print(f"{C.SOURCE}  frame {C.FRAME_INDEX}")
    for d in dets:
        K = d["unit_length_K"]
        extra = (f'  L={d["length_mm"]:.1f}mm  W={d["weight_g"]:.1f}g'
                 if d["length_mm"] is not None else "")
        print(f'  #{d["det_id"]:<3} {d["cls"]:<8} conf={d["conf"]:.3f}  {d["status"]:<18}'
              f'{"" if K is None else f"K={K:.4f}"}{extra}  {d["flags"]}')
    c = count_fish(dets)
    print(f'whole={c["whole"]}  partial={c["partial"]}')
    print(f"csv -> {out_dir / (fid + '.csv')}")
    if SAVE_OVERLAY:
        cv2.imwrite(str(out_dir / f"{fid}.jpg"), draw(proc, dets))
        print(f"overlay -> {out_dir / (fid + '.jpg')}")


if __name__ == "__main__":
    main()
