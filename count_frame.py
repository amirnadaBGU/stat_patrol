"""Stage 1: count the fish in a single frame.

Runs the existing detector from ../fish_images (fishmeasure) on one frame and
splits the detections into whole and partial fish by exact class name.

All settings are global variables in config.py; edit them and run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import config as C

sys.path.insert(0, str(C.FISH_IMAGES_DIR))
from fishmeasure.detector_yolo import detect_boxes  # noqa: E402
from fishmeasure.preprocess import apply_filters    # noqa: E402

SAVE_OVERLAY = True     # writes output/frames/<clip>_f<frame>_count.jpg
SAVE_CSV = True         # writes output/frames/<clip>_f<frame>_count.csv


def read_frame(path: str | Path, frame_idx: int = 0):
    """Read one frame from a video (by index) or an image file, in BGR."""
    path = Path(path)
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        # GoPro rotation hints must be ignored, as in fishmeasure/video.py
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"cannot read frame {frame_idx} from {path}")
        return frame
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cannot read image {path}")
    return frame


class FrameReader:
    """Reads frames of one video in increasing order, as fast as possible.

    Seeking costs about 1 s on these 5.3K HEVC clips; decoding the next frame
    about 0.01 s. So short gaps are walked frame by frame and long gaps are
    seeked. Both give the same pixels (verified: max difference 0).
    """
    SEEK_GAP = 100     # frames; above this seeking is cheaper than decoding

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)   # as in read_frame
        self.pos = 0                                      # index of the next frame

    def read(self, idx: int):
        gap = idx - self.pos
        if gap < 0 or gap > self.SEEK_GAP:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        else:
            for _ in range(gap):
                self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"cannot read frame {idx} from {self.path}")
        self.pos = idx + 1
        return frame

    def close(self):
        self.cap.release()


def brighten(img_bgr, gamma: float | None = None):
    """Gamma on the Lightness channel (LAB L): L' = 255 * (L / 255) ** gamma.

    Runs AFTER fishmeasure's apply_filters (CLAHE on L + sharpen), so the
    brightening really reaches YOLO and SAM; before CLAHE it would be undone.
    Only L changes, so colours keep their hue; gamma < 1 lifts the dark and
    middle tones while 255 stays 255, so nothing is clipped to white.
    gamma defaults to config.BRIGHTEN_GAMMA; 1 (or None) = unchanged.
    """
    g = C.BRIGHTEN_GAMMA if gamma is None else gamma
    if g is None or g == 1:
        return img_bgr
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** g + 0.5, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def night_filters(frame_bgr):
    """NIGHT_ENHANCE preprocessing, in place of fishmeasure's apply_filters.

        1. bilateral denoise (d=9, sigmaColor=40, sigmaSpace=9)
        2. CLAHE on L (clip 2.0, 8x8 tiles), as apply_filters
        3. unsharp mask, amount 0.8 as apply_filters, but a wide radius
           (Gaussian 99x99, sigma 16 instead of 55x55, sigma ~8.6): YOLO sees
           the frame ~8x smaller, so sharpening must act at that scale
        4. gamma NIGHT_GAMMA on L
    Chosen with sharpen_check.py on night frames of experiments 35/40.
    """
    img = cv2.bilateralFilter(frame_bgr, 9, 40, 9)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (99, 99), 16)
    img = cv2.addWeighted(img, 1.8, blur, -0.8, 0)
    return brighten(img, C.NIGHT_GAMMA)


def preprocess(frame_bgr, gamma: float | None = None):
    """The image YOLO and SAM work on.

    NIGHT_ENHANCE on : night_filters (BRIGHTEN_GAMMA is not used)
    NIGHT_ENHANCE off: fishmeasure apply_filters (CLAHE + sharpen), then
                       gamma on L (BRIGHTEN_GAMMA, or ``gamma`` if given)
    """
    if C.NIGHT_ENHANCE:
        return night_filters(frame_bgr)
    return brighten(apply_filters(frame_bgr), gamma)


def preprocess_label() -> str:
    if C.NIGHT_ENHANCE:
        return f"NIGHT_ENHANCE (denoise + CLAHE + wide sharpen + gamma {C.NIGHT_GAMMA})"
    g = C.BRIGHTEN_GAMMA
    return "original (CLAHE + sharpen)" + ("" if g in (None, 1) else f" + gamma {g}")


def detect_frame(frame_bgr, conf: float = C.CONF_THRESHOLD, gamma: float | None = None):
    """Detect fish in one frame.

    Returns (filtered_frame, detections). Each detection is a dict with
    det_id, cls ('whole' | 'partial'), conf and bbox [x0, y0, x1, y1] in
    full-resolution pixels. Unknown classes are dropped. ``gamma`` overrides
    config.BRIGHTEN_GAMMA (used by brighten_preview.py).
    """
    proc = preprocess(frame_bgr, gamma)
    rows = detect_boxes(proc, weights=str(C.YOLO_WEIGHTS), conf=conf)
    dets = []
    for name, c, box in rows:
        if name == C.WHOLE_CLASS:
            cls = "whole"
        elif name == C.PARTIAL_CLASS:
            cls = "partial"
        else:
            continue
        dets.append({"det_id": len(dets), "cls": cls, "conf": c, "bbox": box})
    return proc, dets


def count_fish(dets):
    whole = sum(d["cls"] == "whole" for d in dets)
    partial = sum(d["cls"] == "partial" for d in dets)
    return {"whole": whole, "partial": partial, "total": whole + partial}


COLORS = {"whole": (0, 220, 0), "partial": (0, 165, 255)}


def draw(frame_bgr, dets):
    out = frame_bgr.copy()
    for d in dets:
        x0, y0, x1, y1 = map(int, d["bbox"])
        col = COLORS[d["cls"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 8)
        cv2.putText(out, f'{d["det_id"]} {d["cls"]} {d["conf"]:.2f}', (x0, max(40, y0 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 5)
    return out


def main():
    source = C.PROJECT_DIR / C.SOURCE
    print(f"{C.SOURCE}  frame {C.FRAME_INDEX}")
    frame = read_frame(source, C.FRAME_INDEX)
    proc, dets = detect_frame(frame, C.CONF_THRESHOLD)
    counts = count_fish(dets)

    for d in dets:
        print(f'  #{d["det_id"]:<3} {d["cls"]:<8} conf={d["conf"]:.3f} '
              f'bbox={[round(v) for v in d["bbox"]]}')
    print(f'whole={counts["whole"]}  partial={counts["partial"]}  total={counts["total"]}')

    fid = f"{source.stem}_f{C.FRAME_INDEX:06d}"
    out_dir = C.OUTPUT_DIR / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_CSV:
        out = out_dir / f"{fid}_count.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame_id", "det_id", "cls", "conf", "x0", "y0", "x1", "y1"])
            for d in dets:
                w.writerow([fid, d["det_id"], d["cls"], round(d["conf"], 4),
                            *(round(v, 1) for v in d["bbox"])])
        print(f"csv -> {out}")

    if SAVE_OVERLAY:
        out = out_dir / f"{fid}_count.jpg"
        cv2.imwrite(str(out), draw(proc, dets))
        print(f"overlay -> {out}")


if __name__ == "__main__":
    main()
