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


def brighten(frame_bgr, percent: float | None = None):
    """Multiply every pixel by (1 + percent/100), capped at 255.

    Runs before fishmeasure's apply_filters. percent defaults to
    config.BRIGHTEN_PERCENT; 0 returns the frame unchanged.
    """
    p = C.BRIGHTEN_PERCENT if percent is None else percent
    if not p:
        return frame_bgr
    return cv2.convertScaleAbs(frame_bgr, alpha=1 + p / 100.0, beta=0)


def detect_frame(frame_bgr, conf: float = C.CONF_THRESHOLD):
    """Detect fish in one frame.

    Returns (filtered_frame, detections). Each detection is a dict with
    det_id, cls ('whole' | 'partial'), conf and bbox [x0, y0, x1, y1] in
    full-resolution pixels. Unknown classes are dropped.
    """
    proc = apply_filters(brighten(frame_bgr))   # brighten, then CLAHE + sharpen
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
