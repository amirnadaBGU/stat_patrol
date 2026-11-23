"""Calibrate BRIGHTEN_GAMMA by hand: compare several gammas on one frame.

For SOURCE / FRAME_INDEX in config.py, runs the detector once per gamma in
BRIGHTEN_PREVIEW_GAMMAS (preprocessing exactly as in a real run: CLAHE +
sharpen, then gamma on L) and prints how many whole / partial fish were found
and their mean confidence. Saves one overlay per gamma and a side-by-side
montage to output/brighten_preview/.

Pick the gamma that finds the fish (more whole fish, sensible boxes) without
false detections, put it in BRIGHTEN_GAMMA and check one or two more frames.
"""
from __future__ import annotations

import cv2
import numpy as np

import config as C
from count_frame import count_fish, detect_frame, draw, read_frame


def main():
    source = C.PROJECT_DIR / C.SOURCE
    frame = read_frame(source, C.FRAME_INDEX)
    out_dir = C.OUTPUT_DIR / "brighten_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    fid = f"{source.stem}_f{C.FRAME_INDEX:06d}"

    print(f"{C.SOURCE}  frame {C.FRAME_INDEX}")
    print(f"{'gamma':>6} {'whole':>6} {'partial':>8} {'mean conf':>10} {'mean L':>7}")
    tiles = []
    for g in C.BRIGHTEN_PREVIEW_GAMMAS:
        proc, dets = detect_frame(frame, C.CONF_THRESHOLD, gamma=g)
        n = count_fish(dets)
        conf = np.mean([d["conf"] for d in dets]) if dets else float("nan")
        L = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        print(f"{g:>6.2f} {n['whole']:>6} {n['partial']:>8} {conf:>10.3f} {L:>7.1f}")
        img = draw(proc, dets)
        cv2.putText(img, f"gamma {g:.2f}  whole {n['whole']}  partial {n['partial']}",
                    (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 0), 20)
        cv2.putText(img, f"gamma {g:.2f}  whole {n['whole']}  partial {n['partial']}",
                    (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)
        cv2.imwrite(str(out_dir / f"{fid}_g{g:.2f}.jpg"), img)
        tiles.append(cv2.resize(img, None, fx=0.25, fy=0.25))

    # montage: two columns, small tiles
    blank = np.zeros_like(tiles[0])
    if len(tiles) % 2:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    out = out_dir / f"{fid}_montage.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"-> {out_dir}  (one image per gamma + {out.name})")


if __name__ == "__main__":
    main()
