"""Stage 3: measure N random frames from one video of an experiment.

Intended for one video at a time (config.VIDEO). Reads the TIMES table in
EXPERIMENT_DIR for that video's start/end, draws N_FRAMES random frames in
between, and runs measure_frame on each. Results are appended to two CSVs
after every frame, so an interrupted run resumes where it stopped (same
RANDOM_SEED -> same frames). Everything goes to the video's own folder:

    output/<experiment>/<video>/detections.csv   one row per detection
    output/<experiment>/<video>/frames.csv       one row per frame
    output/<experiment>/<video>/overlays/        one image (the frame with
                                                 the most detections)
    output/<experiment>/<video>/summary.xlsx     POST EXPERIMENT

With VIDEO = None all videos in TIMES run together into output/<experiment>/all/.

All settings are global variables in config.py; edit them and run.
"""
from __future__ import annotations

import csv
import datetime as dt
import random
import time
from pathlib import Path

import cv2

import config as C
from count_frame import FrameReader, count_fish
from footprint import pre_calculations
from post_experiment import output_dir, post_experiment
from measure_frame import CSV_COLUMNS, draw, frame_id, measure_frame

DET_COLUMNS = ["clip", "frame", "time_s"] + CSV_COLUMNS
FRAME_COLUMNS = ["clip", "frame", "time_s", "frame_id", "n_whole", "n_partial",
                 "n_failed", "seconds"]


# ---------------------------------------------------------------------------
# Clip table
# ---------------------------------------------------------------------------

def parse_time(v) -> float:
    """Seconds from a number, 'mm:ss', 'hh:mm:ss', or an Excel time cell."""
    if v is None or v == "":
        raise ValueError("empty time")
    if isinstance(v, dt.time):
        return v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6
    if isinstance(v, dt.timedelta):
        return v.total_seconds()
    if isinstance(v, dt.datetime):
        return parse_time(v.time())
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(p) for p in str(v).strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def read_table_rows(path: Path):
    """[(row_number, video, start, end)] by position: columns A, B, C.

    A first row whose start cell is not a time is taken as a header and skipped.
    """
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    else:
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True).active
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    out = []
    for i, r in enumerate(rows, start=1):
        r = (list(r) + [None, None, None])[:3]
        if all(c in (None, "") for c in r):
            continue
        if not out and i == 1:
            try:
                parse_time(r[1])
            except ValueError:
                continue          # header row
        out.append((i, *r))
    return out


def read_clip_table(exp_dir: Path, table: Path | None = None):
    """[(video_path, start_s, end_s)] in table order."""
    if table is None:
        table = next((exp_dir / n for n in C.CLIPS_TABLE_NAMES if (exp_dir / n).is_file()), None)
        if table is None:
            raise SystemExit(f"no clip table in {exp_dir} (looked for {', '.join(C.CLIPS_TABLE_NAMES)})")
    print(f"clip table: {table}")
    clips = []
    for i, name, start, end in read_table_rows(table):
        name = str(name or "").strip()
        if not name:
            continue
        path = exp_dir / name
        if not path.suffix:
            path = path.with_suffix(".MP4")
        if not path.is_file():
            print(f"  ! row {i}: {path.name} not found in {exp_dir} - skipped")
            continue
        try:
            start, end = parse_time(start), parse_time(end)
        except ValueError as exc:
            print(f"  ! row {i}: {name}: bad start/end ({exc}) - skipped")
            continue
        if end <= start:
            print(f"  ! row {i}: {name}: end {end} <= start {start} - skipped")
            continue
        clips.append((path, start, end))
    return clips


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def sample_frames(path: Path, start_s: float, end_s: float, n: int, seed: int):
    """n distinct random frame indices in [start, end], sorted. Seeded per clip."""
    fps, total = video_info(path)
    first = max(0, int(round(start_s * fps)))
    last = min(total - 1, int(round(end_s * fps)))
    if last < first:
        print(f"  ! {path.name}: start/end outside the clip ({total} frames) - skipped")
        return fps, []
    pool = range(first, last + 1)
    rng = random.Random(f"{seed}:{path.name}")
    picks = sorted(rng.sample(pool, min(n, len(pool))))
    if len(pool) < n:
        print(f"  ! {path.name}: only {len(pool)} frames between start and end")
    return fps, picks


# ---------------------------------------------------------------------------
# Output (appended per frame, so a run can resume)
# ---------------------------------------------------------------------------

def append_rows(path: Path, columns, rows):
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def done_frames(path: Path):
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["clip"], int(r["frame"])) for r in csv.DictReader(fh)}


def overlay_mode():
    """SAVE_OVERLAYS -> None (save none), "ALL", or a positive int."""
    v = C.SAVE_OVERLAYS
    if v is None or v == 0:
        return None
    if isinstance(v, str):
        if v.strip().upper() == "ALL":
            return "ALL"
        if v.strip().upper() == "NONE":
            return None
        raise SystemExit(f"SAVE_OVERLAYS must be None, 'ALL' or an int, not {v!r}")
    if isinstance(v, int) and v > 0:
        return v
    raise SystemExit(f"SAVE_OVERLAYS must be None, 'ALL' or an int > 0, not {v!r}")


def save_overlay(out_dir, video, idx, proc, dets):
    cv2.imwrite(str(out_dir / "overlays" / f"{frame_id(video, idx)}.jpg"), draw(proc, dets))


def _r(v, nd):
    return None if v is None else round(v, nd)


def detection_rows(clip, idx, t, fid, dets):
    rows = []
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        rows.append({
            "clip": clip, "frame": idx, "time_s": round(t, 3), "frame_id": fid,
            "det_id": d["det_id"], "cls": d["cls"], "conf": round(d["conf"], 4),
            "x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1),
            "status": d["status"], "flags": d["flags"],
            "length_px": _r(d["length_px"], 1), "unit_length_K": _r(d["unit_length_K"], 6),
            "rho": _r(d["rho"], 4), "length_mm": _r(d["length_mm"], 1),
            "weight_g": _r(d["weight_g"], 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment(exp_dir: Path, table: Path | None = None, video: str | None = None):
    """Run every clip in the TIMES table, or only ``video`` if given."""
    exp_dir = Path(exp_dir)
    out_dir = output_dir(exp_dir, video)          # output/<experiment>/<video>/
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    det_csv, frame_csv = out_dir / "detections.csv", out_dir / "frames.csv"

    clips = read_clip_table(exp_dir, table)
    if video:
        stem = Path(video).stem.lower()
        clips = [c for c in clips if c[0].stem.lower() == stem]
        if not clips:
            raise SystemExit(f"{video} is not listed in the TIMES table of {exp_dir}")

    pre_calculations([c[0] for c in clips])
    if C.PRE_CALCULATIONS:
        print("PRE_CALCULATIONS is on - stopping here.")
        return
    if C.POST_EXPERIMENT:
        print("POST_EXPERIMENT is on - summarising existing measurements only.")
        post_experiment(exp_dir, video)
        return

    done = done_frames(frame_csv)
    print(f"experiment {exp_dir.name}: {len(clips)} clips, {C.N_FRAMES} frames each"
          f"{f', {len(done)} frames already done' if done else ''}")

    plan = []
    for video, start, end in clips:
        fps, frames = sample_frames(video, start, end, C.N_FRAMES, C.RANDOM_SEED)
        plan.append((video, start, end, fps, frames,
                     [f for f in frames if (video.name, f) not in done]))
    n_total = sum(len(p[5]) for p in plan)

    t_run, n_run = time.time(), 0
    for ci, (video, start, end, fps, frames, todo) in enumerate(plan, start=1):
        print(f"[clip {ci}/{len(clips)}] {video.name}  {start:.1f}-{end:.1f}s  "
              f"{len(frames)} frames, {len(todo)} to do")
        keep_n = overlay_mode()
        best = []     # [(n_dets, idx, proc, dets)] - the top keep_n frames of this clip
        reader = FrameReader(video)       # todo is sorted -> sequential reads
        for fi, idx in enumerate(todo, start=1):
            t0 = time.time()
            proc, dets = measure_frame(video, idx, frame=reader.read(idx))
            secs = time.time() - t0
            fid, t = frame_id(video, idx), idx / fps
            c = count_fish(dets)
            n_failed = sum(d["status"] != "ok" for d in dets)
            append_rows(det_csv, DET_COLUMNS, detection_rows(video.name, idx, t, fid, dets))
            append_rows(frame_csv, FRAME_COLUMNS, [{
                "clip": video.name, "frame": idx, "time_s": round(t, 3), "frame_id": fid,
                "n_whole": c["whole"], "n_partial": c["partial"],
                "n_failed": n_failed, "seconds": round(secs, 2)}])
            n_run += 1
            el = time.time() - t_run
            eta = el / n_run * (n_total - n_run)
            print(f"    [{fi}/{len(todo)} | total {n_run}/{n_total}] frame {idx:>5} ({t:5.1f}s)"
                  f"  whole={c['whole']:<2} partial={c['partial']:<2}"
                  f"{f' failed={n_failed}' if n_failed else ''}"
                  f"  {secs:.1f}s  ETA {eta / 60:.1f} min")
            if keep_n == "ALL":
                save_overlay(out_dir, video, idx, proc, dets)
            elif keep_n:
                best.append((len(dets), idx, proc, dets))
                best.sort(key=lambda b: -b[0])
                del best[keep_n:]                 # keeps memory to keep_n images
        reader.close()
        for _, idx, proc, dets in best:
            save_overlay(out_dir, video, idx, proc, dets)

    if n_run:
        el = time.time() - t_run
        print(f"done: {n_run} frames in {el / 60:.1f} min ({el / n_run:.1f} s/frame)")
    print(f"-> {det_csv}\n-> {frame_csv}")
    post_experiment(exp_dir, video)


def main():
    run_experiment(C.PROJECT_DIR / C.EXPERIMENT_DIR, video=C.VIDEO)


if __name__ == "__main__":
    main()
