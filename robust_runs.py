"""Robust runs: every video several times with different random seeds.

For each experiment folder in ROBUST_DIRS and each video in its TIMES table:
  1. ROBUST_N_RUNS random seeds are drawn once and kept in seeds.txt
     (re-running reads them back, so an interrupted batch resumes and every
     run can be reproduced; raising ROBUST_N_RUNS only draws the extra seeds)
  2. the full pipeline (measure + post experiment) runs once per seed, into
     output/robust/<experiment>/<video>/seed_<seed>/
  3. robust_<video>.xlsx: mean / sd over the seeds, per method and plane
Per experiment (= actual fish count):
  4. robust_<experiment>.xlsx: mean of the video means, only for videos with
     avoidance degree (TIMES column D) <= ROBUST_AVOID_THRESHOLD; videos
     without a value in D are left out with a warning

Methods: every wall option and centre window configured in config.py, each
for the floor and the fish-plane footprint.

Night enhancement: on for experiments with NIGHT_FROM_FISH fish or more,
off below (config.NIGHT_ENHANCE is ignored here).
"""
from __future__ import annotations

import random
import statistics as st
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

import config as C
import count_frame
from run_experiment import read_clip_table, read_table_rows, run_experiment

NIGHT_FROM_FISH = 40            # night videos: NIGHT_ENHANCE on from this fish count
ROBUST_DIR = C.OUTPUT_DIR / "robust"
PLANES = (("floor", "floor"), ("fish_plane", "fish plane"))
HEAD = PatternFill("solid", fgColor="1F3A4D")
WARN = PatternFill("solid", fgColor="FBE4E4")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def avoidance(exp_dir: Path):
    """{video stem (lower): avoidance degree or None} from TIMES column D."""
    table = next((exp_dir / n for n in C.CLIPS_TABLE_NAMES if (exp_dir / n).is_file()), None)
    out = {}
    if table is None:
        return out
    for _, name, _, _, d in read_table_rows(table, ncols=4):
        name = str(name or "").strip()
        if not name:
            continue
        try:
            out[Path(name).stem.lower()] = None if d in (None, "") else float(d)
        except (TypeError, ValueError):
            out[Path(name).stem.lower()] = None
    return out


def seeds_for(video_out: Path, n: int):
    """The video's seeds: read from seeds.txt, extended with new random ones."""
    path = video_out / "seeds.txt"
    seeds = []
    if path.is_file():
        seeds = [int(s) for s in path.read_text(encoding="utf-8").split() if s.strip()]
    rng = random.SystemRandom()
    while len(seeds) < n:
        s = rng.randrange(1, 1_000_000)
        if s not in seeds:
            seeds.append(s)
    video_out.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, seeds)) + "\n", encoding="utf-8")
    return seeds[:n]


def fish_count(exp_dir: Path):
    name = exp_dir.name.split()[0].split("-")[0]
    return int(name) if name.isdigit() else None


# ---------------------------------------------------------------------------
# One run -> flat values
# ---------------------------------------------------------------------------

def run_values(result):
    """{(method, plane): {count, population, biomass_g, error_pct}} of one run."""
    out = {}
    if result is None:
        return out
    true = result["true_count"]
    for o, e in zip(result["opts"], result["ests"]):
        for key, plane in PLANES:
            p = e[key]
            pop = p["population"]
            out[(o["name"], plane)] = {
                "count_per_frame": o["mean_count"],
                "population": pop,
                "biomass_g": p["biomass_g"],
                "error_pct": ((pop - true) / true * 100) if (pop is not None and true) else None,
            }
    return out


def _mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def _sd(v):
    v = [x for x in v if x is not None]
    return st.stdev(v) if len(v) > 1 else None


def _r(v, nd=3):
    return round(v, nd) if isinstance(v, float) else v


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _sheet(wb, title, cols, rows, width=16, warn=None):
    ws = wb.create_sheet(title)
    ws.append(cols)
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), HEAD
    for r in rows:
        ws.append([_r(v) for v in r])
        if warn and warn(r):
            for c in ws[ws.max_row]:
                c.fill = WARN
    for i in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.column_dimensions["A"].width = 22
    ws.freeze_panes = "A2"


FIELDS = ("count_per_frame", "population", "biomass_g", "error_pct")


def write_video_xlsx(path, label, true, seeds, runs):
    """runs: {seed: run_values} (empty dict for a run without estimate)."""
    keys = []
    for v in runs.values():
        keys += [k for k in v if k not in keys]
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    rows = []
    for m, plane in keys:
        vals = {f: [runs[s][(m, plane)][f] for s in seeds if (m, plane) in runs[s]] for f in FIELDS}
        rows.append((m, plane, len(vals["population"]),
                     *(x for f in FIELDS for x in (_mean(vals[f]), _sd(vals[f])))))
    _sheet(wb, "means", ["method", "plane", "n runs"] +
           [f"{f} {s}" for f in FIELDS for s in ("mean", "sd")], rows)
    rows = []
    for s in seeds:
        if not runs[s]:
            rows.append((s, "", "", *[""] * len(FIELDS), "NO ESTIMATE (no reference whole fish)"))
        for (m, plane), v in runs[s].items():
            rows.append((s, m, plane, *(v[f] for f in FIELDS), ""))
    _sheet(wb, "runs", ["seed", "method", "plane", *FIELDS, "note"], rows,
           warn=lambda r: bool(r[-1]))
    _sheet(wb, "info", ["item", "value"],
           [("video", label), ("actual fish count", true), ("seeds", ", ".join(map(str, seeds))),
            ("frames per run", C.N_FRAMES), ("preprocessing", count_frame.preprocess_label())], 40)
    wb.save(path)


def write_experiment_xlsx(path, exp_name, true, videos):
    """videos: [{video, avoidance, included, reason, means: {(m, plane): {field: mean}}}]"""
    inc = [v for v in videos if v["included"]]
    keys = []
    for v in inc:
        keys += [k for k in v["means"] if k not in keys]
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    rows = []
    for m, plane in keys:
        per = {f: [v["means"][(m, plane)][f] for v in inc if (m, plane) in v["means"]] for f in FIELDS}
        rows.append((m, plane, len(per["population"]),
                     *(x for f in FIELDS for x in (_mean(per[f]), _sd(per[f])))))
    _sheet(wb, "means", ["method", "plane", "n videos"] +
           [f"{f} {s}" for f in FIELDS for s in ("mean", "sd between videos")], rows)
    rows = [(v["video"], v["avoidance"], "yes" if v["included"] else "no", v["reason"],
             *(v["means"].get(k, {}).get("population") for k in keys)) for v in videos]
    _sheet(wb, "videos", ["video", "avoidance (D)", "included", "reason"] +
           [f"population {m} / {p}" for m, p in keys], rows,
           warn=lambda r: r[2] == "no")
    _sheet(wb, "info", ["item", "value"],
           [("experiment", exp_name), ("actual fish count", true),
            ("avoidance threshold (<=)", C.ROBUST_AVOID_THRESHOLD),
            ("runs per video", C.ROBUST_N_RUNS), ("frames per run", C.N_FRAMES),
            ("preprocessing", count_frame.preprocess_label())], 40)
    wb.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def robust_experiment(exp_dir: Path):
    exp_dir = Path(exp_dir)
    true = fish_count(exp_dir)
    C.NIGHT_ENHANCE = bool(true is not None and true >= NIGHT_FROM_FISH)
    exp_out = ROBUST_DIR / exp_dir.name
    print("#" * 64)
    print(f"ROBUST: {exp_dir.name}  ({true} fish)  preprocessing: {count_frame.preprocess_label()}")
    print("#" * 64)

    clips = read_clip_table(exp_dir)
    avoid = avoidance(exp_dir)
    videos = []
    for path, _, _ in clips:
        stem = path.stem
        video_out = exp_out / stem
        seeds = seeds_for(video_out, C.ROBUST_N_RUNS)
        runs = {}
        for i, s in enumerate(seeds, start=1):
            print(f"\n=== {exp_dir.name} / {stem}  run {i}/{len(seeds)}  seed {s} ===")
            result = run_experiment(exp_dir, video=stem, out_dir=video_out / f"seed_{s}",
                                    seed=s, standalone=False)
            runs[s] = run_values(result)
            if result is None:
                print(f"  ! {stem} seed {s}: no estimate (no reference whole fish)")
        write_video_xlsx(video_out / f"robust_{stem}.xlsx", f"{exp_dir.name} / {stem}", true, seeds, runs)
        print(f"-> {video_out / f'robust_{stem}.xlsx'}")

        means = {}
        for s in seeds:
            for k in runs[s]:
                means.setdefault(k, None)
        means = {k: {f: _mean([runs[s][k][f] for s in seeds if k in runs[s]]) for f in FIELDS}
                 for k in means}
        a = avoid.get(stem.lower())
        if a is None:
            included, reason = False, "no avoidance degree in column D"
            print(f"  ! WARNING: {stem} has no avoidance degree in TIMES column D - not in the average")
        elif a > C.ROBUST_AVOID_THRESHOLD:
            included, reason = False, f"avoidance {a:g} > {C.ROBUST_AVOID_THRESHOLD}"
        elif not means:
            included, reason = False, "no estimate in any run"
        else:
            included, reason = True, ""
        videos.append({"video": stem, "avoidance": a, "included": included,
                       "reason": reason, "means": means})

    out = exp_out / f"robust_{exp_dir.name}.xlsx"
    write_experiment_xlsx(out, exp_dir.name, true, videos)
    n_in = sum(v["included"] for v in videos)
    print(f"\n{exp_dir.name}: {n_in}/{len(videos)} videos in the average -> {out}")


def main():
    dirs = C.ROBUST_DIRS if isinstance(C.ROBUST_DIRS, (list, tuple)) else [C.ROBUST_DIRS]
    C.SAVE_OVERLAYS = C.ROBUST_SAVE_OVERLAYS
    for d in dirs:
        robust_experiment(C.PROJECT_DIR / d)


if __name__ == "__main__":
    main()
