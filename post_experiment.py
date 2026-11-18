"""POST EXPERIMENT: turn existing measurements into a count and biomass estimate.

Works only from what run_experiment already wrote (detections.csv, frames.csv);
nothing is re-measured. Restricted to config.VIDEO when it is set.

Logic (agreed):
  reference fish  whole, status ok, not cut_off  -> K_mean, L_mean, W_mean
                  (W_mean = mean of individual weights, not weight of L_mean)
  whole fish      count 1 each, also when cut_off or measurement failed
  partial fish    Fraction = K / K_mean (unit length against unit length);
                  failed partial -> 0 and documented; Fraction > 1 warned, kept
  per frame       Count = n_whole + sum(Fractions); empty frames count as 0
  population      mean(Count) * Pond_Area / Footprint, for two footprints:
                  on the floor (H) and in the fish plane (H - K_HEIGHT * L_mean)
  biomass         population * W_mean

Output: output/<experiment>/<video>/summary.xlsx
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

import config as C
from footprint import footprint_zfov
from measure_frame import LFP, lens_of

HEAD = PatternFill("solid", fgColor="1F3A4D")
WARN = PatternFill("solid", fgColor="FBE4E4")


def _f(v):
    return None if v in (None, "") else float(v)


def output_dir(exp_dir: Path, video: str | None) -> Path:
    """output/<experiment>/<video>/  (or .../all/ when no video is given)."""
    return C.OUTPUT_DIR / Path(exp_dir).name / (Path(video).stem if video else "all")


def load(exp_out: Path, video: str | None):
    det_csv, frame_csv = exp_out / "detections.csv", exp_out / "frames.csv"
    if not frame_csv.is_file():
        raise SystemExit(f"no measurements found: {frame_csv} (run the experiment first)")
    with open(frame_csv, newline="", encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))
    dets = []
    if det_csv.is_file():
        with open(det_csv, newline="", encoding="utf-8") as fh:
            dets = list(csv.DictReader(fh))
    if video:
        stem = Path(video).stem.lower()
        frames = [r for r in frames if Path(r["clip"]).stem.lower() == stem]
        dets = [r for r in dets if Path(r["clip"]).stem.lower() == stem]
        if not frames:
            raise SystemExit(f"no measured frames for {video} in {frame_csv}")
    return frames, dets


def mean(v):
    return st.mean(v) if v else None


def sd(v):
    return st.stdev(v) if len(v) > 1 else None


def analyse(frames, dets):
    # --- reference values from valid whole fish
    for d in dets:
        d["K"] = _f(d["unit_length_K"])
        d["L"] = _f(d["length_mm"])
        d["W"] = _f(d["weight_g"])
        d["cut_off"] = "cut_off" in (d["flags"] or "")
        d["ok"] = d["status"] == "ok"
        d["is_reference"] = d["cls"] == "whole" and d["ok"] and not d["cut_off"]
    ref = [d for d in dets if d["is_reference"]]
    if not ref:
        return None
    K_mean = mean([d["K"] for d in ref])
    L_mean = mean([d["L"] for d in ref])
    W_mean = mean([d["W"] for d in ref])

    # --- contribution of every detection to its frame's count
    for d in dets:
        if d["cls"] == "whole":
            d["fraction"], d["note"] = 1.0, ("" if d["is_reference"] else
                                              "counted, not in means" + (" (cut_off)" if d["cut_off"] else " (measure failed)"))
        elif d["ok"] and d["K"] is not None:
            d["fraction"] = d["K"] / K_mean
            d["note"] = "FRACTION > 1" if d["fraction"] > 1 else ""
        else:
            d["fraction"], d["note"] = 0.0, f"failed partial: {d['status']}"

    by_frame = defaultdict(list)
    for d in dets:
        by_frame[(d["clip"], d["frame"])].append(d)
    for f in frames:
        ds = by_frame.get((f["clip"], f["frame"]), [])
        f["n_whole"] = sum(d["cls"] == "whole" for d in ds)
        f["n_partial"] = sum(d["cls"] == "partial" for d in ds)
        f["sum_fractions"] = sum(d["fraction"] for d in ds if d["cls"] == "partial")
        f["count"] = f["n_whole"] + f["sum_fractions"]
        f["biomass_in_footprint_g"] = f["count"] * W_mean

    counts = [f["count"] for f in frames]
    return {
        "ref": ref, "K_mean": K_mean, "L_mean": L_mean, "W_mean": W_mean,
        "L_sd": sd([d["L"] for d in ref]), "W_sd": sd([d["W"] for d in ref]),
        "mean_count": mean(counts), "sd_count": sd(counts),
    }


def wall_factor() -> float:
    """Average share of the footprint that is tank floor (1.0 = no correction).

    WALL_SEGMENTS = [(share of patrol, floor fraction), ...]
    factor = sum(share * fraction) / sum(share)
    e.g. 4/7 * 1 + 3/7 * 2/3 = 6/7 = 0.857
    """
    if not C.WALL_CORRECTION:
        return 1.0
    total = sum(s for s, _ in C.WALL_SEGMENTS)
    return sum(s * f for s, f in C.WALL_SEGMENTS) / total


def estimate(res, lens, pond_m2):
    """Population and biomass for both footprints.

    footprint_eff = footprint (ZFOV + Snell) * wall_factor()
    population    = mean_count * pond_area / footprint_eff
    biomass       = population * W_mean
    """
    H, n = C.CAMERA_HEIGHT_MM, LFP.N_WATER
    d_fish = C.K_HEIGHT * res["L_mean"]
    wf = wall_factor()
    out = {}
    for key, h in (("floor", H), ("fish_plane", H - d_fish)):
        fp_full = footprint_zfov(lens["zfov"], lens["vres"], h, n)["area_m2"]
        fp = fp_full * wf                                   # <- wall correction
        pop = res["mean_count"] * pond_m2 / fp if pond_m2 else None
        out[key] = {"distance_mm": h, "footprint_full_m2": fp_full,
                    "wall_factor": wf, "footprint_m2": fp, "population": pop,
                    "biomass_g": pop * res["W_mean"] if pop is not None else None}
    return out


def err(est, true):
    if est is None or true is None:
        return None, None
    return est - true, (est - true) / true * 100


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _sheet(wb, title, cols, rows, widths=None, warn_col=None):
    ws = wb.create_sheet(title)
    ws.append(cols)
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), HEAD
    for r in rows:
        ws.append(r)
        if warn_col is not None and r[warn_col]:
            for c in ws[ws.max_row]:
                c.fill = WARN
    for i, w in enumerate(widths or [14] * len(cols), start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    return ws


def _round(v, nd=3):
    return round(v, nd) if isinstance(v, float) else v


def write_xlsx(path, label, frames, dets, res, est, true_count, n_frac_gt1):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    whole = [d for d in dets if d["cls"] == "whole"]
    partial = [d for d in dets if d["cls"] == "partial"]
    S = []
    S += [("Experiment / video", label, ""),
          ("Number of frames", len(frames), "sampled frames, empty ones included"),
          ("Total whole detections", len(whole), ""),
          ("  of which reference (ok, not cut_off)", len(res["ref"]), "used for K/L/W means"),
          ("  whole cut_off (counted, not in means)", sum(d["cut_off"] and d["ok"] for d in whole), ""),
          ("  whole measure failed (counted)", sum(not d["ok"] for d in whole), ""),
          ("Total partial detections", len(partial), ""),
          ("  partial failed (contribute 0)", sum(not d["ok"] for d in partial), ""),
          ("  partial Fraction > 1 (kept)", n_frac_gt1, "warning"),
          ("", "", ""),
          ("Mean unit length K_mean", res["K_mean"], "reference for partial fractions"),
          ("Mean fish length L_mean [mm]", res["L_mean"], f"sd {_round(res['L_sd'], 1)}"),
          ("Mean fish weight W_mean [g]", res["W_mean"], f"sd {_round(res['W_sd'], 1)}; mean of individual weights"),
          ("Mean estimated count per frame", res["mean_count"], f"sd {_round(res['sd_count'])}"),
          ("", "", ""),
          ("Camera height H [mm]", C.CAMERA_HEIGHT_MM, "lens to floor"),
          ("Fish plane distance H - k*L_mean [mm]", est["fish_plane"]["distance_mm"], f"k = {C.K_HEIGHT:.4f}"),
          ("Pond area [m^2]", C.POND_AREA_M2, "" if C.POND_AREA_M2 else "NOT SET in config.py")]
    for key, name in (("floor", "floor"), ("fish_plane", "fish plane")):
        e = est[key]
        S += [("", "", ""),
              (f"[{name}] Camera footprint, full [m^2]", e["footprint_full_m2"], "ZFOV + Snell"),
              (f"[{name}] Wall factor", e["wall_factor"],
               "WALL_SEGMENTS" if C.WALL_CORRECTION else "WALL_CORRECTION off"),
              (f"[{name}] Camera footprint, effective [m^2]", e["footprint_m2"], "full * wall factor"),
              (f"[{name}] Estimated total population", e["population"], ""),
              (f"[{name}] Estimated total biomass [g]", e["biomass_g"], "")]
        a, p = err(e["population"], true_count)
        S += [(f"[{name}] Population error (abs)", a, ""),
              (f"[{name}] Population error (%)", p, "")]
    S += [("", "", ""),
          ("Actual fish count", true_count, "from the experiment folder name"),
          ("Actual total biomass [g]", None, "not available yet")]

    ws = _sheet(wb, "summary", ["item", "value", "note"],
                [(k, _round(v, 4), n) for k, v, n in S], [44, 16, 50])
    for r in ws.iter_rows(min_row=2):
        if r[2].value and ("NOT SET" in str(r[2].value) or r[2].value == "warning") and r[1].value:
            r[1].fill = WARN

    _sheet(wb, "frames",
           ["clip", "frame", "time_s", "n_whole", "n_partial", "sum_fractions",
            "estimated_count", "L_mean_mm", "W_mean_g", "biomass_in_footprint_g"],
           [(f["clip"], int(f["frame"]), _f(f["time_s"]), f["n_whole"], f["n_partial"],
             _round(f["sum_fractions"]), _round(f["count"]), _round(res["L_mean"], 1),
             _round(res["W_mean"], 1), _round(f["biomass_in_footprint_g"], 1)) for f in frames])

    _sheet(wb, "detections",
           ["clip", "frame", "det_id", "cls", "conf", "status", "flags", "unit_length_K",
            "length_mm", "weight_g", "reference", "fraction", "note"],
           [(d["clip"], int(d["frame"]), int(d["det_id"]), d["cls"], _f(d["conf"]), d["status"],
             d["flags"], d["K"], d["L"], d["W"], d["is_reference"], _round(d["fraction"], 4),
             d["note"]) for d in dets],
           [14, 8, 7, 8, 8, 18, 22, 13, 10, 10, 10, 10, 34], warn_col=12)

    per_clip = defaultdict(list)
    for f in frames:
        per_clip[f["clip"]].append(f["count"])
    _sheet(wb, "per_clip", ["clip", "frames", "mean_count", "sd_count"],
           [(c, len(v), _round(mean(v)), _round(sd(v))) for c, v in per_clip.items()])

    wb.save(path)


# ---------------------------------------------------------------------------

def post_experiment(exp_dir: Path, video: str | None = None):
    exp_dir = Path(exp_dir)
    exp_out = output_dir(exp_dir, video)
    frames, dets = load(exp_out, video)
    label = f"{exp_dir.name} / {Path(video).stem if video else 'all clips'}"

    print("=" * 64)
    print(f"POST EXPERIMENT: {label}")
    print("=" * 64)
    res = analyse(frames, dets)
    if res is None:
        print("no valid whole fish -> no reference for K_mean / L_mean / W_mean;")
        print("the method cannot produce an estimate for this data.")
        return None

    clip_name = Path(video).stem if video else frames[0]["clip"]
    lens = lens_of(next(exp_dir.glob(f"{Path(clip_name).stem}.*")))
    est = estimate(res, lens, C.POND_AREA_M2)
    true_count = int(exp_dir.name) if exp_dir.name.isdigit() else None
    gt1 = [d for d in dets if d["cls"] == "partial" and d["fraction"] > 1]

    print(f"frames                : {len(frames)}")
    print(f"whole / partial dets  : {sum(d['cls'] == 'whole' for d in dets)} / "
          f"{sum(d['cls'] == 'partial' for d in dets)}   reference whole fish: {len(res['ref'])}")
    print(f"K_mean                : {res['K_mean']:.4f}")
    print(f"L_mean                : {res['L_mean']:.1f} mm")
    print(f"W_mean                : {res['W_mean']:.1f} g")
    print(f"mean count / frame    : {res['mean_count']:.3f}")
    if gt1:
        print(f"  ! {len(gt1)} partial fish with Fraction > 1 (kept): "
              + ", ".join(f"{d['clip']}#{d['frame']}/{d['det_id']}={d['fraction']:.2f}" for d in gt1[:5])
              + (" ..." if len(gt1) > 5 else ""))
    if C.WALL_CORRECTION:
        print(f"wall correction       : factor {wall_factor():.4f}  "
              + " + ".join(f"{s:.3f}*{f:.3f}" for s, f in C.WALL_SEGMENTS))
    else:
        print("wall correction       : off")
    for key, name in (("floor", "floor"), ("fish_plane", "fish plane")):
        e = est[key]
        line = (f"[{name:<10}] footprint {e['footprint_full_m2']:.4f} -> {e['footprint_m2']:.4f} m^2"
                f" at {e['distance_mm']:.1f} mm")
        if e["population"] is not None:
            a, p = err(e["population"], true_count)
            line += (f"  population {e['population']:.1f}  biomass {e['biomass_g'] / 1000:.2f} kg"
                     + (f"  error {a:+.1f} ({p:+.1f}%)" if a is not None else ""))
        print(line)
    if not C.POND_AREA_M2:
        print("  ! POND_AREA_M2 not set in config.py - no population / biomass")
    if true_count is not None:
        print(f"actual fish count     : {true_count}")

    out = exp_out / "summary.xlsx"
    write_xlsx(out, label, frames, dets, res, est, true_count, len(gt1))
    print(f"-> {out}")
    return out


def main():
    post_experiment(C.PROJECT_DIR / C.EXPERIMENT_DIR, C.VIDEO)


if __name__ == "__main__":
    main()
