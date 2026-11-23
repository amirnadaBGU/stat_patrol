"""POST EXPERIMENT: turn existing measurements into a count and biomass estimate.

Works only from what run_experiment already wrote (detections.csv, frames.csv);
nothing is re-measured. Restricted to config.VIDEO when it is set.

Logic (agreed):
  reference fish  whole, status ok, not cut_off  -> K_mean, L_mean, W_mean
                  (W_mean = mean of individual weights, not weight of L_mean)
  whole fish      count 1 each, also when cut_off or measurement failed
  partial fish    Fraction = K / K_mean (unit length against unit length);
                  failed partial -> 0 and documented; Fraction > 1 capped at 1
                  (a part is never more than one fish), documented
  per frame       Count = n_whole + sum(Fractions); empty frames count as 0
  population      mean(Count) * Pond_Area / Footprint, for two footprints:
                  on the floor (H) and in the fish plane (H - K_HEIGHT * L_mean)
  biomass         population * W_mean

Output: output/<experiment>/<video>/summary.xlsx
"""
from __future__ import annotations

import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import cv2
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

import config as C
from footprint import footprint_zfov
from measure_frame import COLORS, LFP, center_box, draw_window, lens_of, window_color, window_list

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
            d["fraction_raw"] = d["K"] / K_mean
            d["fraction"] = min(1.0, d["fraction_raw"])     # a part is never more than one fish
            d["note"] = (f"Fraction {d['fraction_raw']:.2f} > 1 -> capped at 1"
                         if d["fraction_raw"] > 1 else "")
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


def wall_sets():
    """The wall-segment options in config, as a list of segment lists.

    WALL_SEGMENTS may be one option   [(share, fraction), ...]
    or several options                [[(share, fraction), ...], [...], ...]
    With WALL_CORRECTION off there is one option, no correction.
    """
    if not C.WALL_CORRECTION:
        return [None]
    ws = C.WALL_SEGMENTS
    if ws and isinstance(ws[0][0], (int, float)):    # a single option
        return [ws]
    return list(ws)


def wall_factor(segments) -> float:
    """Average share of the footprint that is tank floor (1.0 = no correction).

    segments = [(share of patrol, floor fraction), ...]
    factor = sum(share * fraction) / sum(share)
    e.g. 4/7 * 1 + 3/7 * 2/3 = 6/7 = 0.857
    """
    if segments is None:
        return 1.0
    total = sum(s for s, _ in segments)
    return sum(s * f for s, f in segments) / total


def window_box(lens, frac: float):
    """Central window (x0, y0, x1, y1) covering ``frac`` of width and height."""
    return center_box(*lens["vres"], frac)


def _poly_area(lens, x0, y0, x1, y1, step=16):
    """Floor area (per unit height^2) seen by a pixel rectangle, POLY + Snell."""
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    border = ([(x, y0) for x in range(x0, x1, step)] + [(x1, y) for y in range(y0, y1, step)] +
              [(x, y1) for x in range(x1, x0, -step)] + [(x0, y) for y in range(y1, y0, -step)])
    pts = [LFP.ground(lens, x, y) for x, y in border]
    return 0.5 * abs(sum(a * d - c * b for (a, b), (c, d) in zip(pts, pts[1:] + pts[:1])))


def window_area_ratio(lens, box) -> float:
    """Floor area of the window / floor area of the whole frame (lens-exact)."""
    w, h = lens["vres"]
    return _poly_area(lens, *box) / _poly_area(lens, 0, 0, w - 1, h - 1)


def parse_keypoints(s):
    """'x,y;x,y;...' -> [(x, y), ...]; None when missing."""
    if not s:
        return None
    return [tuple(float(v) for v in p.split(",")) for p in s.split(";")]


def clip_segment(p, q, box):
    """Part of segment p-q inside the rectangle (Liang-Barsky); None if outside.

    The returned end points that lie on the window border are the added
    'cut' points of the midline.
    """
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for den, num in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if den == 0:
            if num < 0:
                return None
            continue
        t = num / den
        if den < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return ((p[0] + t0 * dx, p[1] + t0 * dy), (p[0] + t1 * dx, p[1] + t1 * dy))


def inside_midline(pts, box):
    """The midline pieces inside the window: list of (a, b) pixel segments."""
    return [s for s in (clip_segment(pts[i], pts[i + 1], box) for i in range(len(pts) - 1))
            if s is not None and s[0] != s[1]]


def k_inside(lens, segments) -> float:
    """Unit length of the midline inside the window, as LFP.k_of measures K:
    every point projected through POLY + Snell first, then summed."""
    return sum(math.dist(LFP.ground(lens, *a), LFP.ground(lens, *b)) for a, b in segments)


def center_windows():
    """CENTER_WINDOW as a list of fractions (see measure_frame.window_list)."""
    return window_list()


def wtag(frac: float) -> str:
    """Short name of a window: 0.5 -> 'C50%'."""
    return f"C{frac:.0%}"


def _add_note(d, text):
    d["note"] = (d["note"] + "; " if d["note"] else "") + text


def analyse_center(frames, dets, res, box, lens, tag):
    """Centre-window count with relative fish parts, for one window.

    Every detection's midline is cut at the window border; the part inside is
    measured like K (K_in).
        whole fish, midline entirely inside   -> 1 (one fish, whatever its length)
        whole fish, partly inside             -> K_in / K_own   (its own K)
        partial fish                          -> K_in / K_mean  (all reference fish)
        any part                              -> at most 1
        detection without midline (measure failed) touching the window
                                              -> 0, reported (note + summary row)
    Weight: W_mean of all reference fish of the video (the window is not used).
    Results per detection in d["centre"][tag], per frame in f["centre"][tag].
    Returns None when detections.csv has no keypoints column.
    """
    if dets and "keypoints" not in dets[0]:
        return None
    x0, y0, x1, y1 = box
    n_failed = 0
    for d in dets:
        pts = parse_keypoints(d.get("keypoints"))
        segs = inside_midline(pts, box) if pts else []
        K_in = k_inside(lens, segs) if segs else 0.0
        full = (d["cls"] == "whole" and pts is not None
                and all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in pts))
        if full:
            raw, frac = 1.0, 1.0
        else:
            denom = d["K"] if d["cls"] == "whole" and d["K"] else res["K_mean"]
            raw = K_in / denom
            frac = min(1.0, raw)                   # a part is never more than one fish
            if raw > 1:
                _add_note(d, f"{tag}: part {raw:.2f} > 1 -> capped at 1")
        failed = False
        if not pts:
            bx0, by0, bx1, by1 = (float(d[k]) for k in ("x0", "y0", "x1", "y1"))
            if bx1 > x0 and bx0 < x1 and by1 > y0 and by0 < y1:
                failed = True
                n_failed += 1
                _add_note(d, f"{tag}: FAILED detection in window, no midline -> counted 0")
        d.setdefault("centre", {})[tag] = {
            "segments": segs, "K_in": K_in, "full": full, "fraction": frac,
            "in": K_in > 0, "failed": failed,
            "denominator": "full=1" if full else ("K_own" if d["cls"] == "whole" and d["K"] else "K_mean")}
    by_frame = defaultdict(list)
    for d in dets:
        if d["centre"][tag]["in"]:
            by_frame[(d["clip"], d["frame"])].append(d)
    for f in frames:
        ds = by_frame.get((f["clip"], f["frame"]), [])
        f.setdefault("centre", {})[tag] = {
            "n_whole": sum(d["cls"] == "whole" for d in ds),
            "n_partial": sum(d["cls"] == "partial" for d in ds),
            "count": sum(d["centre"][tag]["fraction"] for d in ds)}
    counts = [f["centre"][tag]["count"] for f in frames]
    return {"mean_count": mean(counts), "sd_count": sd(counts), "n_failed": n_failed}


def options(res, lens, frames, dets):
    """All estimation methods; each becomes one column in the summary."""
    opts = []
    for i, s in enumerate(wall_sets(), start=1):
        opts.append({
            "name": f"wall option {i}" if s is not None else "no wall correction",
            "rule": "whole 1, partial K/K_mean",
            "segments": ("off" if s is None else "; ".join(f"({a:.3f}, {b:.3f})" for a, b in s)),
            "factor": wall_factor(s), "factor_name": "wall factor",
            "mean_count": res["mean_count"], "sd_count": res["sd_count"],
            "W_mean": res["W_mean"], "W_note": f"{len(res['ref'])} reference fish",
            "n_failed": ""})
    for frac in center_windows():
        tag = wtag(frac)
        box = window_box(lens, frac)
        c = analyse_center(frames, dets, res, box, lens, tag)
        if c is None:
            print(f"  ! centre {frac:.0%}: detections.csv has no keypoints "
                  "(measured before they were saved) - re-measure this video for this method")
            continue
        opts.append({
            "name": f"centre {frac:.0%}", "tag": tag,
            "rule": "whole inside = 1; whole partly = K_in/K_own; partial = K_in/K_mean; max 1",
            "segments": "none",
            "factor": window_area_ratio(lens, box), "factor_name": "window area fraction",
            "mean_count": c["mean_count"], "sd_count": c["sd_count"],
            "W_mean": res["W_mean"], "W_note": f"{len(res['ref'])} reference fish (whole video)",
            "n_failed": c["n_failed"]})
    return opts


def estimate(opt, res, lens, pond_m2):
    """Population and biomass for both footprints, for one method.

    footprint_eff = footprint (ZFOV + Snell) * factor (wall factor / window area)
    population    = mean_count * pond_area / footprint_eff
    biomass       = population * W_mean
    """
    H, n = C.CAMERA_HEIGHT_MM, LFP.N_WATER
    d_fish = C.K_HEIGHT * res["L_mean"]
    out = {}
    for key, h in (("floor", H), ("fish_plane", H - d_fish)):
        fp_full = footprint_zfov(lens["zfov"], lens["vres"], h, n)["area_m2"]
        fp = fp_full * opt["factor"]                        # <- wall / window correction
        pop = opt["mean_count"] * pond_m2 / fp if pond_m2 else None
        out[key] = {"distance_mm": h, "footprint_full_m2": fp_full,
                    "factor": opt["factor"], "footprint_m2": fp, "population": pop,
                    "biomass_g": pop * opt["W_mean"] if pop is not None else None}
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


WALL_HEAD = PatternFill("solid", fgColor="2E6B9E")
CENTRE_HEAD = PatternFill("solid", fgColor="A0309A")
USED = PatternFill("solid", fgColor="DDF2DD")       # contributes to count / weight
UNUSED = PatternFill("solid", fgColor="E6E6E6")     # contributes nothing

LEGEND = [
    ("WALL: ...", "methods 'wall option N' in the summary (whole frame, wall-corrected footprint)"),
    ("C50%: ... (C<N>%)", "method 'centre N%' in the summary: only the central N% of width and height, no wall factor"),
    ("WALL: counts as", "contribution to the frame count: whole = 1, partial = min(1, K / K_mean), failed partial = 0"),
    ("in weight", "yes = reference fish (whole, ok, not cut_off): enters K_mean, L_mean, W_mean - in ALL methods"),
    ("C<N>%: K_in", "unit length of the midline part inside the window (midline cut at the window border)"),
    ("C<N>%: counts as", "whole fish fully inside = 1; whole fish partly inside = K_in / its own K; "
                        "partial = K_in / K_mean; never more than 1"),
    ("C<N>%: rule", "which of the above was used: full=1 / K_own / K_mean"),
    ("failed in window", "detection with no midline (measurement failed) touching the window: counted 0; "
                         "see the note column (red row) and the summary row 'Failed detections touching window'"),
    ("green cell", "this detection is used by that method"),
    ("grey cell", "this detection is NOT used by that method"),
    ("red row", "note / warning (failed detection, part > 1 capped at 1, ...)"),
    ("overlays_counted/*.jpg", "dashed boxes = centre windows (one colour each); per detection: W=<wall contribution>, "
                               "C<N>=<centre contribution>, '*' = in weight; thick line = midline part inside the window"),
]


def _color_groups(ws):
    """Colour WALL / CENTRE headers and mark used (green) / unused (grey) cells."""
    for c in ws[1]:
        name = str(c.value)
        if name.startswith("WALL:"):
            c.fill = WALL_HEAD
        elif name.startswith("C") and "%:" in name:
            c.fill = CENTRE_HEAD
        else:
            continue
        if "counts as" in name or "in weight" in name:
            for (cell,) in ws.iter_rows(min_row=2, min_col=c.column, max_col=c.column):
                used = cell.value not in (0, 0.0, "", None)
                cell.fill = USED if used else UNUSED


def write_xlsx(path, label, frames, dets, res, opts, ests, true_count, n_frac_gt1):
    """``opts``/``ests``: one method (wall options, centre window), one column each.

    Rows that do not depend on the wall option fill only the first column.
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    est = ests[0]                       # for the rows shared by all options

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
          ("  partial Fraction > 1 (capped at 1)", n_frac_gt1, "warning"),
          ("", "", ""),
          ("Mean unit length K_mean", res["K_mean"], "reference for partial fractions"),
          ("Mean fish length L_mean [mm]", res["L_mean"], f"sd {_round(res['L_sd'], 1)}"),
          ("Mean fish weight W_mean [g]", res["W_mean"], f"sd {_round(res['W_sd'], 1)}; mean of individual weights"),
          ("Mean estimated count per frame", res["mean_count"], f"sd {_round(res['sd_count'])}"),
          ("", "", ""),
          ("Camera height H [mm]", C.CAMERA_HEIGHT_MM, "lens to floor"),
          ("Fish plane distance H - k*L_mean [mm]", est["fish_plane"]["distance_mm"], f"k = {C.K_HEIGHT:.4f}"),
          ("Pond area [m^2]", C.POND_AREA_M2, "" if C.POND_AREA_M2 else "NOT SET in config.py"),
          ("Actual fish count", true_count, "from the experiment folder name"),
          ("Actual total biomass [g]", None, "not available yet")]
    n_opt = len(ests)
    blank = [""] * (n_opt - 1)
    rows = [(k, _round(v, 4), *blank, n) for k, v, n in S]      # shared rows

    # method-dependent rows: one value per option
    rows.append(("", *[""] * n_opt, ""))
    for item, key, note in (
            ("Count rule", "rule", ""),
            ("Wall segments (share, floor fraction)", "segments", ""),
            ("Mean count per frame (this method)", "mean_count", ""),
            ("SD count per frame", "sd_count", ""),
            ("W_mean used [g]", "W_mean", ""),
            ("  W_mean source", "W_note", ""),
            ("Failed detections touching window (counted 0)", "n_failed", "centre only; listed in detections -> note"),
            ("Footprint factor", "factor", "wall: sum(share*fraction)/sum(share); centre: window area / frame area"),
            ("  factor type", "factor_name", "")):
        rows.append((item, *[_round(o[key], 4) for o in opts], note))
    for key, name in (("floor", "floor"), ("fish_plane", "fish plane")):
        rows.append(("", *[""] * n_opt, ""))
        per = [e[key] for e in ests]
        errs = [err(e["population"], true_count) for e in per]
        for item, vals, note in (
                (f"[{name}] Camera footprint, full [m^2]", [e["footprint_full_m2"] for e in per], "ZFOV + Snell"),
                (f"[{name}] Camera footprint, effective [m^2]", [e["footprint_m2"] for e in per], "full * factor"),
                (f"[{name}] Estimated total population", [e["population"] for e in per], ""),
                (f"[{name}] Estimated total biomass [g]", [e["biomass_g"] for e in per], ""),
                (f"[{name}] Population error (abs)", [a for a, _ in errs], ""),
                (f"[{name}] Population error (%)", [p for _, p in errs], "")):
            rows.append((item, *[_round(v, 4) for v in vals], note))

    head = ["item"] + [o["name"] for o in opts] + ["note"]
    ws = _sheet(wb, "summary", head, rows, [44] + [22] * n_opt + [44])
    for r in ws.iter_rows(min_row=2):
        note = r[-1].value
        if note and ("NOT SET" in str(note) or note == "warning") and r[1].value:
            r[1].fill = WARN
        if str(r[0].value).startswith("Failed detections"):
            for cell in r[1:-1]:
                if isinstance(cell.value, int) and cell.value > 0:
                    cell.fill = WARN

    tags = [o["tag"] for o in opts if o.get("tag")]
    fcols = ["clip", "frame", "time_s",
             "WALL: n_whole", "WALL: n_partial", "WALL: sum_fractions", "WALL: count"]
    for t in tags:
        fcols += [f"{t}: n_whole", f"{t}: n_partial", f"{t}: count"]
    fcols += ["L_mean_mm", "W_mean_g", "WALL: biomass_in_footprint_g"]
    ws = _sheet(wb, "frames", fcols,
                [(f["clip"], int(f["frame"]), _f(f["time_s"]), f["n_whole"], f["n_partial"],
                  _round(f["sum_fractions"]), _round(f["count"]),
                  *(v for t in tags for v in (f["centre"][t]["n_whole"], f["centre"][t]["n_partial"],
                                              _round(f["centre"][t]["count"]))),
                  _round(res["L_mean"], 1), _round(res["W_mean"], 1),
                  _round(f["biomass_in_footprint_g"], 1)) for f in frames],
                [14, 8, 8] + [12] * (len(fcols) - 3))
    _color_groups(ws)

    dcols = ["clip", "frame", "det_id", "cls", "conf", "status", "flags", "unit_length_K",
             "length_mm", "weight_g", "in weight", "WALL: counts as"]
    for t in tags:
        dcols += [f"{t}: K_in", f"{t}: counts as", f"{t}: rule"]
    dcols += ["note"]

    def crow(d, t):
        c = d["centre"][t]
        if c["failed"]:
            return (None, 0.0, "FAILED in window")
        return (_round(c["K_in"], 4), _round(c["fraction"], 4), c["denominator"] if c["in"] else "")

    ws = _sheet(wb, "detections", dcols,
                [(d["clip"], int(d["frame"]), int(d["det_id"]), d["cls"], _f(d["conf"]), d["status"],
                  d["flags"], d["K"], d["L"], d["W"], "yes" if d["is_reference"] else "",
                  _round(d["fraction"], 4),
                  *(v for t in tags for v in crow(d, t)),
                  d["note"]) for d in dets],
                [14, 8, 7, 8, 8, 18, 22, 13, 10, 10, 10] + [12] * (len(dcols) - 12) + [44],
                warn_col=len(dcols) - 1)
    _color_groups(ws)
    for c in ws[1]:
        if c.value == "in weight":
            for (cell,) in ws.iter_rows(min_row=2, min_col=c.column, max_col=c.column):
                cell.fill = USED if cell.value else UNUSED
    _sheet(wb, "legend", ["column / mark", "meaning"], LEGEND, [34, 100])

    per_clip = defaultdict(list)
    for f in frames:
        per_clip[f["clip"]].append(f["count"])
    _sheet(wb, "per_clip", ["clip", "frames", "mean_count", "sd_count"],
           [(c, len(v), _round(mean(v)), _round(sd(v))) for c, v in per_clip.items()])

    wb.save(path)


# ---------------------------------------------------------------------------
# Overlays: what each method counted
# ---------------------------------------------------------------------------

def annotate_overlays(exp_out: Path, frames, dets) -> int:
    """Copy overlays/*.jpg to overlays_counted/ with what entered the estimate.

    Magenta dashed box = centre window. Per detection a tag under its box:
    W=<wall contribution> C<N>=<centre contribution> per window, '*' = enters
    the weight. The midline part inside each window is drawn in its colour.
    Header: frame counts.
    """
    src = exp_out / "overlays"
    if not src.is_dir():
        return 0
    dst = exp_out / "overlays_counted"
    dst.mkdir(exist_ok=True)
    fracs = center_windows()
    tags = [wtag(fr) for fr in fracs]
    if dets and "centre" not in dets[0]:
        tags, fracs = [], []           # no keypoints -> no centre results
    by_fid = defaultdict(list)
    for d in dets:
        by_fid[d["frame_id"]].append(d)
    frame_of = {f["frame_id"]: f for f in frames}
    n = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for jpg in sorted(src.glob("*.jpg")):
        f = frame_of.get(jpg.stem)
        img = cv2.imread(str(jpg))
        if f is None or img is None:
            continue
        draw_window(img)
        for d in by_fid.get(jpg.stem, []):
            x0, y0, x1, y1 = (int(float(d[k])) for k in ("x0", "y0", "x1", "y1"))
            tag = f"W={d['fraction']:.2f}{'*' if d['is_reference'] else ''}"
            for t in tags:
                c = d["centre"][t]
                tag += f"  {t[:-1]}={'FAIL' if c['failed'] else format(c['fraction'], '.2f')}"
            # larger windows first, wider, so the smaller ones stay visible on top
            order = sorted(range(len(tags)), key=lambda i: -fracs[i])
            for rank, i in enumerate(order):
                width = 22 + 16 * (len(order) - 1 - rank)
                for a, b in d["centre"][tags[i]]["segments"]:
                    cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), window_color(i), width)
            for i in order:
                for a, b in d["centre"][tags[i]]["segments"]:
                    for q in (a, b):
                        cv2.circle(img, tuple(map(int, q)), 18, (255, 255, 255), -1)
            ty = min(img.shape[0] - 20, y1 + 75)
            cv2.putText(img, tag, (x0, ty), font, 2.2, (0, 0, 0), 14)
            cv2.putText(img, tag, (x0, ty), font, 2.2, COLORS[d["cls"]], 5)
        head = f"WALL count {f['count']:.2f}"
        for t in tags:
            head += f"   {t} count {f['centre'][t]['count']:.2f}"
        # below the 'second.frame' label that the overlay already carries
        cv2.putText(img, head, (40, 260), font, 3.5, (0, 0, 0), 18)
        cv2.putText(img, head, (40, 260), font, 3.5, (255, 255, 255), 7)
        cv2.imwrite(str(dst / jpg.name), img)
        n += 1
    return n


# ---------------------------------------------------------------------------

def post_experiment(exp_dir: Path, video: str | None = None, exp_out: Path | None = None):
    """Summarise existing measurements. ``exp_out`` overrides the folder
    (default output/<experiment>/<video>/).

    Returns {"summary": path, "res", "opts", "ests", "true_count"}, or None
    when there is no reference whole fish.
    """
    exp_dir = Path(exp_dir)
    exp_out = Path(exp_out) if exp_out else output_dir(exp_dir, video)
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
    opts = options(res, lens, frames, dets)
    ests = [estimate(o, res, lens, C.POND_AREA_M2) for o in opts]
    true_count = int(exp_dir.name) if exp_dir.name.isdigit() else None
    gt1 = [d for d in dets if d["cls"] == "partial" and d.get("fraction_raw", 0) > 1]

    print(f"frames                : {len(frames)}")
    print(f"whole / partial dets  : {sum(d['cls'] == 'whole' for d in dets)} / "
          f"{sum(d['cls'] == 'partial' for d in dets)}   reference whole fish: {len(res['ref'])}")
    print(f"K_mean                : {res['K_mean']:.4f}")
    print(f"L_mean                : {res['L_mean']:.1f} mm")
    print(f"W_mean                : {res['W_mean']:.1f} g")
    print(f"mean count / frame    : {res['mean_count']:.3f}")
    if gt1:
        print(f"  ! {len(gt1)} partial fish with Fraction > 1 (capped at 1): "
              + ", ".join(f"{d['clip']}#{d['frame']}/{d['det_id']}={d['fraction_raw']:.2f}" for d in gt1[:5])
              + (" ..." if len(gt1) > 5 else ""))
    for o, est in zip(opts, ests):
        print(f"{o['name']:<22}: {o['factor_name']} {o['factor']:.4f}   "
              f"count/frame {o['mean_count']:.3f}   W_mean {o['W_mean']:.1f} g   [{o['segments']}]")
        if o["n_failed"]:
            print(f"  ! {o['n_failed']} FAILED detections (no midline) touching the window, counted 0 "
                  "- see summary.xlsx: summary row 'Failed detections touching window', detections -> note")
        for key, name in (("floor", "floor"), ("fish_plane", "fish plane")):
            e = est[key]
            line = (f"  [{name:<10}] footprint {e['footprint_full_m2']:.4f} -> {e['footprint_m2']:.4f} m^2"
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
    write_xlsx(out, label, frames, dets, res, opts, ests, true_count, len(gt1))
    print(f"-> {out}")
    n = annotate_overlays(exp_out, frames, dets)
    if n:
        print(f"-> {exp_out / 'overlays_counted'}  ({n} images)")
    return {"summary": out, "res": res, "opts": opts, "ests": ests, "true_count": true_count}


def main():
    post_experiment(C.PROJECT_DIR / C.EXPERIMENT_DIR, C.VIDEO)


if __name__ == "__main__":
    main()
