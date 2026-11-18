"""Camera footprint on the tank floor, from the clip's own metadata.

Main method (ZFOV + Snell):
    ZFOV        diagonal field of view in air, read from the GoPro GPMF block
    theta_air   = ZFOV / 2
    theta_water = asin( sin(theta_air) / n_water )        Snell at the flat port
    diagonal    = 2 * H * tan(theta_water)                on the floor
    width, height = diagonal split by the image aspect ratio (W:H pixels)
    area        = width * height

Cross-check (POLY): every pixel on the frame border is sent through the
clip's POLY lens model and Snell (loo_fit_poly.ground), landing on the floor;
the area of that outline is the footprint without assuming a pinhole lens.
"""
from __future__ import annotations

import math
from pathlib import Path

import config as C
from measure_frame import LFP, lens_of


def footprint_zfov(zfov_deg: float, vres, height_mm: float, n_water: float):
    w_px, h_px = vres
    th_air = math.radians(zfov_deg / 2)
    th_water = math.asin(math.sin(th_air) / n_water)
    diag = 2 * height_mm * math.tan(th_water)
    d_px = math.hypot(w_px, h_px)
    width, height = diag * w_px / d_px, diag * h_px / d_px
    return {
        "half_angle_air_deg": math.degrees(th_air),
        "half_angle_water_deg": math.degrees(th_water),
        "diag_mm": diag, "width_mm": width, "height_mm": height,
        "area_m2": width * height / 1e6,
    }


def footprint_poly(lens, height_mm: float, step: int = 16):
    """Project the frame border through POLY + Snell; shoelace area on the floor."""
    w, h = lens["vres"]
    border = ([(x, 0) for x in range(0, w, step)] +
              [(w - 1, y) for y in range(0, h, step)] +
              [(x, h - 1) for x in range(w - 1, 0, -step)] +
              [(0, y) for y in range(h - 1, 0, -step)])
    pts = [LFP.ground(lens, x, y) for x, y in border]
    area = 0.5 * abs(sum(x0 * y1 - x1 * y0
                         for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])))
    cy, cx = (h - 1) / 2, (w - 1) / 2
    width = LFP.ground(lens, w - 1, cy)[0] - LFP.ground(lens, 0, cy)[0]
    height = LFP.ground(lens, cx, h - 1)[1] - LFP.ground(lens, cx, 0)[1]
    return {"width_mm": width * height_mm, "height_mm": height * height_mm,
            "area_m2": area * height_mm ** 2 / 1e6}


def pre_calculations(videos):
    """Print the footprint for the clips about to be processed. Returns the ZFOV result."""
    H, n = C.CAMERA_HEIGHT_MM, LFP.N_WATER
    lenses = {Path(v).name: lens_of(Path(v)) for v in videos}
    zfovs = {round(l["zfov"], 4) for l in lenses.values()}
    first_name, lens = next(iter(lenses.items()))

    print("=" * 64)
    print("PRE-CALCULATIONS: camera footprint on the floor")
    print("=" * 64)
    print(f"clips             : {len(lenses)}  (metadata read from each)")
    if len(zfovs) > 1:
        print(f"  ! clips differ in ZFOV: {sorted(zfovs)} - using {first_name}")
    print(f"ZFOV (metadata)   : {lens['zfov']:.4f} deg   diagonal, in air  [{first_name}]")
    print(f"resolution        : {lens['vres'][0]} x {lens['vres'][1]} px")
    print(f"camera height H   : {H:.1f} mm  (lens to floor)")
    print(f"n water           : {n}")

    z = footprint_zfov(lens["zfov"], lens["vres"], H, n)
    print("-- ZFOV + Snell " + "-" * 48)
    print(f"half angle in air : {z['half_angle_air_deg']:.3f} deg")
    print(f"half angle water  : {z['half_angle_water_deg']:.3f} deg"
          f"   = asin(sin({z['half_angle_air_deg']:.3f}) / {n})")
    print(f"diagonal on floor : {z['diag_mm']:.1f} mm   = 2 * {H:.1f} * tan({z['half_angle_water_deg']:.3f})")
    print(f"width x height    : {z['width_mm']:.1f} x {z['height_mm']:.1f} mm")
    print(f"footprint area    : {z['area_m2']:.4f} m^2")

    p = footprint_poly(lens, H)
    print("-- cross-check: POLY lens + Snell, frame border projected " + "-" * 5)
    print(f"width x height    : {p['width_mm']:.1f} x {p['height_mm']:.1f} mm  (through centre)")
    print(f"footprint area    : {p['area_m2']:.4f} m^2"
          f"   ({(p['area_m2'] / z['area_m2'] - 1) * 100:+.1f}% vs ZFOV)")

    print("-" * 64)
    if C.POND_AREA_M2:
        print(f"pond area         : {C.POND_AREA_M2:.3f} m^2")
        print(f"pond / footprint  : {C.POND_AREA_M2 / z['area_m2']:.2f}   (scale-up factor)")
    else:
        print("pond area         : not set (POND_AREA_M2 in config.py)")
    print("=" * 64)
    return z
