"""
Plan a safe rover path between two points on a real height map
(e.g. a Mars / Moon DEM), using Laplace's equation, and draw the
result as a topographic contour map.

Supported input formats
-----------------------
  .img / .imq   PDS3 binary (HiRISE, MOLA, LROC …) — pixel size read automatically
  .tif / .png   GeoTIFF or image — pass --mpp
  .npy          NumPy array — pass --mpp
  .csv / .txt   Comma- / space-delimited — pass --mpp

Examples
--------
  uv run python plan_on_heightmap.py
  uv run python plan_on_heightmap.py data/kasei.IMG --start 250 500 --goal 2400 5800 --max-slope 25 --rover-radius 20 --out outputs/kasei
  uv run python plan_on_heightmap.py data/mars.tif --mpp 463 --start 120 80 --goal 700 520 --palette mars
  uv run python plan_on_heightmap.py dem.png --height-range -4000 2500 --mpp 200 --pick
"""

import argparse
import os
import sys

import numpy as np

from rover_path_finder import terrain as trn
from rover_path_finder import planners
from rover_path_finder import visualization as viz

PALETTES = {"blue": "#3fa9f5", "green": "#2fa66b", "mars": "#d9803f", "paper": "#f3ead7"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "heightmap", nargs="?",
        help="height map file (.img/.tif/.png/.npy/.csv). Omit for synthetic demo.",
    )
    ap.add_argument("--start", nargs=2, type=int, metavar=("X", "Y"),
                    help="start pixel in the ORIGINAL image (x right, y down)")
    ap.add_argument("--goal", nargs=2, type=int, metavar=("X", "Y"),
                    help="goal pixel in the ORIGINAL image")
    ap.add_argument("--pick", action="store_true",
                    help="click start and goal interactively (needs a desktop window)")
    ap.add_argument("--ticks", action="store_true",
                    help="label the contour map axes in original pixels (to read off --start/--goal)")
    ap.add_argument("--mpp", type=float, default=None,
                    help="metres per pixel of the ORIGINAL map (read from .img labels automatically)")
    ap.add_argument("--height-range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="metres for min/max pixel value (needed for plain images)")
    ap.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                    help="use only this sub-window of the original map")
    ap.add_argument("--max-size", type=int, default=150,
                    help="downsample so the longest side ≤ this many cells (default 150)")
    ap.add_argument("--art-size", type=int, default=700,
                    help="max cells on longest side for the contour-art pass (default 700)")
    ap.add_argument("--max-slope", type=float, default=20.0,
                    help="steepest slope the rover can drive, degrees (default 20)")
    ap.add_argument("--rover-radius", type=float, default=100.0,
                    help="safety margin around hazards, metres (default 100)")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Gaussian blur radius in pixels before planning (default 1; 0 = off)")
    ap.add_argument("--levels", type=int, default=40,
                    help="number of contour levels (default 40)")
    ap.add_argument("--palette", default="blue",
                    help="blue | green | mars | paper | any matplotlib colour")
    ap.add_argument("--out", default="outputs",
                    help="output directory (default: outputs/)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    bg = PALETTES.get(a.palette, a.palette)

    # ---- load terrain ----
    if a.heightmap:
        raw, meta = trn.load_heightmap(a.heightmap, a.height_range)
        src = a.heightmap
    else:
        raw = trn.synthetic_mars()
        meta = {"mpp": 50.0, "missing": None}
        src = "synthetic Mars-like terrain (demo)"
        a.crop = None

    if a.crop:
        x0, y0, x1, y1 = a.crop
        raw = raw[y0:y1, x0:x1]
    else:
        x0 = y0 = 0

    mpp = a.mpp or meta.get("mpp")
    if not mpp:
        mpp = 50.0
        print("warning: pixel size unknown, assuming 50 m/pixel. Pass --mpp for correct distances.")

    missing = meta.get("missing")

    # ---- two-pass downsampling: art resolution + planning resolution ----
    fa = int(np.ceil(max(raw.shape) / a.art_size))
    art, frac_art = trn.block_mean(raw, fa, missing)
    nod_art = frac_art < 0.5
    art[nod_art] = np.nan

    if nod_art.all():
        sys.exit("the selected area contains no elevation data")

    fill = float(np.nanmean(art))

    # Edge-aware blur: no-data pixels must not bleed into real data
    if a.smooth > 0 and nod_art.any():
        valid = (~nod_art).astype(float)
        num = trn.smooth(np.where(nod_art, 0.0, art), a.smooth)
        den = trn.smooth(valid, a.smooth)
        art_f = np.where(nod_art, fill, num / np.maximum(den, 1e-9))
    else:
        art_f = trn.smooth(np.where(nod_art, fill, art), a.smooth)

    art_plot = np.ma.masked_where(nod_art, art_f)

    fp = int(np.ceil(max(art_f.shape) / a.max_size))
    h, _ = trn.block_mean(art_f, fp)
    nod_plan = trn.block_mean(nod_art.astype(float), fp)[0] > 0.3
    f_total = fa * fp
    cell = mpp * f_total
    shape = h.shape

    print(
        f"map: {src} | {raw.shape[1]}x{raw.shape[0]} px of {mpp:g} m"
        f" -> planning grid {h.shape[1]}x{h.shape[0]} cells of {cell:.0f} m"
        f" | relief {np.nanmin(art):.0f}..{np.nanmax(art):.0f} m"
        + (f" | {100 * nod_art.mean():.0f}% no-data" if nod_art.any() else "")
    )

    # ---- choose start / goal ----
    ticks_arg = (x0, y0, f_total) if a.ticks else None

    if a.pick:
        start, goal = viz.pick_points(art_plot, fp, shape, a.levels, bg)
    else:
        sp = a.start or (x0 + int(0.12 * raw.shape[1]), y0 + int(0.90 * raw.shape[0]))
        gp = a.goal  or (x0 + int(0.85 * raw.shape[1]), y0 + int(0.10 * raw.shape[0]))
        def conv(p):
            return (
                min(max((p[0] - x0) // f_total, 1), shape[1] - 2),
                min(max((p[1] - y0) // f_total, 1), shape[0] - 2),
            )
        start, goal = conv(sp), conv(gp)

    # ---- contour art map ----
    viz.fig_art(art_plot, fp, shape, a.levels, bg,
                f"{a.out}/map_contours.png", ticks=ticks_arg)

    # ---- plan ----
    info = planners.plan(h, cell, start, goal, a.max_slope, a.rover_radius)

    for note in info["notes"]:
        print("note:", note)
    print(
        f"hazard cells: {int(info['raw'].sum())} real, "
        f"{int(info['hazard'].sum())} with margin, of {h.size}"
    )

    if info["ok"]:
        print(
            f"route found: {len(info['path'])} cells | "
            f"length {info['length_m'] / 1000:.2f} km "
            f"(shortest safe {info['shortest_m'] / 1000:.2f} km, "
            f"+{100 * (info['length_m'] / info['shortest_m'] - 1):.0f}%) | "
            f"max slope on route {info['max_path_slope']:.1f}° (limit {a.max_slope}) | "
            f"climb {info['climb_m']:.0f} m, descent {info['descent_m']:.0f} m | "
            f"solver {info['iters']} iters, {info['solve_s']:.1f}s"
        )
        viz.fig_profile(info, a.max_slope, f"{a.out}/route_profile.png")
        with open(f"{a.out}/route.csv", "w") as fh:
            fh.write("x_original_px,y_original_px,elevation_m,distance_m\n")
            for (cx, cy), z, d in zip(info["path"], info["profile_z"], info["dist_m"]):
                fh.write(
                    f"{x0 + cx * f_total + f_total // 2},"
                    f"{y0 + cy * f_total + f_total // 2},"
                    f"{z:.1f},{d:.0f}\n"
                )
    else:
        print("no route")

    viz.fig_route(art_plot, fp, shape, info, a.levels, bg, cell,
                  f"{a.out}/map_route.png", ticks=ticks_arg)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
