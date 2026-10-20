"""
Plan a safe rover path between two points you choose on a real height map
(e.g. a Mars DEM), using Laplace's equation, and draw the result as a
topographic contour map.

Examples
--------
    python plan_on_heightmap.py                                   # built-in synthetic Mars-like demo
    python plan_on_heightmap.py mars.tif --mpp 463 --start 120 80 --goal 700 520
    python plan_on_heightmap.py dem.png --height-range -4000 2500 --mpp 200 --pick
"""

import argparse
import os
import sys

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
        help="height map file (.tif / .png / .npy / .csv). Omit for the synthetic demo.",
    )
    ap.add_argument("--start", nargs=2, type=int, metavar=("X", "Y"),
                    help="start pixel in the ORIGINAL image (x right, y down)")
    ap.add_argument("--goal", nargs=2, type=int, metavar=("X", "Y"),
                    help="goal pixel in the ORIGINAL image")
    ap.add_argument("--pick", action="store_true",
                    help="click start and goal on the map (needs a desktop window)")
    ap.add_argument("--mpp", type=float, default=50.0,
                    help="metres per pixel of the ORIGINAL map (default 50)")
    ap.add_argument("--height-range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="metres for min/max pixel value (needed for plain images)")
    ap.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                    help="use only this window of the original map")
    ap.add_argument("--max-size", type=int, default=150,
                    help="downsample so the longest side ≤ this many cells (default 150)")
    ap.add_argument("--max-slope", type=float, default=20.0,
                    help="steepest slope the rover can drive, degrees (default 20)")
    ap.add_argument("--rover-radius", type=float, default=100.0,
                    help="safety margin around hazards, metres (default 100)")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="blur the map by this many pixels before planning (default 1; 0 = off)")
    ap.add_argument("--levels", type=int, default=40,
                    help="number of contour levels (default 40)")
    ap.add_argument("--palette", default="blue",
                    help="blue | green | mars | paper | any matplotlib colour")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    bg = PALETTES.get(a.palette, a.palette)

    # ---- load / generate terrain ----
    if a.heightmap:
        h0 = trn.load_heightmap(a.heightmap, a.height_range)
        src = a.heightmap
    else:
        h0 = trn.synthetic_mars()
        src = "synthetic Mars-like terrain (demo)"

    if a.crop:
        x0, y0, x1, y1 = a.crop
        h0 = h0[y0:y1, x0:x1]
    else:
        x0 = y0 = 0

    h0 = trn.smooth(h0, a.smooth)
    h, f = trn.downsample(h0, a.max_size)
    cell = a.mpp * f
    shape = h.shape
    print(
        f"map: {src} | {h0.shape[1]}×{h0.shape[0]} px → {h.shape[1]}×{h.shape[0]} cells "
        f"of {cell:.0f} m | relief {h.min():.0f}..{h.max():.0f} m"
    )

    # ---- choose start / goal ----
    if a.pick:
        start, goal = viz.pick_points(h0, f, shape, a.levels, bg)
    else:
        sp = a.start or (int(0.12 * h0.shape[1]) + x0, int(0.90 * h0.shape[0]) + y0)
        gp = a.goal or (int(0.85 * h0.shape[1]) + x0, int(0.10 * h0.shape[0]) + y0)
        def conv(p: tuple[int, int]) -> tuple[int, int]:
            return (
                min(max((p[0] - x0) // f, 1), h.shape[1] - 2),
                min(max((p[1] - y0) // f, 1), h.shape[0] - 2),
            )
        start, goal = conv(sp), conv(gp)

    # ---- plan ----
    viz.fig_art(h0, f, shape, a.levels, bg, f"{a.out}/map_contours.png")
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
                fh.write(f"{x0 + cx * f + f // 2},{y0 + cy * f + f // 2},{z:.1f},{d:.0f}\n")
    else:
        print("no route")

    viz.fig_route(h0, f, shape, info, a.levels, bg, cell, f"{a.out}/map_route.png")
    print("saved to", a.out)


if __name__ == "__main__":
    main()
