"""
Matplotlib figure helpers for the Rover Path Finder project.

Each function saves one figure to disk and closes it, so they can be
called safely in non-interactive (headless) environments.

Functions
---------
fig_terrain         : elevation, slope, and hazard overview (3 panels)
fig_solution        : side-by-side Formulation A vs B potential fields + path
fig_numerics        : convergence order and solver comparison log-log plots
fig_trap            : U-shaped trap test — APF stuck vs Laplace success
fig_route           : topographic contour map with the planned route overlaid
fig_profile         : elevation and slope profiles along the route
fig_art             : clean contour map without the route (for visual reference)
draw_contours       : shared helper — draw contour lines on an Axes
pick_points         : interactive start/goal selection (desktop only)
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Validation / solver figures (rover_laplace demo)
# ---------------------------------------------------------------------------

def fig_terrain(T: dict, out_dir: str) -> None:
    """Three-panel overview: elevation, slope magnitude, and hazard map.

    Parameters
    ----------
    T       : terrain dict returned by :func:`terrain.make_terrain`
    out_dir : directory where ``fig1_terrain.png`` is saved
    """
    f, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    im = ax[0].imshow(T["h"], origin="lower", cmap="terrain")
    ax[0].set_title("Synthetic lunar elevation")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    im = ax[1].imshow(T["slope"], origin="lower", cmap="magma")
    ax[1].set_title("Slope |grad h|")
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    show = np.zeros(T["h"].shape)
    show[T["hazard"]] = 1
    show[T["raw"]] = 2
    ax[2].imshow(show, origin="lower", cmap="gray_r")
    ax[2].set_title("Hazards (dark) + safety margin (grey)")

    for a in ax:
        a.plot(*T["start"], "go")
        a.plot(*T["goal"], "r*", ms=12)

    f.tight_layout()
    f.savefig(f"{out_dir}/fig1_terrain.png", dpi=130)
    plt.close(f)


def fig_solution(
    T: dict,
    phiA: np.ndarray,
    pathA: list,
    phiB: np.ndarray,
    pathB: list,
    out_dir: str,
) -> None:
    """Three-panel comparison: Formulation A vs B potential fields and final path.

    Parameters
    ----------
    T       : terrain dict
    phiA    : Dirichlet potential (Formulation A)
    pathA   : path from Formulation A descent
    phiB    : Neumann potential (Formulation B)
    pathB   : path from Formulation B descent
    out_dir : output directory for ``fig2_solution_and_path.png``
    """
    f, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for a, phi, path, t in (
        (ax[0], phiA, pathA, "A: hazards at phi=1 (Dirichlet) — phi is flat, path fails"),
        (ax[1], phiB, pathB, "B: no-flux walls, start=1, target=0 — path succeeds"),
    ):
        m = np.ma.masked_where(T["hazard"], phi)
        im = a.imshow(m, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=a, fraction=0.046, label="phi")
        p = np.array(path)
        a.plot(p[:, 0], p[:, 1], "r-", lw=2)
        a.plot(*T["start"], "wo", ms=8)
        a.plot(*T["goal"], "w*", ms=14)
        a.set_title(t, fontsize=10)

    ax[2].imshow(
        np.where(T["raw"], 2, np.where(T["hazard"], 1, 0)),
        origin="lower", cmap="gray_r",
    )
    p = np.array(pathB)
    ax[2].plot(p[:, 0], p[:, 1], "c-", lw=2.2, label="rover path (B)")
    ax[2].plot(*T["start"], "go", ms=9, label="start")
    ax[2].plot(*T["goal"], "r*", ms=15, label="target")
    ax[2].legend(loc="upper left")
    ax[2].set_title("Path on the hazard map (dark = hazard, grey = safety margin)", fontsize=10)

    f.tight_layout()
    f.savefig(f"{out_dir}/fig2_solution_and_path.png", dpi=130)
    plt.close(f)


def fig_numerics(
    Nv: list[int],
    ev: list[float],
    Nc: list[int],
    res: dict[str, list[int]],
    out_dir: str,
) -> None:
    """Two log-log plots: convergence order and solver iteration counts.

    Parameters
    ----------
    Nv      : grid sizes used for validation
    ev      : max errors for each grid size
    Nc      : grid sizes used for solver comparison
    res     : dict mapping solver name → list of iteration counts
    out_dir : output directory for ``fig3_validation_and_solvers.png``
    """
    f, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ax[0].loglog(Nv, ev, "o-", label="max error")
    ax[0].loglog(Nv, [ev[0] * (Nv[0] / n) ** 2 for n in Nv], "k--", label="slope −2 (O(h²))")
    ax[0].set_xlabel("grid points N")
    ax[0].set_ylabel("max |numerical − exact|")
    ax[0].set_title("Validation: 2nd-order convergence")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    for k, v in res.items():
        ax[1].loglog(Nc, v, "o-", label=k)
    ax[1].set_xlabel("grid points N")
    ax[1].set_ylabel("iterations to tol 1e-6")
    ax[1].set_title("Solver comparison")
    ax[1].legend()
    ax[1].grid(True, which="both", alpha=0.3)

    f.tight_layout()
    f.savefig(f"{out_dir}/fig3_validation_and_solvers.png", dpi=130)
    plt.close(f)


def fig_trap(
    hz: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    P_apf: np.ndarray,
    p_apf: list,
    ok_apf: bool,
    phi: np.ndarray,
    p_lap: list,
    ok_lap: bool,
    out_dir: str,
) -> None:
    """Side-by-side comparison on the U-shaped trap map.

    Parameters
    ----------
    hz      : hazard mask for the trap map
    start, goal : ``(x, y)`` coordinates
    P_apf   : APF potential field
    p_apf   : APF path
    ok_apf  : whether APF reached the goal
    phi     : Laplace potential field
    p_lap   : Laplace path
    ok_lap  : whether Laplace reached the goal
    out_dir : output directory for ``fig4_trap_comparison.png``
    """
    f, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for a, pth, ok, t in (
        (ax[0], p_apf, ok_apf, "Classic potential field: STUCK"),
        (ax[1], p_lap, ok_lap, "Laplace potential: reaches target"),
    ):
        a.imshow(hz, origin="lower", cmap="gray_r")
        p = np.array(pth)
        a.plot(p[:, 0], p[:, 1], "c-", lw=2.2)
        a.plot(*start, "go", ms=9)
        a.plot(*goal, "r*", ms=15)
        a.set_title(t if (ok == (a is ax[1])) else t + " (!)")

    f.tight_layout()
    f.savefig(f"{out_dir}/fig4_trap_comparison.png", dpi=130)
    plt.close(f)


# ---------------------------------------------------------------------------
# Height-map planner figures (plan_on_heightmap CLI)
# ---------------------------------------------------------------------------

def draw_contours(
    ax: plt.Axes,
    hfull: np.ndarray,
    f: int,
    shape: tuple[int, int],
    levels: int,
    bg: str,
    lw: float = 0.7,
) -> None:
    """Draw contour lines from the full-resolution map in planning-grid coordinates.

    Parameters
    ----------
    ax     : target Axes
    hfull  : full-resolution height array (before downsampling)
    f      : downsampling factor
    shape  : ``(rows, cols)`` of the planning grid
    levels : number of contour levels
    bg     : background colour string
    lw     : contour line width
    """
    ax.set_facecolor(bg)
    r, c = shape
    hf = hfull[:r * f, :c * f]
    X, Y = np.meshgrid(
        (np.arange(hf.shape[1]) + 0.5) / f - 0.5,
        (np.arange(hf.shape[0]) + 0.5) / f - 0.5,
    )
    ax.contour(
        X, Y, hf,
        levels=np.linspace(hf.min(), hf.max(), levels + 2)[1:-1],
        colors="#1a1a1a",
        linewidths=lw,
    )
    ax.set_xlim(0, c - 1)
    ax.set_ylim(r - 1, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_art(
    hfull: np.ndarray,
    f: int,
    shape: tuple[int, int],
    levels: int,
    bg: str,
    path: str,
) -> None:
    """Save a clean contour map (no route overlay).

    Parameters
    ----------
    hfull  : full-resolution height array
    f      : downsampling factor
    shape  : ``(rows, cols)`` of the planning grid
    levels : number of contour levels
    bg     : background colour
    path   : output file path
    """
    f_, ax = plt.subplots(
        figsize=(7, 7 * shape[0] / shape[1]),
        facecolor=bg,
    )
    draw_contours(ax, hfull, f, shape, levels, bg)
    f_.subplots_adjust(0, 0, 1, 1)
    f_.savefig(path, dpi=160, facecolor=bg)
    plt.close(f_)


def fig_route(
    hfull: np.ndarray,
    f: int,
    shape: tuple[int, int],
    info: dict,
    levels: int,
    bg: str,
    cell: float,
    path: str,
    show_hazard: bool = True,
) -> None:
    """Save a contour map with the planned route overlaid.

    Parameters
    ----------
    hfull       : full-resolution height array
    f           : downsampling factor
    shape       : ``(rows, cols)`` of the planning grid
    info        : result dict from :func:`planners.plan`
    levels      : number of contour levels
    bg          : background colour
    cell        : cell size in metres (used for title)
    path        : output file path
    show_hazard : whether to shade hazard cells
    """
    fg, ax = plt.subplots(
        figsize=(7.5, 7.5 * shape[0] / shape[1] + 0.8),
        facecolor=bg,
    )
    draw_contours(ax, hfull, f, shape, levels, bg)

    if show_hazard:
        ax.contourf(info["hazard"].astype(float), levels=[0.5, 1.5], colors=["#000000"], alpha=0.30)
        ax.contour(info["hazard"].astype(float), levels=[0.5], colors=["#7a0000"], linewidths=0.8)

    p = np.array(info["path"])
    ax.plot(p[:, 0], p[:, 1], "-", color="black", lw=4.2, solid_capstyle="round")
    ax.plot(p[:, 0], p[:, 1], "-", color="#fff200", lw=2.4, solid_capstyle="round", label="safe route (Laplace)")
    ax.plot(*info["start"], "o", ms=11, mfc="#22dd44", mec="black", mew=1.6, label="start")
    ax.plot(*info["goal"], "*", ms=18, mfc="#ff2a2a", mec="black", mew=1.4, label="goal")

    title = "Safe route over height map"
    if info["ok"]:
        title += f" — {info['length_m'] / 1000:.2f} km, max slope on route {info['max_path_slope']:.1f}°"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fg.tight_layout()
    fg.savefig(path, dpi=160, facecolor=bg)
    plt.close(fg)


def fig_profile(info: dict, max_slope: float, path: str) -> None:
    """Save elevation and slope profiles along the planned route.

    Parameters
    ----------
    info      : result dict from :func:`planners.plan`
    max_slope : slope limit in degrees (drawn as a reference line)
    path      : output file path
    """
    f, ax = plt.subplots(1, 2, figsize=(11, 3.4))

    ax[0].plot(info["dist_m"] / 1000, info["profile_z"], "-", color="#b5541a")
    ax[0].set_xlabel("distance along route (km)")
    ax[0].set_ylabel("elevation (m)")
    ax[0].set_title("Elevation profile along the route")
    ax[0].grid(alpha=0.3)

    sl = info["slope"][np.array(info["path"])[:, 1], np.array(info["path"])[:, 0]]
    ax[1].plot(info["dist_m"] / 1000, sl, "-", color="#333")
    ax[1].axhline(max_slope, color="r", ls="--", label=f"limit {max_slope}°")
    ax[1].set_xlabel("distance along route (km)")
    ax[1].set_ylabel("slope (deg)")
    ax[1].set_title("Slope along the route")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    f.tight_layout()
    f.savefig(path, dpi=140)
    plt.close(f)


# ---------------------------------------------------------------------------
# Interactive utilities
# ---------------------------------------------------------------------------

def pick_points(
    hfull: np.ndarray,
    f: int,
    shape: tuple[int, int],
    levels: int,
    bg: str,
) -> list[tuple[int, int]]:
    """Open an interactive window and let the user click start and goal.

    Tries available GUI backends in order.  Exits the process if fewer than
    two clicks are received.

    Parameters
    ----------
    hfull  : full-resolution height array
    f      : downsampling factor
    shape  : ``(rows, cols)`` of the planning grid
    levels : number of contour levels
    bg     : background colour

    Returns
    -------
    [(x_start, y_start), (x_goal, y_goal)]
    """
    for backend in ("MacOSX", "QtAgg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue

    fg, ax = plt.subplots(figsize=(8, 8 * shape[0] / shape[1]))
    draw_contours(ax, hfull, f, shape, levels, bg)
    ax.set_title("Click START, then GOAL")
    pts = fg.ginput(2, timeout=0)
    plt.close(fg)

    if len(pts) < 2:
        sys.exit("need two clicks")

    return [(int(round(x)), int(round(y))) for x, y in pts]
