"""
Path-planning primitives for the Laplace-equation rover planner.

Functions
---------
reachable           : 4-connected flood fill to find cells that can reach the goal
dijkstra_length     : shortest 8-connected path length (reference benchmark)
dirichlet_potential : Formulation A — hazards fixed at φ = 1, target at φ = 0
neumann_potential   : Formulation B — no-flux walls, start = 1, target = 0  ← main method
apf_potential       : classic artificial potential field (attractive + repulsive)
descend             : greedy steepest descent over 8 neighbours
path_stats          : path length and minimum clearance to raw hazards
nearest_free        : snap a point to the nearest non-hazard cell
trap_map            : build the U-shaped dead-end test map
plan                : high-level planner for an arbitrary height map
"""

import time
from collections import deque

import heapq
import numpy as np

from .solvers import sor, sor_neumann, omega_opt
from .terrain import dilate, slope_map


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------

def reachable(hazard: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """4-connected flood fill: find all free cells that can reach *goal*.

    Parameters
    ----------
    hazard : 2-D boolean hazard mask
    goal   : ``(x, y)`` target coordinate

    Returns
    -------
    Boolean array — True where the cell can reach the goal.
    """
    n = hazard.shape[0]
    seen = np.zeros_like(hazard)
    q: deque[tuple[int, int]] = deque([goal])
    seen[goal[1], goal[0]] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < n and 0 <= b < n and not hazard[b, a] and not seen[b, a]:
                seen[b, a] = True
                q.append((a, b))
    return seen


def dijkstra_length(
    hazard: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> float:
    """Length of the shortest 8-connected obstacle-free path (reference).

    Parameters
    ----------
    hazard : 2-D boolean hazard mask
    start  : ``(x, y)`` start coordinate
    goal   : ``(x, y)`` goal coordinate

    Returns
    -------
    Path length in cells (``np.inf`` if no path exists).
    """
    n = hazard.shape[0]
    dist = np.full((n, n), np.inf)
    dist[start[1], start[0]] = 0.0
    pq = [(0.0, start[0], start[1])]
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        if (x, y) == goal:
            return d
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                a, b = x + dx, y + dy
                if (dx or dy) and 0 <= a < n and 0 <= b < n and not hazard[b, a]:
                    nd = d + np.hypot(dx, dy)
                    if nd < dist[b, a]:
                        dist[b, a] = nd
                        heapq.heappush(pq, (nd, a, b))
    return np.inf


def nearest_free(
    hazard: np.ndarray,
    p: tuple[int, int],
    reach: np.ndarray,
) -> tuple[int, int]:
    """Return the nearest non-hazard cell to *p* within *reach*.

    Parameters
    ----------
    hazard : 2-D boolean hazard mask (unused — kept for API clarity)
    p      : ``(x, y)`` query point
    reach  : boolean mask of candidate cells

    Returns
    -------
    ``(x, y)`` of the nearest free cell
    """
    ys, xs = np.nonzero(reach)
    i = int(np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2))
    return (int(xs[i]), int(ys[i]))


# ---------------------------------------------------------------------------
# Potential-field builders
# ---------------------------------------------------------------------------

def dirichlet_potential(
    hazard: np.ndarray,
    goal: tuple[int, int],
) -> tuple[np.ndarray, int, float]:
    """Formulation A: hazards held at φ = 1, target at φ = 0 (Dirichlet).

    .. warning::
        This formulation **fails on rugged terrain** — φ is essentially flat
        in narrow passages (exponential decay ∝ e^{-πd/w}).

    Returns
    -------
    phi   : potential field
    iters : number of SOR iterations
    secs  : wall-clock time
    """
    n = hazard.shape[0]
    phi = np.full((n, n), 0.5)
    fixed = hazard.copy()
    phi[hazard] = 1.0
    gx, gy = goal
    phi[gy, gx] = 0.0
    fixed[gy, gx] = True
    t0 = time.time()
    phi, hist = sor(phi, fixed, omega_opt(n), tol=1e-11)
    return phi, len(hist), time.time() - t0


def neumann_potential(
    hazard: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[np.ndarray, int, float]:
    """Formulation B: no-flux walls, start = source (φ = 1), target = sink (φ = 0).

    This is the main planning formulation.  Hazards satisfy ∂φ/∂n = 0
    (Neumann condition), so the field varies smoothly through every passage
    and steepest descent cannot stall.

    Returns
    -------
    phi   : potential field
    iters : number of SOR iterations
    secs  : wall-clock time
    """
    n, m = hazard.shape
    free = ~hazard
    phi = np.full((n, m), 0.5)
    fixed = np.zeros((n, m), bool)
    phi[start[1], start[0]] = 1.0
    fixed[start[1], start[0]] = True
    phi[goal[1], goal[0]] = 0.0
    fixed[goal[1], goal[0]] = True
    t0 = time.time()
    phi, hist = sor_neumann(phi, fixed, free, omega_opt(n) * 0.97, tol=1e-10)
    return phi, len(hist), time.time() - t0


def apf_potential(
    hazard: np.ndarray,
    goal: tuple[int, int],
    d0: float = 10.0,
    k_att: float = 0.002,
    k_rep: float = 200.0,
) -> np.ndarray:
    """Classic artificial potential field (attractive + repulsive).

    Included for comparison only — it can get stuck in local minima.

    Returns
    -------
    P : 2-D potential array
    """
    n = hazard.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    hy, hx = np.nonzero(hazard)
    d = np.full((n, n), 1e9)
    for i in range(0, len(hy), 200):
        d = np.minimum(
            d,
            np.hypot(xx[..., None] - hx[i:i + 200],
                     yy[..., None] - hy[i:i + 200]).min(axis=2),
        )
    P = 0.5 * k_att * ((xx - goal[0]) ** 2 + (yy - goal[1]) ** 2)
    dd = np.maximum(d, 1.0)
    P += np.where(d < d0, 0.5 * k_rep * (1 / dd - 1 / d0) ** 2, 0.0)
    return P


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def descend(
    P: np.ndarray,
    hazard: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    maxsteps: int = 5000,
) -> tuple[list[tuple[int, int]], bool]:
    """Greedy steepest descent over the 8 neighbours.

    Parameters
    ----------
    P        : potential field
    hazard   : boolean hazard mask
    start    : ``(x, y)`` start
    goal     : ``(x, y)`` target
    maxsteps : maximum steps before giving up

    Returns
    -------
    path     : list of ``(x, y)`` cells visited
    reached  : True if the goal was reached
    """
    x, y = start
    path = [(x, y)]
    for _ in range(maxsteps):
        best = (P[y, x], x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx or dy) and not hazard[y + dy, x + dx]:
                    if P[y + dy, x + dx] < best[0] - 1e-14:
                        best = (P[y + dy, x + dx], x + dx, y + dy)
        if (best[1], best[2]) == (x, y):
            return path, False
        x, y = best[1], best[2]
        path.append((x, y))
        if (x, y) == goal:
            return path, True
    return path, False


def path_stats(
    path: list[tuple[int, int]],
    raw: np.ndarray,
) -> tuple[float, float]:
    """Compute path length and minimum clearance to raw hazard cells.

    Parameters
    ----------
    path : list of ``(x, y)`` cells
    raw  : raw (un-dilated) hazard mask

    Returns
    -------
    length    : total path length in cells
    clearance : minimum Euclidean distance to any raw hazard cell
    """
    p = np.array(path)
    length = float(np.hypot(*np.diff(p, axis=0).T).sum())
    hy, hx = np.nonzero(raw)
    clearance = float(
        np.hypot(p[:, None, 0] - hx[None, :],
                 p[:, None, 1] - hy[None, :]).min()
    )
    return length, clearance


# ---------------------------------------------------------------------------
# Test maps
# ---------------------------------------------------------------------------

def trap_map(N: int = 100) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Build a U-shaped dead-end test map.

    The rover start is placed *inside* the U; the goal is on the far side.
    A classic potential field gets stuck; the Laplace planner escapes.

    Returns
    -------
    hazard : boolean hazard mask
    start  : ``(x, y)``
    goal   : ``(x, y)``
    """
    hz = np.zeros((N, N), bool)
    hz[0, :] = hz[-1, :] = hz[:, 0] = hz[:, -1] = True
    hz[30:70, 60:65] = True   # back wall of the U (facing the goal)
    hz[30:35, 35:65] = True   # lower arm
    hz[65:70, 35:65] = True   # upper arm — U is open on the left
    return hz, (48, 50), (90, 50)


# ---------------------------------------------------------------------------
# High-level planner (for plan_on_heightmap CLI)
# ---------------------------------------------------------------------------

def plan(
    h: np.ndarray,
    cell: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_slope_deg: float,
    rover_radius_m: float,
) -> dict:
    """Plan a safe route across an arbitrary height map.

    Parameters
    ----------
    h             : 2-D height array (metres)
    cell          : cell size in metres
    start         : ``(x, y)`` start cell
    goal          : ``(x, y)`` goal cell
    max_slope_deg : maximum drivable slope in degrees
    rover_radius_m: safety margin around hazards in metres

    Returns
    -------
    info : dict with keys ``path``, ``ok``, ``phi``, ``hazard``, ``slope``,
           ``raw``, ``notes``, and (if successful) ``length_m``,
           ``shortest_m``, ``max_path_slope``, ``climb_m``, ``descent_m``,
           ``dist_m``, ``profile_z``, ``iters``, ``solve_s``.
    """
    n, m = h.shape
    slope = slope_map(h, cell)
    raw = slope > max_slope_deg
    raw[0, :] = raw[-1, :] = raw[:, 0] = raw[:, -1] = True

    if rover_radius_m < 0.7 * cell:
        print(
            f"note: safety margin {rover_radius_m:.0f} m is smaller than a grid cell "
            f"({cell:.0f} m), so NO extra margin is applied. "
            f"Use --max-size larger (finer cells) or a bigger --rover-radius."
        )

    hazard = dilate(raw, rover_radius_m / cell)
    hazard[0, :] = hazard[-1, :] = hazard[:, 0] = hazard[:, -1] = True
    info: dict = dict(slope=slope, raw=raw, hazard=hazard, notes=[])

    # snap chosen points to the nearest safe cell if they fall on a hazard
    for name, p in (("start", start), ("goal", goal)):
        if hazard[p[1], p[0]]:
            q = nearest_free(hazard, p, ~hazard)
            info["notes"].append(
                f"{name} {p} was on a hazard/margin; moved to nearest safe cell {q} "
                f"({np.hypot(q[0] - p[0], q[1] - p[1]) * cell:.0f} m away)"
            )
            if name == "start":
                start = q
            else:
                goal = q

    reach = reachable(hazard, goal)
    info["start"], info["goal"] = start, goal

    if not reach[start[1], start[0]]:
        info["ok"] = False
        info["path"] = [start]
        info["notes"].append(
            "NO SAFE ROUTE: the start cannot reach the goal. "
            "Raise --max-slope or lower --rover-radius."
        )
        return info

    free = ~hazard
    phi = np.full((n, m), 0.5)
    fixed = np.zeros((n, m), bool)
    phi[start[1], start[0]] = 1.0
    fixed[start[1], start[0]] = True
    phi[goal[1], goal[0]] = 0.0
    fixed[goal[1], goal[0]] = True

    t0 = time.time()
    phi, hist = sor_neumann(phi, fixed, free, omega_opt(max(n, m)) * 0.97, tol=1e-9)
    info["solve_s"] = time.time() - t0
    info["iters"] = len(hist)
    info["phi"] = phi

    path, ok = descend(phi, hazard, start, goal, maxsteps=20 * (n + m))
    info["path"], info["ok"] = path, ok

    if ok:
        p = np.array(path)
        steps = np.hypot(*np.diff(p, axis=0).T)
        info["length_m"] = float(steps.sum() * cell)
        info["shortest_m"] = dijkstra_length(hazard, start, goal) * cell
        info["max_path_slope"] = float(slope[p[:, 1], p[:, 0]].max())
        z = h[p[:, 1], p[:, 0]]
        dz = np.diff(z)
        info["climb_m"] = float(dz[dz > 0].sum())
        info["descent_m"] = float(-dz[dz < 0].sum())
        info["dist_m"] = np.concatenate([[0], np.cumsum(steps)]) * cell
        info["profile_z"] = z

    return info
