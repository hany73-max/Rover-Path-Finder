"""
Terrain generation, loading, pre-processing, and hazard handling.

Functions
---------
make_terrain        : synthesise a 100×100 lunar terrain with craters and boulders
synthetic_mars      : synthesise a Mars-like fractal terrain (for the heightmap demo)
load_heightmap      : load a height map from .npy / .csv / image file
smooth              : separable Gaussian blur
downsample          : average-pool to a target maximum dimension
slope_map           : compute per-cell slope in degrees from a height array
dilate              : morphological dilation of a boolean hazard mask
"""

import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic terrains
# ---------------------------------------------------------------------------

def make_terrain(
    N: int = 100,
    seed: int = 3,
    slope_limit: float = 0.8,
    rover_radius: float = 2.0,
) -> dict:
    """Build a synthetic 100×100 lunar terrain with craters and boulders.

    Parameters
    ----------
    N            : grid size (N×N cells)
    seed         : random seed for boulder placement
    slope_limit  : |∇h| threshold above which a cell is a steep-slope hazard
    rover_radius : dilation radius in cells for the safety margin

    Returns
    -------
    dict with keys:
        h        – elevation array (N×N)
        slope    – |∇h| array
        raw      – raw hazard mask (steep slopes + boulders + walls)
        hazard   – inflated hazard mask (safety margin applied)
        start    – (x, y) start coordinate
        goal     – (x, y) goal coordinate
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N]
    h = np.zeros((N, N))

    crater_params = [
        (.30, .30, .10), (.62, .45, .12), (.40, .72, .09),
        (.76, .76, .08), (.20, .58, .06), (.82, .25, .07),
    ]
    for cx, cy, r in crater_params:
        R = r * N
        d = np.hypot(xx - cx * N, yy - cy * N) / R
        h += R * (-0.9 * np.exp(-(d / 0.75) ** 2) + 0.35 * np.exp(-((d - 1.0) / 0.3) ** 2))

    gy, gx = np.gradient(h)
    slope = np.hypot(gx, gy)
    steep = slope > slope_limit

    boulders = np.zeros((N, N), bool)
    for _ in range(14):
        bx = rng.uniform(8, N - 8)
        by = rng.uniform(8, N - 8)
        br = rng.uniform(1.2, 2.2)
        boulders |= np.hypot(xx - bx, yy - by) <= br

    raw = steep | boulders
    start, goal = (6, 6), (92, 90)
    for sx, sy in (start, goal):
        raw &= np.hypot(xx - sx, yy - sy) > 6

    wall = np.zeros((N, N), bool)
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True

    inflated = dilate(raw, rover_radius) | wall
    for sx, sy in (start, goal):
        inflated &= np.hypot(xx - sx, yy - sy) > 3

    return dict(h=h, slope=slope, raw=raw | wall, hazard=inflated, start=start, goal=goal)


def synthetic_mars(rows: int = 420, cols: int = 300, seed: int = 7, relief: float = 1700.0) -> np.ndarray:
    """Mars-like fractal terrain (power-law spectrum) with craters, in metres.

    Parameters
    ----------
    rows, cols : output shape
    seed       : random seed
    relief     : vertical range in metres

    Returns
    -------
    h : 2-D float array of elevations
    """
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.fftfreq(cols)[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1
    spec = (k ** -1.8) * np.exp(2j * np.pi * rng.random((rows, cols)))
    spec[0, 0] = 0
    h = np.fft.ifft2(spec).real
    h = (h - h.min()) / (h.max() - h.min())

    yy, xx = np.mgrid[0:rows, 0:cols]
    for _ in range(12):
        cx, cy = rng.uniform(0, cols), rng.uniform(0, rows)
        r = rng.uniform(10, 28)
        d = np.hypot(xx - cx, yy - cy) / r
        h += 0.07 * (-np.exp(-(d / 0.75) ** 2) + 0.35 * np.exp(-((d - 1.0) / 0.3) ** 2)) * (r / 20)

    return (h - h.min()) / (h.max() - h.min()) * relief


# ---------------------------------------------------------------------------
# Loading and pre-processing
# ---------------------------------------------------------------------------

def load_heightmap(path: str, height_range: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Load a height map from disk.

    Supports ``.npy``, ``.txt``, ``.csv``, and any image format that PIL
    can open (16-bit PNG, float/int TIFF, …).

    Parameters
    ----------
    path         : path to the file
    height_range : optional ``(lo, hi)`` metres to rescale pixel values

    Returns
    -------
    h : 2-D float array of heights
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        h = np.load(path).astype(float)
    elif ext in (".txt", ".csv"):
        h = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        from PIL import Image  # optional dependency
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path)
        if im.mode in ("RGB", "RGBA", "P", "LA"):
            im = im.convert("L")
        h = np.asarray(im, dtype=float)

    if h.ndim == 3:
        h = h[..., 0]

    # replace common "no data" sentinel values
    h = np.where(np.abs(h) > 1e30, np.nan, h)
    if np.isnan(h).any():
        print(f"warning: {int(np.isnan(h).sum())} no-data pixels replaced by the mean height")
        h = np.where(np.isnan(h), np.nanmean(h), h)

    if height_range is not None:
        lo, hi = height_range
        h = lo + (h - h.min()) / max(h.max() - h.min(), 1e-12) * (hi - lo)

    return h


def smooth(h: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur.

    Parameters
    ----------
    h     : 2-D height array
    sigma : blur radius in pixels (0 → no-op)

    Returns
    -------
    blurred copy of ``h``
    """
    if sigma <= 0:
        return h
    r = int(3 * sigma) + 1
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    p = np.pad(h, r, mode="edge")
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, p)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, p)


def downsample(h: np.ndarray, max_size: int) -> tuple[np.ndarray, int]:
    """Average-pool ``h`` so its longest side is ≤ ``max_size``.

    Parameters
    ----------
    h        : 2-D height array
    max_size : target maximum dimension

    Returns
    -------
    h_ds : downsampled array
    f    : downsampling factor (1 means no change)
    """
    f = int(np.ceil(max(h.shape) / max_size))
    if f <= 1:
        return h, 1
    r = (h.shape[0] // f) * f
    c = (h.shape[1] // f) * f
    return h[:r, :c].reshape(r // f, f, c // f, f).mean(axis=(1, 3)), f


def slope_map(h: np.ndarray, cell: float) -> np.ndarray:
    """Per-cell slope in degrees.

    Parameters
    ----------
    h    : 2-D height array
    cell : cell size in metres

    Returns
    -------
    slope in degrees
    """
    gy, gx = np.gradient(h, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


# ---------------------------------------------------------------------------
# Hazard utilities
# ---------------------------------------------------------------------------

def dilate(mask: np.ndarray, r: float) -> np.ndarray:
    """Morphological dilation of a boolean mask by radius ``r`` pixels.

    Parameters
    ----------
    mask : 2-D boolean hazard array
    r    : dilation radius in pixels

    Returns
    -------
    dilated boolean array
    """
    R = int(np.ceil(r))
    out = mask.copy()
    p = np.pad(mask, R)
    n, m = mask.shape
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            if dx * dx + dy * dy <= r * r:
                out |= p[R + dy:R + dy + n, R + dx:R + dx + m]
    return out
