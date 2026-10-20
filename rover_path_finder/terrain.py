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
import re
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

def read_pds3(path: str) -> tuple[np.ndarray, dict]:
    """Minimal reader for PDS3 rasters with an attached label (e.g. HiRISE DTM .IMG).

    Returns a memory-mapped array (nothing is read until you slice it) and a
    metadata dict with keys ``mpp`` (metres/pixel), ``missing`` (no-data value),
    and ``label_type`` (SAMPLE_TYPE string).

    Raises
    ------
    ValueError
        If no attached PDS3 label is found or the pointer is detached.
    """
    with open(path, "rb") as fh:
        head = fh.read(400_000)
    end = re.search(rb"\r?\nEND\s*(\r?\n|$)", head)
    if not end:
        raise ValueError(
            "no attached PDS3 label found; "
            "convert with: gdal_translate file.IMG file.tif"
        )
    label = head[: end.end()].decode("latin-1")

    def get(key: str) -> Optional[str]:
        m = re.search(
            r"^\s*" + re.escape(key) + r"\s*=\s*(.+?)\s*$", label, re.M
        )
        return m.group(1).strip() if m else None

    def num(v: str) -> float:
        """Parse a PDS3 numeric, including radix literals like 16#FF7FFFFB#."""
        tok = re.split(r"[\s<(,]", v.strip().strip('"'))[0]
        radix = re.match(r"(\d+)#([0-9A-Fa-f]+)#", tok)
        if radix:
            return float(int(radix.group(2), int(radix.group(1))))
        return float(tok)

    ptr = get("^IMAGE")
    if ptr is None or '"' in ptr or ptr.startswith("("):
        raise ValueError(
            "label is detached; convert with: gdal_translate file.IMG file.tif"
        )
    off = int(num(ptr)) - 1
    if "<BYTES>" not in ptr:
        off *= int(num(get("RECORD_BYTES")))
    lines = int(num(get("LINES")))
    samples = int(num(get("LINE_SAMPLES")))
    bits = int(num(get("SAMPLE_BITS")))
    st = (get("SAMPLE_TYPE") or "").strip('"').upper()
    little = st.startswith(("PC", "LSB"))
    kind = "f" if "REAL" in st else ("u" if "UNSIGNED" in st else "i")
    dtype = np.dtype(("<" if little else ">") + kind + str(bits // 8))
    arr = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=(lines, samples))
    meta = {
        "mpp": num(get("MAP_SCALE")) if get("MAP_SCALE") else None,
        "missing": num(get("MISSING_CONSTANT")) if get("MISSING_CONSTANT") else None,
        "label_type": st,
    }
    return arr, meta


def block_mean(
    h: np.ndarray,
    f: int,
    missing: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average f×f blocks, ignoring no-data pixels.

    Works in row bands to keep memory low for large memory-mapped arrays.

    Parameters
    ----------
    h       : 2-D array (may be a memory-map)
    f       : block side length in pixels
    missing : sentinel value to treat as no-data (in addition to non-finite)

    Returns
    -------
    out  : (rows//f) × (cols//f) float array of block means
    frac : same shape — fraction of valid pixels in each block (0..1)
    """
    r = (h.shape[0] // f) * f
    c = (h.shape[1] // f) * f
    out = np.empty((r // f, c // f))
    frac = np.empty_like(out)
    band = f * max(1, 1500 // f)
    for i in range(0, r, band):
        blk = np.asarray(h[i : min(i + band, r), :c], dtype=np.float64)
        bad = ~np.isfinite(blk) | (np.abs(blk) > 1e30)
        if missing is not None:
            bad |= blk == missing
        good = (~bad).astype(float)
        blk = np.where(bad, 0.0, blk)
        n = blk.shape[0] // f
        s = blk.reshape(n, f, c // f, f).sum(axis=(1, 3))
        g = good.reshape(n, f, c // f, f).sum(axis=(1, 3))
        out[i // f : i // f + n] = np.where(g > 0, s / np.maximum(g, 1), np.nan)
        frac[i // f : i // f + n] = g / (f * f)
    return out, frac


def load_heightmap(
    path: str,
    height_range: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, dict]:
    """Load a height map from disk.

    Supports ``.img`` / ``.imq`` (PDS3 attached label), ``.npy``, ``.txt``,
    ``.csv``, and any image format that PIL can open (16-bit PNG, float TIFF…).

    Parameters
    ----------
    path         : path to the file
    height_range : optional ``(lo, hi)`` metres to rescale pixel values
                   (only used for plain image formats)

    Returns
    -------
    h    : 2-D float array of heights
    meta : dict with ``mpp`` (float or None) and ``missing`` (float or None)
    """
    ext = os.path.splitext(path)[1].lower()
    meta: dict = {"mpp": None, "missing": None}

    if ext in (".img", ".imq"):
        h, meta = read_pds3(path)
    elif ext == ".npy":
        h = np.load(path, mmap_mode="r")
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

    if height_range is not None:
        h = np.asarray(h, dtype=float)
        lo, hi = height_range
        h = lo + (h - h.min()) / max(h.max() - h.min(), 1e-12) * (hi - lo)

    return h, meta


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
