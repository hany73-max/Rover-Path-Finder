# Rover Path Finder

> Safe planetary rover path planning using **Laplace's equation** — implemented from scratch with NumPy only. No SciPy, no graph libraries, no ML.

*College PDE project — tested on synthetic lunar terrain and compatible with real Mars DEMs.*

---

## The Idea

Most path planners for robots use A* search or **Artificial Potential Fields (APF)**. APF is fast but has a known fatal flaw — the rover can get permanently trapped in a local minimum of the potential, even when the goal is reachable.

This project uses **Laplace's equation** instead:

$$
\frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} = 0
$$

We solve this PDE over the obstacle-free region of the terrain, with:
- **Start:** $\phi = 1$ (Dirichlet source)
- **Goal:** $\phi = 0$ (Dirichlet sink)
- **Obstacles:** $\frac{\partial\phi}{\partial n} = 0$ (Neumann no-flux wall)

The rover then follows steepest descent of $\phi$. Because $\phi$ is **harmonic** (satisfies Laplace's equation), the **Maximum Principle** guarantees it has no interior local minima — the rover *cannot get stuck*, ever.

---

## Project Structure

```
Rover-Path-Finder/
│
├── rover_path_finder/          ← Python package (all the real math lives here)
│   ├── __init__.py
│   ├── solvers.py              ← Jacobi, Gauss-Seidel, SOR, Neumann-SOR
│   ├── terrain.py              ← terrain generation, DEM loading, dilation, slope
│   ├── planners.py             ← potential builders, steepest descent, Dijkstra
│   └── visualization.py       ← all Matplotlib figures
│
├── main.py                     ← CLI: runs the full lunar demo (reproduces REPORT.md)
├── plan_on_heightmap.py        ← CLI: plan a route on any height map (real or synthetic)
│
├── pyproject.toml              ← project metadata + dependencies (uv-managed)
├── uv.lock                     ← locked dependency versions
├── REPORT.md                   ← full project write-up (theory, derivation, results)
└── outputs/                    ← generated figures & logs (git-ignored)
```

---

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) (fast Python package manager).

```bash
# 1. Install dependencies into a managed virtual environment
uv sync

# 2. Run the full lunar demo (~35 s)
#    Reproduces every number and figure from REPORT.md
uv run python main.py
#    → outputs/fig1_terrain.png
#    → outputs/fig2_solution_and_path.png
#    → outputs/fig3_validation_and_solvers.png
#    → outputs/fig4_trap_comparison.png
#    → outputs/results.txt

# 3. Run the height-map planner (synthetic Mars terrain demo)
uv run python plan_on_heightmap.py
#    → outputs/map_contours.png
#    → outputs/map_route.png
#    → outputs/route_profile.png
#    → outputs/route.csv
```

---

## Using Real Mars DEMs

**Yes — the planner works on real Mars elevation data.** The `plan_on_heightmap.py` script accepts `.tif`, `.png`, `.npy`, and `.csv` files from NASA/ESA archives.

### Where to download Mars DEMs

| Source | Resolution | Format |
|--------|-----------|--------|
| [MOLA (NASA PDS)](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html) | ~463 m/px (global) | GeoTIFF |
| [HRSC (ESA)](https://www.cosmos.esa.int/web/psa) | ~12–50 m/px | GeoTIFF |
| [HiRISE DTMs](https://www.uahirise.org/dtm/) | ~1 m/px (targeted) | GeoTIFF |
| [OpenTopography Mars](https://portal.opentopography.org/) | ~6 m/px (CTX) | GeoTIFF |

### Example usage

```bash
# MOLA global map — plan across Tharsis plateau
uv run python plan_on_heightmap.py mola_tharsis.tif \
    --mpp 463 \
    --crop 800 400 1400 900 \
    --max-slope 15 \
    --rover-radius 200 \
    --max-size 300 \
    --palette mars \
    --out outputs/tharsis_route

# HiRISE 1 m/px DEM — very high resolution local planning
uv run python plan_on_heightmap.py hirise_dem.tif \
    --mpp 1 \
    --start 200 150 \
    --goal 800 600 \
    --max-slope 20 \
    --rover-radius 5 \
    --max-size 400

# Interactive: click start and goal on the map
uv run python plan_on_heightmap.py mars.tif --mpp 463 --pick
```

### All CLI options

```
plan_on_heightmap.py [heightmap] [options]

Arguments:
  heightmap             Path to DEM file (.tif/.png/.npy/.csv). Omit for synthetic demo.

Options:
  --start X Y           Start pixel in the original image (x right, y down)
  --goal  X Y           Goal pixel in the original image
  --pick                Click start and goal interactively (needs a display)
  --mpp   FLOAT         Metres per pixel of the original map (default: 50)
  --height-range LO HI  Map pixel values to this elevation range in metres
  --crop  X0 Y0 X1 Y1   Use only this sub-region of the original map
  --max-size  INT       Downsample so longest side ≤ N cells (default: 150)
  --max-slope FLOAT     Max drivable slope in degrees (default: 20)
  --rover-radius FLOAT  Safety margin around hazards in metres (default: 100)
  --smooth FLOAT        Gaussian blur radius in pixels before planning (default: 1)
  --levels INT          Number of contour lines on the output map (default: 40)
  --palette NAME        Colour palette: blue | green | mars | paper (default: blue)
  --out DIR             Output directory (default: outputs)
```

---

## Key Results (see REPORT.md for full derivation)

| Check | Result |
|-------|--------|
| Solver accuracy | $O(h^2)$ — verified against analytical solution |
| SOR vs Jacobi   | ~66× fewer iterations at $N = 160$ |
| Path vs Dijkstra shortest | +17% (cost of the safety guarantee) |
| Min clearance to real hazard | 2.24 cells |
| Never-stuck test | **400 / 400** random starts reached the goal |
| Trap test (U-shaped dead end) | APF stuck · Laplace ✓ exits and reaches goal |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | All numerical computation |
| `matplotlib` | All figures |
| `pillow` | Loading image-format DEMs (`.tif`, `.png`) |

All installed automatically by `uv sync`.