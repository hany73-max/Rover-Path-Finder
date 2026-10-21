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

### For Non-Technical Readers: The Water Flow Analogy 🌊

If you don't care about the partial differential equations, think of this method as **water flowing across a landscape**:

1. **Imagine the start point is a water hose** continuously pumping water into the map.
2. **The goal is an open drain** pulling water out.
3. **Obstacles (cliffs, steep crater walls) are waterproof dams** that water cannot cross.
4. **The rover is a small floating rubber duck.** Once the water fills the maze and begins flowing smoothly from the hose to the drain, you drop the duck at the start. The duck simply floats along the water current until it arrives safely at the goal.

#### Why does it guarantee the rover never gets stuck?
In traditional video-game or robotics navigation (APF), a rover is like a marble rolling down a bumpy hill — it easily gets trapped in small potholes, dead ends, or U-shaped crater pockets. With Laplace's equation, **there are no potholes**. Water always flows toward the lowest point (the drain). By mathematical law (the Maximum Principle), water currents cannot form stagnant pools inside the terrain.

#### Why is the path NOT the shortest path? (Why does it take wide turns or dive into craters?)
* **Water spreads out:** Water doesn't travel like a laser beam or the shortest geometric line. It spreads across the whole plain to reduce pressure. As a result, the paths take wide, gentle curves instead of hugging obstacle corners.
* **The equation only sees 2D slope limits, not elevation:** The math only asks: *"Is this slope under 20 degrees?"* If a giant crater has a gentle ramp into it, water flows right through the bottom of the crater and out the other side rather than walking the long way around the rim!
* **Safety over speed:** You trade off distance optimality (paths are typically 15% to 68% longer than the absolute shortest route) in exchange for a **100% guarantee that the rover will never get stuck in a trap**.

---

## Project Structure

```
Rover-Path-Finder/
│
├── rover_path_finder/          ← Python package (all the real math lives here)
│   ├── __init__.py
│   ├── solvers.py              ← Jacobi, Gauss-Seidel, SOR, Neumann-SOR
│   ├── terrain.py              ← terrain generation, DEM loading, PDS3 reader, dilation
│   ├── planners.py             ← potential builders, steepest descent, Dijkstra
│   └── visualization.py       ← all Matplotlib figures
│
├── main.py                     ← CLI: runs the full lunar demo (reproduces REPORT.md)
├── plan_on_heightmap.py        ← CLI: plan a route on any height map (.img/.tif/.png/…)
├── run_all.ps1                 ← reproduces all real-DEM runs end-to-end
│
├── data/                       ← downloaded DEM files (git-ignored)
│   ├── kasei.IMG               ← HiRISE DTM, Kasei Valles, 2 m/px (~78 MB)
│   └── opportunity.IMG         ← HiRISE DTM, Opportunity site, 1 m/px (~9 MB)
│
├── outputs/                    ← generated figures & logs (git-ignored)
│   ├── kasei/                  ← Kasei Valles route plan
│   ├── opportunity/            ← Opportunity site route plan
│   └── look/                   ← ticked look-pass maps (for picking coordinates)
│
├── pyproject.toml              ← project metadata + dependencies (uv-managed)
├── uv.lock                     ← locked dependency versions
├── REPORT.md                   ← full project write-up (theory, derivation, results)
└── small_mars_dtms.csv         ← catalogue of downloadable HiRISE DEMs
```

---

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) (fast Python package manager).

```powershell
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

**Yes — the planner works on real Mars elevation data.** The `plan_on_heightmap.py` script reads **PDS3 `.IMG` files** (HiRISE, MOLA) directly with automatic pixel-size detection, and also accepts `.tif`, `.png`, `.npy`, `.csv`.

### Where to download Mars DEMs

| Source | Resolution | Format |
|--------|-----------|--------|
| [MOLA (NASA PDS)](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html) | ~463 m/px (global) | GeoTIFF |
| [HRSC (ESA)](https://www.cosmos.esa.int/web/psa) | ~12–50 m/px | GeoTIFF |
| [HiRISE DTMs](https://www.uahirise.org/dtm/) | ~1–2 m/px (targeted) | PDS3 `.IMG` |
| [OpenTopography Mars](https://portal.opentopography.org/) | ~6 m/px (CTX) | GeoTIFF |

### Two-step workflow for `.IMG` files

```powershell
# Step 1: look pass — generates a pixel-labelled contour map
#         so you can read off --start / --goal coordinates
uv run python plan_on_heightmap.py data/kasei.IMG `
    --ticks --palette mars --out outputs/look

# Step 2: plan the route with the coordinates you picked
uv run python plan_on_heightmap.py data/kasei.IMG `
    --start 250 500 `
    --goal 2400 5800 `
    --max-slope 25 `
    --rover-radius 20 `
    --palette mars `
    --out outputs/kasei
```

**Actual result on Kasei Valles (HiRISE DTM, 2 m/px, ~78 MB):**
```
map: 3164x6422 px of 2.02 m → 63x128 cells of 101 m | 24% no-data
route found: 107 cells | 14.42 km | max slope 24.9° | climb 651 m | 2.7 s
```

> **Tip:** Run `.\ run_all.ps1` to reproduce all runs automatically, including downloading the DEMs.

### Interactive Testing: How to Pick Start & Goal Coordinates on a Map 🗺️

Want to pick custom start and goal coordinates and see how the rover navigates? Here is the easiest way:

#### Step 1: Generate the reference map with coordinate tick marks
Run the planner with `--ticks` to print pixel coordinates directly on the axes:

```powershell
uv run python plan_on_heightmap.py data/opportunity.IMG --ticks --palette mars --out outputs/my_map
```

Open `outputs/my_map/map_contours.png`. You will see grid marks with numbers:
* **X axis (horizontal):** reads from `0` on the left to `1260` on the right.
* **Y axis (vertical):** reads from `0` at the top to `1680` at the bottom.
* **The center ring:** The dense rings of contour lines in the middle are the steep walls of the crater. The wide open space around it is the flat Martian plain.

#### Step 2: Run your test with your chosen coordinates
Pass your chosen `(X, Y)` for `--start` and `--goal`:

```powershell
uv run python plan_on_heightmap.py data/opportunity.IMG `
    --start 250 1500 `
    --goal 1100 200 `
    --ticks `
    --palette mars `
    --out outputs/my_test
```

#### Step 3: Check your results
Inside `outputs/my_test/`, you will find:
* `map_route.png`: Shows the rover's path (yellow line) from start (green dot) to goal (red star).
* `route_profile.png`: Graphs showing the elevation changes and slopes along the route.
* `route.csv`: Exact coordinates, distances, and elevations along the planned path.

#### Example scenarios to test:
1. **The Crater Crossing:** Start at `250 1500` (bottom-left) and Goal at `1100 200` (top-right). Watch whether it chooses to drive across the crater floor or skirt the rim.
2. **The Crater Escape (Trap Test):** Start *inside* the crater bottom at `600 850` and Goal outside at `1100 200`. Watch the solver find a drivable breach in the crater wall to climb out.
3. **Plains Highway:** Start at `1100 1500` and Goal at `1100 200` to see it cruise along the flat plains.

### All CLI options

```
plan_on_heightmap.py [heightmap] [options]

Arguments:
  heightmap             Path to DEM file (.img/.tif/.png/.npy/.csv). Omit for synthetic demo.

Options:
  --start X Y           Start pixel in the original image (x right, y down)
  --goal  X Y           Goal pixel in the original image
  --pick                Click start and goal interactively (needs a display)
  --ticks               Label contour map axes in original pixels (to pick start/goal)
  --mpp   FLOAT         Metres per pixel (auto-read from .img labels; default: 50)
  --height-range LO HI  Map pixel values to this elevation range in metres
  --crop  X0 Y0 X1 Y1   Use only this sub-region of the original map
  --max-size  INT       Downsample so longest side ≤ N cells (default: 150)
  --art-size  INT       Max cells for contour-art pass (default: 700)
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