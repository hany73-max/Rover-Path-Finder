# Safe Path Planning for a Planetary Rover Using Laplace's Equation

**Course:** Partial Differential Equations  
**Subject:** Application of Laplace's Equation to Autonomous Navigation  

---

## 1. Introduction and Motivation

One of the first problems any autonomous rover must solve is: *how do I get from here to there without driving into something dangerous?* For a rover operating on the Moon or Mars, "dangerous" means craters, steep rocky slopes, and large boulders. A wrong move is permanent — there is no recovery crew.

The standard engineering approach is to use a **graph search** (like A* or Dijkstra's algorithm) on a grid map, or a technique called **artificial potential fields (APF)**. Both have well-known failure modes. Graph search can struggle with large maps and does not naturally incorporate safety margins. APF is simple and elegant but has a fatal flaw: the rover can get permanently stuck in a local minimum of the potential field, even when a clear path to the goal exists.

In this project, we explore a different approach: using **Laplace's equation** — the same PDE that describes electrostatic potential, steady heat flow, and fluid streamlines — as the basis for path planning. The mathematical structure of Laplace's equation guarantees, by the **maximum principle**, that the resulting potential field has no interior local minima. This means a rover following the gradient of this field can *never* get stuck. The goal of this project is to implement this idea entirely from scratch using only NumPy, verify the numerical method rigorously, and test it on a synthetic lunar terrain.

---

## 2. Mathematical Background: Laplace's Equation from the Ground Up

This section walks through the mathematics carefully, since the PDE and its numerical treatment are the core of the project.

### 2.1 What is Laplace's Equation?

Laplace's equation is a second-order linear PDE. For a scalar field $\phi$ defined over a 2D domain, it is written as:

$$
\nabla^2\phi = 0
$$

Expanding the Laplacian operator $\nabla^2$ in Cartesian coordinates gives us the explicit form:

$$
\frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} = 0
$$

This equation says: at every point in the domain, the **sum of the second partial derivatives is zero**. A function satisfying this equation everywhere in a domain is called a **harmonic function**.

To understand why this is useful, it helps to have an intuition for what the second derivative means. Recall that the first derivative $\frac{\partial\phi}{\partial x}$ measures *how fast $\phi$ changes* in the $x$-direction. The second derivative $\frac{\partial^2\phi}{\partial x^2}$ measures *how fast that rate of change is itself changing* — it measures **curvature**. When this is positive, the function is concave-up (bowl shape) in that direction. When negative, concave-down.

Laplace's equation says: **the concavity in $x$ and the concavity in $y$ must cancel out exactly**. This is a very strong constraint. It rules out the function being bowl-shaped in both $x$ and $y$ simultaneously — which means it cannot form a local minimum or maximum in the interior of the domain.

### 2.2 Boundary Conditions

Laplace's equation on its own is underdetermined — there are infinitely many harmonic functions. To get a unique solution, we must specify **boundary conditions** on the edges of the domain. There are two classical types:

**Dirichlet boundary conditions** fix the *value* of $\phi$ on a boundary:

$$
\phi = g(x, y) \quad \text{on boundary } \Gamma_D
$$

For example, "the potential at the target is $\phi = 0$" is a Dirichlet condition.

**Neumann boundary conditions** fix the *normal derivative* of $\phi$ on a boundary:

$$
\frac{\partial\phi}{\partial n} = h(x, y) \quad \text{on boundary } \Gamma_N
$$

where $\mathbf{n}$ is the outward-pointing unit normal vector. The special case $h = 0$ is called the **no-flux** or **insulating** condition — it says the field does not flow through that boundary at all. This is the one we use for hazard cells.

A problem that mixes both types is called a **mixed boundary value problem**, and that is exactly what we have here.

### 2.3 Why Harmonic Functions Cannot Have Interior Extrema

The reason Laplace's equation is perfect for path planning is the **Maximum Principle**, which states:

> *A non-constant harmonic function on a bounded connected domain cannot attain its maximum or minimum value at an interior point. The extrema must occur on the boundary.*

We can give an informal proof. Suppose $\phi$ has a local minimum at some interior point $P$. By definition of a local minimum, the function curves *upward* in every direction away from $P$. That means both $\frac{\partial^2\phi}{\partial x^2} \ge 0$ and $\frac{\partial^2\phi}{\partial y^2} \ge 0$ at $P$ (with at least one being strictly positive). But then $\frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} > 0$, which contradicts Laplace's equation. Therefore, no interior minimum can exist.

The implication for rover navigation is profound: if we set up $\phi$ so that the **target is the global minimum** ($\phi = 0$ at the goal), then the only minimum in the whole domain is the goal itself. A rover that always moves toward lower values of $\phi$ will therefore always eventually reach the goal — it can never get stuck.

### 2.4 The Two Formulations We Tried

#### Formulation A — Dirichlet on Hazards (Failure)

Our first attempt placed Dirichlet conditions everywhere:
- **Hazard cells:** $\phi = 1$ (high potential, repulsive)
- **Target:** $\phi = 0$ (low potential, attractive)
- **Free cells:** solve $\nabla^2\phi = 0$

This seems reasonable, but it fails on real terrain. Consider a narrow corridor between two hazard walls. The potential at the walls is fixed at 1. By the maximum principle, $\phi$ inside the corridor must be between 0 and 1. But it turns out that $\phi$ decays from 1 *exponentially* as you move away from the wall — roughly like $e^{-\pi d / w}$ where $d$ is the distance from the wall and $w$ is the corridor width. For narrow corridors, this decay is so sharp that by the time you are even one cell away from the wall, the gradient has already dropped below floating-point precision. The field looks completely flat, and the rover has no gradient to follow. This formulation simply does not work on complex terrain.

#### Formulation B — Neumann on Hazards (Success)

The fix is to change the boundary condition on hazards from Dirichlet to **Neumann**:
- **Hazard cells and map edges:** $\frac{\partial\phi}{\partial n} = 0$ (no flux through walls)
- **Start cell:** $\phi = 1$ (Dirichlet source)
- **Target cell:** $\phi = 0$ (Dirichlet sink)

Now the hazards are "insulating walls" rather than voltage sources. The field is not forced to a constant value near the walls — it is free to vary smoothly. Current (the gradient $\nabla\phi$) simply cannot flow *through* a hazard, but it flows freely *around* them. This produces a smooth, well-conditioned field throughout the whole free region, even in narrow passages.

### 2.5 Finite Difference Discretization

To solve this PDE numerically, we discretize the domain into a uniform grid with spacing $h$ in both $x$ and $y$. Let $\phi_{i,j}$ denote the value of $\phi$ at grid point $(i, j)$.

We approximate the second derivatives using the standard **central difference formulas**. These come from Taylor expanding $\phi$ around the point $(i, j)$:

$$
\phi(x+h) = \phi(x) + h\cdot\phi'(x) + \frac{h^2}{2}\phi''(x) + \frac{h^3}{6}\phi'''(x) + \dots
$$

$$
\phi(x-h) = \phi(x) - h\cdot\phi'(x) + \frac{h^2}{2}\phi''(x) - \frac{h^3}{6}\phi'''(x) + \dots
$$

Adding these two expansions:

$$
\phi(x+h) + \phi(x-h) = 2\phi(x) + h^2\cdot\phi''(x) + O(h^4)
$$

Solving for $\phi''(x)$:

$$
\phi''(x) \approx \frac{\phi(x+h) - 2\phi(x) + \phi(x-h)}{h^2} + O(h^2)
$$

This approximation is **second-order accurate** — the error is proportional to $h^2$. Applying this to both $x$ and $y$ on our grid:

$$
\frac{\partial^2\phi}{\partial x^2} \approx \frac{\phi_{i,j+1} - 2\phi_{i,j} + \phi_{i,j-1}}{h^2}
$$

$$
\frac{\partial^2\phi}{\partial y^2} \approx \frac{\phi_{i+1,j} - 2\phi_{i,j} + \phi_{i-1,j}}{h^2}
$$

Setting their sum to zero (Laplace's equation) and multiplying through by $h^2$:

$$
(\phi_{i,j+1} - 2\phi_{i,j} + \phi_{i,j-1}) + (\phi_{i+1,j} - 2\phi_{i,j} + \phi_{i-1,j}) = 0
$$

Rearranging:

$$
4\phi_{i,j} = \phi_{i+1,j} + \phi_{i-1,j} + \phi_{i,j+1} + \phi_{i,j-1}
$$

Or equivalently:

$$
\phi_{i,j} = \frac{1}{4} \left( \phi_{i+1,j} + \phi_{i-1,j} + \phi_{i,j+1} + \phi_{i,j-1} \right)
$$

This is the **5-point stencil**: the value at each grid point equals the **average of its four axis-aligned neighbours**. This has a beautiful interpretation — a harmonic function satisfies the **mean value property**: its value at any point equals the average over any circle (or, on a grid, over its neighbours) centered at that point.

For Neumann conditions (no-flux walls), when a neighbour cell is a hazard, we simply **drop that neighbour from the average** and divide by however many free neighbours remain. This is the correct finite-difference implementation of $\frac{\partial\phi}{\partial n} = 0$.

### 2.6 Iterative Solvers

The discretized system gives us one equation per free grid cell, and each equation couples a cell to its neighbours. For a 100×100 grid, this is a system of up to 10,000 coupled linear equations. Direct solvers (like Gaussian elimination) are expensive. Instead, we use **iterative methods** that start from an initial guess and repeatedly update each cell until convergence.

**Jacobi Method**

The simplest approach. At each sweep, update every free cell simultaneously using values from the *previous* sweep:

$$
\phi_{i,j}^{(k+1)} = \frac{1}{4} \left( \phi_{i+1,j}^{(k)} + \phi_{i-1,j}^{(k)} + \phi_{i,j+1}^{(k)} + \phi_{i,j-1}^{(k)} \right)
$$

Jacobi is simple to implement but converges slowly — roughly $O(N^2)$ iterations for an $N \times N$ grid.

**Gauss-Seidel Method**

A simple improvement: use the most recently computed values as soon as they are available. When updating cell $(i,j)$, use the already-updated values of $(i-1,j)$ and $(i,j-1)$:

$$
\phi_{i,j}^{(k+1)} = \frac{1}{4} \left( \phi_{i+1,j}^{(k)} + \phi_{i-1,j}^{(k+1)} + \phi_{i,j+1}^{(k)} + \phi_{i,j-1}^{(k+1)} \right)
$$

Gauss-Seidel converges about twice as fast as Jacobi for Laplace's equation, since it incorporates new information faster within each sweep.

**Successive Over-Relaxation (SOR)**

SOR takes Gauss-Seidel one step further by **over-shooting** in the direction of the correction:

$$
\phi_{i,j}^{(k+1)} = (1 - \omega)\phi_{i,j}^{(k)} + \omega \cdot (\text{Gauss-Seidel update})
$$

where $\omega \in (1, 2)$ is the **relaxation factor**. The intuition is: if we know we need to move $\phi$ in a certain direction, why not move it a little further to make up for future corrections? The optimal value is:

$$
\omega_{\text{opt}} = \frac{2}{1 + \sin(\pi / N)}
$$

For $N = 100$, this gives $\omega \approx 1.94$. With this optimal $\omega$, SOR converges in $O(N)$ iterations — far better than the $O(N^2)$ of Jacobi and Gauss-Seidel.

We implement SOR using a **red-black ordering**: we split the grid cells into two interleaved sets (like a checkerboard pattern) and update them alternately. This allows a vectorized NumPy implementation that updates all cells of one colour simultaneously.

---

## 3. Terrain Model

We built a synthetic 100×100 grid elevation map $h(x, y)$ to test the planner. The terrain features:

- **Six craters**, each modelled as a Gaussian bowl with a raised rim (a common morphology for impact craters)
- **14 random boulders**, each modelled as a circular obstacle
- **Steep slopes**, detected by computing $|\nabla h|$ numerically and marking any cell with slope > 0.8 as a hazard

To keep the rover safe, every hazard is **inflated by 2 cells** using a morphological dilation operation. This means the planned path will always keep at least 2 cells of clearance from any real obstacle — roughly matching the rover's own footprint.

After inflation, 2,787 of the 10,000 cells are classified as hazard (compared to 1,773 before inflation). The start is placed at (6, 6) and the goal at (92, 90) — diagonally opposite corners of the map.

---

## 4. Implementation Details

All numerical code is written from scratch in Python using only NumPy. No scientific computing libraries (SciPy, scikit-learn, etc.) were used.

The code is organized into four modules inside the `rover_path_finder/` package:

- **`solvers.py`** — Jacobi, Gauss-Seidel, SOR, and the Neumann-SOR variant
- **`terrain.py`** — terrain generation, loading external DEMs, dilation, slope computation
- **`planners.py`** — potential field builders, steepest descent path extractor, Dijkstra reference
- **`visualization.py`** — all figures

The Neumann condition is implemented as follows: for each free cell, we precompute the count of free neighbours. During the SOR update, we replace the standard 4-neighbour average with a weighted average over *only the free neighbours*, dividing by that precomputed count. This is mathematically equivalent to setting the flux through hazard faces to zero.

---

## 5. Verification and Results

### 5.1 Validation Against an Exact Solution

Before trusting the solver on the terrain problem, we verified its accuracy against a known analytical solution. We used the unit square with boundary conditions:

$$
\phi = \sin(\pi x) \quad \text{on top edge}, \quad \phi = 0 \quad \text{on other three edges.}
$$

The exact solution to Laplace's equation for this problem is:

$$
\phi(x, y) = \frac{\sin(\pi x) \cdot \sinh(\pi y)}{\sinh(\pi)}
$$

We solved the same problem numerically at four grid resolutions and measured the maximum error:

| N   | Max error   | Ratio (prev/current) |
|-----|-------------|----------------------|
| 16  | $1.109 \times 10^{-3}$ | — |
| 32  | $2.780 \times 10^{-4}$ | 3.99 |
| 64  | $6.963 \times 10^{-5}$ | 3.99 |
| 128 | $1.741 \times 10^{-5}$ | 4.00 |

Each time $N$ doubles ($h$ halves), the error drops by a factor of 4 — confirming **second-order convergence $O(h^2)$**, exactly as expected from the central-difference formula we derived in Section 2.5.

### 5.2 Solver Comparison

We compared the three solvers on the same unit-square problem at increasing grid sizes, counting how many iterations each needs to reach a tolerance of $10^{-6}$:

| N   | Jacobi  | Gauss-Seidel | SOR  |
|-----|---------|--------------|------|
| 20  | 669     | 364          | 47   |
| 40  | 2,233   | 1,230        | 88   |
| 80  | 7,136   | 4,019        | 167  |
| 160 | 21,356  | 12,477       | 322  |

Jacobi and Gauss-Seidel both grow roughly like $N^2$ (doubling $N$ roughly quadruples the iteration count). Gauss-Seidel needs about half as many iterations as Jacobi. SOR with the optimal $\omega$ grows only like $N$ — far superior. At $N = 160$, SOR needs about 66× fewer iterations than Jacobi.

### 5.3 Rover Path Planning Results

**Formulation A (Dirichlet hazards):** As predicted in Section 2.4, this fails. After convergence, $\phi$ over all free cells sits in the range $[1.000000000000, 1.000000000003]$. The gradient is completely undetectable — steepest descent immediately stalls.

**Formulation B (Neumann hazards):** The solver converged in 8,826 SOR iterations, taking about 4.2 seconds. The resulting path successfully reaches the goal with:
- Path length: **153.5 cells**
- Shortest possible path (Dijkstra): **131.3 cells** → **17% longer**
- Minimum clearance to any real hazard: **2.24 cells**

The 17% path length overhead is the cost of the safety guarantee. The planner prioritises *certainty of success* over shortest distance.

**Classic APF comparison:** The classic artificial potential field fails to reach the goal on the same map, getting stuck in a local minimum as expected.

**Never-stuck test:** We ran steepest descent on the Formulation B field starting from 400 random positions across the reachable free area. Every single attempt reached the goal. (14 cells are in fully enclosed crater pockets that no planner of any kind can escape — they are excluded from the test.)

### 5.4 Trap Test

We constructed a worst-case scenario for APF: a U-shaped dead end with the rover starting *inside* the U, and the goal placed on the far side of the closed back wall. The classic APF planner gets trapped at cell (52, 50) and can never escape. The Laplace planner correctly identifies the open side of the U, exits, and reaches the goal.

---

## 6. Limitations and Honest Reflections

No project is perfect, and it is important to be upfront about where this approach falls short:

1. **Binary hazard model.** Slopes are either safe or forbidden — there is no graded cost for "steeper but still drivable" terrain. A more realistic model would penalize steep slopes continuously.

2. **Path length overhead.** The Laplace path is 17% longer than optimal. For real missions where battery life matters, this may be unacceptable. One could combine Laplace planning with a post-processing step to shorten the path.

3. **Static, fully-known map.** The field is computed once and assumes the terrain is fully known and does not change. For a real rover encountering unexpected rocks, the field would need to be recomputed — which takes a few seconds on a 100×100 grid.

4. **No rover dynamics.** The planned path is a sequence of grid cells. A real rover has a turning radius, limited wheel torque on slopes, and suspension travel. These are not modelled.

5. **Jagged discrete paths.** 8-neighbour grid paths have visible staircase artifacts. A path-smoothing step (e.g., Bézier fitting) would be needed before sending commands to actual actuators.

6. **The Formulation A failure analysis.** We explained the exponential decay of the gradient near narrow corridors as a qualitative argument. A more rigorous treatment would require solving a 1D boundary layer problem analytically for a canonical corridor geometry.

---

## 7. Running the Code

### Setup

```powershell
# Clone the repository and install dependencies
uv sync
```

### Run the full lunar demo (reproduces all figures and numbers from this report)

```powershell
uv run python main.py
# Output saved to: outputs/
#   fig1_terrain.png               — elevation, slope, and hazard map
#   fig2_solution_and_path.png     — potential fields (A vs B) and the rover path
#   fig3_validation_and_solvers.png — convergence order and solver comparison
#   fig4_trap_comparison.png       — APF stuck vs Laplace success
#   results.txt                    — all printed numbers
```

### Run the height-map planner on the built-in synthetic Mars terrain

```powershell
uv run python plan_on_heightmap.py
# Output saved to: outputs/
#   map_contours.png   — topographic contour map
#   map_route.png      — route overlaid on the contour map
#   route_profile.png  — elevation and slope profiles along the route
#   route.csv          — waypoint table (x, y, elevation, distance)
```

### Run on a real Mars DEM (HiRISE .IMG — pixel size read automatically)

First, generate a ticked map so you can read off start/goal pixel coordinates:

```powershell
# Look pass — produces map_contours.png with x/y pixel labels on the axes
uv run python plan_on_heightmap.py data/kasei.IMG `
    --ticks --palette mars --out outputs/look
```

Then plan the route using the coordinates you read from the map:

```powershell
# Kasei Valles constriction — needs 25° limit (channel walls ~25°)
uv run python plan_on_heightmap.py data/kasei.IMG `
    --start 250 500 `
    --goal 2400 5800 `
    --max-slope 25 `
    --rover-radius 20 `
    --palette mars `
    --out outputs/kasei
```

**Actual output from this run:**
```
map: data/kasei.IMG | 3164x6422 px of 2.02 m -> planning grid 63x128 cells of 101 m
     relief -2126..-1059 m | 24% no-data
hazard cells: 1567 real, 1567 with margin, of 8064
route found: 107 cells | length 14.42 km (shortest safe 12.50 km, +15%)
             max slope on route 24.9° (limit 25.0) | climb 651 m, descent 935 m
             solver 10666 iters, 2.7s
```

See Section 8 for details on obtaining real Mars elevation data.

---

## 8. Using Real Mars Height Maps

The planner fully supports real planetary Digital Elevation Models (DEMs). The `plan_on_heightmap.py` script accepts **PDS3 `.IMG` files** (HiRISE, MOLA, LROC) directly — the pixel size is read automatically from the label. It also accepts `.tif`, `.png`, `.npy`, and `.csv`.

### Where to get Mars DEMs

| Source | Resolution | Format | Link |
|--------|-----------|--------|------|
| **MOLA** (Mars Orbiter Laser Altimeter) | ~463 m/px globally | GeoTIFF | [NASA PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html) |
| **HRSC** (Mars Express High Resolution Stereo Camera) | ~12–50 m/px | GeoTIFF | [ESA Planetary Science Archive](https://www.cosmos.esa.int/web/psa) |
| **HiRISE DEMs** (Mars Reconnaissance Orbiter) | ~1–2 m/px (targeted areas) | PDS3 `.IMG` | [HiRISE DTM portal](https://www.uahirise.org/dtm/) |
| **CTX DEMs** | ~6 m/px (many areas) | GeoTIFF | [OpenTopography Mars](https://portal.opentopography.org/) |

### Practical example — Kasei Valles (HiRISE DTM, 2 m/px)

```powershell
# Download the DEM into data/ (~78 MB)
curl.exe -L -o data/kasei.IMG `
    "https://www.uahirise.org/PDS/DTM/ESP/ORB_032100_032199/ESP_032193_1990_ESP_031837_1990/DTEED_032193_1990_031837_1990_A01.IMG"

# Step 1: look pass — generates a pixel-labelled contour map so you can pick coordinates
uv run python plan_on_heightmap.py data/kasei.IMG `
    --ticks --palette mars --out outputs/look

# Step 2: plan the route (25° needed — channel walls exceed 20°)
uv run python plan_on_heightmap.py data/kasei.IMG `
    --start 250 500 `
    --goal 2400 5800 `
    --max-slope 25 `
    --rover-radius 20 `
    --palette mars `
    --out outputs/kasei
```

### Key parameters for real DEMs

| Parameter | What it does | Typical value |
|-----------|-------------|---------------|
| `--mpp` | Metres per pixel of the *original* image (auto-read from `.IMG`) | 463 (MOLA), 6 (CTX), 1–2 (HiRISE) |
| `--ticks` | Label contour map axes in original pixels to read off start/goal | Use on first run |
| `--max-slope` | Maximum drivable slope in degrees | 15–25° for wheeled rovers |
| `--rover-radius` | Safety clearance around hazards in metres | 5–500 m depending on scale |
| `--max-size` | Downsample to this many cells on longest side | 150–400 (trades accuracy vs speed) |
| `--crop X0 Y0 X1 Y1` | Use only a sub-region of the original map | Needed for very large DEMs |
| `--out DIR` | Output directory for figures and route.csv | e.g. `outputs/kasei` |
| `--pick` | Click start/goal interactively on the map | Useful for exploration |

### Why the same algorithm works at any scale

The Laplace equation itself has no intrinsic length scale — it is scale-invariant. Whether the grid cells represent 1 metre or 10 kilometres, the same finite-difference stencil applies, and the same maximum principle guarantees the rover cannot get stuck. The only thing that changes with scale is the number of SOR iterations needed (O(N)), which grows modestly with grid size.

---

## 9. Conclusion

This project demonstrates that Laplace's equation — a classical tool from mathematical physics — provides an elegant and provably correct solution to the rover path-planning problem. The key insight is that the PDE structure itself encodes the safety guarantee: harmonic functions cannot have interior local minima, so a rover following the negative gradient can never get stuck.

The numerical implementation confirmed second-order convergence of the finite-difference scheme and showed that SOR with the optimal relaxation parameter converges roughly 66× faster than Jacobi at the tested grid sizes. The planner successfully navigated a complex synthetic lunar terrain and passed all correctness tests, including the classic U-shaped trap that defeats standard potential field methods.

The extension to real Mars DEMs is straightforward — the same code handles any height-map file, at any resolution, with appropriate scaling of the physical parameters.
