"""
Safe lunar-rover path planning with Laplace's equation (∇²φ = 0).

Everything numerical is written from scratch with NumPy only — no SciPy,
no ML libraries.

Run
---
    python main.py [output_dir]

Reproduces every number and figure from the report and saves them to
``output_dir`` (default: ``outputs/``).
"""

import os
import sys
import numpy as np

from rover_path_finder import solvers, terrain, planners, visualization as viz

LOG: list[str] = []


def log(s: str) -> None:
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------------------
# Validation against an exact analytical solution
# ---------------------------------------------------------------------------

def exact_problem(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-square Laplace problem with known exact solution.

    Boundary condition: φ = sin(πx) on the top edge, 0 elsewhere.
    Exact solution: sin(πx) · sinh(πy) / sinh(π).
    """
    x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x)
    exact = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)
    phi = np.zeros_like(exact)
    fixed = np.zeros_like(exact, bool)
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    phi[-1, :] = np.sin(np.pi * x)
    return phi, fixed, exact


def run_validation() -> tuple[list[int], list[float]]:
    """Validate the SOR solver against the exact solution at 4 grid sizes."""
    Ns = [16, 32, 64, 128]
    errs: list[float] = []
    log("== Validation against the exact solution (Laplace on the unit square) ==")
    for N in Ns:
        phi, fixed, exact = exact_problem(N)
        sol, _ = solvers.sor(phi, fixed, solvers.omega_opt(N), tol=1e-12)
        errs.append(float(np.abs(sol - exact).max()))
        ratio = f"  (ratio to previous = {errs[-2] / errs[-1]:.2f})" if len(errs) > 1 else ""
        log(f"N={N:4d}  max error = {errs[-1]:.3e}{ratio}")
    return Ns, errs


# ---------------------------------------------------------------------------
# Solver convergence study
# ---------------------------------------------------------------------------

def run_convergence_study() -> tuple[list[int], dict[str, list[int]]]:
    """Compare Jacobi, Gauss-Seidel, and SOR iteration counts."""
    Ns = [20, 40, 80, 160]
    res: dict[str, list[int]] = {"Jacobi": [], "Gauss-Seidel": [], "SOR": []}
    log("== Iterations needed to reach tol=1e-6 (same problem, different grid sizes) ==")
    for N in Ns:
        phi, fixed, _ = exact_problem(N)
        _, h1 = solvers.jacobi(phi, fixed, tol=1e-6)
        _, h2 = solvers.sor(phi, fixed, 1.0, tol=1e-6)
        _, h3 = solvers.sor(phi, fixed, solvers.omega_opt(N), tol=1e-6)
        res["Jacobi"].append(len(h1))
        res["Gauss-Seidel"].append(len(h2))
        res["SOR"].append(len(h3))
        log(f"N={N:4d}  Jacobi={len(h1):6d}  Gauss-Seidel={len(h2):6d}  SOR={len(h3):5d}")
    return Ns, res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: str = "outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Numerics validation
    Nv, ev = run_validation()
    Nc, res = run_convergence_study()
    viz.fig_numerics(Nv, ev, Nc, res, out_dir)

    # 2. Synthetic lunar terrain
    T = terrain.make_terrain()
    hz = T["hazard"]
    st, gl = T["start"], T["goal"]
    log("\n== Terrain / planning ==")
    log(
        f"hazard cells (real): {int(T['raw'].sum())}  "
        f"with rover-size margin: {int(hz.sum())} of {hz.size}"
    )
    viz.fig_terrain(T, out_dir)

    # 3. Formulation A (Dirichlet — expected to fail)
    phiA, itA, _ = planners.dirichlet_potential(hz, gl)
    pathA, okA = planners.descend(phiA, hz, st, gl)
    log(
        f"Formulation A (hazards phi=1): {itA} iterations | reached goal: {okA} | "
        f"phi range over free cells = [{phiA[~hz].min():.2e}, {phiA[~hz].max():.12f}]"
        f"  -> essentially flat, no gradient to follow"
    )

    # 4. Formulation B (Neumann — main method)
    phiB, itB, sB = planners.neumann_potential(hz, st, gl)
    pathB, okB = planners.descend(phiB, hz, st, gl)
    L, clr = planners.path_stats(pathB, T["raw"])
    Lmin = planners.dijkstra_length(hz, st, gl)
    log(
        f"Formulation B (no-flux walls): {itB} iterations, {sB:.1f}s | "
        f"reached goal: {okB} | path length {L:.1f} vs shortest possible {Lmin:.1f} "
        f"({100 * (L / Lmin - 1):.0f}% longer) | "
        f"min clearance to a real hazard {clr:.2f} cells"
    )
    viz.fig_solution(T, phiA, pathA, phiB, pathB, out_dir)

    # 5. Classic APF comparison
    Pa = planners.apf_potential(hz, gl)
    _, okp = planners.descend(Pa, hz, st, gl)
    log(f"Classic potential field on the same map: reached goal: {okp}")

    # 6. "Never-stuck" test — 400 random starting cells
    reach = planners.reachable(hz, gl)
    pockets = int((~hz).sum() - reach.sum())
    rng = np.random.default_rng(1)
    fy, fx = np.nonzero(reach)
    idx = rng.choice(len(fy), 400, replace=False)
    good = sum(
        planners.descend(phiB, hz, (int(fx[i]), int(fy[i])), gl)[1]
        for i in idx
    )
    log(
        f"Never-stuck test: steepest descent on phi_B reached the target from "
        f"{good} of 400 random reachable cells "
        f"({pockets} free cells sit in enclosed pockets and are unreachable by any planner)"
    )

    # 7. Trap test (U-shaped dead end)
    log("\n== Trap test (U-shaped dead end; start inside the U, target behind its closed back wall) ==")
    hzt, s2, g2 = planners.trap_map()
    Pa2 = planners.apf_potential(hzt, g2)
    pa, oka = planners.descend(Pa2, hzt, s2, g2)
    ph, _, _ = planners.neumann_potential(hzt, s2, g2)
    pl, okl = planners.descend(ph, hzt, s2, g2)
    log(f"Classic potential field: reached goal = {oka} (stuck at {pa[-1]})")
    log(f"Laplace (formulation B): reached goal = {okl} (path length {planners.path_stats(pl, hzt)[0]:.1f} cells)")
    viz.fig_trap(hzt, s2, g2, Pa2, pa, oka, ph, pl, okl, out_dir)

    # 8. Save log
    results_path = os.path.join(out_dir, "results.txt")
    with open(results_path, "w") as fh:
        fh.write("\n".join(LOG))
    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    main(out_dir)
