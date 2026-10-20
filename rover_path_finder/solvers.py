"""
Iterative solvers for Laplace's equation (nabla^2 phi = 0).

All solvers operate on 2-D NumPy arrays and are written entirely from
scratch — no SciPy or other numerical libraries are used.

Functions
---------
omega_opt       : compute the optimal SOR relaxation parameter
jacobi          : Jacobi iteration
sor             : Red-black SOR (omega = 1 → Gauss-Seidel)
sor_neumann     : Red-black SOR with no-flux (Neumann) wall boundary conditions
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg(phi: np.ndarray) -> np.ndarray:
    """Average of the 4 axis-aligned neighbours (5-point finite-difference stencil)."""
    return 0.25 * (
        np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1) + np.roll(phi, -1, 1)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def omega_opt(N: int) -> float:
    """Optimal over-relaxation parameter for an N×N Laplace problem."""
    return 2.0 / (1.0 + np.sin(np.pi / N))


def jacobi(
    phi: np.ndarray,
    fixed: np.ndarray,
    tol: float = 1e-8,
    maxit: int = 500_000,
) -> tuple[np.ndarray, list[float]]:
    """Jacobi iteration for Laplace's equation.

    Parameters
    ----------
    phi   : initial guess (modified in-place on a copy)
    fixed : boolean mask — cells that keep their Dirichlet value
    tol   : convergence tolerance (max absolute change per sweep)
    maxit : maximum number of sweeps

    Returns
    -------
    phi  : converged field
    hist : list of per-sweep max residuals
    """
    phi = phi.copy()
    hist: list[float] = []
    for _ in range(maxit):
        new = _avg(phi)
        new[fixed] = phi[fixed]
        d = float(np.abs(new - phi).max())
        phi = new
        hist.append(d)
        if d < tol:
            break
    return phi, hist


def sor(
    phi: np.ndarray,
    fixed: np.ndarray,
    omega: float = 1.0,
    tol: float = 1e-8,
    maxit: int = 500_000,
) -> tuple[np.ndarray, list[float]]:
    """Red-black successive over-relaxation (SOR).

    ``omega = 1`` is equivalent to Gauss-Seidel.
    Use :func:`omega_opt` to get the optimal relaxation factor.

    Parameters
    ----------
    phi   : initial guess
    fixed : boolean mask of Dirichlet cells
    omega : relaxation factor (1 = Gauss-Seidel, < 2 for convergence)
    tol   : convergence tolerance
    maxit : maximum sweeps

    Returns
    -------
    phi  : converged field
    hist : per-sweep max residuals
    """
    phi = phi.copy()
    n, m = phi.shape
    yy, xx = np.mgrid[0:n, 0:m]
    parity = (xx + yy) % 2
    masks = [(parity == c) & ~fixed for c in (0, 1)]
    hist: list[float] = []
    for _ in range(maxit):
        d = 0.0
        for mk in masks:
            old = phi[mk]
            phi[mk] = (1 - omega) * old + omega * _avg(phi)[mk]
            d = max(d, float(np.abs(phi[mk] - old).max()))
        hist.append(d)
        if d < tol:
            break
    return phi, hist


def sor_neumann(
    phi: np.ndarray,
    fixed: np.ndarray,
    free: np.ndarray,
    omega: float = 1.0,
    tol: float = 1e-10,
    maxit: int = 400_000,
) -> tuple[np.ndarray, list[float]]:
    """Red-black SOR for Laplace's equation with no-flux (Neumann) walls.

    Hazard cells are excluded from every neighbour average, which is the
    finite-difference form of ``∂φ/∂n = 0`` on the hazard boundary.
    ``fixed`` cells retain their Dirichlet value unchanged.

    Parameters
    ----------
    phi   : initial guess
    fixed : boolean mask of Dirichlet cells (e.g. start and target)
    free  : boolean mask of cells that are not hazards
    omega : relaxation factor
    tol   : convergence tolerance
    maxit : maximum sweeps

    Returns
    -------
    phi  : converged field
    hist : per-sweep max residuals
    """
    phi = phi.copy()
    n, m = phi.shape
    f = free.astype(float)
    cnt = sum(np.roll(f, s, a) for a in (0, 1) for s in (1, -1))
    cnt = np.maximum(cnt, 1.0)
    yy, xx = np.mgrid[0:n, 0:m]
    parity = (xx + yy) % 2
    masks = [(parity == c) & free & ~fixed for c in (0, 1)]
    phi[~free] = 0.0
    hist: list[float] = []
    for _ in range(maxit):
        d = 0.0
        for mk in masks:
            pf = phi * f
            avg = (
                np.roll(pf, 1, 0) + np.roll(pf, -1, 0)
                + np.roll(pf, 1, 1) + np.roll(pf, -1, 1)
            ) / cnt
            old = phi[mk]
            phi[mk] = (1 - omega) * old + omega * avg[mk]
            d = max(d, float(np.abs(phi[mk] - old).max()))
        hist.append(d)
        if d < tol:
            break
    return phi, hist
