"""
secular.py
-----------
Module for computing secular perturbation matrices (eccentricity and inclination).
Results are converted to degrees per year (deg/yr), as in Murray & Dermott
eqs. (7.34) and (7.35). Includes diagonalization and matching of eigenvectors
to the conventions of eq. (7.42).
"""

import numpy as np
from scipy.integrate import quad
import math

# Conversion factors
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_ARCSEC = 3600.0


def laplaceCoefficient(s: float, j: int, alpha: float) -> float:
    """Compute Laplace coefficient b_s^(j)(alpha)."""
    def integrand(psi):
        denom = (1 - 2 * alpha * math.cos(psi) + alpha**2)
        return math.cos(j * psi) / (denom ** s)

    result, _ = quad(
        integrand,
        0.0,
        2 * math.pi,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=500,
    )
    return result / math.pi


def meanMotion(G: float, M0: float, a: float, m_planet: float = 0.0) -> float:
    """Mean motion n = sqrt(G*(M0 + m_planet)/a^3) [rad/yr]."""
    return math.sqrt(G * (M0 + m_planet) / (a ** 3))


def computeMatrixA(planets, constants, toDeg: bool = True) -> np.ndarray:
    """
    Compute secular matrix A (eccentricity terms).
    Implements Eqs. (7.9)–(7.12) from Murray & Dermott.
    """
    G = constants["G"]
    M0 = constants["M0"]

    p1, p2 = planets.iloc[0], planets.iloc[1]

    m1, a1 = float(p1["mass"]), float(p1["a"])
    m2, a2 = float(p2["mass"]), float(p2["a"])

    n1 = meanMotion(G, M0, a1, m1)
    n2 = meanMotion(G, M0, a2, m2)

    alpha = a1 / a2

    b32_1 = laplaceCoefficient(1.5, 1, alpha)
    b32_2 = laplaceCoefficient(1.5, 2, alpha)

    # barAlpha factors: inner planet gets alpha, outer gets 1
    barAlpha_inner = alpha
    barAlpha_outer = 1.0

    A11 = 0.25 * n1 * (m2 / M0) * alpha * barAlpha_inner * b32_1
    A12 = -0.25 * n1 * (m2 / M0) * alpha * barAlpha_inner * b32_2
    A21 = -0.25 * n2 * (m1 / M0) * alpha * barAlpha_outer * b32_2
    A22 = 0.25 * n2 * (m1 / M0) * alpha * barAlpha_outer * b32_1

    A = np.array([[A11, A12],
                  [A21, A22]])

    if toDeg:
        A *= RAD_TO_DEG  # rad/yr -> deg/yr

    return A


def computeMatrixB(planets, constants, toDeg: bool = True) -> np.ndarray:
    """
    Compute secular matrix B (inclination terms).
    Implements Eqs. (7.9)–(7.12) from Murray & Dermott.
    """
    G = constants["G"]
    M0 = constants["M0"]

    p1, p2 = planets.iloc[0], planets.iloc[1]

    m1, a1 = float(p1["mass"]), float(p1["a"])
    m2, a2 = float(p2["mass"]), float(p2["a"])

    n1 = meanMotion(G, M0, a1, m1)
    n2 = meanMotion(G, M0, a2, m2)

    alpha = a1 / a2

    b32_1 = laplaceCoefficient(1.5, 1, alpha)

    # barAlpha factors: inner planet gets alpha, outer gets 1
    barAlpha_inner = alpha
    barAlpha_outer = 1.0

    B11 = -0.25 * n1 * (m2 / M0) * alpha * barAlpha_inner * b32_1
    B12 = +0.25 * n1 * (m2 / M0) * alpha * barAlpha_inner * b32_1
    B21 = +0.25 * n2 * (m1 / M0) * alpha * barAlpha_outer * b32_1
    B22 = -0.25 * n2 * (m1 / M0) * alpha * barAlpha_outer * b32_1

    B = np.array([[B11, B12],
                  [B21, B22]])

    if toDeg:
        B *= RAD_TO_DEG  # rad/yr -> deg/yr

    return B


# =========================
# Diagonalization utilities
# =========================

def normalizeColumns(V: np.ndarray) -> np.ndarray:
    """Normalize columns of matrix V to unit Euclidean norm."""
    Vn = V.copy().astype(float)
    for j in range(Vn.shape[1]):
        col = Vn[:, j]
        norm = np.linalg.norm(col)
        if norm != 0:
            Vn[:, j] = col / norm
    return Vn


def orientColumnsToReference(V: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Flip sign of each column of V so that dot(V[:,j], ref[:,j]) > 0.
    """
    Vout = V.copy()
    for j in range(ref.shape[1]):
        d = np.dot(Vout[:, j], ref[:, j])
        if d < 0:
            Vout[:, j] = -Vout[:, j]
    return Vout


def diagonalizeAndMatch(M: np.ndarray,
                        inDeg: bool = True,
                        toArcsec: bool = True,
                        referenceVecs: np.ndarray = None,
                        sortBy: str = "magnitude"):
    """
    Diagonalize matrix M and optionally reorder/orient eigenvectors to match
    reference (e.g. Murray & Dermott eq. 7.42).

    Parameters
    ----------
    M : np.ndarray
        2x2 matrix (in deg/yr if inDeg=True).
    inDeg : bool
        If True, M is in deg/yr.
    toArcsec : bool
        If True, eigenvalues are returned in arcsec/yr.
    referenceVecs : np.ndarray or None
        Reference eigenvectors (2x2) to match orientation/order.
    sortBy : str
        "magnitude" (default) or "value".

    Returns
    -------
    eigvals_out : np.ndarray
        Eigenvalues (arcsec/yr if toArcsec=True, else deg/yr).
    eigvecs_out : np.ndarray
        Eigenvectors (normalized, columns).
    """
    eigvals, eigvecs = np.linalg.eig(M)

    # Sorting
    if sortBy == "magnitude":
        idx = np.argsort(np.abs(eigvals))
    else:
        idx = np.argsort(eigvals)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Normalize columns
    eigvecs = normalizeColumns(eigvecs)

    # Match to reference if given
    if referenceVecs is not None:
        # Normalize reference
        ref = normalizeColumns(referenceVecs)
        # Try both permutations (2x2)
        cost01 = 1 - abs(np.dot(eigvecs[:, 0], ref[:, 0])) + \
                 1 - abs(np.dot(eigvecs[:, 1], ref[:, 1]))
        cost10 = 1 - abs(np.dot(eigvecs[:, 1], ref[:, 0])) + \
                 1 - abs(np.dot(eigvecs[:, 0], ref[:, 1]))
        if cost10 < cost01:
            eigvecs = eigvecs[:, [1, 0]]
            eigvals = eigvals[[1, 0]]
        eigvecs = orientColumnsToReference(eigvecs, ref)

    # Convert eigenvalues if requested
    if toArcsec and inDeg:
        eigvals_out = eigvals * DEG_TO_ARCSEC
    else:
        eigvals_out = eigvals

    return eigvals_out, eigvecs
