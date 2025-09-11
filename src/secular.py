"""
secular.py
-----------
Module for computing secular perturbation matrices (eccentricity and inclination).
Results are converted to degrees per year (deg/yr), as in Murray & Dermott
eqs. (7.34) and (7.35).
"""

import numpy as np
from scipy.integrate import quad
import math

# Conversion factor
RAD_TO_DEG = 180.0 / math.pi


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
    Implements Eqs. (7.9)–(7.12) for B, as in Murray & Dermott.
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

def diagonalizeMatrix(M: np.ndarray):
    """
    Diagonalize secular matrix (A or B).

    Parameters
    ----------
    M : np.ndarray
        2x2 secular matrix (in deg/yr).

    Returns
    -------
    eigenvalues : np.ndarray
        Frequencies (deg/yr).
    eigenvectors : np.ndarray
        Normalized eigenvectors (modes).
    """
    eigvals, eigvecs = np.linalg.eig(M)
    # Ordenar por valor absoluto (opcional, para consistência com o livro)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

