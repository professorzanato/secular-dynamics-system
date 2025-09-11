from src.reader import loadPlanets
from src.constants import loadConstants
from src.secular import computeMatrixA, computeMatrixB, diagonalizeAndMatch
import numpy as np


def normCols(X: np.ndarray) -> np.ndarray:
    """Normalize columns of a 2x2 matrix."""
    Xc = X.copy().astype(float)
    for j in range(Xc.shape[1]):
        norm = np.linalg.norm(Xc[:, j])
        if norm != 0:
            Xc[:, j] /= norm
    return Xc


def main():
    # ----------------------------
    # Load input data
    # ----------------------------
    planets = loadPlanets("data/inputPlanets.json")
    constants = loadConstants("data/constants.json")

    # ----------------------------
    # Compute secular matrices
    # ----------------------------
    A = computeMatrixA(planets, constants, toDeg=True)
    B = computeMatrixB(planets, constants, toDeg=True)

    # ----------------------------
    # Reference eigenvectors (Murray & Dermott, eq. 7.42)
    # ----------------------------
    ref_g = np.array([[-0.777991,  0.332842],
                      [-0.628275, -1.01657 ]])   # eccentricity modes
    ref_s = np.array([[ 0.707107, -0.40797 ],
                      [ 0.707107,  1.00624 ]])   # inclination modes

    ref_g = normCols(ref_g)
    ref_s = normCols(ref_s)

    # ----------------------------
    # Diagonalize and match
    # ----------------------------
    g_vals, g_vecs = diagonalizeAndMatch(A, inDeg=True, toArcsec=True, referenceVecs=ref_g)
    s_vals, s_vecs = diagonalizeAndMatch(B, inDeg=True, toArcsec=True, referenceVecs=ref_s)

    # ----------------------------
    # Print results
    # ----------------------------
    print("======================================")
    print(" Secular matrices (deg/yr)")
    print("======================================")
    print("Matrix A (eccentricity terms):")
    print(A)
    print()
    print("Matrix B (inclination terms):")
    print(B)
    print("======================================")
    print(" Eigenfrequencies g (arcsec/yr):")
    print(g_vals)
    print(" Eigenvectors g (matched to M&D 7.42):")
    print(g_vecs)
    print("--------------------------------------")
    print(" Eigenfrequencies s (arcsec/yr):")
    print(s_vals)
    print(" Eigenvectors s (matched to M&D 7.42):")
    print(s_vecs)
    print("======================================")


if __name__ == "__main__":
    main()
