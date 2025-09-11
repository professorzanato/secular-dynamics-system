from src.reader import loadPlanets
from src.constants import loadConstants
from src.secular import computeMatrixA, computeMatrixB, diagonalizeMatrix


def main():
    # Carregar dados
    planets = loadPlanets("data/inputPlanets.json")
    constants = loadConstants("data/constants.json")

    # Calcular matrizes seculares
    A = computeMatrixA(planets, constants, toDeg=True)
    B = computeMatrixB(planets, constants, toDeg=True)

    # Diagonalizar
    g_vals, g_vecs = diagonalizeMatrix(A)
    s_vals, s_vecs = diagonalizeMatrix(B)

    # Mostrar resultados
    print("======================================")
    print(" Secular matrices (deg/yr)")
    print("======================================")
    print("Matrix A (eccentricity terms):")
    print(A)
    print()
    print("Matrix B (inclination terms):")
    print(B)
    print("======================================")
    print(" Eigenfrequencies g (eccentricities):")
    print(g_vals)
    print(" Eigenvectors g:")
    print(g_vecs)
    print("--------------------------------------")
    print(" Eigenfrequencies s (inclinations):")
    print(s_vals)
    print(" Eigenvectors s:")
    print(s_vecs)
    print("======================================")


if __name__ == "__main__":
    main()
