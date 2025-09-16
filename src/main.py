# -*- coding: utf-8 -*-
"""
main.py
-------
Pipeline completo e genérico (dados em /data/*.json).
Mostra resultados intermediários no console (com espaçamento e nomes dos planetas)
e gera os arquivos de saída (CSV, NPZ, PNG).
"""

import os
import numpy as np
from .secular import (
    load_bodies_from_json,
    load_constants,
    build_AB_from_bodies,
    secular_eigendecomp,
    save_intermediates,
    build_time_series,
    hk_to_e_varpi,
    pq_to_i_Omega,
    initial_conditions_from_json,
    SecularMatrices,
    ARCSEC_PER_RAD,
)
from .plotting import plot_fig71


def _ensure_output_dir(path: str = "output") -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def print_matrix_with_labels(matrix, names, title):
    """Imprime matriz NxN com nomes dos planetas como rótulos."""
    print(f"\n{title}:")
    header = "          " + "".join([f"{name:>12}" for name in names])
    print(header)
    for i, row in enumerate(matrix):
        row_str = " ".join([f"{val:12.6f}" for val in row])
        print(f"{names[i]:>10} {row_str}")


def print_vector_with_labels(vec, names, title):
    """Imprime vetor com nomes dos planetas."""
    print(f"\n{title}:")
    for name, val in zip(names, vec):
        print(f"  {name:<10}: {val:12.6f}")


def print_matrix_vectors(matrix, names, title):
    """Imprime matriz de autovetores, cada coluna corresponde a um modo."""
    print(f"\n{title}:")
    for i, row in enumerate(matrix):
        row_str = " ".join([f"{val:12.6f}" for val in row])
        print(f"  {names[i]:<10} {row_str}")


def print_initial_conditions(h0, k0, p0, q0, names):
    """Imprime condições iniciais orbitais em formato legível."""
    print("\n" + "="*80)
    print(" Initial conditions (from inputPlanets.json)")
    print("="*80)
    for i, name in enumerate(names):
        print(f"  {name:<10}: h0={h0[i]:.6f}, k0={k0[i]:.6f}, "
              f"p0={p0[i]:.6f}, q0={q0[i]:.6f}")


def main():
    _ensure_output_dir("output")

    # (1) Leitura dos JSONs
    bodies = load_bodies_from_json("data/inputPlanets.json")
    consts = load_constants("data/constants.json")
    names = [b["name"] for b in bodies]

    # (2) Matrizes A e B
    M: SecularMatrices = build_AB_from_bodies(bodies, consts)
    print("\n" + "="*80)
    print(" Secular matrices (deg/yr)")
    print("="*80)
    print_matrix_with_labels(M.A_degyr, names, "Matrix A (eccentricity terms)")
    print_matrix_with_labels(M.B_degyr, names, "Matrix B (inclination terms)")

    # (3) Autovalores/autovetores
    eig = secular_eigendecomp(M)

    print("\n" + "="*80)
    print(" Eigenfrequencies g (arcsec/yr)")
    print("="*80)
    print_vector_with_labels(eig.g_radyr * ARCSEC_PER_RAD, names, "Frequencies g")
    print_matrix_vectors(np.real(eig.Sg), names, "Eigenvectors Sg")

    print("\n" + "="*80)
    print(" Eigenfrequencies s (arcsec/yr)")
    print("="*80)
    print_vector_with_labels(eig.s_radyr * ARCSEC_PER_RAD, names, "Frequencies s")
    print_matrix_vectors(np.real(eig.Ss), names, "Eigenvectors Ss")

    # (4) Salvar intermediários (CSV)
    save_intermediates(M, eig, out_dir="output")

    # (5) Condições iniciais
    h0, k0, p0, q0 = initial_conditions_from_json(bodies)
    print_initial_conditions(h0, k0, p0, q0, names)

    # (6) Malha temporal [-1e5, +1e5] anos
    t_years = np.linspace(-1.0e5, 1.0e5, 5000)

    # (7) Reconstrução temporal
    h, k, p, q = build_time_series(eig, h0, k0, p0, q0, t_years)

    # (8) Conversão para elementos orbitais
    e, varpi = hk_to_e_varpi(h, k)
    inc, Omega = pq_to_i_Omega(p, q)

    # (9) Salvar séries completas
    np.savez("output/series_full.npz",
             t_years=t_years, h=h, k=k, p=p, q=q,
             e=e, varpi=varpi, inc=inc, Omega=Omega)

    # Exemplos CSV (corpo 0) para inspeção rápida:
    np.savetxt("output/t.csv", t_years, delimiter=",")
    np.savetxt("output/e_body0.csv", e[:, 0], delimiter=",")
    np.savetxt("output/i_deg_body0.csv", np.degrees(inc[:, 0]), delimiter=",")

    # (10) Gráficos
    plot_fig71(t_years, e, inc, bodies_names=names, out_dir="output")

    print("\n" + "="*80)
    print(" Cálculo concluído. Resultados salvos em /output/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
