# -*- coding: utf-8 -*-
"""
main.py
-------
Pipeline completo e genérico (dados em /data/*.json).
Mostra resultados intermediários no console e gera Fig. 7.1.
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

def main():
    _ensure_output_dir("output")

    # (1) Leitura dos JSONs
    bodies = load_bodies_from_json("data/inputPlanets.json")
    consts = load_constants("data/constants.json")
    names = [b["name"] for b in bodies]

    # (2) Matrizes A e B
    M: SecularMatrices = build_AB_from_bodies(bodies, consts)
    print("="*80)
    print(" Secular matrices (deg/yr)")
    print("="*80)
    print("Matrix A (eccentricity terms):")
    print(M.A_degyr)
    print("-"*80)
    print("Matrix B (inclination terms):")
    print(M.B_degyr)

    # (3) Autovalores/autovetores
    eig = secular_eigendecomp(M)

    print("="*80)
    print(" Eigenfrequencies g (arcsec/yr)")
    print("="*80)
    print(eig.g_radyr * ARCSEC_PER_RAD)
    print("-"*80)
    print(" Eigenvectors Sg:")
    print(np.real(eig.Sg))

    print("="*80)
    print(" Eigenfrequencies s (arcsec/yr)")
    print("="*80)
    print(eig.s_radyr * ARCSEC_PER_RAD)
    print("-"*80)
    print(" Eigenvectors Ss:")
    print(np.real(eig.Ss))

    # (4) Salvar intermediários
    save_intermediates(M, eig, out_dir="output")

    # (5) Condições iniciais
    h0, k0, p0, q0 = initial_conditions_from_json(bodies)

    # (6) Malha temporal [-1e5, +1e5] anos
    t_years = np.linspace(-1.0e5, 1.0e5, 5000)

    # (7) Reconstrução temporal
    h, k, p, q = build_time_series(eig, h0, k0, p0, q0, t_years)

    # (8) Conversão para elementos orbitais
    e, varpi = hk_to_e_varpi(h, k)
    inc, Omega = pq_to_i_Omega(p, q)

    # (9) Salvar séries
    np.savez("output/series_full.npz",
             t_years=t_years, h=h, k=k, p=p, q=q,
             e=e, varpi=varpi, inc=inc, Omega=Omega)

    # (10) Gráfico Fig. 7.1
    plot_fig71(t_years, e, inc, bodies_names=names, out_dir="output")

if __name__ == "__main__":
    main()
