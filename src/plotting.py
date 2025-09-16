# -*- coding: utf-8 -*-
"""
plotting.py
-----------
Plota os resultados seculares no estilo da Fig. 7.1 de M&D:
- e(t) em um gráfico separado
- i(t) (em graus) em outro gráfico separado
- linha tracejada em t=0
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def _ensure_output_dir(path: str = "output") -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_fig71(t_years: np.ndarray, e: np.ndarray, inc_rad: np.ndarray,
               bodies_names=None, out_dir: str = "output") -> None:
    """
    Gera dois gráficos separados (como em M&D Fig. 7.1):
    - fig71_e_t.png : evolução da excentricidade
    - fig71_i_t.png : evolução da inclinação (graus)

    Tempo: de -100000 a +100000 anos
    Linha tracejada em t=0
    """
    _ensure_output_dir(out_dir)
    T, n = e.shape
    if bodies_names is None:
        bodies_names = [f"corpo {j}" for j in range(n)]

    # --- Gráfico 1: excentricidade
    plt.figure(figsize=(6,4))
    for j in range(n):
        plt.plot(t_years, e[:, j], label=bodies_names[j])
    plt.axvline(x=0, color="k", linestyle="--")  # linha tracejada em t=0
    plt.xlim(-1.0e5, 1.0e5)
    plt.ylim(0, max(e.max()*1.1, 0.1))
    plt.xlabel("Time (years)")
    plt.ylabel("e")
    plt.title("Secular evolution of eccentricity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig71_e_t.png"), dpi=300)
    plt.close()

    # --- Gráfico 2: inclinação
    plt.figure(figsize=(6,4))
    for j in range(n):
        plt.plot(t_years, np.degrees(inc_rad[:, j]), label=bodies_names[j])
    plt.axvline(x=0, color="k", linestyle="--")  # linha tracejada em t=0
    plt.xlim(-1.0e5, 1.0e5)
    plt.ylim(0, max(np.degrees(inc_rad).max()*1.1, 3))
    plt.xlabel("Time (years)")
    plt.ylabel("I [deg]")
    plt.title("Secular evolution of inclination")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig71_i_t.png"), dpi=300)
    plt.close()
