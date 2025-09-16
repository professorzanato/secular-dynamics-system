# -*- coding: utf-8 -*-
"""
inspect_output.py
-----------------
Script auxiliar para inspecionar o arquivo 'series_full.npz' gerado pelo sistema.
Mostra as chaves, dimensões, alguns valores de exemplo e explica
o significado físico de cada array.
"""

import numpy as np

# Dicionário explicativo para cada chave
EXPLANATIONS = {
    "t_years": "Tempo em anos (de -100000 até +100000).",
    "h": "h = e * cos(varpi), variável secular associada à excentricidade.",
    "k": "k = e * sin(varpi), variável secular associada à excentricidade.",
    "p": "p = sin(I/2) * cos(Omega), variável secular associada à inclinação.",
    "q": "q = sin(I/2) * sin(Omega), variável secular associada à inclinação.",
    "e": "Excentricidade e(t) reconstruída a partir de h e k.",
    "varpi": "Longitude do periastro varpi(t), em radianos.",
    "inc": "Inclinação I(t), em radianos.",
    "Omega": "Longitude do nó ascendente Omega(t), em radianos."
}

def main():
    fname = "output/series_full.npz"
    data = np.load(fname)

    print("="*80)
    print(f" Inspeção do arquivo {fname}")
    print("="*80)
    print("\nArrays disponíveis:", list(data.files))

    for key in data.files:
        arr = data[key]
        print("\n" + "-"*80)
        print(f"{key} -> shape {arr.shape}, dtype {arr.dtype}")
        print("Descrição:", EXPLANATIONS.get(key, "Sem descrição."))
        # mostra os primeiros 5 valores (ou linhas, se 2D)
        if arr.ndim == 1:
            print("Primeiros valores:", arr[:5])
        else:
            print("Primeiras linhas:\n", arr[:5, :])

    print("\n" + "="*80)
    print(" Fim da inspeção")
    print("="*80)

if __name__ == "__main__":
    main()
