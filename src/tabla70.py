#!/usr/bin/env python3
"""
tabla70.py — junta en una sola tabla todas las corridas de 70 epocas.

Lee los JSON que dejan las barridas (e70_*.json y c70_*.json) y los ordena por
perdida final, que es lo unico que permite comparar de un vistazo arquitecturas
y tasas distintas.

Uso:  $PY tabla70.py [carpeta]        (por defecto /workspace/kan_snap)
"""

import json
import sys
from pathlib import Path


DIVERGE = 1e6      # perdida inicial por encima de esto = el modelo reviento
SUELO = (5.0e-2, 6.0e-2)   # el suelo degenerado medido el 2 de octubre


def filas(raiz):
    for f in (sorted(raiz.glob("e70_*.json")) + sorted(raiz.glob("c70_*.json"))
              + sorted(raiz.glob("s70_*.json"))):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            yield f.stem, None, None, None, None, f"json ilegible: {e}"
            continue
        for r in d.values():
            c = r.get("curva") or []
            p = [x[1] for x in c]
            yield (f.stem, r.get("lr"), r.get("semilla"),
                   p[0] if p else None, p[-1] if p else None,
                   r.get("fallo") or ("" if p else "sin curva"))


def main():
    raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/kan_snap")
    datos = list(filas(raiz))
    if not datos:
        print(f"no hay JSON de 70 epocas en {raiz}")
        return 1

    print(f"{'corrida':20s} {'lr':>8s} {'sem':>5s} {'p inicial':>12s} "
          f"{'p final':>12s} {'caida':>9s}  nota")
    print("-" * 88)

    def clave(t):
        return t[4] if t[4] is not None else float("inf")

    for nombre, lr, sem, p0, pf, nota in sorted(datos, key=clave):
        if p0 is None or pf is None:
            print(f"{nombre:20s} {'':>8s} {'':>5s} {'':>12s} {'':>12s} "
                  f"{'':>9s}  {nota[:40]}")
            continue
        caida = (1 - pf / p0) * 100
        marca = []
        if p0 > DIVERGE:
            marca.append("DIVERGIO al arrancar")
        if SUELO[0] <= pf <= SUELO[1]:
            marca.append("SUELO DEGENERADO")
        aviso = "  ".join(marca) or nota[:40]
        print(f"{nombre:20s} {lr:8.0e} {sem if sem else 0:5d} "
              f"{p0:12.4e} {pf:12.4e} {caida:8.2f}%  {aviso}")

    print()
    print("Ordenado por perdida final, pero NO se lee de arriba abajo:")
    print()
    print(" - 'DIVERGIO al arrancar': perdida inicial > 1e6. Con esa tasa el")
    print("   modelo reviento en el primer paso. La fila no mide nada.")
    print(" - 'SUELO DEGENERADO': perdida final entre 5.0e-2 y 6.0e-2. El 2 de")
    print("   octubre siete configuraciones distintas —MLP, gaussiana, B-spline")
    print("   y Chebyshev, con y sin normalizacion, dos tasas, dos semillas—")
    print("   cayeron todas en 5.4e-2 con dos cifras iguales. Es un estado al")
    print("   que cae la optimizacion, no un ajuste. Esas filas no se comparan.")
    print(" - Solo son comparables entre si las filas SIN marca y con la MISMA")
    print("   tasa: ahi la arquitectura es la unica variable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
