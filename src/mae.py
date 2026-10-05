#!/usr/bin/env python3
"""
mae.py — errores fisicos de cada corrida, a partir de perconfig.dat y peratom.dat.

La perdida de entrenamiento no distingue un modelo bueno de uno que predice
cero. El 2 de octubre de 2026 siete configuraciones distintas cayeron en la
misma perdida, 5.4e-2, y al mirar las predicciones resulto que todas predecian
fuerza cero y energia constante: ese numero es la media del cuadrado de las
fuerzas verdaderas, una propiedad del conjunto de datos.

Este lector calcula lo que si compara modelos:

  E MAE   eV/atomo, |E_verdad - E_pred| / Natoms
  F MAE   eV/A, sobre las tres componentes de cada atomo
  F/cero  el mismo error dividido por el de predecir cero: 1.000 = no aprendio

y, sobre todo, dos diagnosticos que habrian ahorrado la sesion entera:

  F/cero ~ 1                 el modelo predice CERO: no aprendio nada
  desv(E pred) / desv(E verdad) ~ 0   el modelo predice una CONSTANTE

Referencia del grupo (LiF_metrics.dat del modelo base):
  E MAE 2.790e-03 eV/atomo,  F MAE 1.108e-02 eV/A

Uso:
    $PY mae.py /workspace/kan_snap/s70_* /workspace/kan_snap/c70_* ...
    $PY mae.py                 # busca solo en /workspace/kan_snap
"""

import glob
import sys
from pathlib import Path

import numpy as np


def leer(ruta, columnas):
    """Lee un .dat de FitSNAP con cabecera de nombres separados por espacios."""
    lineas = Path(ruta).read_text(encoding="utf-8").splitlines()
    if not lineas:
        return None
    cab = lineas[0].split()
    idx = {}
    for c in columnas:
        if c not in cab:
            return None
        idx[c] = cab.index(c)
    datos = {c: [] for c in columnas}
    for l in lineas[1:]:
        p = l.split()
        if len(p) < len(cab):
            continue
        for c in columnas:
            datos[c].append(p[idx[c]])
    return datos


def analizar(carpeta):
    pc = Path(carpeta) / "perconfig.dat"
    pa = Path(carpeta) / "peratom.dat"
    if not pc.exists() or not pa.exists():
        return None

    d = leer(pc, ["Natoms", "Energy_Truth", "Energy_Pred", "Testing_Bool"])
    if d is None:
        return None
    nat = np.array(d["Natoms"], float)
    ev = np.array(d["Energy_Truth"], float) / nat
    ep = np.array(d["Energy_Pred"], float) / nat
    test_e = np.array([x.lower().startswith("t") for x in d["Testing_Bool"]])

    f = leer(pa, ["Fx_Truth", "Fy_Truth", "Fz_Truth",
                  "Fx_Pred", "Fy_Pred", "Fz_Pred", "Testing_Bool"])
    if f is None:
        return None
    fv = np.array([f["Fx_Truth"], f["Fy_Truth"], f["Fz_Truth"]], float).ravel()
    fp = np.array([f["Fx_Pred"], f["Fy_Pred"], f["Fz_Pred"]], float).ravel()
    t = np.array([x.lower().startswith("t") for x in f["Testing_Bool"]])
    test_f = np.concatenate([t, t, t])

    def mae(a, b, m):
        return float(np.abs(a[m] - b[m]).mean()) if m.any() else float("nan")

    # El patron de comparacion es predecir CERO, cuyo error medio es
    # exactamente la media del valor absoluto de la fuerza verdadera. Asi la
    # razon vale 1.000 para un modelo que no predice nada, sin depender de la
    # forma de la distribucion (contra la raiz cuadratica media daria 0.80
    # para fuerzas gaussianas, que es un umbral traicionero).
    f_cero = float(np.abs(fv).mean())
    return dict(
        n_cfg=len(ev), n_test=int(test_e.sum()),
        e_mae_tr=mae(ev, ep, ~test_e), e_mae_te=mae(ev, ep, test_e),
        f_mae_tr=mae(fv, fp, ~test_f), f_mae_te=mae(fv, fp, test_f),
        f_cero=f_cero,
        razon_f=mae(fv, fp, ~test_f) / f_cero if f_cero else float("nan"),
        razon_e=float(ep.std() / ev.std()) if ev.std() else float("nan"),
    )


def main():
    patrones = sys.argv[1:] or ["/workspace/kan_snap/*70_*"]
    carpetas = []
    for p in patrones:
        for c in sorted(glob.glob(p)):
            for sub in ("barrida_lr_tmp", "barrida_modo_tmp", "sonda_tmp", ""):
                d = Path(c) / sub if sub else Path(c)
                if (d / "perconfig.dat").exists():
                    carpetas.append((Path(c).name, d))
                    break
    if not carpetas:
        print("no se encontro ningun perconfig.dat en:", patrones)
        return 1

    print("Referencia del grupo:  E MAE 2.790e-03 eV/atomo   F MAE 1.108e-02 eV/A")
    print()
    print(f"{'corrida':22s} {'E MAE tr':>10s} {'E MAE te':>10s} "
          f"{'F MAE tr':>10s} {'F MAE te':>10s} {'F/cero':>7s} {'sE/sE':>7s}  veredicto")
    print("-" * 104)
    for nombre, d in carpetas:
        r = analizar(d)
        if r is None:
            print(f"{nombre:22s}  ilegible")
            continue
        v = []
        if r["razon_f"] > 0.90:
            v.append("NO MEJOR QUE PREDECIR CERO")
        if r["razon_e"] < 0.05:
            v.append("ENERGIA CONSTANTE")
        if not v:
            v.append("predice algo")
        print(f"{nombre:22s} {r['e_mae_tr']:10.3e} {r['e_mae_te']:10.3e} "
              f"{r['f_mae_tr']:10.3e} {r['f_mae_te']:10.3e} "
              f"{r['razon_f']:7.3f} {r['razon_e']:7.3f}  {' + '.join(v)}")

    print()
    print("F/cero  = error de fuerza dividido por el que daria predecir CERO.")
    print("          1.000 significa que el modelo no aprendio nada de fuerzas,")
    print("          y entonces su perdida, por baja que sea, no mide nada.")
    print("          Por debajo de 1 hay senal; cuanto mas bajo, mejor.")
    print("sE/sE   = desviacion de la energia predicha sobre la de la verdadera.")
    print("          Cerca de 0 significa que predice una constante.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
