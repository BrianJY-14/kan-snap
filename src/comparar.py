#!/usr/bin/env python3
"""
comparar.py — envoltorio de `compare_runs.py`, de fitsnap-custom (Arbues).

`compare_runs.py` busca las corridas en una carpeta `07_experimentos/` situada
junto a el mismo. Este envoltorio hace que ese nombre apunte a la carpeta de
corrida que se le indique y le pasa el resto de argumentos tal cual, **sin
modificar su archivo**: asi se puede reemplazar por una version suya mas nueva
copiandola encima.

Uso:
    python comparar.py /workspace/kan_snap/corrida_20261001_0952
    python comparar.py <carpeta> --system
    python comparar.py <carpeta> --plots            # parity plots + curvas
    python comparar.py <carpeta> --winners-only

Lo que produce, por dataset (aqui el grupo de FitSNAP):

  - Tabla comparativa con Δ% de cada corrida contra el baseline. El baseline lo
    detecta por nombre: cualquier corrida cuyo nombre contenga "run_01". Como
    `trabajador.py` nombra las carpetas con el prefijo numerico de la condicion,
    `1_mlp_base` cae en `run_01_...` y queda como baseline automaticamente.
  - Con --plots, un parity plot por corrida desde `perconfig.dat`. Un modelo que
    no aprendio sale como una linea horizontal: se ve de un vistazo, que es lo
    que no se vio el 1 de octubre mirando solo numeros de perdida.
  - Con --system, picos de RAM y tiempo desde el `system_metrics.csv` de cada
    corrida.

Requiere que la salida siga la convencion que escribe `trabajador.py`:

    <carpeta>/<grupo>/run_NN_<id>_s<semilla>/<id>_s<semilla>_metrics.dat
"""

import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).parent.resolve()
ENLACE = AQUI / "07_experimentos"
SCRIPT = AQUI / "compare_runs.py"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    destino = Path(sys.argv[1]).resolve()
    resto = sys.argv[2:]

    if not SCRIPT.exists():
        print(f"falta {SCRIPT.name}. Copiarlo de workspace/ de fitsnap-custom.")
        return 1
    if not destino.is_dir():
        print(f"no existe la carpeta: {destino}")
        return 1

    metricas = list(destino.rglob("*_metrics.dat"))
    if not metricas:
        print(f"en {destino} no hay ningun *_metrics.dat.")
        print("Comprobar que las corridas se hicieron con el trabajador nuevo:")
        print("  <carpeta>/<grupo>/run_NN_<id>_s<semilla>/<id>_s<semilla>_metrics.dat")
        return 1
    print(f"{len(metricas)} corridas con metricas en {destino}")

    # Apuntar 07_experimentos a la carpeta pedida, sin tocar compare_runs.py.
    if ENLACE.is_symlink() or ENLACE.exists():
        if ENLACE.is_symlink():
            ENLACE.unlink()
        else:
            print(f"{ENLACE} existe y no es un enlace. Moverlo o borrarlo a mano.")
            return 1
    try:
        ENLACE.symlink_to(destino, target_is_directory=True)
    except OSError as e:
        print(f"no se pudo crear el enlace ({e}).")
        print("Alternativa: copiar compare_runs.py dentro de la carpeta padre")
        print("de las corridas y renombrar esa carpeta a 07_experimentos.")
        return 1

    try:
        r = subprocess.run([sys.executable, str(SCRIPT)] + resto, cwd=str(AQUI))
    finally:
        if ENLACE.is_symlink():
            ENLACE.unlink()
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
