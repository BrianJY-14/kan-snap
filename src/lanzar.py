#!/usr/bin/env python3
"""
lanzar.py — reparte las corridas entre trabajadores y los vigila.

Un proceso por trabajador, cada uno envuelto en monitor_fases.py, cada uno con
su tarjeta asignada por CUDA_VISIBLE_DEVICES. Los descriptores se calculan una
vez por trabajador.

Uso:
    # comprobacion previa, sin gastar GPU
    python lanzar.py --raiz /workspace/kan_snap --solo-comprobar

    # validacion: las 8 condiciones, una semilla, pocas epocas
    python lanzar.py --raiz /workspace/kan_snap --semillas 1111 --epocas 3

    # rejilla completa
    python lanzar.py --raiz /workspace/kan_snap --epocas 70 --trabajadores 4

Antes de lanzar imprime el reparto y el tiempo estimado, y no arranca sin
confirmacion salvo que se pase --si.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(AQUI))

from kan_lib import CONDICIONES, SEMILLAS          # noqa: E402


def gpus_disponibles():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def repartir(tareas, n):
    cajas = [[] for _ in range(n)]
    for i, t in enumerate(tareas):
        cajas[i % n].append(t)
    return [c for c in cajas if c]


def comprobar():
    """Comprobacion estatica de los archivos del marco. Un segundo."""
    archivos = [AQUI / "kan_lib.py", AQUI / "trabajador.py"]
    faltan = [a for a in archivos if not a.exists()]
    if faltan:
        print("FALTAN ARCHIVOS: " + ", ".join(str(a.name) for a in faltan))
        return 1
    r = subprocess.run([sys.executable, str(AQUI / "verificar_lib.py")]
                       + [str(a) for a in archivos])
    return r.returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", help="carpeta que contiene JSON/")
    ap.add_argument("--grupo", default="LiF_64_kjpaw",
                    help="un grupo, varios separados por coma, o 'todos'")
    ap.add_argument("--epocas", type=int, default=70)
    ap.add_argument("--trabajadores", type=int, default=None,
                    help="por defecto, tantos como tarjetas")
    ap.add_argument("--semillas", nargs="+", type=int, default=None,
                    help="por defecto las cinco de kan_lib")
    ap.add_argument("--condiciones", nargs="+", default=None,
                    help="ids a correr; por defecto las ocho")
    ap.add_argument("--salida", default=None,
                    help="carpeta de resultados; por defecto RAIZ/corrida_<hora>")
    ap.add_argument("--dtype-setting", type=int, default=None,
                    help="1 float32, 2 float64")
    ap.add_argument("--seg-por-epoca", type=float, default=8.4,
                    help="para el estimado; medido el 28-09 con lote 10")
    ap.add_argument("--seg-descriptores", type=float, default=452.0,
                    help="para el estimado; medido el 28-09 con FRACCION 0.25")
    ap.add_argument("--solo-comprobar", action="store_true")
    ap.add_argument("--si", action="store_true", help="no pedir confirmacion")
    args = ap.parse_args()

    rc = comprobar()
    if rc != 0:
        print("\nLa comprobacion estatica ha fallado. No se lanza nada.")
        return rc
    if args.solo_comprobar:
        return 0
    if not args.raiz:
        ap.error("falta --raiz")

    raiz = Path(args.raiz)
    semillas = args.semillas if args.semillas is not None else list(SEMILLAS)
    condiciones = [c for c in CONDICIONES
                   if args.condiciones is None or c[0] in args.condiciones]
    if not condiciones:
        print("ninguna condicion coincide con --condiciones")
        return 1

    tareas = [dict(id=cid, capas=capas, salida=sal, kw=kw, semilla=s)
              for (cid, capas, sal, kw) in condiciones for s in semillas]

    n_gpu = gpus_disponibles()
    n = args.trabajadores or max(1, n_gpu)
    cajas = repartir(tareas, n)
    por_trabajador = max(len(c) for c in cajas)

    salida = Path(args.salida) if args.salida else \
        raiz / f"corrida_{time.strftime('%Y%m%d_%H%M')}"
    salida.mkdir(parents=True, exist_ok=True)

    seg_est = args.seg_descriptores + por_trabajador * args.epocas * args.seg_por_epoca

    print()
    print(f"grupo            {args.grupo}")
    print(f"condiciones      {len(condiciones)}")
    print(f"semillas         {len(semillas)}  {semillas}")
    print(f"corridas         {len(tareas)}")
    print(f"epocas           {args.epocas}")
    print(f"tarjetas         {n_gpu}")
    print(f"trabajadores     {n}")
    print(f"dtype_setting    {args.dtype_setting if args.dtype_setting else 'por defecto (1 = float32)'}")
    print(f"salida           {salida}")
    print()
    for i, c in enumerate(cajas):
        gpu = i % max(1, n_gpu)
        print(f"  trabajador {i} -> GPU {gpu} -> {len(c)} corridas: "
              f"{', '.join(t['id'] for t in c[:3])}"
              f"{'...' if len(c) > 3 else ''}")
    print()
    print(f"estimado  {por_trabajador} corridas por trabajador, "
          f"~{seg_est/60:.0f} min de reloj")
    print("          (descriptores una vez por trabajador + epocas x corridas)")
    print("          El estimado usa los segundos por epoca medidos el 28-09 en")
    print("          un proceso solo. Con varios en paralelo sera mayor; la")
    print("          primera corrida real da el numero bueno.")
    print()

    if not args.si:
        try:
            if input("lanzar? [s/N] ").strip().lower() not in ("s", "si", "y"):
                print("cancelado")
                return 0
        except EOFError:
            print("sin terminal interactiva; usar --si para lanzar")
            return 0

    procesos = []
    t0 = time.perf_counter()
    for i, caja in enumerate(cajas):
        entorno = dict(os.environ)
        if n_gpu:
            entorno["CUDA_VISIBLE_DEVICES"] = str(i % n_gpu)
        if args.dtype_setting is not None:
            entorno["KAN_DTYPE_SETTING"] = str(args.dtype_setting)
        cmd = [sys.executable, str(AQUI / "monitor_fases.py"),
               "--csv", str(salida / f"metricas_w{i}.csv"),
               # Sin --silencioso: en modo silencioso el monitor se traga la
               # salida del hijo, traceback incluido. El 2 de octubre los ocho
               # trabajadores murieron con codigo 1 y el log solo contenia el
               # pie del monitor. Va a un archivo, no a pantalla: que sobre.
               "--",
               sys.executable, str(AQUI / "trabajador.py"),
               json.dumps(caja), str(raiz), args.grupo, str(args.epocas),
               str(salida), str(i)]
        log = (salida / f"w{i}.log").open("w", encoding="utf-8")
        procesos.append((i, subprocess.Popen(cmd, env=entorno, stdout=log,
                                             stderr=subprocess.STDOUT), log))
        print(f"  lanzado trabajador {i} en GPU {i % max(1, n_gpu)}: "
              f"{len(caja)} corridas")

    print()
    print(f"{len(procesos)} trabajadores corriendo.")
    print(f"seguimiento:  tail -f {salida}/w0.log")
    print()

    codigos = []
    for i, p, log in procesos:
        rc = p.wait()
        log.close()
        codigos.append(rc)
        print(f"  trabajador {i} termino con codigo {rc} "
              f"({(time.perf_counter()-t0)/60:.1f} min)")

    print()
    print(f"TOTAL: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"resultados en {salida}")

    # Recuento a partir de los JSONL, que es la fuente fiable: un codigo 0 no
    # garantiza que las corridas entrenaran.
    ok = fallo = sin_entrenar = 0
    for reg in sorted(salida.glob("trabajador_*.jsonl")):
        for linea in reg.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            d = json.loads(linea)
            if d.get("evento") != "corrida":
                continue
            if not d.get("ok"):
                fallo += 1
            elif not d.get("entreno", True):
                sin_entrenar += 1
            else:
                ok += 1
    print()
    print(f"corridas con resultado y pesos movidos : {ok}")
    print(f"corridas que no movieron los pesos     : {sin_entrenar}")
    print(f"corridas que fallaron                  : {fallo}")
    if sin_entrenar:
        print()
        print("Las que no movieron los pesos NO cuentan como resultado, aunque")
        print("hayan escrito metrics.dat.")
    return 0 if (fallo == 0 and sin_entrenar == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
