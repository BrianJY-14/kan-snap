#!/usr/bin/env python3
"""
sonda_gradientes.py — donde se queda el gradiente, medido dentro del pipeline.

Hasta ahora el diagnostico se ha hecho mirando la perdida por fuera. Eso dice
que las KAN no aprenden, no por que. Esta sonda engancha a cada parametro del
modelo y a la entrada de la ultima capa, corre unas pocas epocas reales de
FitSNAP y responde tres preguntas:

  1. La ultima capa, ¿recibe gradiente? Si su |grad| es comparable al de la
     misma capa en el MLP, el problema no es que la base este muerta.
  2. El tronco, ¿recibe gradiente a traves de la cabecera KAN? Una base con
     pesos pequenos tiene derivada d(salida)/d(entrada) casi nula, asi que
     puede estrangular al tronco aunque ella misma se mueva.
  3. ¿Cuanto se mueven de verdad los parametros, por grupo, en terminos
     relativos? El `cambio_max_pesos` del trabajador esta dominado por la capa
     de estandarizacion de FitSNAP y no sirve para esto.

La comparacion que importa es correr esto dos veces, con --arquitectura linear
y con gaussiana, y poner las dos tablas una al lado de la otra.

Uso:
    CUDA_VISIBLE_DEVICES=4 $PY sonda_gradientes.py --raiz /workspace/kan_snap \\
        --arquitectura gaussiana --lr 5e-3 --epocas 2

    KAN_NORMALIZAR=layernorm CUDA_VISIBLE_DEVICES=5 $PY sonda_gradientes.py ...

Cuesta lo que cuestan los descriptores (~7 min) mas las epocas que se pidan.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import kan_lib
from kan_lib import (ajustes_fitsnap, aplicar_todos_los_parches,
                     cambiar_arquitectura)


# El gancho tiene que ser una funcion de modulo, no una local: al terminar
# perform_fit, FitSNAP guarda el modelo con torch.save del objeto entero, y los
# ganchos de forward viajan dentro. Una funcion local no se puede serializar y
# el guardado revienta despues de haber entrenado, perdiendo la medicion.
ENTRADA = {}


def mira_entrada(mod, args_in):
    x = args_in[0].detach()
    if x.dim() < 2:
        return
    foto = (float(x.mean()), float(x.std()), float(x.std(0).mean()),
            float(x.min()), float(x.max()))
    ENTRADA.setdefault("primera", foto)
    ENTRADA["ultima"] = foto


def grupo_de(nombre):
    """Agrupa los parametros por capa para que la tabla quepa y se lea."""
    partes = nombre.split(".")
    if len(partes) >= 3:
        return f"capa {partes[1]} ({partes[2]})"
    return nombre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--grupo", default="LiF_64_kjpaw")
    ap.add_argument("--arquitectura", default="gaussiana",
                    choices=["linear", "gaussiana", "bspline", "chebyshev"])
    ap.add_argument("--capas", default="num_desc 256 128 64 64 1")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--semilla", type=int, default=1111)
    args = ap.parse_args()

    print(f"arquitectura {args.arquitectura}   lr {args.lr:g}   "
          f"epocas {args.epocas}   modo rango {getattr(kan_lib,'MODO_KAN','?')}")

    aplicar_todos_los_parches()
    from fitsnap3lib.fitsnap import FitSnap

    trabajo = Path.cwd() / "sonda_tmp"
    trabajo.mkdir(parents=True, exist_ok=True)
    previo = os.getcwd()
    os.chdir(trabajo)

    try:
        t0 = time.perf_counter()
        cfg = ajustes_fitsnap(Path(args.raiz) / "JSON", semilla=args.semilla,
                              epocas=args.epocas, grupos=[args.grupo])
        cfg["PYTORCH"]["learning_rate"] = args.lr
        fs = FitSnap(cfg, arglist=["--overwrite"])
        fs.scrape_configs()
        fs.process_configs()
        print(f"descriptores en {(time.perf_counter()-t0)/60:.1f} min")

        fs.solver.learning_rate = args.lr
        fs.config.sections["PYTORCH"].learning_rate = args.lr
        cambiar_arquitectura(fs, args.capas, args.arquitectura, {},
                             semilla=args.semilla)
        fs.config.sections["PYTORCH"].num_epochs = args.epocas
        torch.manual_seed(args.semilla)
        np.random.seed(args.semilla)

        modelo = fs.solver.model
        antes = {k: p.detach().float().cpu().clone()
                 for k, p in modelo.named_parameters()}

        # --- 1. gradiente por parametro, acumulado en cada backward ---------
        suma = defaultdict(float)
        veces = defaultdict(int)
        pico = defaultdict(float)

        def engancha(nombre):
            def fn(g):
                if g is not None:
                    a = g.detach().abs()
                    suma[nombre] += float(a.mean())
                    pico[nombre] = max(pico[nombre], float(a.max()))
                    veces[nombre] += 1
            return fn

        asas = [p.register_hook(engancha(k))
                for k, p in modelo.named_parameters() if p.requires_grad]

        # --- 2. que entra a la ultima capa ----------------------------------
        ultima_capa = modelo.networks[0][-1]
        asas.append(ultima_capa.register_forward_pre_hook(mira_entrada))
        print(f"ultima capa: {type(ultima_capa).__name__}")

        t1 = time.perf_counter()
        try:
            fs.perform_fit()
            print(f"ajuste en {(time.perf_counter()-t1)/60:.1f} min\n")
        except Exception as e:
            # El ajuste puede haber terminado y fallar solo al escribir el
            # modelo. La medicion ya esta tomada: se informa y se sigue.
            print(f"\nperform_fit lanzo: {type(e).__name__}: {str(e)[:200]}")
            print(f"(a los {(time.perf_counter()-t1)/60:.1f} min). "
                  "Si el fallo fue al guardar, la tabla de abajo es valida.\n")
        finally:
            for a in asas:
                a.remove()

        # --- resultados ------------------------------------------------------
        if ENTRADA:
            for cual in ("primera", "ultima"):
                if cual in ENTRADA:
                    m, s, sc, lo, hi = ENTRADA[cual]
                    print(f"entrada a la ultima capa, {cual} pasada: "
                          f"media {m:.4g}  desv global {s:.4g}  "
                          f"desv por canal {sc:.4g}  rango [{lo:.4g}, {hi:.4g}]")
        else:
            print("no se capturo la entrada a la ultima capa")
        print()

        print(f"{'grupo':26s} {'n par':>7s} {'|grad| medio':>13s} "
              f"{'|grad| max':>12s} {'backwards':>10s} {'cambio rel':>11s}")
        print("-" * 86)
        agg = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])
        for k, p in modelo.named_parameters():
            g = grupo_de(k)
            d = agg[g]
            d[0] += p.numel()
            n = max(1, veces[k])
            d[1] += suma[k] / n
            d[2] = max(d[2], pico[k])
            d[3] = max(d[3], veces[k])
            if k in antes and p.shape == antes[k].shape:
                dif = (p.detach().float().cpu() - antes[k]).norm().item()
                base = antes[k].norm().item()
                d[4] += dif
                d[5] += base
        for g in sorted(agg):
            n_par, gmed, gmax, nb, dif, base = agg[g]
            rel = dif / base if base > 0 else float("nan")
            print(f"{g:26s} {n_par:7d} {gmed:13.3e} {gmax:12.3e} "
                  f"{nb:10d} {rel:11.3e}")

        print()
        print("Como leerlo.")
        print(" - 'backwards' en 0 para un grupo: ese grupo no recibio gradiente")
        print("   nunca. Es el caso roto de verdad.")
        print(" - La ultima capa con |grad| sano y el tronco con |grad| muy por")
        print("   debajo del caso linear: la cabecera esta estrangulando al")
        print("   tronco, que es lo que hace una base con pesos pequenos.")
        print(" - 'cambio rel' es ||delta|| / ||inicial|| por grupo. A diferencia")
        print("   de cambio_max_pesos, no lo domina la capa de estandarizacion.")
    finally:
        os.chdir(previo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
