#!/usr/bin/env python3
"""
barrida.py — calibracion de tasa, de tamano de lote o de modo de rango.

    --que lr     barre tasas            (por defecto 5e-6 5e-5 5e-4 5e-3)
    --que lote   barre tamanos de lote  (por defecto 10 25 50 100 200)
    --que modo   barre la gestion del rango de entrada del bloque KAN
                 (por defecto no canal layernorm cuantiles)

El modo solo tiene sentido con una arquitectura KAN: con `linear` no hay base
fija que se pueda quedar fuera del rango y el programa se detiene. Con
`chebyshev` se salta `cuantiles`, que no aplica a una base global.

Por que existe este archivo. Las barridas del 27 y 28 de septiembre no
midieron nada. La de tasas dio 0.00 % de caida con cuatro valores separados
por tres ordenes de magnitud, y de la de lotes solo la primera fila entreno.
El motivo esta en fitsnap3lib/solvers/pytorch.py, dentro de evaluate_configs:

    if hasattr(self, "model_best"):
        self.model = self.model_best      # deepcopy: otro objeto

El optimizador se construye una sola vez, en Solver.__init__, y nunca se
rehace. Despues del primer ajuste queda ligado a los tensores del modelo
anterior, asi que `step()` actualiza parametros que ya nadie evalua. Las
barridas viejas restauraban los pesos con load_state_dict sobre ese
optimizador viejo, y por eso la perdida se quedaba congelada.

Esta pasa por `cambiar_arquitectura`, que reconstruye el optimizador sobre el
modelo vigente y borra `model_best`, y comprueba por identidad, antes de cada
ajuste, que optimizador y modelo comparten los mismos tensores.

Los descriptores se calculan una sola vez y se reutilizan entre valores: son
~7.5 minutos y no dependen ni de la tasa ni del lote. La arquitectura y la
semilla son las mismas en todos los casos, asi que el punto de partida es
identico.

Uso:
    python barrida.py --que lote --raiz /workspace/kan_snap
    python barrida.py --que lr   --raiz /workspace/kan_snap
    python barrida.py --que modo --raiz /workspace/kan_snap \\
        --arquitectura gaussiana --semillas 1111 2222

    # bajo el monitor por fases, que es como conviene:
    python monitor_fases.py --csv barrida_lote.csv -- \\
        python barrida.py --que lote --raiz /workspace/kan_snap

ORDEN. El lote va primero. Lote y tasa interactuan: con lotes mayores el
gradiente tiene menos ruido y admite pasos mayores. Decidir el lote despues de
barrer las tasas invalida la barrida de tasas.

Si la comprobacion de acoplamiento falla, el programa se detiene y lo dice. En
ese caso hay que volver a un proceso por valor:

    for v in 5e-6 5e-5 5e-4 5e-3; do
        python barrida.py --que lr --valores $v --raiz ... --salida lr_$v.json
    done
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import kan_lib
from kan_lib import (ajustes_fitsnap, carpeta_datos, aplicar_todos_los_parches,
                     cambiar_arquitectura, _curva_desde_texto)

POR_DEFECTO = {
    "lr": [5e-6, 5e-5, 5e-4, 5e-3],      # la primera es la del grupo
    "lote": [10, 25, 50, 100, 200],      # 50 es el del grupo
    "modo": ["no", "canal", "layernorm", "cuantiles"],   # "no" es el control
}

# MODO_KAN es un global de kan_lib que `_normalizar_entrada` lee en cada
# llamada, asi que cambiarlo aqui basta: la capa nueva que crea
# `cambiar_arquitectura` se calibra desde cero con el modo vigente.
def fijar_modo(v):
    kan_lib.MODO_KAN = v
    kan_lib.NORMALIZAR_KAN = v != "no"


def optimizador_acoplado(fs):
    """Identidad, no valor: es lo unico que distingue un tensor de su copia."""
    ids_modelo = {id(p) for p in fs.solver.model.parameters()}
    ids_optim = {id(p) for g in fs.solver.optimizer.param_groups
                 for p in g["params"]}
    if not ids_optim:
        return False, "el optimizador no tiene parametros"
    if ids_optim - ids_modelo:
        return False, f"{len(ids_optim - ids_modelo)} tensores huerfanos"
    if ids_modelo - ids_optim:
        return False, f"{len(ids_modelo - ids_optim)} parametros fuera"
    return True, None


def vram_pico_mib():
    """Lo que el modelo pidio de verdad, no lo que nvidia-smi ve reservado."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--que", choices=["lr", "lote", "modo"], default="lr")
    ap.add_argument("--valores", nargs="+", default=None,
                    help="si no se da, los de POR_DEFECTO segun --que")
    ap.add_argument("--raiz", required=True, help="carpeta que contiene JSON/")
    ap.add_argument("--grupo", default="LiF_64_kjpaw")
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=1111)
    ap.add_argument("--semillas", nargs="+", type=int, default=None,
                    help="repite cada valor con varias semillas, reusando "
                         "los descriptores. Por defecto, solo --semilla")
    ap.add_argument("--capas", default="num_desc 256 128 64 64 1")
    ap.add_argument("--arquitectura", default="linear",
                    choices=["linear", "gaussiana", "bspline", "chebyshev"])
    ap.add_argument("--dtype-setting", type=int, default=None,
                    help="1 float32, 2 float64. Por defecto KAN_DTYPE_SETTING o 1")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    valores = args.valores if args.valores is not None else POR_DEFECTO[args.que]
    if args.que == "lote":
        valores = [int(v) for v in valores]
    elif args.que == "lr":
        valores = [float(v) for v in valores]
    else:
        valores = [str(v).strip().lower() for v in valores]
        desconocidos = [v for v in valores if v not in POR_DEFECTO["modo"]]
        if desconocidos:
            print(f"modos no reconocidos: {desconocidos}")
            print(f"los validos son {POR_DEFECTO['modo']}")
            return 1
        if args.arquitectura == "linear":
            print("--que modo necesita una arquitectura KAN. Con 'linear' no hay")
            print("base fija que pueda quedar fuera del rango: usar, por ejemplo,")
            print("  --arquitectura gaussiana")
            return 1
        if args.arquitectura == "chebyshev" and "cuantiles" in valores:
            print("aviso: 'cuantiles' no aplica al Chebyshev (base global, sin")
            print("rejilla). Se salta.")
            valores = [v for v in valores if v != "cuantiles"]
        if not hasattr(kan_lib, "MODO_KAN"):
            print("kan_lib no tiene MODO_KAN: falta aplicar parche_variantes.py")
            return 1

    semillas = args.semillas or [args.semilla]
    salida_json = args.salida or f"barrida_{args.que}.json"

    raiz = Path(args.raiz)
    aplicar_todos_los_parches()
    from fitsnap3lib.fitsnap import FitSnap

    trabajo = Path.cwd() / f"barrida_{args.que}_tmp"
    trabajo.mkdir(parents=True, exist_ok=True)
    previo = os.getcwd()
    os.chdir(trabajo)

    resultados = {}
    try:
        etiqueta = {"lr": "tasa de aprendizaje", "lote": "tamano de lote",
                    "modo": "modo de gestion del rango"}[args.que]
        print(f"barrida de {etiqueta}: {valores}")
        print(f"grupo {args.grupo}, {args.epocas} epocas por valor, "
              f"arquitectura {args.arquitectura}")
        if len(semillas) > 1:
            print(f"semillas: {semillas}  ->  {len(valores)*len(semillas)} ajustes")
        t0 = time.perf_counter()
        cfg = ajustes_fitsnap(carpeta_datos(raiz), semilla=args.semilla,
                              epocas=args.epocas, grupos=[args.grupo],
                              dtype_setting=args.dtype_setting)
        fs = FitSnap(cfg, arglist=["--overwrite"])
        fs.scrape_configs()
        fs.process_configs()
        print(f"descriptores listos en {(time.perf_counter()-t0)/60:.1f} min "
              f"(se reutilizan para los {len(valores)*len(semillas)} ajustes)")
        print(f"dtype del solver: {getattr(fs.solver, 'dtype', '?')}")
        print()

        for v, semilla in [(v, s) for v in valores for s in semillas]:
            ti = time.perf_counter()
            if args.que == "lr":
                fs.solver.learning_rate = float(v)
                fs.config.sections["PYTORCH"].learning_rate = float(v)
            elif args.que == "lote":
                # FitSNAP llama a create_datasets dentro de cada perform_fit y
                # lee batch_size de la configuracion viva, asi que basta
                # cambiarlo aqui.
                fs.config.sections["PYTORCH"].batch_size = int(v)
            else:
                # Antes de crear la cabecera: la capa nueva se calibra con el
                # modo vigente en su primer lote de entrenamiento.
                fijar_modo(v)

            with contextlib.redirect_stdout(io.StringIO()):
                cambiar_arquitectura(fs, args.capas, args.arquitectura, {},
                                     semilla=semilla)

            ok, motivo = optimizador_acoplado(fs)
            if not ok:
                print(f"DETENIDO: {motivo}")
                print("El optimizador no apunta al modelo. Ver la cabecera de "
                      "este archivo: hay que correr un valor por proceso.")
                return 1

            lr_real = fs.solver.optimizer.param_groups[0]["lr"]
            lote_real = fs.config.sections["PYTORCH"].batch_size
            torch.manual_seed(semilla)
            np.random.seed(semilla)
            fs.config.sections["PYTORCH"].num_epochs = args.epocas

            antes = {k: p.detach().float().cpu().clone()
                     for k, p in fs.solver.model.named_parameters()}
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            sub = trabajo / f"{args.que}_{v}_s{semilla}"
            sub.mkdir(parents=True, exist_ok=True)
            aqui = os.getcwd()
            os.chdir(sub)
            buf = io.StringIO()
            fallo = None
            try:
                with contextlib.redirect_stdout(buf):
                    fs.perform_fit()
            except Exception as e:
                fallo = str(e)[:300]
            finally:
                os.chdir(aqui)

            delta = 0.0
            for k, p in fs.solver.model.named_parameters():
                if k in antes and p.shape == antes[k].shape:
                    delta = max(delta, (p.detach().float().cpu()
                                        - antes[k]).abs().max().item())

            curva = _curva_desde_texto(buf.getvalue())
            seg = time.perf_counter() - ti
            n_ep = max(1, len(curva))
            clave = str(v) if len(semillas) == 1 else f"{v}_s{semilla}"
            resultados[clave] = dict(
                valor=v, que=args.que, semilla=semilla, lr=lr_real,
                lote=lote_real, modo=getattr(kan_lib, "MODO_KAN", None),
                segundos=seg, seg_por_epoca=seg / n_ep,
                vram_mib=vram_pico_mib(), cambio_max_pesos=delta,
                curva=curva, fallo=fallo, salida=buf.getvalue()[-1500:])

            print(f"=== {args.que} = {v}"
                  f"{'' if len(semillas) == 1 else f', semilla {semilla}'} ===")
            if fallo:
                print(f"  FALLO: {fallo}")
                print()
                continue
            print(f"  lr en el optimizador       {lr_real:.2e}")
            print(f"  batch_size en uso          {lote_real}")
            print(f"  cambio maximo en los pesos {delta:.3e}"
                  f"{'   <-- NO SE MOVIERON' if delta == 0 else ''}")
            if curva:
                p0, pf = curva[0][1], curva[-1][1]
                print(f"  epocas leidas   {len(curva)}")
                print(f"  perdida inicial {p0:.4e}")
                print(f"  perdida final   {pf:.4e}")
                print(f"  caida           {(1-pf/p0)*100:6.2f} %")
            else:
                print("  no se pudo leer la curva")
            print(f"  seg por epoca   {seg/n_ep:.1f}")
            print(f"  VRAM pico real  {vram_pico_mib():.0f} MiB")
            print()
    finally:
        os.chdir(previo)

    Path(salida_json).write_text(json.dumps(resultados, indent=1),
                                 encoding="utf-8")

    print("=" * 88)
    print(f"{args.que:>10s} {'semilla':>8s} {'lr optim':>10s} {'lote':>6s} "
          f"{'p inicial':>12s} {'p final':>12s} {'caida':>8s} {'s/epoca':>8s} "
          f"{'delta':>10s}")
    print("-" * 88)
    for k, r in resultados.items():
        c = r["curva"]
        p0 = c[0][1] if c else float("nan")
        pf = c[-1][1] if c else float("nan")
        caida = (1 - pf / p0) * 100 if c else float("nan")
        print(f"{str(r['valor']):>10s} {r['semilla']:8d} {r['lr']:10.2e} "
              f"{r['lote']:6d} {p0:12.4e} {pf:12.4e} {caida:7.2f}% "
              f"{r['seg_por_epoca']:8.1f} {r['cambio_max_pesos']:10.2e}")
    print()
    print(f"guardado en {salida_json}")
    print()
    print("Como leerlo. Si la columna 'delta' es 0 en alguna fila, esa fila no")
    print("entreno y su caida no significa nada. Solo cuando todas son > 0 la")
    print("tabla mide lo que debe medir.")
    if args.que == "lote":
        print()
        print("Para el lote, ademas: un lote mayor da menos actualizaciones de")
        print("pesos por epoca, asi que puede ir mas rapido por epoca y aprender")
        print("menos por epoca. Hay que mirar 's/epoca' y 'caida' juntas.")
    if args.que == "modo":
        print()
        print("Para el modo: 'no' es el control. Los cuatro modos tienen el mismo")
        print("numero de parametros, asi que la diferencia no es capacidad. Si")
        print("'no' se queda plano y los otros tres caen, el problema era el")
        print("rango de entrada y no la arquitectura KAN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
