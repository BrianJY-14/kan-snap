#!/usr/bin/env python3
"""
trabajador.py — ejecuta un lote de corridas en un proceso propio.

Los descriptores se calculan UNA vez por trabajador y se reutilizan en todas
sus corridas; lo unico que cambia entre corridas es la cabecera de la red, que
`cambiar_arquitectura` sustituye en caliente.

DISPOSICION DE SALIDA. Sigue la convencion de `workspace/07_experimentos/` del
repositorio fitsnap-custom de Arbues, para que su `compare_runs.py` lea estas
corridas sin modificaciones:

    <salida>/<grupo>/run_NN_<id>_s<semilla>/
        <id>_s<semilla>_metrics.dat     <- renombrado; FitSNAP lo llama metrics.dat
        loss_vs_epochs.dat
        perconfig.dat                   <- lo usa compare_runs.py para los parity plots
        peratom.dat
        system_metrics.csv              <- recursos de ESTA corrida
        pot.mod, pot.mliap.descriptor, FitTorch_Pytorch.pt, modelo.pt

El numero NN sale del prefijo del id de la condicion, asi que `1_mlp_base` cae
en `run_01_...` y `compare_runs.py` lo toma como baseline automaticamente: su
`_sort_key()` busca "baseline" o "run_01" en el nombre.

POR QUE SE MUEVEN LOS ARCHIVOS. FitSNAP resuelve metrics.dat, peratom.dat,
perconfig.dat y los modelos al CONSTRUIR el objeto, no al escribir, asi que
todas las corridas de un mismo trabajador escriben en la carpeta del trabajador
y se pisan entre si. El 1 de octubre de 2026 solo sobrevivio el metrics.dat de
la ultima corrida de cada trabajador. Aqui se mueven por fecha de modificacion,
sin depender de internos de FitSNAP.

Uso:
    python trabajador.py TAREAS_JSON RAIZ GRUPO EPOCAS SALIDA ETIQUETA

    TAREAS_JSON  lista JSON de tareas. Cada tarea:
                 {"id": str, "capas": str, "salida": str, "kw": dict,
                  "semilla": int, "lr": float opcional}

Variables de entorno:
    CUDA_VISIBLE_DEVICES   tarjeta asignada
    KAN_DTYPE_SETTING      sin valor = el defecto de FitSNAP (float32)
    KAN_NORMALIZAR         canal | layernorm | cuantiles | no  (defecto: canal)
                           como se gestiona el rango de entrada del bloque KAN
    KAN_FRACCION, KAN_BATCH_SIZE, KAN_LR
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from kan_lib import *          # noqa: F401,F403
from kan_lib import (ajustes_fitsnap, carpeta_datos, aplicar_todos_los_parches,
                     cambiar_arquitectura)

# Mismas columnas que monitor_training.py de fitsnap-custom, para que
# compare_runs.py las encuentre por nombre.
COLUMNAS = ["Time_s", "Phase", "CPU_Percent", "RAM_GB", "Proc_RSS_GB",
            "Disk_Read_MB", "Disk_Write_MB", "GPU_Util_Percent", "GPU_Mem_GB"]


class MuestreadorRecursos:
    """Muestrea recursos de ESTA corrida y escribe un system_metrics.csv.

    El monitor externo produce un CSV por trabajador; compare_runs.py espera
    uno por corrida. Este lo genera desde dentro, con las mismas columnas.
    """

    def __init__(self, ruta, fase="Ajustando modelo", cada=1.0):
        self.ruta, self.fase, self.cada = Path(ruta), fase, cada
        self._parar = threading.Event()
        self._hilo = None
        self.filas = []

    def _psutil(self):
        try:
            import psutil
            return psutil
        except ImportError:
            return None

    def _muestra(self, t):
        ps = self._psutil()
        cpu = ram = rss = leido = escrito = ""
        if ps is not None:
            try:
                cpu = f"{ps.cpu_percent(None):.1f}"
                vm = ps.virtual_memory()
                ram = f"{(vm.total - vm.available) / 1024**3:.2f}"
                p = ps.Process()
                total = p.memory_info().rss
                for h in p.children(recursive=True):
                    try:
                        total += h.memory_info().rss
                    except Exception:
                        pass
                rss = f"{total / 1024**3:.3f}"
                d = ps.disk_io_counters()
                leido = f"{d.read_bytes / 1024**2:.2f}"
                escrito = f"{d.write_bytes / 1024**2:.2f}"
            except Exception:
                pass
        util, gmem = "", ""
        try:
            if torch.cuda.is_available():
                gmem = f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}"
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4)
                if r.returncode == 0 and r.stdout.strip():
                    util = f"{float(r.stdout.strip().splitlines()[0]):.1f}"
            except Exception:
                pass
        return [f"{t:.2f}", self.fase, cpu, ram, rss, leido, escrito, util, gmem]

    def _bucle(self, t0):
        while not self._parar.wait(self.cada):
            self.filas.append(self._muestra(time.time() - t0))

    def __enter__(self):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        t0 = time.time()
        self.filas = [self._muestra(0.0)]
        self._hilo = threading.Thread(target=self._bucle, args=(t0,), daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *a):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=self.cada + 1)
        return False

    def volcar(self, destino):
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNAS)
            w.writerows(self.filas)


def optimizador_acoplado(fs):
    """¿El optimizador apunta a los parametros del modelo que se evalua?

    Esta es la comprobacion que habria ahorrado la sesion del 28 de septiembre.
    `perform_fit` termina sustituyendo `solver.model` por `model_best`, que es
    un deepcopy, mientras el optimizador sigue ligado a los tensores del modelo
    anterior. A partir de ahi `step()` actualiza parametros huerfanos y la
    perdida se queda congelada, con cualquier tasa de aprendizaje.

    Compara por identidad de objeto, no por valor: es lo unico que distingue un
    tensor de su copia.
    """
    ids_modelo = {id(p) for p in fs.solver.model.parameters()}
    ids_optim = {id(p) for g in fs.solver.optimizer.param_groups
                 for p in g["params"]}
    if not ids_optim:
        return False, "el optimizador no tiene parametros"
    huerfanos = ids_optim - ids_modelo
    if huerfanos:
        return False, (f"{len(huerfanos)} de {len(ids_optim)} tensores del "
                       f"optimizador no pertenecen al modelo")
    faltan = ids_modelo - ids_optim
    if faltan:
        return False, (f"{len(faltan)} parametros del modelo no estan en el "
                       f"optimizador: no se entrenarian")
    return True, None


def numero_de_corrida(id_condicion):
    """run_01 para `1_mlp_base`, etc. El 01 es el baseline de compare_runs.py."""
    cabeza = str(id_condicion).split("_")[0]
    return f"run_{int(cabeza):02d}" if cabeza.isdigit() else "run_99"


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        return 2

    tareas = json.loads(sys.argv[1])
    raiz = Path(sys.argv[2])
    grupo = sys.argv[3]
    # Un grupo, varios separados por coma, o "todos" para el conjunto completo.
    # El presupuesto se cuenta en pasos de optimizador, y los pasos por epoca
    # dependen de cuantas configuraciones entren: por eso esto es una palanca.
    if grupo == "todos":
        _base = carpeta_datos(Path(sys.argv[2]))
        _grupos = sorted(d.name for d in _base.iterdir() if d.is_dir())
    else:
        _grupos = [g for g in grupo.split(",") if g]
    epocas = int(sys.argv[4])
    salida = Path(sys.argv[5])
    etiqueta = sys.argv[6]

    aplicar_todos_los_parches()
    from fitsnap3lib.fitsnap import FitSnap

    salida.mkdir(parents=True, exist_ok=True)
    registro = salida / f"trabajador_{etiqueta}.jsonl"

    def anota(d):
        d["t"] = time.strftime("%H:%M:%S")
        with registro.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")

    _v = os.environ.get("KAN_DTYPE_SETTING", "").strip()
    dtype_setting = int(_v) if _v else None
    anota({"evento": "inicio", "grupo": grupo, "epocas": epocas,
           "tareas": [t["id"] for t in tareas],
           "dtype_setting": dtype_setting,
           "modo_kan": str(globals().get("MODO_KAN", "canal")),
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "(todas)")})

    t0 = time.perf_counter()
    taller = salida / f"w{etiqueta}"          # donde FitSNAP escribe de verdad
    taller.mkdir(parents=True, exist_ok=True)
    previo = os.getcwd()
    os.chdir(taller)

    fallos = 0
    try:
        cfg = ajustes_fitsnap(carpeta_datos(raiz), semilla=1111, epocas=epocas,
                              grupos=_grupos, dtype_setting=dtype_setting)
        fs = FitSnap(cfg, arglist=["--overwrite"])
        fs.scrape_configs()
        fs.process_configs()
        seg_desc = time.perf_counter() - t0
        anota({"evento": "descriptores", "seg": round(seg_desc, 1),
               "min": round(seg_desc / 60, 2),
               "dtype_modelo": str(getattr(fs.solver, "dtype", "?"))})
        print(f"[trabajador {etiqueta}] descriptores en {seg_desc/60:.1f} min")

        for t in tareas:
            ti = time.perf_counter()
            ti_reloj = time.time()
            nombre = f'{t["id"]}_s{t["semilla"]}'
            # <salida>/<grupo>/run_NN_<id>_s<semilla>/ — convencion de
            # 07_experimentos/ de fitsnap-custom.
            _etiq = grupo if len(_grupos) == 1 else "todos"
            destino = salida / _etiq / f'{numero_de_corrida(t["id"])}_{nombre}'
            destino.mkdir(parents=True, exist_ok=True)
            try:
                lr = t.get("lr")
                if lr is not None:
                    fs.solver.learning_rate = float(lr)
                    fs.config.sections["PYTORCH"].learning_rate = float(lr)

                # Reconstruye modelo Y optimizador, y borra model_best.
                cambiar_arquitectura(fs, t["capas"], t["salida"],
                                     t.get("kw") or {}, semilla=t["semilla"])

                ok, motivo = optimizador_acoplado(fs)
                if not ok:
                    raise RuntimeError(
                        f"optimizador desacoplado antes del ajuste: {motivo}")

                torch.manual_seed(t["semilla"])
                np.random.seed(t["semilla"])
                fs.config.sections["PYTORCH"].num_epochs = epocas

                antes = {k: v.detach().float().cpu().clone()
                         for k, v in fs.solver.model.named_parameters()}

                aqui = os.getcwd()
                os.chdir(destino)
                try:
                    with MuestreadorRecursos(destino) as mon:
                        fs.perform_fit()
                        fs.write_output()
                finally:
                    os.chdir(aqui)
                mon.volcar(destino / "system_metrics.csv")

                # Mover lo que escribio esta corrida en la carpeta del taller.
                movidos = []
                for f in sorted(taller.iterdir()):
                    if f.is_file() and f.stat().st_mtime >= ti_reloj - 1:
                        f.replace(destino / f.name)
                        movidos.append(f.name)

                # Renombrar metrics.dat como espera compare_runs.py: *_metrics.dat
                ruta_metrics = destino / f"{nombre}_metrics.dat"
                if (destino / "metrics.dat").exists():
                    (destino / "metrics.dat").replace(ruta_metrics)

                # ¿Se movieron los pesos? Si no, el ajuste no ha ocurrido y el
                # resultado no vale, aunque metrics.dat exista.
                delta = 0.0
                for k, v in fs.solver.model.named_parameters():
                    if k in antes and v.shape == antes[k].shape:
                        delta = max(delta, (v.detach().float().cpu()
                                            - antes[k]).abs().max().item())

                anota({"evento": "corrida", "id": t["id"],
                       "semilla": t["semilla"], "lr": lr,
                       "min": round((time.perf_counter() - ti) / 60, 2),
                       "ok": True, "cambio_max_pesos": delta,
                       "entreno": bool(delta > 0),
                       "carpeta": str(destino), "salidas": movidos,
                       "metrics": str(ruta_metrics)})
                aviso = "" if delta > 0 else "   <-- LOS PESOS NO SE MOVIERON"
                print(f"[trabajador {etiqueta}] {destino.name} "
                      f"{(time.perf_counter()-ti)/60:.1f} min  "
                      f"delta={delta:.3e}{aviso}")

            except Exception as e:
                fallos += 1
                anota({"evento": "corrida", "id": t["id"],
                       "semilla": t["semilla"], "ok": False,
                       "carpeta": str(destino),
                       "error": str(e)[:400],
                       "traza": traceback.format_exc()[-1500:]})
                print(f"[trabajador {etiqueta}] {nombre} FALLO: "
                      f"{str(e)[:200]}")
    finally:
        os.chdir(previo)

    total = (time.perf_counter() - t0) / 60
    anota({"evento": "fin", "min": round(total, 2), "fallos": fallos})
    print(f"[trabajador {etiqueta}] fin: {total:.1f} min, {fallos} fallos")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
