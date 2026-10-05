#!/usr/bin/env python3
"""
monitor_fases.py — envoltorio que mide recursos por FASE.

Basado en la arquitectura de monitoreo de `workspace/monitor_training.py` del
repositorio fitsnap-custom de Arbués, ofrecida en la reunión del 27 de
septiembre de 2026. Se conservan los nombres de columna de su CSV para que sus
scripts de comparación (`compare_runs.py`, `generate_paper_figures.py`) puedan
leer estas corridas sin cambios.

Qué añade respecto al monitor que se usaba antes:

1. ETIQUETA DE FASE. Deduce la etapa leyendo las líneas de temporización que
   FitSNAP ya imprime. Sin esto, la utilización de GPU se promedia sobre toda
   la corrida, y como el 99 % del tiempo son descriptores en CPU, el promedio
   sale entre 0 y 9 % y no significa nada. El número que interesa es el de la
   fase de ajuste.

2. VRAM POR PROCESO. Usa `nvidia-smi --query-compute-apps=pid,used_memory`, que
   atribuye la memoria a cada PID, en lugar de leer el total de la tarjeta. El
   28 de septiembre el total incluía 22 GiB de caché de PyTorch del kernel del
   notebook, y se interpretó como consumo del trabajador.

3. RSS DEL PROCESO además de la RAM del sistema, incluyendo los hijos.

Uso:
    python monitor_fases.py --csv metricas.csv -- python trabajador.py ARGS...

El comando va después de `--`. La salida del comando pasa tal cual a la
consola, con un prefijo de tiempo y fase.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time

# Marcadores que FitSNAP imprime al terminar cada etapa. Al verlos, la fase que
# EMPIEZA es la siguiente. Si una versión de FitSNAP cambia estos textos, el
# CSV saldrá con la fase "Desconocida": comprobarlo en la primera corrida.
TRANSICIONES = [
    ("'scrape_configs' took",  "Calculando descriptores"),
    ("'process_configs' took", "Ajustando modelo"),
    ("'fit' took",             "Analisis de error"),
    ("'error_analysis' took",  "Escribiendo salida"),
]

COLUMNAS = ["Time_s", "Phase", "CPU_Percent", "RAM_GB", "Proc_RSS_GB",
            "Disk_Read_MB", "Disk_Write_MB", "GPU_Util_Percent", "GPU_Mem_GB"]


class Estado:
    def __init__(self):
        self.fase = "Leyendo configuracion"
        self.fin = False


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _gpu_visible():
    """Índice físico de la tarjeta que verá el hijo, según CUDA_VISIBLE_DEVICES."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not v:
        return 0
    try:
        return int(v.split(",")[0])
    except ValueError:
        return 0


def _muestra_gpu(indice, pids):
    """Devuelve (utilizacion %, memoria GiB de NUESTROS procesos).

    La memoria se suma solo de los PID del árbol vigilado. Si nvidia-smi no
    puede atribuirla por proceso, se devuelve NaN en vez del total de la
    tarjeta: es preferible un hueco a un número que invita a leerlo mal.
    """
    if not shutil.which("nvidia-smi"):
        return float("nan"), float("nan")
    util = float("nan")
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={indice}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            util = float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    mem = float("nan")
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            total = 0.0
            visto = False
            for linea in r.stdout.strip().split("\n"):
                if not linea.strip():
                    continue
                partes = [p.strip() for p in linea.split(",")]
                if len(partes) < 2:
                    continue
                try:
                    pid, usada = int(partes[0]), float(partes[1])
                except ValueError:
                    continue
                if pid in pids:
                    total += usada
                    visto = True
            if visto:
                mem = total / 1024.0
            elif pids:
                mem = 0.0
    except Exception:
        pass
    return util, mem


def _arbol(psutil_mod, proceso):
    """PIDs del proceso y sus descendientes, tolerando que mueran."""
    pids = set()
    if psutil_mod is None:
        return pids
    try:
        p = psutil_mod.Process(proceso.pid)
        pids.add(p.pid)
        for h in p.children(recursive=True):
            pids.add(h.pid)
    except Exception:
        pass
    return pids


def muestrear(proceso, estado, ruta_csv, cada, t0):
    psutil_mod = _psutil()
    indice = _gpu_visible()
    p_raiz = None
    if psutil_mod is not None:
        try:
            p_raiz = psutil_mod.Process(proceso.pid)
            p_raiz.cpu_percent(None)
        except Exception:
            p_raiz = None

    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(COLUMNAS)
        while not estado.fin:
            t = time.time() - t0
            cpu = ram = rss = leido = escrito = float("nan")
            pids = _arbol(psutil_mod, proceso)
            if psutil_mod is not None:
                try:
                    cpu = psutil_mod.cpu_percent(None)
                    vm = psutil_mod.virtual_memory()
                    ram = (vm.total - vm.available) / 1024 ** 3
                except Exception:
                    pass
                total_rss = 0.0
                for pid in pids:
                    try:
                        total_rss += psutil_mod.Process(pid).memory_info().rss
                    except Exception:
                        pass
                rss = total_rss / 1024 ** 3
                try:
                    d = psutil_mod.disk_io_counters()
                    leido = d.read_bytes / 1024 ** 2
                    escrito = d.write_bytes / 1024 ** 2
                except Exception:
                    pass
            util, gmem = _muestra_gpu(indice, pids)
            escritor.writerow([
                f"{t:.2f}", estado.fase,
                "" if cpu != cpu else f"{cpu:.1f}",
                "" if ram != ram else f"{ram:.2f}",
                "" if rss != rss else f"{rss:.3f}",
                "" if leido != leido else f"{leido:.2f}",
                "" if escrito != escrito else f"{escrito:.2f}",
                "" if util != util else f"{util:.1f}",
                "" if gmem != gmem else f"{gmem:.3f}",
            ])
            fh.flush()
            time.sleep(cada)


def leer_salida(proceso, estado, t0, silencioso):
    for linea in iter(proceso.stdout.readline, ""):
        if not linea:
            break
        for marca, siguiente in TRANSICIONES:
            if marca in linea:
                estado.fase = siguiente
                break
        if not silencioso:
            sys.stdout.write(f"[{time.time()-t0:7.1f}s][{estado.fase}] {linea}")
            sys.stdout.flush()
    estado.fin = True


def resumen(ruta_csv):
    """Imprime, por fase, duración y los máximos que importan."""
    filas = []
    with open(ruta_csv, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            filas.append(f)
    if not filas:
        print("sin muestras")
        return

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    fases = []
    for f in filas:
        if not fases or fases[-1][0] != f["Phase"]:
            fases.append([f["Phase"], [], []])
        fases[-1][1].append(num(f["Time_s"]))
        fases[-1][2].append(f)

    print()
    print(f"{'fase':26s} {'seg':>8s} {'GPU med':>8s} {'GPU max':>8s} "
          f"{'VRAM GiB':>9s} {'RSS GiB':>8s} {'CPU max':>8s}")
    print("-" * 80)
    for nombre, tiempos, muestras in fases:
        dur = max(tiempos) - min(tiempos)
        def col(k):
            v = [num(m[k]) for m in muestras]
            return [x for x in v if x == x]
        g, vm = col("GPU_Util_Percent"), col("GPU_Mem_GB")
        rs, cp = col("Proc_RSS_GB"), col("CPU_Percent")
        print(f"{nombre:26s} {dur:8.1f} "
              f"{(sum(g)/len(g) if g else float('nan')):8.1f} "
              f"{(max(g) if g else float('nan')):8.1f} "
              f"{(max(vm) if vm else float('nan')):9.2f} "
              f"{(max(rs) if rs else float('nan')):8.2f} "
              f"{(max(cp) if cp else float('nan')):8.1f}")
    print()
    print("La fila que sirve para dimensionar es 'Ajustando modelo'. Las demas")
    print("miden preprocesado y escritura, que no son el objeto de comparacion.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="system_metrics.csv",
                    help="archivo de salida (por defecto system_metrics.csv)")
    ap.add_argument("--cada", type=float, default=1.0,
                    help="segundos entre muestras (por defecto 1.0)")
    ap.add_argument("--silencioso", action="store_true",
                    help="no repetir la salida del comando")
    ap.add_argument("comando", nargs=argparse.REMAINDER,
                    help="-- seguido del comando a ejecutar")
    args = ap.parse_args()

    cmd = args.comando
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("falta el comando. Usar: monitor_fases.py --csv x.csv -- python ...")

    if _psutil() is None:
        print("AVISO: psutil no esta instalado; CPU, RAM y disco saldran vacios.")
        print("       pip install psutil")

    t0 = time.time()
    estado = Estado()
    proceso = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    hilo_csv = threading.Thread(target=muestrear,
                                args=(proceso, estado, args.csv, args.cada, t0),
                                daemon=True)
    hilo_csv.start()
    leer_salida(proceso, estado, t0, args.silencioso)
    rc = proceso.wait()
    estado.fin = True
    hilo_csv.join(timeout=args.cada + 2)

    print()
    print(f"codigo de salida: {rc}   tiempo total: {(time.time()-t0)/60:.1f} min")
    print(f"metricas en: {args.csv}")
    try:
        resumen(args.csv)
    except Exception as e:
        print(f"(no se pudo resumir: {e})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
