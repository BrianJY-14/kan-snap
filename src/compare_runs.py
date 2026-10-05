#!/usr/bin/env python3
"""
compare_runs.py — FitSNAP Experiment Comparison Tool
=====================================================
Escanea workspace/07_experimentos/ buscando todas las corridas completadas
y genera una tabla comparativa con deltas respecto al baseline.

Uso:
    python compare_runs.py                      # Reporte completo en consola + .md
    python compare_runs.py --dataset LiBF4      # Filtrar por dataset
    python compare_runs.py --plots              # Parity plots + curvas de loss
    python compare_runs.py --system             # Incluir métricas de RAM/tiempo
    python compare_runs.py --winners-only       # Solo runs que superan al baseline

El script detecta automáticamente archivos *_metrics.dat en el árbol de directorios
(cualquier profundidad, vía Path.rglob). Cada directorio que contiene uno de estos
archivos es una "corrida" (run); el dataset y el nombre del run se derivan de los
últimos 1-2 componentes de la ruta relativa a 07_experimentos/ (ver find_runs()).

Archivos leídos por corrida (todos opcionales salvo *_metrics.dat):
    <algo>_metrics.dat    Formato: "Group  Train/Test  Property  Count  MAE  RMSE"
                          CON cabecera (se descarta la línea que empieza con "Group").
                          Solo se usan las filas donde Group == "*ALL".
    loss_vs_epochs.dat    Formato: "epoch  train_loss  val_loss", SIN cabecera.
    system_metrics.csv    Generado por monitor_training.py, CON cabecera
                          (columnas detectadas por nombre, ver parse_system_metrics()).
    perconfig.dat         Usado solo por --plots (parity plots).
    *.in / *.in.txt       Referenciado como config_file, no se parsea su contenido.

Detección del baseline: dentro de cada dataset, las corridas se ordenan con
_sort_key() — cualquier run cuyo nombre contenga "baseline" o "run_01" (sin
distinguir mayúsculas) se antepone; el resto se ordena alfabéticamente por
run_name. El PRIMERO de esa lista ordenada es el baseline usado para los Δ%
de la tabla. Si ningún run tiene "baseline"/"run_01" en el nombre, el baseline
termina siendo simplemente el primero por orden alfabético — no necesariamente
el primero cronológicamente ejecutado.

Salida: imprime el reporte en consola y lo escribe en Markdown en
07_experimentos/comparison_report.md (o en --output). Con --plots, además
escribe PNGs (parity_<dataset>_<run>.png y loss_curves_<dataset>.png) en
07_experimentos/plots/.
"""

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

# ── Importaciones opcionales ───────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── Dataclass de métricas por corrida ─────────────────────────────────────────

@dataclass
class RunMetrics:
    dataset: str
    run_name: str
    run_path: Path

    # Precisión — de *_metrics.dat
    train_energy_mae: Optional[float] = None
    train_energy_rmse: Optional[float] = None
    train_force_mae: Optional[float] = None
    train_force_rmse: Optional[float] = None
    test_energy_mae: Optional[float] = None
    test_energy_rmse: Optional[float] = None
    test_force_mae: Optional[float] = None
    test_force_rmse: Optional[float] = None

    # Convergencia — de loss_vs_epochs.dat
    num_epochs: Optional[int] = None
    final_train_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    final_val_loss: Optional[float] = None

    # Recursos (opcional) — de system_metrics.csv
    peak_ram_gb: Optional[float] = None
    total_time_s: Optional[float] = None

    config_file: Optional[Path] = None
    notes: str = ""


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_metrics_dat(metrics_file: Path) -> Dict[str, Dict[str, float]]:
    """
    Parsea el archivo *_metrics.dat de FitSNAP.
    Formato:  Group  Train/Test  Property  Count  MAE  RMSE
    Retorna un dict con claves como 'Train_Energy', 'Test_Force', etc.
    """
    result: Dict[str, Dict[str, float]] = {}
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6 or parts[0] == "Group":
                    continue
                group, split, prop = parts[0], parts[1], parts[2]
                if group != "*ALL":
                    continue
                try:
                    mae, rmse = float(parts[4]), float(parts[5])
                except ValueError:
                    continue
                key = f"{split}_{prop}"
                result[key] = {"mae": mae, "rmse": rmse}
    except Exception as exc:
        print(f"  ⚠️  No se pudo leer {metrics_file.name}: {exc}")
    return result


def parse_loss_dat(loss_file: Path) -> Dict[str, float]:
    """
    Parsea loss_vs_epochs.dat.
    Formato:  epoch  train_loss  val_loss  (sin cabecera, separado por espacios)
    """
    result: Dict[str, float] = {}
    epochs, train_losses, val_losses = [], [], []
    try:
        with open(loss_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        epochs.append(float(parts[0]))
                        train_losses.append(float(parts[1]))
                        val_losses.append(float(parts[2]))
                    except ValueError:
                        continue
        if epochs:
            result["num_epochs"] = int(max(epochs)) + 1
            result["final_train_loss"] = train_losses[-1]
            result["final_val_loss"] = val_losses[-1]
            result["best_val_loss"] = min(val_losses)
    except Exception as exc:
        print(f"  ⚠️  No se pudo leer {loss_file.name}: {exc}")
    return result


def parse_system_metrics(csv_file: Path) -> Dict[str, float]:
    """
    Parsea system_metrics.csv (generado por monitor_training.py).

    Formato esperado (CON cabecera). Columnas actuales de monitor_training.py:
        Time_s, Phase, CPU_Percent, RAM_GB, Proc_RSS_GB, Disk_Read_MB,
        Disk_Write_MB, GPU_Util_Percent, GPU_Mem_GB
    (versiones antiguas del script no tenían "Phase" ni "Proc_RSS_GB"; no
    importa, aquí se busca por nombre, no por posición).

    - peak_ram_gb      = máximo de la PRIMERA columna cuyo nombre (en
                         minúsculas) contiene "ram" o "mem", EXCLUYENDO
                         cualquier columna que contenga "gpu" o "rss" (para no
                         confundir RAM del sistema con GPU_Mem_GB — antes de
                         este fix, GPU_Mem_GB pisaba el valor real de RAM_GB
                         porque ambas matchean "mem" y GPU_Mem_GB se procesa
                         después en el orden de columnas — ni con Proc_RSS_GB,
                         que es una métrica distinta, ver abajo).
    - peak_proc_rss_gb = máximo de la primera columna cuyo nombre contiene
                         "rss" (típicamente "Proc_RSS_GB"). Es la RAM real del
                         proceso de FitSNAP (+ hijos, ej. ranks de mpirun), a
                         diferencia de peak_ram_gb que es RAM de TODO el
                         sistema (se contamina con otros procesos de la
                         máquina) — usar esta para comparaciones defendibles
                         en el paper. Ausente en CSVs generados antes de esta
                         columna existir.
    - total_time_s     = máximo de la primera columna cuyo nombre contiene
                         "time" o "elapsed" (típicamente "Time_s").
    - total_disk_read_mb / total_disk_write_mb = máximo (= valor final, ya que
                         monitor_training.py acumula desde el inicio de la
                         corrida) de las columnas "disk_read"/"disk_write".
                         Es I/O acumulado del proceso, no el tamaño del
                         archivo `.h5` de disk_cache (ver find_h5_file()).
    """
    result: Dict[str, float] = {}
    if not HAS_PANDAS:
        return result
    try:
        df = pd.read_csv(csv_file)
        for col in df.columns:
            lc = col.lower()
            if "gpu" in lc:
                continue
            if "peak_proc_rss_gb" not in result and "rss" in lc:
                result["peak_proc_rss_gb"] = df[col].max()
                continue
            if "peak_ram_gb" not in result and ("ram" in lc or "mem" in lc):
                result["peak_ram_gb"] = df[col].max()
            if "total_time_s" not in result and ("time" in lc or "elapsed" in lc):
                result["total_time_s"] = df[col].max()
            if "total_disk_read_mb" not in result and "disk_read" in lc:
                result["total_disk_read_mb"] = df[col].max()
            if "total_disk_write_mb" not in result and "disk_write" in lc:
                result["total_disk_write_mb"] = df[col].max()
    except Exception:
        pass
    return result


def find_h5_file(run_path: Path) -> Optional[Path]:
    """Busca un archivo `*.h5` (disk_cache) dentro del directorio de la corrida.

    No hay convención de nombre fija (se define en el `.in` con
    `disk_cache = <lo-que-sea>.h5`), así que se toma el primero que aparezca.
    Devuelve None si el run no usó disk_cache (caso normal para runs InRAM).
    """
    matches = sorted(run_path.glob("*.h5"))
    return matches[0] if matches else None


# ── Descubrimiento de corridas ─────────────────────────────────────────────────

def find_runs(
    experiments_dir: Path,
    dataset_filter: Optional[str] = None,
) -> List[RunMetrics]:
    """
    Escanea experiments_dir buscando directorios con *_metrics.dat.

    Estructura esperada (flexible):
        experiments_dir/<dataset>/<run_name>/*_metrics.dat   ← nuevo convenio
        experiments_dir/<run_name>/<run_name>/*_metrics.dat  ← legado (LiBF4_final)
        experiments_dir/<run_name>/*_metrics.dat             ← plano
    """
    runs: List[RunMetrics] = []
    metrics_files = sorted(experiments_dir.rglob("*_metrics.dat"))

    if not metrics_files:
        return runs

    for mf in metrics_files:
        run_path = mf.parent
        rel = run_path.relative_to(experiments_dir)
        parts = rel.parts

        if len(parts) == 1:
            dataset, run_name = "default", parts[0]
        elif len(parts) == 2:
            dataset, run_name = parts[0], parts[1]
        else:
            # Más de 2 niveles: tomar los últimos dos
            dataset, run_name = parts[-2], parts[-1]

        if dataset_filter and dataset.lower() != dataset_filter.lower():
            continue

        run = RunMetrics(dataset=dataset, run_name=run_name, run_path=run_path)

        # Métricas de precisión
        m = parse_metrics_dat(mf)
        run.train_energy_mae  = m.get("Train_Energy", {}).get("mae")
        run.train_energy_rmse = m.get("Train_Energy", {}).get("rmse")
        run.train_force_mae   = m.get("Train_Force",  {}).get("mae")
        run.train_force_rmse  = m.get("Train_Force",  {}).get("rmse")
        run.test_energy_mae   = m.get("Test_Energy",  {}).get("mae")
        run.test_energy_rmse  = m.get("Test_Energy",  {}).get("rmse")
        run.test_force_mae    = m.get("Test_Force",   {}).get("mae")
        run.test_force_rmse   = m.get("Test_Force",   {}).get("rmse")

        # Convergencia
        lf = run_path / "loss_vs_epochs.dat"
        if lf.exists():
            ld = parse_loss_dat(lf)
            run.num_epochs       = ld.get("num_epochs")
            run.final_train_loss = ld.get("final_train_loss")
            run.final_val_loss   = ld.get("final_val_loss")
            run.best_val_loss    = ld.get("best_val_loss")

        # Recursos (opcional)
        sf = run_path / "system_metrics.csv"
        if sf.exists():
            sd = parse_system_metrics(sf)
            run.peak_ram_gb  = sd.get("peak_ram_gb")
            run.total_time_s = sd.get("total_time_s")

        # Archivo .in
        in_files = list(run_path.glob("*.in")) + list(run_path.glob("*.in.txt"))
        if in_files:
            run.config_file = in_files[0]

        runs.append(run)
        print(f"  ✓ {dataset}/{run_name}")

    return runs


# ── Formateo ──────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float], spec: str = ".4e") -> str:
    return "N/A" if value is None else format(value, spec)


def _delta(current: Optional[float], baseline: Optional[float]) -> str:
    """Porcentaje de cambio con iconos."""
    if current is None or baseline is None or baseline == 0.0:
        return "—"
    pct = (current - baseline) / abs(baseline) * 100.0
    if pct < -10:
        return f"✅ ▼{abs(pct):.1f}%"
    elif pct < -1:
        return f"🟩 ▼{abs(pct):.1f}%"
    elif pct < 1:
        return f"➡️  {pct:+.1f}%"
    elif pct < 10:
        return f"🟨 ▲{abs(pct):.1f}%"
    else:
        return f"⚠️  ▲{abs(pct):.1f}%"


def _sort_key(run: RunMetrics) -> str:
    """Ordenar baseline primero, luego por nombre."""
    if "baseline" in run.run_name.lower() or "run_01" in run.run_name.lower():
        return "000_" + run.run_name
    return run.run_name


# ── Generación del reporte ────────────────────────────────────────────────────

# Objetivos de calidad comunes para FFs de litio (ajustar según el sistema)
TARGETS = {
    "test_energy_rmse": 0.005,   # 5 meV/atom
    "test_force_rmse":  0.020,   # 20 meV/Å
}


def generate_report(
    runs_by_dataset: Dict[str, List[RunMetrics]],
    show_system: bool = False,
) -> str:
    lines = [
        "# 📊 FitSNAP — Reporte Comparativo de Experimentos\n",
        "*Generado automáticamente por `compare_runs.py`*\n",
        "---\n",
        "## 🎯 Objetivos de Calidad\n",
        f"- **Energía RMSE (test)** < {TARGETS['test_energy_rmse']} eV/átomo  (≡ 5 meV/át.)\n",
        f"- **Fuerzas RMSE (test)** < {TARGETS['test_force_rmse']} eV/Å\n",
        "\n---\n",
    ]

    for dataset, runs in sorted(runs_by_dataset.items()):
        if not runs:
            continue
        ordered = sorted(runs, key=_sort_key)
        baseline = ordered[0]

        lines.append(f"## Dataset: `{dataset}`\n")
        lines.append(f"**Total de corridas**: {len(ordered)}  |  **Baseline**: `{baseline.run_name}`\n\n")

        # ── Tabla de precisión ────────────────────────────────────────────────
        lines.append("### Métricas de Precisión\n")
        lines.append(
            "| Run | E-RMSE test (eV/át) | Δ base | F-RMSE test (eV/Å) | Δ base "
            "| E-MAE test | F-MAE test | Épocas | Mejor val-loss |"
        )
        lines.append(
            "|-----|---------------------|--------|---------------------|--------|"
            "-----------|------------|--------|----------------|"
        )

        for run in ordered:
            is_base = run is baseline
            tag = " ⭐" if is_base else ""
            name = f"`{run.run_name}`{tag}"
            lines.append(
                f"| {name} "
                f"| {_fmt(run.test_energy_rmse)} "
                f"| {'—' if is_base else _delta(run.test_energy_rmse, baseline.test_energy_rmse)} "
                f"| {_fmt(run.test_force_rmse)} "
                f"| {'—' if is_base else _delta(run.test_force_rmse, baseline.test_force_rmse)} "
                f"| {_fmt(run.test_energy_mae)} "
                f"| {_fmt(run.test_force_mae)} "
                f"| {run.num_epochs or 'N/A'} "
                f"| {_fmt(run.best_val_loss, '.4e')} |"
            )

        lines.append("\n")

        # ── Tabla de convergencia ─────────────────────────────────────────────
        lines.append("### Convergencia\n")
        lines.append("| Run | Loss inicial | Loss final (train) | Loss final (val) | Mejor val-loss |")
        lines.append("|-----|-------------|-------------------|-------------------|----------------|")
        for run in ordered:
            lines.append(
                f"| `{run.run_name}` "
                f"| — "
                f"| {_fmt(run.final_train_loss, '.4e')} "
                f"| {_fmt(run.final_val_loss, '.4e')} "
                f"| {_fmt(run.best_val_loss, '.4e')} |"
            )
        lines.append("\n")

        # ── Check de objetivos ────────────────────────────────────────────────
        lines.append("### Logro de Objetivos\n")
        for run in ordered:
            e_ok = run.test_energy_rmse is not None and run.test_energy_rmse < TARGETS["test_energy_rmse"]
            f_ok = run.test_force_rmse  is not None and run.test_force_rmse  < TARGETS["test_force_rmse"]
            lines.append(
                f"- **{run.run_name}**: "
                f"Energía {'✅' if e_ok else '❌'} ({_fmt(run.test_energy_rmse, '.4e')} eV/át.)  "
                f"| Fuerzas {'✅' if f_ok else '❌'} ({_fmt(run.test_force_rmse, '.4e')} eV/Å)"
            )
        lines.append("\n")

        # ── Recursos (opcional) ───────────────────────────────────────────────
        if show_system and any(r.peak_ram_gb is not None for r in ordered):
            lines.append("### Uso de Recursos (opcional)\n")
            lines.append("| Run | RAM pico (GB) | Tiempo total (s) |")
            lines.append("|-----|---------------|-----------------|")
            for run in ordered:
                # _fmt ya retorna "N/A" si el valor es None; no usar un `if run.x`
                # aquí porque 0.0 es un valor legítimo (falsy) y se mostraría
                # "N/A" incorrectamente.
                lines.append(
                    f"| `{run.run_name}` "
                    f"| {_fmt(run.peak_ram_gb, '.1f')} "
                    f"| {_fmt(run.total_time_s, '.0f')} |"
                )
            lines.append("\n")

        lines.append("---\n")

    return "\n".join(lines)


# ── Gráficas ──────────────────────────────────────────────────────────────────

def generate_parity_plots(
    runs_by_dataset: Dict[str, List[RunMetrics]],
    output_dir: Path,
) -> None:
    """Genera parity plots (Predicción vs DFT) de energía por corrida."""
    if not HAS_PLOTS:
        print("⚠️  matplotlib/numpy no disponibles. Omitiendo plots.")
        return

    for dataset, runs in runs_by_dataset.items():
        for run in runs:
            pc_file = run.run_path / "perconfig.dat"
            if not pc_file.exists():
                continue
            try:
                truth, pred, is_test, natoms = [], [], [], []
                with open(pc_file, "r") as f:
                    f.readline()  # cabecera
                    for line in f:
                        p = line.split()
                        if len(p) >= 6:
                            natoms.append(int(p[2]))
                            truth.append(float(p[3]))
                            pred.append(float(p[4]))
                            is_test.append(p[5].strip().lower() == "true")

                natoms_arr = np.array(natoms, dtype=float)
                truth_pa = np.array(truth) / natoms_arr
                pred_pa  = np.array(pred)  / natoms_arr
                is_test  = np.array(is_test)

                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(truth_pa[~is_test], pred_pa[~is_test],
                           alpha=0.4, s=16, color="#3B82F6", label="Train")
                ax.scatter(truth_pa[is_test], pred_pa[is_test],
                           alpha=0.8, s=24, color="#EF4444", label="Test", zorder=5)
                lims = [min(truth_pa.min(), pred_pa.min()), max(truth_pa.max(), pred_pa.max())]
                ax.plot(lims, lims, "k--", lw=1.5, label="Perfecto")
                ax.set_xlabel("Energía DFT (eV/átomo)", fontsize=12)
                ax.set_ylabel("Energía Predicha (eV/átomo)", fontsize=12)
                ax.set_title(f"Parity Plot — {dataset} / {run.run_name}", fontsize=13)
                ax.legend(framealpha=0.9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                out = output_dir / f"parity_{dataset}_{run.run_name}.png"
                plt.savefig(out, dpi=150)
                plt.close()
                print(f"  💾 {out.name}")
            except Exception as exc:
                print(f"  ⚠️  Parity plot error ({run.run_name}): {exc}")


def generate_loss_curves(
    runs_by_dataset: Dict[str, List[RunMetrics]],
    output_dir: Path,
) -> None:
    """Genera curvas de loss superpuestas por dataset."""
    if not HAS_PLOTS:
        return

    for dataset, runs in runs_by_dataset.items():
        runs_with_loss = [r for r in runs if (r.run_path / "loss_vs_epochs.dat").exists()]
        if not runs_with_loss:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10.colors

        for i, run in enumerate(sorted(runs_with_loss, key=_sort_key)):
            lf = run.run_path / "loss_vs_epochs.dat"
            epochs, val_l = [], []
            with open(lf, "r") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 3:
                        try:
                            epochs.append(float(p[0]))
                            val_l.append(float(p[2]))
                        except ValueError:
                            continue
            if not epochs:
                continue
            label = f"{run.run_name}  (mejor={_fmt(run.best_val_loss, '.2e')})"
            ax.semilogy(epochs, val_l, color=colors[i % 10], label=label, lw=2)

        ax.set_xlabel("Época", fontsize=12)
        ax.set_ylabel("Val Loss (escala log)", fontsize=12)
        ax.set_title(f"Curvas de Convergencia — {dataset}", fontsize=13)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        out = output_dir / f"loss_curves_{dataset}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  💾 {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FitSNAP — Herramienta de Comparación de Experimentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Ejemplos:
              python compare_runs.py                      # Reporte completo
              python compare_runs.py --dataset LiBF4      # Solo ese dataset
              python compare_runs.py --plots              # + parity plots y curvas de loss
              python compare_runs.py --system             # + métricas de RAM/tiempo
              python compare_runs.py --winners-only       # Solo corridas que mejoran
              python compare_runs.py --output mi_rep.md   # Nombre de archivo custom
            """
        ),
    )
    parser.add_argument("--dataset",      type=str, default=None)
    parser.add_argument("--plots",        action="store_true")
    parser.add_argument("--system",       action="store_true")
    parser.add_argument("--winners-only", action="store_true", dest="winners_only")
    parser.add_argument("--output",       type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    experiments_dir = script_dir / "07_experimentos"

    print("🔍  Buscando corridas en", experiments_dir)
    runs = find_runs(experiments_dir, dataset_filter=args.dataset)

    if not runs:
        print("❌  No se encontraron corridas completas (ningún *_metrics.dat).")
        print(f"    Verificar que {experiments_dir} existe y contiene corridas finalizadas.")
        sys.exit(1)

    # Agrupar por dataset
    runs_by_dataset: Dict[str, List[RunMetrics]] = {}
    for run in runs:
        runs_by_dataset.setdefault(run.dataset, []).append(run)

    # Filtro --winners-only
    if args.winners_only:
        filtered: Dict[str, List[RunMetrics]] = {}
        for ds, dlist in runs_by_dataset.items():
            ordered = sorted(dlist, key=_sort_key)
            base = ordered[0]
            winners = [base] + [
                r for r in ordered[1:]
                if r.test_energy_rmse is not None
                and base.test_energy_rmse is not None
                and r.test_energy_rmse < base.test_energy_rmse
            ]
            filtered[ds] = winners
        runs_by_dataset = filtered

    # Reporte de texto
    print("\n📝  Generando reporte...\n")
    report = generate_report(runs_by_dataset, show_system=args.system)

    print("=" * 80)
    print(report)
    print("=" * 80)

    out_path = Path(args.output) if args.output else experiments_dir / "comparison_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n✅  Reporte guardado → {out_path}")

    # Plots opcionales
    if args.plots:
        plots_dir = experiments_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        print(f"\n🎨  Generando gráficas en {plots_dir}/")
        generate_parity_plots(runs_by_dataset, plots_dir)
        generate_loss_curves(runs_by_dataset, plots_dir)
        print("✅  Gráficas listas.")


if __name__ == "__main__":
    main()
