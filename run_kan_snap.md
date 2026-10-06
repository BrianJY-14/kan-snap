# Run KAN-SNAP Experiments on the Cluster

Replaces the last layer of a FitSNAP neural-network potential (Li–F, SNAP
bispectrum descriptors) with a Kolmogorov–Arnold block, and compares eight
output-block variants at equal capacity, learning rate and optimizer budget.

## Prerequisites

- **SLURM** with a GPU partition: **1 NVIDIA GPU (≥ 8 GB VRAM) and 8 CPU cores** per array task
- **conda / mamba** — Step 1 installs Miniforge if none is on `PATH`
- **~20 GB free disk** on `$SCRATCH` and **internet during Step 1 only**
- A **read token** for the dataset, which is private. Ask for one; no Hugging
  Face account is needed to use it. A single 70 MB transfer

## Step 1 — Setup

```bash
cd "$SCRATCH"
git clone https://github.com/BrianJY-14/kan-snap.git
cd kan-snap
export KAN_RAIZ=$PWD
mamba env create -f environment.yml
conda activate fitsnap
export HF_TOKEN=hf_...                          # the token sent with this sheet
bash scripts/bajar_dataset.sh BrianJY-14/lif-snap-dataset
```

Everything comes from conda-forge — LAMMPS built with ML-SNAP, FitSNAP 3.1.0.4,
PyTorch GPU; nothing is compiled. About 15 min, and the only step that needs the
network. If the cluster already provides LAMMPS with the ML-SNAP package as a
module, that module can replace the `lammps` line in `environment.yml`; the rest
is pure Python.

The last line pulls **23 883 configurations** of Li–F into `dataset/` — a
single 70 MB archive that expands to ~225 MB, one folder per system. Each
configuration is a frame of a Quantum ESPRESSO molecular-dynamics run:
positions, forces, energy, stress and cell. The DFT itself is already done and
nothing here recomputes it. The script refuses to continue if fewer than 23 883
files land, because a partial set silently changes the steps per epoch and that
is the one thing this campaign cannot afford to get wrong.

The data is a private Hugging Face dataset rather than part of the repository:
it belongs to the group that produced the DFT, and access is granted per
person. The converter that built it, `scripts/convertir_qe.py`, is in the
repository, so the set can be rebuilt from the raw QE output if a system needs
to be added — but that is a separate 19 GB of logs and is not needed here.

## Step 2 — Preflight (no GPU, seconds)

```bash
conda activate fitsnap
cd "$KAN_RAIZ/src"
python verificar_lib.py kan_lib.py trabajador.py barrida.py
python prueba_variantes.py
python -c "from kan_lib import resumen_kan; print(resumen_kan())"
python lanzar.py --raiz "$KAN_RAIZ" --solo-comprobar
```

Four checks that cost nothing and have each caught a dead campaign: the modules
parse, the four range modes agree with their closed forms, the KAN block reports
the configuration actually in effect, and the launcher finds the data, the GPUs
and the eight conditions. **Do not queue a node until all four pass.**

## Step 3 — Fix the budget (one GPU task, ~10 min)

```bash
export KAN_FRACCION=1.0
export KAN_RESIDUO=1          # the residual activation; off by default
export KAN_REJILLA_CADA=200   # grid refresh, in optimizer steps
python lanzar.py --raiz "$KAN_RAIZ" --grupo todos --semillas 1111 --epocas 3 --si
python mae.py "$KAN_RAIZ/corrida_*"
```

This is an end-to-end validation, not a result: three epochs train nothing. What
it produces is the **seconds per epoch and the steps per epoch on this machine**,
and those fix the epoch count for Step 4:

```
EPOCHS = ceil(35300 / steps_per_epoch)
```

35 300 is the optimizer-step count of the group's reference model (`lr 5e-6`,
70 epochs, batch 50, full dataset), which reaches **1.108e-02 eV/Å** on forces.
A campaign that does fewer steps than that is comparing initializations, not
architectures — measured on 2 October 2026, when twenty models at 70 epochs all
landed on the degenerate floor because the run performed 2.6 % of the reference
budget.

Two ways to spend the same 35 300 steps, and they are not equivalent. All twelve
systems at `KAN_FRACCION=1.0` gives 16 718 training configurations, 335 steps per
epoch, **about 106 epochs** — this is the one to run. A single system gives 13
steps per epoch and would need ~2 520 epochs for the same budget: identical GPU
cost, a twenty-fifth of the data, and an overfit that says nothing about the
architecture. The single-system path exists only as a cheap smoke test.

`mae.py` is the acceptance criterion, not the loss. It prints the energy MAE per
atom, the force MAE, and `razon_f`, the ratio against predicting zero force:
**`razon_f` ≈ 1.000 means the model learned nothing**, whatever the loss says.

## Step 4 — Run the grid (this is the only step that needs the GPU queue)

```bash
cd "$KAN_RAIZ"
export EPOCHS=106                       # from Step 3
export KAN_FRACCION=1.0
export KAN_RESIDUO=1
export KAN_REJILLA_CADA=200
export CONDA_BASE=$(conda info --base)
cat > enviar_kan.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=kan-snap
#SBATCH --account=ADJUST_ACCOUNT
#SBATCH --partition=ADJUST_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/kan_%A_%a.out
#SBATCH --error=logs/kan_%A_%a.err
set -euo pipefail
mkdir -p logs
# ADJUST: module load cuda/12.4
source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate fitsnap
cd "$KAN_RAIZ/src"
COND=(1_mlp_base 2_mlp_compacto 3_kan_gauss 4_kan_bspline 5_kan_cheby \
      6_kan_compacto 7_kan_iso 8_mlp_ampliado)
SEMILLAS=(1111 2222 3333 4444 5555)
I=$SLURM_ARRAY_TASK_ID
C=${COND[$((I / 5))]}
S=${SEMILLAS[$((I % 5))]}
python -c "from kan_lib import resumen_kan; print(resumen_kan())"
python lanzar.py --raiz "$KAN_RAIZ" --grupo todos \
    --condiciones "$C" --semillas "$S" \
    --epocas $EPOCHS --trabajadores 1 \
    --salida "$KAN_RAIZ/campana_$SLURM_ARRAY_JOB_ID" --si
EOF
sbatch enviar_kan.sbatch
```

`KAN_RAIZ`, `EPOCHS`, `KAN_FRACCION`, `KAN_RESIDUO`, `KAN_REJILLA_CADA` and
`CONDA_BASE` travel to the job through the default `--export=ALL`. **The two KAN
variables are off by default and the campaign is meaningless without them**: they
switch on the residual activation and the on-the-fly grid update that Liu et al.
(2024) declare necessary for a KAN to train. `resumen_kan()`, printed at the top
of every task, must say `residuo=si`. Forty tasks: **eight output-block conditions × five seeds**, one independent
process each, so a failed task takes nothing else with it and can be resubmitted
alone. The conditions are two MLP baselines, three KAN bases (Gaussian,
B-spline, Chebyshev), a compact pair and two parameter-matched controls — the
KAN block is 0.127 % of the parameters, so the controls are what make a
difference attributable to the architecture rather than to capacity. Those two
controls are sized for `KAN_RESIDUO=1` and are wrong without it: the residual
adds 64 parameters to the KAN layer, which is why `7_kan_iso` is 57 wide and
`8_mlp_ampliado` is 72. Both land within 16 parameters of their target.

Lines marked `ADJUST` are the account, the partition and whatever loads CUDA.
Expect **~13 h of GPU per task** on an RTX 3090, extrapolated from 1.37 s per
optimizer step measured on 2 October 2026, plus a **descriptor phase of ~3 h on
CPU** — LAMMPS computing the bispectrum for all twelve systems, once per task,
before any training starts. Budget **24 h of walltime and ~640 node-hours** for
the array; both figures drop on A100 or H100. Each task writes to
`$KAN_RAIZ/campana_<jobid>/<condition>_s<seed>/`:

- `<id>_s<seed>_metrics.dat` — FitSNAP metrics for the run
- `loss_vs_epochs.dat` — training curve
- `perconfig.dat`, `peratom.dat` — truth vs. prediction, per configuration and per atom
- `system_metrics.csv` — GPU, VRAM and RAM sampled during that run

## Step 5 — Send me the results

```bash
cd "$KAN_RAIZ"
python src/mae.py "campana_*" > resumen_mae.txt
tar czf kan_output.tar.gz campana_* resumen_mae.txt logs
```

Attach `kan_output.tar.gz`. `resumen_mae.txt` already has the energy and force
MAE of every run and its ratio against predicting zero force, which is the
single number that says whether the campaign measured anything.

---

## Troubleshooting

| Problem | Likely fix |
|---------|-----------|
| `Found unmatched variable in PYTORCH section` | `dtype_setting` reached FitSNAP. It is read but not accepted; check 3 of the module cell must print `True`. Everything runs in float32 |
| `ML-SNAP` reported missing | Reinstall `lammps` from conda-forge. Without that package there are no descriptors and nothing else matters |
| `cuda.h: No such file or directory` | Triton compiles a C module. `export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH` |
| `strtobool`, `shuffle`, `create_atoms`, `index_add_`, `extract_atom_*` errors | Seven 2024-era API calls. `aplicar_todos_los_parches()` fixes all of them, but it must run **before** the first `import fitsnap3lib` |
| Tasks die in seconds and the log holds only the monitor footer | The monitor discards child output. Rerun one worker without `--silencioso` to get the traceback |
| Every condition ends at a loss of ~5.4e-02 | That is the degenerate floor — the mean square of the true forces. The models predict zero force and constant energy. Read `mae.py`, not the loss |
| `razon_f` = 1.000 in `mae.py` | Same thing, stated exactly: the error equals predicting zero |
| Loss grows past 1e+6 in the first epochs | Learning rate above the divergence threshold of this pipeline, which sits between 2e-2 and 5e-2 |
| A sweep row shows `delta = 0` | The weights never moved: the optimizer lost its link to the model. Go through `cambiar_arquitectura`, which rebuilds it and clears `model_best` |
| B-spline condition gets exactly zero gradient | The inputs left the grid; a compact-support basis vanishes there instead of decaying. Enable `KAN_REJILLA_CADA=200` |
| KAN trains far worse than the MLP | Check `resumen_kan()` says `residuo=si`. Without the residual path the edge has no linear route for the gradient |
| `CUDA OOM` | One task per GPU. A worker was measured at 5 743 MiB of VRAM; more than one per card is what overflows |
| Host RAM exhausted | ~18 GB per worker. Keep `--trabajadores` below RAM/18 |
| `nvidia-smi` shows the GPU near idle | Expected during the descriptor phase, which is LAMMPS on CPU: 452 s of the first 456. Measure with `monitor_fases.py`, which separates the phases |
| A task spends hours before the first epoch | Expected: that is LAMMPS computing bispectrum descriptors for all twelve systems on CPU. They are computed once per task and reused across its runs |
| A task needs more than 24 h | Split the array: `--array=0-19` and `--array=20-39` write into the same campaign folder |
