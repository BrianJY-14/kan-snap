# kan-snap

A Kolmogorov–Arnold output block for SNAP machine-learned interatomic
potentials, on Li–F.

The goal: find out whether replacing the linear read-out of a FitSNAP neural
network with a Kolmogorov–Arnold layer improves the fit, at equal capacity,
learning rate and optimizer budget. Everything else in the pipeline — SNAP
bispectrum descriptors, the loss weights, the per-atom normalization — stays as
the group's reference model has it, so the numbers land in the same
`metrics.dat` and can be compared line by line.

## Quickstart

```bash
git clone https://github.com/BrianJY-14/kan-snap.git
cd kan-snap
mamba env create -f environment.yml
conda activate fitsnap
export KAN_RAIZ=$PWD
bash scripts/bajar_dataset.sh BrianJY-14/lif-snap-dataset
python src/prueba_variantes.py
```

`run_kan_snap.md` is the run sheet for a SLURM cluster: prerequisites, the five
steps, and a troubleshooting table.

## What is in here

| Path | Contents |
|---|---|
| `scripts/bajar_dataset.sh` | Fetches the **23 883 Li–F configurations** into `dataset/` |
| `src/` | The experimental framework: KAN layers, the FitSNAP patches, the launcher, the measurement tools |
| `scripts/convertir_qe.py` | Rebuilds `dataset/` from raw Quantum ESPRESSO output |
| `notebooks/` | The protocol, section by section, runnable top to bottom |
| `run_kan_snap.md` | How to run the campaign on a cluster |

## The data

The training data is **not in this repository**. It is a private Hugging Face
dataset, and `scripts/bajar_dataset.sh` pulls it into `dataset/` as the first
thing anyone does after cloning. Access is granted per person: ask for a read
token before running the script, and expect `hf auth login` to be needed once.

A sample would not have helped. The comparison this repository exists to make
is between architectures at a fixed optimizer budget, and a subset changes the
number of steps per epoch — which is the one variable that invalidated the
October campaign. Either the whole set is there or the run means nothing.

Each file under `dataset/<system>/output_<n>.json` is one frame of a Quantum
ESPRESSO molecular-dynamics run: positions, forces, total energy, stress and
cell. The DFT is already done — nothing here recomputes it. Twelve systems:

| System | Configurations | Atoms |
|---|---:|---:|
| `LiF_64_kjpaw`, `LiF_64_NPT`, `LiF_64_isolated` | 7 413 | 64 |
| `BCC_54_kjpaw`, `BCC_54_NPT`, `BCC_54_isolated` | 5 297 | 54 |
| `LiwithF`, `LiwithF_NPT`, `LiwithF_isolated` | 6 184 | 54 |
| `LiFinterface_kjpaw`, `LiFinterface_NPT` | 4 889 | 122 |
| `Special` | 100 | 2 |

Once expanded, `carpeta_datos()` in `src/kan_lib.py` finds `dataset/` on its
own; nothing needs to be configured.

The `_NPT` sets come from constant-pressure runs and carry a varying cell; the
`_isolated` sets are the same systems in a large box, which is what pins the
energy zero. `Special` holds Li and F dimers at a range of separations — the
boron dimers of the original set are excluded, since a Li–F fit declares only
two element types.

## The framework

`src/kan_lib.py` patches FitSNAP in memory and never edits an installed file:
`aplicar_todos_los_parches()` must run **before** the first `import fitsnap3lib`.
It swaps `create_torch_network` for one whose output block is selectable —
`linear`, `gaussiana`, `bspline` or `chebyshev` — and fixes seven 2024-era API
calls that newer PyTorch, LAMMPS, Python and setuptools have moved or renamed.

Three things worth knowing before reading any result:

- **The criterion is the physical error**, not the loss. `src/mae.py` prints the
  energy MAE per atom, the force MAE, and the ratio against predicting zero
  force. A ratio of 1.000 means the model learned nothing, whatever the loss
  curve looks like.
- **The budget is counted in optimizer steps**, not epochs. The group's
  reference model takes 35 300 of them to reach 1.108e-02 eV/Å on forces. A run
  with fewer is comparing initializations.
- **The degenerate floor of this dataset is 5.4e-02**, the mean square of the
  true forces. Several architectures landing on the same loss near that value
  means all of them collapsed, not that they tie.

## Credits

Dataset and reference model: Prof. Luis A. Selis's group, UNI — Texas A&M.
Architecture line and this framework: Brian Jara, UNI.
