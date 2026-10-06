"""
kan_lib.py — modulo del marco experimental KAN x FitSNAP.

GENERADO UNA VEZ desde el notebook por construir_kan_lib.py, y a partir de
ahi editado a mano como cualquier otro archivo fuente. No se regenera en
cada corrida: es la fuente de verdad que importan tanto el notebook como
los trabajadores.

Importarlo NO importa fitsnap3lib ni aplica ningun parche. Para eso:

    from kan_lib import *
    aplicar_todos_los_parches()
    from fitsnap3lib.fitsnap import FitSnap

Variables de entorno reconocidas (ver el bloque final del archivo):
    KAN_FRACCION        fraccion del dataset          (por defecto la del notebook)
    KAN_BATCH_SIZE      tamano de lote
    KAN_LR              tasa de aprendizaje
    KAN_DTYPE_SETTING   1 = float32 (defecto de FitSNAP), 2 = float64
"""

import os
import sys
import math
import copy
import json
import time
import glob
import shutil
import importlib
import contextlib
import threading
import subprocess
import re
import io as _io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as _lrs

# ======================================================================
# Celda 8 del notebook — Capas KAN: KANGaussiana, KANBSpline, KANChebyshev, BASES
# ======================================================================

import math, torch, torch.nn as nn

class KANGaussiana(nn.Module):
    """Base RBF gaussiana. Recomendada por Wang et al. (2024).
    b_k(x) = exp(-((x-c_k)/h)^2). Derivada analitica C-infinito: apta para fuerzas."""
    def __init__(self, n_in, n_out, K=8, rango=(-2.0, 2.0)):
        super().__init__()
        self.n_in, self.n_out, self.K = n_in, n_out, K
        self.register_buffer("centros", torch.linspace(rango[0], rango[1], K))
        self.h = (rango[1] - rango[0]) / (K - 1)
        self.w = nn.Parameter(torch.randn(n_in, n_out, K) / math.sqrt(n_in * K))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        x = self._normalizar_entrada(x)
        # `anchos` solo existe en modo cuantiles: un ancho por canal y por nudo.
        h = self.anchos if hasattr(self, "anchos") else self.h
        z = (x.unsqueeze(-1) - self.centros) / h
        return torch.einsum("...ik,iok->...o", torch.exp(-z * z), self.w) + self.bias

class KANBSpline(nn.Module):
    """KAN original de Liu et al. (2024). G=5, k=3 -> continuidad C^2."""
    def __init__(self, n_in, n_out, G=5, k=3, rango=(-2.0, 2.0)):
        super().__init__()
        self.n_in, self.n_out, self.G, self.k = n_in, n_out, G, k
        h = (rango[1] - rango[0]) / G
        self.register_buffer("rejilla",
            torch.arange(-k, G + k + 1, dtype=torch.get_default_dtype()) * h + rango[0])
        self.K = G + k
        self.w = nn.Parameter(torch.randn(n_in, n_out, self.K) / math.sqrt(n_in * self.K))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def _bases(self, x):
        # Indexado con ... para que la rejilla pueda ser (M,) —uniforme— o
        # (n_in, M) —una por canal, en modo cuantiles—.
        g, xx = self.rejilla, x.unsqueeze(-1)
        b = ((xx >= g[..., :-1]) & (xx < g[..., 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            izq = (xx - g[..., : -(p + 1)]) / (g[..., p:-1] - g[..., : -(p + 1)]) * b[..., :-1]
            der = (g[..., p + 1:] - xx) / (g[..., p + 1:] - g[..., 1:-p]) * b[..., 1:]
            b = izq + der
        return b

    def forward(self, x):
        x = self._normalizar_entrada(x)
        return torch.einsum("...ik,iok->...o", self._bases(x), self.w) + self.bias

class KANChebyshev(nn.Module):
    """Polinomios de Chebyshev. Mahmoud et al. (2025). tanh acota el dominio a [-1,1]."""
    def __init__(self, n_in, n_out, grado=5):
        super().__init__()
        self.n_in, self.n_out, self.K = n_in, n_out, grado + 1
        self.w = nn.Parameter(torch.randn(n_in, n_out, self.K) / math.sqrt(n_in * self.K))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        x = self._normalizar_entrada(x)
        t = torch.tanh(x).unsqueeze(-1)
        Ts = [torch.ones_like(t), t]
        for _ in range(2, self.K):
            Ts.append(2 * t * Ts[-1] - Ts[-2])
        return torch.einsum("...ik,iok->...o", torch.cat(Ts[: self.K], -1), self.w) + self.bias

BASES = {"gaussiana": KANGaussiana, "bspline": KANBSpline, "chebyshev": KANChebyshev}

# ======================================================================
# Celda 10 del notebook — crear_red y n_params
# ======================================================================

def crear_red(layer_sizes, salida="linear", **kw):
    """Replica exacta de create_torch_network de FitSNAP, con la ultima capa
    intercambiable.

    salida = "linear"                       -> identica a FitSNAP (linea base)
             "gaussiana"|"bspline"|"chebyshev" -> bloque KAN
    """
    L = [int(s) for s in layer_sizes]
    capas = [nn.Linear(L[0], L[0], bias=True)]        # estandarizacion
    for i in range(len(L) - 1):
        ultima = (i == len(L) - 2)
        if ultima and salida != "linear":
            capas.append(BASES[salida](L[i], L[i + 1], **kw))
        else:
            capas.append(nn.Linear(L[i], L[i + 1], bias=True))
        if not ultima:
            capas.append(nn.Softplus())
    return nn.Sequential(*capas)

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

_LS = [440, 256, 128, 64, 64, 1]

# ======================================================================
# Celda 13 del notebook — MODULOS_A_PARCHEAR, _ORIGINALES, parchear_red, restaurar_red
# ======================================================================

import importlib

MODULOS_A_PARCHEAR = [
    "fitsnap3lib.lib.neural_networks.pytorch",
    "fitsnap3lib.solvers.pytorch",
    "fitsnap3lib.io.sections.solver_sections.pytorch",
]

_ORIGINALES = {}

def parchear_red(salida="gaussiana", **kw):
    """Hace que FitSNAP construya sus redes con el bloque de salida indicado."""
    def fabrica(layer_sizes):
        return crear_red(layer_sizes, salida, **kw)

    for nombre in MODULOS_A_PARCHEAR:
        try:
            mod = importlib.import_module(nombre)
        except Exception:
            continue
        if hasattr(mod, "create_torch_network"):
            _ORIGINALES.setdefault(nombre, mod.create_torch_network)
            mod.create_torch_network = fabrica
    print(f"FitSNAP usara bloque de salida: {salida}  {kw if kw else ''}")

def restaurar_red():
    """Devuelve FitSNAP a su comportamiento original."""
    for nombre, fn in _ORIGINALES.items():
        importlib.import_module(nombre).create_torch_network = fn
    print("FitSNAP restaurado a su version original")

# ======================================================================
# Celda 15 del notebook — Parches de compatibilidad (torch, python, lammps, setuptools, cuda)
# ======================================================================

import torch.optim.lr_scheduler as _lrs

_ARGS_RETIRADOS = ("verbose",)   # anadir aqui si aparecen mas

def parchear_compatibilidad_torch():
    """Hace que FitSNAP funcione con versiones nuevas de PyTorch."""
    orig = getattr(_lrs, "_ReduceLROnPlateau_original", None) or _lrs.ReduceLROnPlateau
    _lrs._ReduceLROnPlateau_original = orig

    import inspect
    acepta = set(inspect.signature(orig.__init__).parameters)
    sobran = [a for a in _ARGS_RETIRADOS if a not in acepta]
    if not sobran:
        print("  PyTorch todavia acepta todos los argumentos: no hace falta parche")
        return

    class ReduceLROnPlateauCompat(orig):
        def __init__(self, *a, **kw):
            for arg in sobran:
                kw.pop(arg, None)
            super().__init__(*a, **kw)

    _lrs.ReduceLROnPlateau = ReduceLROnPlateauCompat
    torch.optim.lr_scheduler.ReduceLROnPlateau = ReduceLROnPlateauCompat
    print(f"  parcheado ReduceLROnPlateau: ignora {sobran}")

def parchear_compatibilidad_python():
    """random.shuffle(x, fuente) dejo de existir en Python 3.11.

    Quitar el segundo argumento NO cambia el resultado: FitSNAP llama a seed()
    justo antes y el argumento que pasaba era `random`, la fuente por defecto.
    """
    import random as _rnd
    import fitsnap3lib.scrapers.scrape as _scrape

    orig = getattr(_scrape, "_shuffle_original", None) or _rnd.shuffle
    _scrape._shuffle_original = orig

    import inspect
    try:
        n_args = len(inspect.signature(orig).parameters)
    except (TypeError, ValueError):
        n_args = 1
    if n_args >= 2:
        print("  Python todavia acepta shuffle(x, fuente): no hace falta parche")
        return

    def shuffle_compat(x, *ignorado, **kw):
        return orig(x)

    _scrape.shuffle = shuffle_compat
    print("  parcheado random.shuffle en el scraper: ignora el 2o argumento")

def parchear_compatibilidad_lammps():
    """LAMMPS renombro create_atoms(id=, type=) a (atomid=, atype=).

    Cambio de nombre puro, para no pisar las palabras reservadas `id` y `type`
    de Python. Verificado comparando el codigo de lammps 2024.8 contra el
    actual: mismo orden de argumentos, mismo cuerpo, misma llamada a la
    libreria en C.
    """
    import inspect
    from lammps.core import lammps as _LMP

    orig = getattr(_LMP, "_create_atoms_original", None) or _LMP.create_atoms
    _LMP._create_atoms_original = orig
    nombres = [p for p in inspect.signature(orig).parameters if p != "self"]

    if "id" in nombres and "type" in nombres:
        print("  LAMMPS usa create_atoms(id=, type=): no hace falta parche")
        return

    RENOMBRES = {"id": "atomid", "type": "atype"}
    faltan = [v for v in RENOMBRES.values() if v not in nombres]
    if faltan:
        print(f"  AVISO: firma inesperada create_atoms{tuple(nombres)}")
        print(f"         no encuentro {faltan}; NO se parchea nada.")
        print("         Revisar a mano antes de confiar en los descriptores.")
        return

    def create_atoms_compat(self, *args, **kw):
        return orig(self, *args, **{RENOMBRES.get(k, k): v for k, v in kw.items()})

    _LMP.create_atoms = create_atoms_compat
    print("  parcheado create_atoms: id->atomid, type->atype")

def parchear_compatibilidad_setuptools():
    """distutils.strtobool devolvia 1/0; el reemplazo de setuptools devuelve True/False.

    FitSNAP interpola ese valor dentro del comando `compute snap` de LAMMPS,
    que ahi exige un entero. Devolver 1/0 restaura el comportamiento original.
    """
    import fitsnap3lib.io.sections.sections as _sec

    orig = getattr(_sec, "_strtobool_original", None) or _sec.strtobool
    _sec._strtobool_original = orig

    if not isinstance(orig("true"), bool):      # ojo: bool ES subclase de int
        print("  strtobool ya devuelve enteros: no hace falta parche")
        return

    def strtobool_entero(val):
        return int(orig(val))

    _sec.strtobool = strtobool_entero
    print("  parcheado strtobool: devuelve 1/0 en vez de True/False")

def parchear_compatibilidad_lammps_numpy():
    """LAMMPS elimino extract_atom_iarray y extract_atom_darray del envoltorio numpy.

    FitSNAP los llama una vez por configuracion para leer ids, posiciones y
    tipos. Se reponen copiando el cuerpo exacto de lammps 2024.8.
    """
    from ctypes import c_int
    from lammps.numpy_wrapper import numpy_wrapper as _NW
    from lammps.constants import (LAMMPS_INT, LAMMPS_INT_2D,
                                  LAMMPS_DOUBLE, LAMMPS_DOUBLE_2D)
    repuestos = []

    if not hasattr(_NW, "extract_atom_iarray"):
        def extract_atom_iarray(self, name, nelem, dim=1):
            if name in ("id", "molecule"):
                tipo = self.lmp.c_tagint
            elif name == "image":
                tipo = self.lmp.c_imageint
            else:
                tipo = c_int
            crudo = self.lmp.extract_atom(name, LAMMPS_INT if dim == 1 else LAMMPS_INT_2D)
            return self.iarray(tipo, crudo, nelem, dim)
        _NW.extract_atom_iarray = extract_atom_iarray
        repuestos.append("extract_atom_iarray")

    if not hasattr(_NW, "extract_atom_darray"):
        def extract_atom_darray(self, name, nelem, dim=1):
            crudo = self.lmp.extract_atom(name, LAMMPS_DOUBLE if dim == 1 else LAMMPS_DOUBLE_2D)
            return self.darray(crudo, nelem, dim)
        _NW.extract_atom_darray = extract_atom_darray
        repuestos.append("extract_atom_darray")

    if repuestos:
        print("  repuestos en el envoltorio numpy:", ", ".join(repuestos))
    else:
        print("  el envoltorio numpy los conserva: no hace falta parche")

def arreglar_cabeceras_cuda():
    """PyTorch 2.13 manda parte del calculo de derivadas a un nucleo de Triton.

    Triton no trae ese nucleo compilado: escribe un archivo en C y llama a gcc
    en el momento. Ese archivo hace #include "cuda.h", asi que gcc necesita
    encontrarlo. Aqui se localiza en la maquina y se le indica a gcc por medio
    de CPATH, que es la variable de entorno que gcc lee como ruta de busqueda
    adicional. No se instala ni se modifica nada.
    """
    import os, sys, glob, subprocess, shutil
    from pathlib import Path

    patrones = [
        f"{sys.prefix}/targets/*/include/cuda.h",
        f"{sys.prefix}/include/cuda.h",
        f"{sys.prefix}/lib/python*/site-packages/triton/backends/nvidia/include/cuda.h",
        "/usr/local/cuda*/include/cuda.h",
        "/usr/local/cuda*/targets/*/include/cuda.h",
        "/usr/include/cuda.h",
        "/opt/conda/targets/*/include/cuda.h",
    ]
    hallados = sorted({h for pat in patrones for h in glob.glob(pat)})

    if hallados:
        carpetas = sorted({os.path.dirname(h) for h in hallados})
        previo = os.environ.get("CPATH", "")
        partes = [c for c in carpetas if c not in previo.split(":")]
        os.environ["CPATH"] = ":".join(partes + ([previo] if previo else []))
        for c in carpetas:
            print(f"  cuda.h encontrado en: {c}")
        print(f"  CPATH = {os.environ['CPATH']}")
        # Triton cachea los fallos de compilacion; se limpia para que reintente.
        cache = Path.home() / ".triton" / "cache"
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
            print("  cache de Triton limpiada")
    else:
        print("  cuda.h NO esta en esta maquina.")
        print("  Instalalo en una terminal y vuelve a ejecutar esta celda:")
        print("      conda install -y -n fitsnap -c conda-forge cuda-driver-dev")

    # Verificacion real: exactamente lo que hara Triton.
    prueba = Path("/tmp/prueba_cuda.c")
    prueba.write_text('#include "cuda.h"\nint main(void){return 0;}\n')
    r = subprocess.run(["gcc", "-fsyntax-only", str(prueba)],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("  gcc compila #include \"cuda.h\": OK")
    else:
        print("  gcc TODAVIA no encuentra cuda.h:")
        print("   ", r.stderr.strip().split("\n")[0])

def parchear_evaluacion_una_config():
    """El analisis de error evalua las configuraciones de una en una.

    En ese camino FitSNAP arma el numero de atomos como un escalar
    (`torch.tensor(config.natoms)`), mientras que al entrenar lo arma como un
    vector (`batch['noa']`). Con el escalar, `torch.zeros(...)` produce un
    tensor sin dimensiones e `index_add_` lo rechaza. PyTorch antes lo
    toleraba tratandolo como si tuviera una dimension.

    El arreglo le anade esa dimension al entrar y se la QUITA al salir.

    Las dos mitades importan. Si solo se anade, la energia predicha sale como
    vector de un elemento en vez de escalar, y aguas abajo el analisis de error
    hace:

        e_pred = energies_model.detach().numpy() / c.natoms
        mae_e[grupo]["train"] += abs(c.energy - e_pred)

    con lo que el acumulador se convierte en un arreglo de numpy. Eso revienta
    al escribir metrics.dat, y ademas escribiria "[-204.65]" con corchetes en
    perconfig.dat. Devolviendo la forma original el parche queda invisible para
    todo lo que viene despues.
    """
    from fitsnap3lib.lib.neural_networks.pytorch import FitTorch

    orig = getattr(FitTorch, "_forward_original", None) or FitTorch.forward
    FitTorch._forward_original = orig

    def forward_compat(self, x, xd, indices, atoms_per_structure, *a, **kw):
        ajustado = (hasattr(atoms_per_structure, "dim")
                    and atoms_per_structure.dim() == 0)
        if ajustado:
            atoms_per_structure = atoms_per_structure.reshape(1)

        salida = orig(self, x, xd, indices, atoms_per_structure, *a, **kw)

        # Deshacer el cambio: devolver la energia con la forma que tenia antes.
        if ajustado and isinstance(salida, tuple) and len(salida) == 2:
            energias, fuerzas = salida
            if (energias is not None and hasattr(energias, "dim")
                    and energias.dim() == 1 and energias.numel() == 1):
                energias = energias.reshape(())
            return (energias, fuerzas)
        return salida

    FitTorch.forward = forward_compat
    print("  parcheado FitTorch.forward: num_atoms escalar <-> vector (ida y vuelta)")

# ======================================================================
# Celda 17 del notebook — GRUPOS, pesos, FRACCION, pesos_de, ajustes_fitsnap
# ======================================================================

GRUPOS = ["BCC_54_kjpaw", "BCC_54_NPT", "BCC_54_isolated",
          "LiF_64_kjpaw", "LiF_64_NPT", "LiF_64_isolated",
          "LiFinterface_kjpaw", "LiFinterface_NPT",
          "LiwithF", "LiwithF_NPT", "LiwithF_isolated", "Special"]

PESO_ENERGIA, PESO_FUERZA, PESO_VIRIAL = 0.2, 1.0, 1.00E-04

FRACCION = 0.25

def pesos_de(_grupo):
    return (f"{0.7 * FRACCION:.4f} {0.3 * FRACCION:.4f} "
            f"{PESO_ENERGIA} {PESO_FUERZA} {PESO_VIRIAL:.2E}")

BATCH_SIZE       = 50      # la seccion 11 lo mide

TASA_APRENDIZAJE = 5e-6    # la seccion 11 lo barre

def ajustes_fitsnap(ruta_json, semilla=1111, epocas=70, grupos=None):
    """Diccionario equivalente a LiF-example.in del equipo."""
    g = {k: pesos_de(k) for k in GRUPOS if grupos is None or k in grupos}
    return {
        "BISPECTRUM": {
            "numTypes": 2, "twojmax": "8 8", "rcutfac": 4.812302818,
            "rfac0": 0.99363, "rmin0": 0.0, "wj": "1.0 1.0",
            "radelem": "0.5 0.5", "type": "Li F",
            "wselfallflag": 0, "chemflag": 1, "bzeroflag": 1,
            "quadraticflag": 0, "bikflag": 1, "dgradflag": 1,
        },
        "CALCULATOR": {"calculator": "LAMMPSSNAP", "energy": 1,
                       "per_atom_energy": 1, "force": 1, "stress": 1, "nonlinear": 1},
        "ESHIFT": {"Li": 0.0, "F": 0.0},
        "PYTORCH": {
            "layer_sizes": "num_desc 256 128 64 64 1",
            "learning_rate": TASA_APRENDIZAJE, "num_epochs": epocas,
            "batch_size": BATCH_SIZE,
            "save_state_output": "modelo.pt",
            "multi_element_option": 2, "num_elements": 2,
            # OJO: FitSNAP no acepta `manual_seed` y con `manual_seed_flag=1`
            # llama a torch.manual_seed(0) SIEMPRE, ignorando lo que pidas.
            # Por eso se apaga aqui y la semilla se fija desde Python.
            "manual_seed_flag": 0,
        },
        "SOLVER": {"solver": "PYTORCH", "compute_testerrs": 1, "detailed_errors": 1},
        "SCRAPER": {"scraper": "JSON"},
        "PATH": {"dataPath": str(ruta_json)},
        "OUTFILE": {"metrics": "metrics.dat", "potential": "pot"},
        "REFERENCE": {"units": "metal", "atom_style": "atomic",
                      "pair_style": "zero 10.0", "pair_coeff1": "* *"},
        "GROUPS": {
            "group_sections": "name training_size testing_size eweight fweight vweight",
            "group_types": "str float float float float float",
            # random_seed FIJO a proposito: el reparto entrenamiento/prueba debe ser
            # IDENTICO en las 8 condiciones. Lo unico que varia entre semillas es la
            # inicializacion de los pesos, no los datos.
            "smartweights": 0, "random_sampling": 1, "random_seed": 12345, **g,
        },
        "EXTRAS": {"dump_peratom": 1, "dump_perconfig": 1},
        "MEMORY": {"override": 0},
    }

# ======================================================================
# Celda 25 del notebook — SEMILLAS y CONDICIONES
# ======================================================================

SEMILLAS = [1111, 2222, 3333, 4444, 5555]   # las 3 primeras son las de Wang et al.

CONDICIONES = [
    # id,                capas,                          salida,      kwargs
    ("1_mlp_base",      "num_desc 256 128 64 64 1",      "linear",    {}),
    ("2_mlp_compacto",  "num_desc 64 1",                 "linear",    {}),
    ("3_kan_gauss",     "num_desc 256 128 64 64 1",      "gaussiana", dict(K=8)),
    ("4_kan_bspline",   "num_desc 256 128 64 64 1",      "bspline",   dict(G=5, k=3)),
    ("5_kan_cheby",     "num_desc 256 128 64 64 1",      "chebyshev", dict(grado=5)),
    ("6_kan_compacto",  "num_desc 64 1",                 "gaussiana", dict(K=8)),
    # Redimensionadas el 6 de octubre de 2026, con el residuo activado.
    # El termino residual anade n_in x n_out = 64 parametros por capa KAN,
    # asi que el ancho que igualaba capacidades sin el ya no la iguala.
    #   7_kan_iso      58 -> 57 : 352 307 contra 352 313 del MLP base  (-6)
    #   8_mlp_ampliado 71 -> 72 : 352 841 contra 352 825 del KAN       (+16)
    # Con el residuo puesto y los anchos viejos el desajuste era +68 y -50.
    ("7_kan_iso",       "num_desc 256 128 64 57 1",      "gaussiana", dict(K=8)),
    ("8_mlp_ampliado",  "num_desc 256 128 64 72 1",      "linear",    {}),
]

# ======================================================================
# Celda 28 del notebook — leer_metrics y tabla_final
# ======================================================================

def leer_metrics(ruta):
    """Lee un metrics.dat de FitSNAP y devuelve un DataFrame."""
    filas = []
    for ln in Path(ruta).read_text().splitlines()[1:]:
        p = ln.split()
        if len(p) >= 6:
            filas.append(dict(grupo=p[0], conjunto=p[1], magnitud=p[2],
                              n=int(p[3]), MAE=float(p[4]), RMSE=float(p[5])))
    return pd.DataFrame(filas)

def tabla_final(resultados):
    """Junta todas las corridas en la tabla del paper."""
    filas = []
    for r in resultados:
        df = leer_metrics(r["metrics"])
        sel = df[(df.grupo == "*ALL") & (df.conjunto == "Test")]
        e = sel[sel.magnitud == "Energy"].iloc[0]
        f = sel[sel.magnitud == "Force"].iloc[0]
        filas.append(dict(condicion=r["condicion"], semilla=r["semilla"],
                          MAE_E_meV_atomo=e.MAE * 1000, RMSE_E=e.RMSE * 1000,
                          MAE_F_meV_A=f.MAE * 1000, RMSE_F=f.RMSE * 1000,
                          seg_ajuste=r["seg_ajuste"]))
    d = pd.DataFrame(filas)
    return (d.groupby("condicion")
             .agg(MAE_E=("MAE_E_meV_atomo", "mean"), MAE_E_sd=("MAE_E_meV_atomo", "std"),
                  MAE_F=("MAE_F_meV_A", "mean"), MAE_F_sd=("MAE_F_meV_A", "std"),
                  MAE_F_peor=("MAE_F_meV_A", "max"),
                  seg=("seg_ajuste", "mean"))
             .sort_values("MAE_F"))

# ======================================================================
# Celda 21 del notebook — una_corrida
# ======================================================================

import time, os, shutil

import numpy as np

from dataclasses import dataclass, asdict

def revisar_descriptores(fs):
    """Cuenta descriptores constantes y avisa. Devuelve cuantos hay.

    FitSNAP estandariza cada descriptor dividiendo por su desviacion tipica.
    Si un descriptor vale lo mismo en TODAS las configuraciones, esa division
    es 1/0 = infinito, y a partir de ahi la red entrena con NaN en silencio.

    Pasa cuando el conjunto declara dos elementos (`type = Li F`) pero los
    grupos elegidos solo contienen uno: los 440 descriptores incluyen las 8
    combinaciones quimicas, y las que involucran al elemento ausente valen
    cero en todas las filas.
    """
    a = fs.pt.shared_arrays["a"].array
    desv = np.std(a, axis=0)
    muertos = int(np.sum(desv == 0))
    if muertos:
        print(f"  AVISO: {muertos} de {a.shape[1]} descriptores son constantes "
              f"(desviacion cero).")
        print(f"         La estandarizacion hara 1/0 y la red entrenara con NaN.")
        print(f"         Causa tipica: los grupos elegidos no contienen los dos "
              f"elementos.")
        print(f"         Solucion: incluir un grupo con Li y F (p.ej. LiF_64_kjpaw).")
    else:
        print(f"  descriptores: los {a.shape[1]} tienen varianza. OK")
    return muertos

def carpeta_datos(raiz):
    """Carpeta padre de los grupos, la que FitSNAP recibe como dataPath.

    Dos disposiciones conviven y las dos son validas:
      - `dataset/` es la del repositorio, donde los JSON viajan con el codigo.
      - `JSON/` es la que deja el conversor cuando se parte de las salidas
        crudas de Quantum ESPRESSO en la maquina de computo.
    Se prefiere la primera si existe.
    """
    raiz = Path(raiz)
    d = raiz / "dataset"
    return d if d.is_dir() else raiz / "JSON"


def una_corrida(ruta_json, salida_kan, kw_kan, semilla, carpeta_salida,
                epocas=70, grupos=None, fs_precargado=None):
    """Corre FitSNAP con el bloque de salida indicado. Devuelve la ruta de metrics.dat.

    ruta_json: carpeta PADRE (dataPath). FitSNAP le agrega el nombre de cada grupo.
               Correcto:   RAIZ / "JSON"
               Incorrecto: RAIZ / "JSON" / "BCC_54_kjpaw"

    Si se pasa `fs_precargado`, reutiliza sus descriptores ya calculados en vez de
    volver a llamar a LAMMPS.
    """
    from fitsnap3lib.fitsnap import FitSnap

    carpeta_salida = Path(carpeta_salida); carpeta_salida.mkdir(parents=True, exist_ok=True)
    previo = os.getcwd(); os.chdir(carpeta_salida)
    try:
        parchear_red(salida_kan, **kw_kan) if salida_kan != "linear" else restaurar_red()
        cfg = ajustes_fitsnap(ruta_json, semilla=semilla, epocas=epocas, grupos=grupos)

        # La semilla se fija AQUI, no en el .in: FitSNAP construye las redes al
        # parsear la configuracion, asi que sembrar antes de FitSnap(...) alcanza.
        # Sembrar una sola vez (y no dentro de la fabrica) hace que las dos redes
        # por elemento arranquen distintas, como corresponde.
        torch.manual_seed(semilla)
        np.random.seed(semilla)

        t0 = time.perf_counter()
        if fs_precargado is None:
            fs = FitSnap(cfg, arglist=["--overwrite"])
            fs.scrape_configs()
            fs.process_configs()          # <- LAMMPS, el paso caro
        else:
            fs = fs_precargado   # descriptores ya calculados
        t_desc = time.perf_counter() - t0

        n_muertas = revisar_descriptores(fs)

        t1 = time.perf_counter()
        fs.perform_fit()
        fs.write_output()
        t_fit = time.perf_counter() - t1

        return dict(metrics=carpeta_salida / "metrics.dat",
                    seg_descriptores=t_desc, seg_ajuste=t_fit,
                    descriptores_muertos=n_muertas, fs=fs)
    finally:
        os.chdir(previo)
        restaurar_red()

# ======================================================================
# Celda 34 del notebook — MonitorHardware, _curva_desde_texto, cambiar_arquitectura
# ======================================================================

import io as _io, re, copy, time, threading, subprocess, contextlib

class MonitorHardware:
    '''Muestrea nvidia-smi y la RAM mientras corre un bloque de codigo.

    Uso:
        with MonitorHardware() as m:
            ...lo que sea...
        m.informe()
    '''

    def __init__(self, cada=2.0):
        self.cada = cada
        self.muestras = []
        self._parar = threading.Event()
        self._hilo = None

    def _leer(self):
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            util, usada, total = [float(x) for x in r.stdout.strip().split(chr(10))[0].split(',')]
        except Exception:
            util = usada = total = float('nan')
        try:
            info = {}
            for l in open('/proc/meminfo'):
                k, v = l.split(':', 1)
                info[k] = float(v.strip().split()[0]) / 1024 / 1024   # GiB
            ram_usada = info['MemTotal'] - info['MemAvailable']
            ram_total = info['MemTotal']
        except Exception:
            ram_usada = ram_total = float('nan')
        return dict(gpu_pct=util, vram_usada=usada, vram_total=total,
                    ram_usada=ram_usada, ram_total=ram_total)

    def _bucle(self):
        while not self._parar.wait(self.cada):
            self.muestras.append(self._leer())

    def __enter__(self):
        self.muestras = [self._leer()]
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *a):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=self.cada + 1)
        self.muestras.append(self._leer())
        return False

    def resumen(self):
        if not self.muestras:
            return {}
        def col(k):
            v = np.array([m[k] for m in self.muestras], dtype=float)
            return v[~np.isnan(v)]
        g, vu = col('gpu_pct'), col('vram_usada')
        vt, ru, rt = col('vram_total'), col('ram_usada'), col('ram_total')
        return dict(
            n=len(self.muestras),
            gpu_media=float(g.mean()) if g.size else float('nan'),
            gpu_max=float(g.max()) if g.size else float('nan'),
            gpu_pct_ociosa=float((g < 10).mean() * 100) if g.size else float('nan'),
            vram_max=float(vu.max()) if vu.size else float('nan'),
            vram_total=float(vt.max()) if vt.size else float('nan'),
            ram_max=float(ru.max()) if ru.size else float('nan'),
            ram_total=float(rt.max()) if rt.size else float('nan'),
        )

    def informe(self):
        r = self.resumen()
        if not r:
            print('  sin muestras'); return r
        print(f"  muestras tomadas         {r['n']}")
        print(f"  GPU utilizacion media    {r['gpu_media']:6.1f} %")
        print(f"  GPU utilizacion maxima   {r['gpu_max']:6.1f} %")
        print(f"  GPU por debajo del 10 %  {r['gpu_pct_ociosa']:6.1f} % del tiempo")
        print(f"  VRAM maxima usada        {r['vram_max']:8.0f} / {r['vram_total']:.0f} MiB"
              f"  ({r['vram_max']/r['vram_total']*100:.1f} %)")
        print(f"  RAM maxima usada         {r['ram_max']:8.1f} / {r['ram_total']:.1f} GiB")
        print()
        if r['gpu_media'] < 30:
            print('  LECTURA: la GPU esta mayormente ociosa. El cuello de botella no es el')
            print('           calculo. Subir batch_size deberia acelerar casi gratis.')
        elif r['gpu_media'] > 70:
            print('  LECTURA: la GPU esta trabajando. Subir batch_size ayudara poco; para')
            print('           acortar el presupuesto habria que recortar la rejilla.')
        else:
            print('  LECTURA: utilizacion intermedia. Vale la pena probar batch_size mayor.')
        if r['vram_max'] / r['vram_total'] < 0.25:
            print(f"  Sobra VRAM ({r['vram_max']/r['vram_total']*100:.0f} % usada):"
                  f" hay margen para subir batch_size varias veces.")
        return r

_RE_EPOCA = re.compile(r'^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$')

def _curva_desde_texto(txt):
    filas = []
    for linea in txt.split(chr(10)):
        m = _RE_EPOCA.match(linea)
        if m:
            filas.append((int(m.group(1)), float(m.group(2)),
                          float(m.group(3)), float(m.group(4))))
    return filas

def cambiar_arquitectura(fs, capas, salida='linear', kw=None, semilla=1111):
    '''Sustituye las redes DENTRO de un FitSnap ya construido.

    Conserva los descriptores ya calculados, que es el paso caro. Necesario
    porque FitSNAP construye la red en el __init__ del solver: reutilizar el
    objeto sin esto haria que todas las condiciones usen la misma red.
    '''
    kw = kw or {}
    n_desc  = fs.solver.model.desc_len
    L       = [n_desc] + [int(x) for x in str(capas).split()[1:]]
    n_redes = len(fs.solver.model.networks)

    # Una semilla, todas las redes: las dos redes por elemento arrancan distintas.
    torch.manual_seed(semilla); np.random.seed(semilla)
    if salida != 'linear':
        parchear_red(salida, **kw)
    else:
        restaurar_red()
    nuevas = [crear_red(L, salida, **kw) for _ in range(n_redes)]

    dt = getattr(fs.solver.model, 'dtype', torch.get_default_dtype())
    for i, red in enumerate(nuevas):
        red.to(dt)
        setattr(fs.solver.model, 'network_architecture' + str(i), red)
    fs.solver.model.networks = nuevas
    fs.solver.model = fs.solver.model.to(fs.solver.device)

    # El optimizador viejo apunta a parametros que ya no existen.
    fs.solver.optimizer = torch.optim.Adam(fs.solver.model.parameters(),
                                           lr=fs.solver.learning_rate)
    fs.solver.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fs.solver.optimizer, mode='min', factor=0.5, patience=49,
        threshold=1e-4, threshold_mode='abs')
    if hasattr(fs.solver, 'model_best'):
        del fs.solver.model_best

    fs.config.sections['PYTORCH'].layer_sizes = L

    # Foto de los pesos recien inicializados, para la medida 3.
    fs.solver.model._estado_inicial = copy.deepcopy(fs.solver.model.state_dict())

    n_par = sum(p.numel() for p in fs.solver.model.parameters())
    ultima = type(fs.solver.model.networks[0][-1]).__name__
    print(f'  arquitectura cambiada: {L}  salida={salida}  ultima capa={ultima}')
    print(f'  parametros totales: {n_par:,}  ({n_redes} redes)')
    return fs.solver.model

# ======================================================================
# Celda 55 del notebook — comprobar_cambio_arquitectura
# ======================================================================

def comprobar_cambio_arquitectura(fs):
    '''Verifica que el cambio de arquitectura realmente cambia la red.'''
    print('comprobacion del cambio en caliente:')
    for capas, sal, kw in [('num_desc 256 128 64 64 1', 'linear', {}),
                           ('num_desc 256 128 64 64 1', 'gaussiana', dict(K=8)),
                           ('num_desc 64 1', 'linear', {})]:
        cambiar_arquitectura(fs, capas, sal, kw)
    print()
    print('  Si las tres lineas muestran ultima capa y conteo distintos, el')
    print('  arreglo funciona y fs_precargado ya es seguro para la rejilla.')

# ======================================================================
# Celda 57 del notebook — Medicion de influencia de las bases KAN
# ======================================================================

def _bases_de(capa, x):
    '''Matriz de funciones base evaluadas en x. Vale para las tres bases.'''
    if hasattr(capa, '_bases'):                       # B-spline
        return capa._bases(x)
    if hasattr(capa, 'centros'):                      # gaussiana
        z = (x.unsqueeze(-1) - capa.centros) / capa.h
        return torch.exp(-z * z)
    t = torch.tanh(x).unsqueeze(-1)                   # chebyshev
    Ts = [torch.ones_like(t), t]
    for _ in range(2, capa.K):
        Ts.append(2 * t * Ts[-1] - Ts[-2])
    return torch.cat(Ts[:capa.K], -1)

def capturar_entradas_ultima_capa(fs, n_config=8):
    '''Engancha la ultima capa y guarda lo que realmente le entra.'''
    capa = fs.solver.model.networks[0][-1]
    guardado = []
    h = capa.register_forward_pre_hook(
        lambda m, inp: guardado.append(inp[0].detach().cpu().double()))
    try:
        n = min(n_config, len(fs.solver.configs))
        for i in range(n):
            fs.solver.evaluate_configs(config_idx=i, standardize_bool=False,
                                       dtype=torch.float64)
    finally:
        h.remove()
    if not guardado:
        return None
    return torch.cat(guardado, 0)

def _suelo_no_linealidad(B, rejilla):
    '''Minimo de no-linealidad alcanzable con esta base.

    Ni K gaussianas pueden reproducir una recta EXACTA, asi que la medida 2
    tiene un suelo distinto de cero. Sin este numero de referencia, un valor
    pequeno se leeria como "algo de no linealidad" cuando en realidad es
    "lo mas lineal que esta base puede ser".
    '''
    try:
        coef = torch.linalg.lstsq(B, rejilla.unsqueeze(-1)).solution
    except Exception:
        return float('nan')
    phi = (B @ coef).squeeze(-1)
    xm = rejilla.mean(); xc = rejilla - xm
    pend = (xc * phi).sum() / (xc * xc).sum()
    recta = pend * rejilla + (phi.mean() - pend * xm)
    total = ((phi - phi.mean()) ** 2).sum()
    return float(((phi - recta) ** 2).sum() / total.clamp_min(1e-30))

def medir_influencia_kan(fs, n_config=8, n_puntos=241):
    '''Tres medidas de cuanto influye el bloque KAN en la prediccion.'''
    capa = fs.solver.model.networks[0][-1]
    if not hasattr(capa, 'w'):
        print('  la ultima capa es Linear (MLP): no hay KAN que medir')
        return None

    X = capturar_entradas_ultima_capa(fs, n_config)
    if X is None or X.numel() == 0:
        print('  no se pudieron capturar activaciones')
        return None
    lo, hi = float(X.min()), float(X.max())
    p01, p99 = [float(torch.quantile(X.flatten(), q)) for q in (0.01, 0.99)]
    print(f'RANGO REAL de entrada a la ultima capa ({X.shape[0]:,} atomos):')
    print(f'  min {lo:+.3f}   p01 {p01:+.3f}   p99 {p99:+.3f}   max {hi:+.3f}')
    rng = getattr(capa, 'rango', None)
    if hasattr(capa, 'centros'):
        c = capa.centros.detach().cpu()
        fuera = int(((c < p01) | (c > p99)).sum())
        print(f'  centros de la base: {[round(float(v), 2) for v in c]}')
        if fuera:
            print(f'  AVISO: {fuera} de {len(c)} centros caen fuera del rango real de datos.')
            print(f'         Esas campanas casi no reciben senal: la capacidad efectiva')
            print(f'         del KAN es menor que sus {capa.n_in*capa.n_out*capa.K} pesos.')
            print(f'         Considerar kw_kan=dict(K=8, rango=({p01:.1f}, {p99:.1f})).')
        else:
            print('  todos los centros caen dentro del rango real de datos. OK')
    print()

    w = capa.w.detach().cpu().double()                 # (n_in, n_out, K)
    n_in, n_out, K = w.shape
    rejilla = torch.linspace(p01, p99, n_puntos, dtype=torch.float64)
    B = _bases_de(capa, rejilla).double()              # (n_puntos, K)
    phi = torch.einsum('pk,iok->pio', B, w)            # (n_puntos, n_in, n_out)

    # --- ajuste de la mejor recta por arista -------------------------------
    xm = rejilla.mean()
    xc = rejilla - xm
    sxx = (xc * xc).sum()
    pend = torch.einsum('p,pio->io', xc, phi) / sxx
    corte = phi.mean(0) - pend * xm
    recta = pend.unsqueeze(0) * rejilla.view(-1, 1, 1) + corte.unsqueeze(0)
    resid = ((phi - recta) ** 2).sum(0)
    total = ((phi - phi.mean(0, keepdim=True)) ** 2).sum(0)
    no_lin = (resid / total.clamp_min(1e-30)).flatten()   # 1 - R^2 por arista

    # --- ablacion: KAN real contra su version linealizada ------------------
    Xd = X.double()
    Bx = _bases_de(capa, Xd.flatten()).double().view(Xd.shape[0], Xd.shape[1], -1)
    e_kan = torch.einsum('nik,iok->no', Bx, w)
    e_lin = torch.einsum('ni,io->no', Xd, pend) + corte.sum(0)
    dif = (e_kan - e_lin).flatten()
    rms_dif = float((dif ** 2).mean().sqrt())
    rms_kan = float((e_kan.flatten() ** 2).mean().sqrt())

    # --- desplazamiento de los parametros ----------------------------------
    mov_kan = mov_resto = float('nan')
    est0 = getattr(fs.solver.model, '_estado_inicial', None)
    if est0 is not None:
        act = fs.solver.model.state_dict()
        d_kan = d_resto = n_kan = n_resto = 0.0
        for k, v0 in est0.items():
            if k not in act:
                continue
            d = float(((act[k].detach().cpu().double() - v0.detach().cpu().double()) ** 2).sum())
            n = v0.numel()
            # la ultima capa de cada red es el bloque de salida
            if k.split('.')[-2:-1] and k.split('.')[-2] == str(len(fs.solver.model.networks[0]) - 1):
                d_kan += d; n_kan += n
            else:
                d_resto += d; n_resto += n
        mov_kan   = (d_kan / n_kan) ** 0.5 if n_kan else float('nan')
        mov_resto = (d_resto / n_resto) ** 0.5 if n_resto else float('nan')

    print('LAS TRES MEDIDAS')
    print(f'  1. Ablacion por linealizacion')
    print(f'     cambio RMS en la energia por atomo al linealizar: {rms_dif:.4e} eV')
    print(f'     magnitud RMS de la salida del bloque KAN:         {rms_kan:.4e} eV')
    if rms_kan > 0:
        print(f'     la no linealidad aporta el {rms_dif/rms_kan*100:.2f} % de la salida')
    print()
    suelo = _suelo_no_linealidad(B, rejilla)
    print(f'  2. No linealidad efectiva por arista (1 - R2 contra la mejor recta)')
    print(f'     mediana {float(no_lin.median()):.4e}   media {float(no_lin.mean()):.4e}'
          f'   maxima {float(no_lin.max()):.4e}')
    print(f'     SUELO de esta base (lo mas lineal que puede ser): {suelo:.4e}')
    if suelo == suelo and suelo > 0:
        print(f'     razon mediana/suelo: {float(no_lin.median())/suelo:.1f}x'
              f'   (cerca de 1 = el KAN es una recta cara)')
    print(f'     aristas por encima de 10x el suelo: '
          f'{int((no_lin > 10*suelo).sum()) if suelo==suelo else 0} de {no_lin.numel()}')
    print()
    print(f'  3. Desplazamiento de los pesos durante el entrenamiento (RMS por parametro)')
    if est0 is None:
        print('     sin foto inicial. Usar cambiar_arquitectura() para guardarla.')
    else:
        print(f'     bloque de salida (el KAN): {mov_kan:.4e}')
        print(f'     resto de la red:           {mov_resto:.4e}')
        if mov_resto > 0:
            print(f'     razon KAN/resto:           {mov_kan/mov_resto:.2f}x')
    print()
    print('COMO LEERLO')
    print('  Medida 1: leerla en PORCENTAJE, no en valor absoluto. El absoluto')
    print('            depende de la escala de los pesos y puede enganar.')
    print('            < 1 %  -> el KAN opera en regimen lineal: NO influye.')
    print('                      Es un resultado reportable, no un fracaso.')
    print('            > 10 % -> la no linealidad aprendida si esta trabajando.')
    print('  Medida 2: compararla contra el SUELO, no contra cero. Una base de K')
    print('            gaussianas no puede hacer una recta exacta, asi que el')
    print('            minimo no es cero. Razon mediana/suelo cerca de 1 = recta.')
    print('  Medida 3: si el KAN se movio mucho menos que el resto de la red,')
    print('            apenas se entreno y la comparacion no es concluyente.')
    return dict(rango=(lo, p01, p99, hi), no_lin=no_lin,
                rms_dif=rms_dif, rms_kan=rms_kan,
                mov_kan=mov_kan, mov_resto=mov_resto,
                pendientes=pend, cortes=corte, phi=phi, rejilla=rejilla)

# ======================================================================
# Celda 59 del notebook — dibujar_funciones_kan
# ======================================================================

def dibujar_funciones_kan(INFLU, n=12, archivo=None):
    '''Dibuja las n aristas mas no lineales, contra su mejor recta.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x   = INFLU['rejilla'].numpy()
    phi = INFLU['phi']
    nl  = INFLU['no_lin']
    idx = torch.argsort(nl, descending=True)[:n]

    filas = (n + 3) // 4
    fig, ejes = plt.subplots(filas, 4, figsize=(13, 2.7 * filas))
    ejes = ejes.flatten()
    for j, plano in enumerate(idx):
        i = int(plano) // phi.shape[2]
        o = int(plano) % phi.shape[2]
        y = phi[:, i, o].numpy()
        recta = INFLU['pendientes'][i, o].item() * x + INFLU['cortes'][i, o].item()
        ejes[j].plot(x, y, lw=1.8, label='aprendida')
        ejes[j].plot(x, recta, lw=1.0, ls='--', color='0.5', label='mejor recta')
        ejes[j].set_title(f'arista {i}  (1-R2 = {float(nl[plano]):.3f})', fontsize=9)
        ejes[j].tick_params(labelsize=7)
        if j == 0:
            ejes[j].legend(fontsize=7)
    for j in range(len(idx), len(ejes)):
        ejes[j].axis('off')
    fig.suptitle('Funciones univariadas aprendidas por el bloque KAN, '
                 'sobre el rango real de datos', fontsize=11)
    fig.tight_layout()
    destino = archivo or (RAIZ / 'funciones_kan.png')
    fig.savefig(destino, dpi=140, bbox_inches='tight')
    print(f'figura guardada en {destino}')
    return destino

# ======================================================================
# Anadido el 2026-09-30, no procede del notebook.
# ======================================================================

def aplicar_todos_los_parches():
    """Aplica los parches de compatibilidad en el orden correcto.

    Hay que llamarla ANTES del primer import de fitsnap3lib.
    """
    parchear_compatibilidad_torch()
    parchear_compatibilidad_python()
    parchear_compatibilidad_lammps()
    parchear_compatibilidad_setuptools()
    parchear_compatibilidad_lammps_numpy()
    parchear_evaluacion_una_config()
    arreglar_cabeceras_cuda()


# Carpeta de trabajo. En el notebook esto era una variable de celda; aqui se
# lee del entorno para que un subproceso la herede. `dibujar_funciones_kan` la
# usa como destino por defecto de las figuras.
RAIZ = Path(os.environ.get("KAN_RAIZ", "/workspace/kan_snap"))


# Sobreescritura por entorno. Permite que un trabajador lanzado como
# subproceso reciba estos valores sin tocar el archivo.
FRACCION = float(os.environ.get("KAN_FRACCION", FRACCION))
BATCH_SIZE = int(os.environ.get("KAN_BATCH_SIZE", BATCH_SIZE))
TASA_APRENDIZAJE = float(os.environ.get("KAN_LR", TASA_APRENDIZAJE))


# dtype_setting: la clave de configuracion real de FitSNAP para el tipo de
# dato. Esta en fitsnap3lib/io/sections/solver_sections/pytorch.py:
#
#     self.dtype_setting = self.get_value("PYTORCH", "dtype_setting", "1", "int")
#     if (self.dtype_setting == 1): self.dtype = torch.float32
#     else:                         self.dtype = torch.float64
#
# `dtype` NO es una clave de entrada, es el atributo derivado. Escribirla en
# la configuracion no tiene ningun efecto, que es lo que paso el 28 de
# septiembre al intentar forzar float64.
_ajustes_fitsnap_sin_dtype = ajustes_fitsnap


def ajustes_fitsnap(ruta_json, semilla=1111, epocas=70, grupos=None,
                    dtype_setting=None):
    """Igual que la del notebook, mas la clave dtype_setting.

    dtype_setting = 1  -> torch.float32 (el defecto de FitSNAP)
                    2  -> torch.float64 (el tipo del modelo de referencia)

    Si no se pasa, se lee de KAN_DTYPE_SETTING, y si tampoco esta, 1.
    """
    cfg = _ajustes_fitsnap_sin_dtype(ruta_json, semilla=semilla,
                                     epocas=epocas, grupos=grupos)
    # FitSNAP 3.1.0.4 lee dtype_setting con get_value, pero _check_section la
    # rechaza como "unmatched variable": no esta en la lista de claves
    # admitidas de la seccion PYTORCH. Escribirla aborta la construccion del
    # objeto. Por eso NO se envia nunca; float64 sigue sin via por aqui.
    if dtype_setting is None:
        _v = os.environ.get("KAN_DTYPE_SETTING", "").strip()
        dtype_setting = int(_v) if _v else None
    if dtype_setting is not None:
        print("aviso: esta version de FitSNAP rechaza dtype_setting en "
              "_check_section; la clave no se envia y se queda en float32")
    return cfg


# ======================================================================
# Anadido el 2026-10-01: normalizacion de la entrada al bloque KAN.
#
# Motivo: la entrada que llega al bloque de salida tiene media ~0.72 y
# desviacion ~0.22 (el Softplus previo solo devuelve positivos), mientras las
# bases se reparten en (-2, 2). Tres de las ocho campanas gaussianas quedan
# fuera del soporte de los datos y no reciben gradiente. Medido el 1 de
# octubre: los cinco KAN bajan entre 0.3 y 2.0 % en 3 epocas mientras los tres
# MLP bajan 96 %.
#
# Las estadisticas se toman del primer lote de entrenamiento y se congelan.
# Viven como buffers, asi que viajan en el state_dict.
#
# KAN_NORMALIZAR=0 lo desactiva, para correr el control sin normalizar.
# ======================================================================

_ALIAS_KAN = {"1": "canal", "si": "canal", "true": "canal",
              "0": "no", "": "no", "off": "no", "false": "no"}
MODO_KAN = _ALIAS_KAN.get(
    os.environ.get("KAN_NORMALIZAR", "no").strip().lower(),
    os.environ.get("KAN_NORMALIZAR", "no").strip().lower())
NORMALIZAR_KAN = MODO_KAN != "no"


def _calibrar_cuantiles(self, x):
    """Mueve la rejilla a los cuantiles de los datos, por canal.

    Es la ruta de pykan (`update_grid_from_samples`): en vez de llevar los datos
    a la rejilla, lleva la rejilla a los datos. A diferencia de estandarizar,
    produce una rejilla NO uniforme — nudos mas densos donde hay mas datos — y
    por eso no es equivalente a `canal`.
    """
    xd = x.detach().reshape(-1, self.n_in)
    if isinstance(self, KANGaussiana):
        K = self.K
        q = torch.tensor([(k + 0.5) / K for k in range(K)],
                         dtype=xd.dtype, device=xd.device)
        c = torch.quantile(xd, q, dim=0).T.contiguous()               # (n_in, K)
        d = (c[:, 1:] - c[:, :-1]).clamp_min(1e-6)
        anchos = torch.cat([d[:, :1], (d[:, :-1] + d[:, 1:]) / 2, d[:, -1:]], 1)
        self.centros = c
        self.register_buffer("anchos", anchos.clamp_min(1e-6))
    elif isinstance(self, KANBSpline):
        G, k = self.G, self.k
        q = torch.linspace(0, 1, G + 1, dtype=xd.dtype, device=xd.device)
        nudos = torch.quantile(xd, q, dim=0).T.contiguous()           # (n_in, G+1)
        paso = ((nudos[:, -1:] - nudos[:, :1]) / G).clamp_min(1e-6)
        izq = nudos[:, :1] - paso * torch.arange(k, 0, -1, dtype=xd.dtype,
                                                 device=xd.device)
        der = nudos[:, -1:] + paso * torch.arange(1, k + 1, dtype=xd.dtype,
                                                  device=xd.device)
        self.rejilla = torch.cat([izq, nudos, der], 1)                # (n_in, G+2k+1)
    # Chebyshev no tiene rejilla: su base es global. En ese caso no se toca nada.


def _normalizar_entrada(self, x):
    """Gestiona el rango de entrada segun MODO_KAN. Ver la cabecera del archivo."""
    if MODO_KAN == "no":
        return x
    if MODO_KAN == "layernorm":
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))
    if not hasattr(self, "calibrado"):
        self.register_buffer("calibrado",
                             torch.zeros(1, dtype=x.dtype, device=x.device))
    primera = (self.training and float(self.calibrado) == 0.0
               and x.dim() >= 2 and x.shape[0] > 1)
    if MODO_KAN == "cuantiles":
        if primera:
            with torch.no_grad():
                _calibrar_cuantiles(self, x)
                self.calibrado.fill_(1.0)
        return x
    # canal (el defecto): estandarizacion por canal, congelada tras el 1er lote.
    if not hasattr(self, "mu"):
        n = self.n_in
        self.register_buffer("mu", torch.zeros(n, dtype=x.dtype, device=x.device))
        self.register_buffer("sigma", torch.ones(n, dtype=x.dtype, device=x.device))
    if primera:
        with torch.no_grad():
            self.mu.copy_(x.detach().mean(0).reshape(-1))
            self.sigma.copy_(x.detach().std(0).reshape(-1).clamp_min(1e-6))
            self.calibrado.fill_(1.0)
    return (x - self.mu) / self.sigma


for _clase_kan in (KANGaussiana, KANBSpline, KANChebyshev):
    _clase_kan._normalizar_entrada = _normalizar_entrada


# ======================================================================
# Anadido el 2026-10-03: termino residual y rejilla adaptativa.
#
# Las dos piezas que Liu et al. (2024) declara necesarias y que esta
# implementacion no tenia. Apagadas por omision: con los valores por defecto
# el modulo se comporta exactamente como antes del parche.
# ======================================================================

RESIDUO_KAN = os.environ.get("KAN_RESIDUO", "0").strip() in ("1", "si", "true")
REJILLA_CADA = int(os.environ.get("KAN_REJILLA_CADA", "0") or 0)


def _residuo_tras_init(clase):
    """Envuelve __init__ para crear w_b al construir la capa.

    Tiene que ser en __init__ y no de forma perezosa: `cambiar_arquitectura`
    construye el optimizador justo despues, y un parametro creado mas tarde
    nunca entraria en el.
    """
    original = clase.__init__

    def nuevo(self, *a, **kw):
        original(self, *a, **kw)
        if RESIDUO_KAN:
            # Misma escala que la de un Linear: 1/sqrt(fan_in). Asi, al
            # inicializar, el camino residual equivale a la ultima capa del MLP.
            self.w_b = nn.Parameter(
                torch.randn(self.n_in, self.n_out) / math.sqrt(self.n_in))

    nuevo.__name__ = "__init__"
    nuevo.__doc__ = original.__doc__
    clase.__init__ = nuevo


def _residuo_en_forward(clase):
    """Suma w_b * silu(x) a la salida de la base, sobre la MISMA entrada.

    El forward original normaliza por dentro; `_normalizar_entrada` deja el
    resultado en `self._x_norm` para que el residuo use exactamente ese tensor
    y no se normalice dos veces.
    """
    original = clase.forward

    def nuevo(self, x):
        y = original(self, x)
        if RESIDUO_KAN and hasattr(self, "w_b"):
            xn = getattr(self, "_x_norm", None)
            if xn is None:
                xn = x
            y = y + torch.einsum("...i,io->...o",
                                 torch.nn.functional.silu(xn), self.w_b)
        return y

    nuevo.__name__ = "forward"
    clase.forward = nuevo


def _normalizar_entrada_v2(self, x):
    """Como la version anterior, mas el contador de la rejilla adaptativa.

    Guarda la entrada ya normalizada en `self._x_norm`, que es lo que consume
    el termino residual.
    """
    if REJILLA_CADA > 0 and self.training:
        n = int(getattr(self, "_pasos", 0))
        self._pasos = n + 1
        if n % REJILLA_CADA == 0 and x.dim() >= 2 and x.shape[0] > 1:
            with torch.no_grad():
                _calibrar_cuantiles(self, x)
    xn = _normalizar_entrada(self, x)
    self._x_norm = xn
    return xn


for _clase_kan in (KANGaussiana, KANBSpline, KANChebyshev):
    _clase_kan._normalizar_entrada = _normalizar_entrada_v2
    _residuo_tras_init(_clase_kan)
    _residuo_en_forward(_clase_kan)


def resumen_kan():
    """Una linea con la configuracion efectiva del bloque KAN.

    Conviene imprimirla en cada corrida: tres de los errores de la campana de
    septiembre y octubre fueron creer que una opcion estaba activa cuando no lo
    estaba.
    """
    rej = ("cada %d pasos" % REJILLA_CADA) if REJILLA_CADA else "no"
    res = "si" if RESIDUO_KAN else "no"
    return ("modo de rango=%s  residuo=%s  rejilla adaptativa=%s  "
            "lr=%g  lote=%s  fraccion=%s"
            % (MODO_KAN, res, rej, TASA_APRENDIZAJE, BATCH_SIZE, FRACCION))
