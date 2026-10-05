#!/usr/bin/env python3
"""
Conversor de salidas de Quantum ESPRESSO a JSON de FitSNAP — parametrizado.

Reproduce exactamente la logica de los scripts del grupo (`qetofitsnap4n.py` y
`qetofitsnapNPT.py`), pero recibiendo por linea de comandos lo que alli estaba
codificado a mano: numero de atomos, dimensiones de celda, submuestreo y rutas.

Eso permite lanzar las 12 conversiones en paralelo sobre los nucleos de la
maquina, en vez de editar el script una vez por sistema.

Conversiones de unidades, identicas a las del grupo:
    fuerzas   Ry/bohr -> eV/A     x 25.711
    stress    kbar    -> bar      x 1000
    energia   Ry      -> eV       x 13.605693,  E = Etot - Ekin

Uso:
    python convertir_qe.py --out BCC_54.out --salida NEWJSON/output \\
                           --nat 54 --celda 10.53 10.53 10.53 --cada 5

    # celda variable (corridas NPT): se lee de CELL_PARAMETERS en cada frame
    python convertir_qe.py --out BCC_54_NPT.out --salida NEWJSON/output \\
                           --nat 54 --celda 10.53 10.53 10.53 --cada 5 --npt

VALIDACION RECOMENDADA: converti un sistema con este script y con el original
del grupo, y compara un JSON de cada uno. Deben ser identicos.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

RY_A_EV = 13.605693
FUERZA_RY_BOHR_A_EV_A = 25.711
KBAR_A_BAR = 1000.0


def convertir(archivo_out: Path, prefijo_salida: Path, nat: int,
              celda: tuple[float, float, float], cada: int = 1,
              npt: bool = False, limite: int = 19999,
              energia_ref: float = 0.0) -> int:
    """Recorre el .out y escribe un JSON por frame. Devuelve cuantos escribio.

    Soporta LOS DOS formatos de energia que produce Quantum ESPRESSO:

      NPT, una sola linea:
          Ekin =  0.15105691 Ry   T =  0.0 K   Etot =  -812.15094366
          -> E = Etot - Ekin

      NVT, tres lineas seguidas:
          kinetic energy (Ekin) =    0.13550970 Ry
          temperature           =  269.12314114 K
          Ekin + Etot (const)   = -812.16654497 Ry
          -> E = (Ekin+Etot) - Ekin

    Ambas dan la energia potencial. Validado contra los JSON del equipo
    (special/Li, special/F y special/B): reproduce los cuatro campos exactos.
    """
    prefijo_salida.parent.mkdir(parents=True, exist_ok=True)

    L_FUERZA = "Forces acting on atoms (cartesian axes, Ry/au):"
    L_STRESS = "total   stress  (Ry/bohr**3)"
    L_POS    = "ATOMIC_POSITIONS (angstrom)"
    L_CELDA  = "CELL_PARAMETERS (angstrom)"
    E_NPT    = "Ekin ="
    E_NVT    = "kinetic energy (Ekin)"

    n_escritos = cont = 0
    n_flags = 5 if npt else 4
    listo = [False] * n_flags
    fuerzas = [[0.0] * 3 for _ in range(nat)]
    posiciones = [[0.0] * 3 for _ in range(nat)]
    stress = [[0.0] * 3 for _ in range(3)]
    red = [[celda[0], 0.0, 0.0], [0.0, celda[1], 0.0], [0.0, 0.0, celda[2]]]
    tipos: list[str] = []
    energia = ekin = 0.0
    modo, resto = None, 0     # maquina de estados: que bloque se esta leyendo

    with archivo_out.open(errors="ignore") as fh:
        for linea in fh:
            if all(listo):
                listo = [False] * n_flags
                cont += 1
                if cont >= cada:
                    _escribir(prefijo_salida, n_escritos, nat,
                              energia + energia_ref * nat, stress,
                              posiciones, red, fuerzas, tipos)
                    n_escritos += 1
                    cont = 0
                    if n_escritos >= limite:
                        break
                tipos = []

            if modo is None:
                if L_FUERZA in linea:
                    modo, resto = "F", nat + 1        # 1 linea en blanco + nat filas
                elif L_STRESS in linea:
                    modo, resto = "S", 3
                elif L_POS in linea:
                    modo, resto = "P", nat
                elif npt and L_CELDA in linea:
                    modo, resto = "C", 3
                elif E_NPT in linea and "Etot" in linea:
                    p = linea.split()
                    energia = float(p[10]) * RY_A_EV - float(p[2]) * RY_A_EV
                    listo[3] = True
                elif E_NVT in linea:
                    ekin = float(linea.split()[4]) * RY_A_EV
                    modo, resto = "E", 2              # saltar temperatura, leer la 3a
                continue

            p = linea.split()
            if modo == "F":
                i = nat + 1 - resto
                if i >= 1 and len(p) >= 9:
                    fuerzas[i - 1] = [float(p[6]) * FUERZA_RY_BOHR_A_EV_A,
                                      float(p[7]) * FUERZA_RY_BOHR_A_EV_A,
                                      float(p[8]) * FUERZA_RY_BOHR_A_EV_A]
                resto -= 1
                if resto == 0:
                    modo, listo[0] = None, True
            elif modo == "S":
                i = 3 - resto
                if len(p) >= 6:
                    stress[i] = [float(p[3]) * KBAR_A_BAR,
                                 float(p[4]) * KBAR_A_BAR,
                                 float(p[5]) * KBAR_A_BAR]
                resto -= 1
                if resto == 0:
                    modo, listo[1] = None, True
            elif modo == "P":
                i = nat - resto
                if len(p) >= 4:
                    posiciones[i] = [float(p[1]), float(p[2]), float(p[3])]
                    tipos.append(p[0])
                resto -= 1
                if resto == 0:
                    modo, listo[2] = None, True
            elif modo == "C":
                i = 3 - resto
                if len(p) >= 3:
                    red[i] = [float(p[0]), float(p[1]), float(p[2])]
                resto -= 1
                if resto == 0:
                    modo, listo[4] = None, True
            elif modo == "E":
                resto -= 1
                if resto == 0:
                    energia = float(p[5]) * RY_A_EV - ekin
                    modo, listo[3] = None, True

    return n_escritos


def _escribir(prefijo, n, nat, energia, stress, posiciones, red, fuerzas, tipos):
    doc = {"Dataset": {"Data": [{
        "Stress": stress, "Positions": posiciones, "Energy": energia,
        "AtomTypes": tipos, "Lattice": red, "NumAtoms": nat, "Forces": fuerzas}],
        "PositionsStyle": "angstrom", "AtomTypeStyle": "chemicalsymbol",
        "Label": f"Example containing 1 configurations, each with {nat} atoms",
        "StressStyle": "bar", "LatticeStyle": "angstrom",
        "EnergyStyle": "electronvolt",
        "ForcesStyle": "electronvoltperangstrom"}}
    destino = Path(f"{prefijo}_{n}.json")
    with destino.open("w") as g:
        g.write("# A test JSON file for QM data\n")
        json.dump(doc, g)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="archivo .out de QE")
    ap.add_argument("--salida", required=True, type=Path,
                    help="prefijo de salida, p.ej. NEWJSON/output")
    ap.add_argument("--nat", required=True, type=int)
    ap.add_argument("--celda", required=True, nargs=3, type=float,
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--cada", type=int, default=1,
                    help="submuestreo: escribe 1 de cada N frames")
    ap.add_argument("--npt", action="store_true",
                    help="lee la celda de cada frame (corridas de celda variable)")
    ap.add_argument("--limite", type=int, default=19999)
    ap.add_argument("--energia-ref", type=float, default=0.0,
                    help="energia de atomo aislado por atomo (0 = como el grupo)")
    a = ap.parse_args()

    if not a.out.exists():
        sys.exit(f"no existe: {a.out}")
    n = convertir(a.out, a.salida, a.nat, tuple(a.celda), a.cada, a.npt,
                  a.limite, a.energia_ref)
    print(f"{a.out.parent.name:34s} -> {n:6d} JSON")


if __name__ == "__main__":
    main()
