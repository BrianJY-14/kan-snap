#!/usr/bin/env python3
"""
verificar_orden.py — ¿este notebook corre de arriba abajo?

Responde en un segundo tres preguntas que normalmente se contestan bajando y
contando celdas a mano:

1. ¿Se ejecuto en orden?  Compara el orden fisico de las celdas con el orden
   en que se ejecutaron (el contador [n] de Jupyter). Si la celda 40 se ejecuto
   antes que la 39, el notebook guardado NO refleja una corrida limpia.

2. ¿Algo se usa antes de definirse?  Para cada nombre, compara la celda donde
   aparece por primera vez con la celda donde se define. Si se usa antes, al
   reiniciar el kernel y correr de arriba abajo ese punto falla.

3. ¿Que celdas sobran?  Lista las que nunca se ejecutaron y las que definen
   algo que no usa nadie.

Uso:
    python verificar_orden.py experimentos_KAN_fitsnap.ipynb

Codigo 0 si no hay problemas de orden, 1 si los hay.
"""

import ast
import builtins
import json
import sys
from pathlib import Path

INCORPORADOS = set(dir(builtins)) | {
    "__file__", "__name__", "self", "cls", "get_ipython",
    "In", "Out", "exit", "quit", "display",
}


def nombres_de_celda(fuente):
    """(definidos, usados) de una celda, sin distinguir ambitos."""
    # Las lineas magicas y de shell no son Python valido.
    limpio = "\n".join(
        "" if l.lstrip().startswith(("!", "%", "?")) else l
        for l in fuente.split("\n")
    )
    try:
        arbol = ast.parse(limpio)
    except SyntaxError:
        return set(), set()
    definidos, usados = set(), set()
    for n in ast.walk(arbol):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definidos.add(n.name)
        elif isinstance(n, ast.arg):
            definidos.add(n.arg)
        elif isinstance(n, ast.Import):
            for a in n.names:
                definidos.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    definidos.add(a.asname or a.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            definidos.add(n.name)
        elif isinstance(n, ast.Name):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                definidos.add(n.id)
            else:
                usados.add(n.id)
    return definidos, usados


def revisar(ruta):
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    celdas = nb["cells"]

    codigo = []            # (indice fisico, contador de ejecucion, fuente)
    for i, c in enumerate(celdas):
        if c["cell_type"] != "code":
            continue
        if not "".join(c["source"]).strip():
            continue
        codigo.append((i, c.get("execution_count"), "".join(c["source"])))

    print(f"notebook: {ruta}")
    print(f"celdas totales: {len(celdas)}   de codigo con contenido: {len(codigo)}")
    print()

    problemas = 0

    # --- 1. orden de ejecucion -------------------------------------------
    ejecutadas = [(i, n) for i, n, _ in codigo if n is not None]
    sin_ejecutar = [i for i, n, _ in codigo if n is None]

    print("1. ORDEN DE EJECUCION")
    if not ejecutadas:
        print("   ninguna celda tiene contador; el notebook se guardo sin ejecutar")
    else:
        desorden = []
        for k in range(1, len(ejecutadas)):
            (i_ant, n_ant), (i, n) = ejecutadas[k - 1], ejecutadas[k]
            if n < n_ant:
                desorden.append((i_ant, n_ant, i, n))
        if not desorden:
            print("   correcto: el orden fisico coincide con el de ejecucion")
        else:
            problemas += len(desorden)
            print(f"   {len(desorden)} saltos hacia atras:")
            for i_ant, n_ant, i, n in desorden:
                print(f"     celda {i:3d} se ejecuto en [{n}], "
                      f"despues de la celda {i_ant} que se ejecuto en [{n_ant}]")
            print("   Un salto hacia atras significa que el resultado guardado en")
            print("   esa celda se produjo con un estado distinto al que tendria")
            print("   corriendo de arriba abajo.")
    if sin_ejecutar:
        print(f"   celdas de codigo sin ejecutar: {sin_ejecutar}")
    print()

    # --- 2. uso antes de definicion --------------------------------------
    print("2. NOMBRES USADOS ANTES DE DEFINIRSE")
    primera_def = {}
    primer_uso = {}
    for i, _, fuente in codigo:
        d, u = nombres_de_celda(fuente)
        for n in d:
            primera_def.setdefault(n, i)
        for n in u:
            primer_uso.setdefault(n, i)

    adelantados = []
    for n, celda_uso in primer_uso.items():
        if n in INCORPORADOS:
            continue
        celda_def = primera_def.get(n)
        if celda_def is None:
            continue          # puede venir de un import *, o no existir
        if celda_def > celda_uso:
            adelantados.append((n, celda_uso, celda_def))

    if not adelantados:
        print("   correcto: nada se usa antes de estar definido")
    else:
        problemas += len(adelantados)
        print(f"   {len(adelantados)} nombres:")
        for n, uso, defn in sorted(adelantados, key=lambda x: x[1]):
            print(f"     {n:28s} se usa en la celda {uso:3d} "
                  f"y se define en la {defn:3d}")
        print("   Con el kernel reiniciado, cada uno de estos es un NameError.")
    print()

    # --- 3. nombres nunca usados ------------------------------------------
    print("3. DEFINICIONES QUE NADIE USA")
    nunca = sorted(n for n in primera_def
                   if n not in primer_uso and not n.startswith("_"))
    if not nunca:
        print("   ninguna")
    else:
        print(f"   {len(nunca)}: {', '.join(nunca[:20])}"
              f"{' ...' if len(nunca) > 20 else ''}")
        print("   No es un error. Suele ser codigo que quedo de versiones previas.")
    print()

    print("=" * 66)
    if problemas == 0:
        print("El notebook corre de arriba abajo.")
        return 0
    print(f"{problemas} problemas de orden. El notebook NO corre de arriba abajo")
    print("sin intervencion manual.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(revisar(sys.argv[1]))
