#!/usr/bin/env python3
"""
verificar_lib.py — comprobacion estatica antes de lanzar nada.

Recorre el arbol sintactico de un archivo .py y lista los nombres que se usan
pero no se definen en ningun sitio: ni como variable de modulo, ni como
argumento, ni como import, ni como incorporado de Python.

Existe por lo que paso el 28 de septiembre: la libreria del trabajador se
generaba volcando funciones sueltas y no arrastraba los globales de los que
dependian. `PESOS`, `pesos_de` y `_ORIGINALES` fueron el mismo fallo tres
veces, y cada uno se descubrio pagando entre 8 y 15 minutos de GPU. Esta
comprobacion tarda menos de un segundo.

Uso:
    python verificar_lib.py kan_lib.py
    python verificar_lib.py kan_lib.py trabajador.py     (varios archivos)

Devuelve codigo 0 si no falta nada, 1 si falta algo. Sirve para encadenarlo:

    python verificar_lib.py kan_lib.py && python lanzar_piloto.py

Limitaciones, para no fiarse de mas: el analisis es estatico y no ejecuta
nada, asi que no ve nombres creados en tiempo de ejecucion (globals(),
setattr, exec, import *). Un resultado limpio no garantiza que el modulo
funcione; un resultado sucio si garantiza que va a fallar.
"""

import ast
import builtins
import sys
from pathlib import Path

# Nombres que aparecen por el entorno y no por una definicion explicita.
EXTRA_DEFINIDOS = {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "self", "cls",
}


class Recolector(ast.NodeVisitor):
    """Reune nombres definidos y nombres usados, sin distinguir ambitos.

    No distinguir ambitos hace el analisis conservador por el lado bueno:
    puede dar por definido algo que solo existe dentro de una funcion, con lo
    que podria callar un fallo, pero nunca inventa un fallo que no existe.
    Para lo que hace falta aqui —detectar globales que no se copiaron— es
    suficiente y no produce ruido.
    """

    def __init__(self):
        self.definidos = set(dir(builtins)) | set(EXTRA_DEFINIDOS)
        self.usados = {}          # nombre -> primera linea donde aparece
        self.importa_estrella = []

    def visit_FunctionDef(self, nodo):
        self.definidos.add(nodo.name)
        self.generic_visit(nodo)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, nodo):
        self.definidos.add(nodo.name)
        self.generic_visit(nodo)

    def visit_arg(self, nodo):
        self.definidos.add(nodo.arg)
        self.generic_visit(nodo)

    def visit_Import(self, nodo):
        for a in nodo.names:
            self.definidos.add((a.asname or a.name).split(".")[0])
        self.generic_visit(nodo)

    def visit_ImportFrom(self, nodo):
        for a in nodo.names:
            if a.name == "*":
                self.importa_estrella.append(nodo.module or "?")
            else:
                self.definidos.add(a.asname or a.name)
        self.generic_visit(nodo)

    def visit_ExceptHandler(self, nodo):
        if nodo.name:
            self.definidos.add(nodo.name)
        self.generic_visit(nodo)

    def visit_Global(self, nodo):
        for n in nodo.names:
            self.definidos.add(n)
        self.generic_visit(nodo)

    visit_Nonlocal = visit_Global

    def visit_Name(self, nodo):
        if isinstance(nodo.ctx, (ast.Store, ast.Del)):
            self.definidos.add(nodo.id)
        else:
            self.usados.setdefault(nodo.id, nodo.lineno)
        self.generic_visit(nodo)


def revisar(rutas):
    """Analiza los archivos como un conjunto: lo definido en uno vale en otro.

    Esto es deliberado. El trabajador hace `from kan_lib import *`, asi que
    lo que kan_lib define esta disponible en trabajador.py, y analizarlos por
    separado daria falsos positivos en todos los nombres del modulo.
    """
    r = Recolector()
    origen = {}
    for ruta in rutas:
        texto = Path(ruta).read_text(encoding="utf-8")
        try:
            arbol = ast.parse(texto, filename=str(ruta))
        except SyntaxError as e:
            print(f"{ruta}: no parsea, linea {e.lineno}: {e.msg}")
            return 1
        antes = set(r.usados)
        r.visit(arbol)
        for n in set(r.usados) - antes:
            origen[n] = ruta
    faltan = sorted(set(r.usados) - r.definidos)
    return r, faltan, origen


def main(rutas):
    resultado = revisar(rutas)
    if resultado == 1:
        return 1
    r, faltan, origen = resultado

    print(f"archivos analizados: {', '.join(str(x) for x in rutas)}")
    print(f"nombres definidos:   {len(r.definidos) - len(dir(builtins)) - len(EXTRA_DEFINIDOS)}")
    print(f"nombres usados:      {len(r.usados)}")

    if r.importa_estrella:
        print(f"\nAviso: hay 'from X import *' de: {', '.join(r.importa_estrella)}")
        print("Lo que venga de ahi no se puede comprobar de forma estatica.")

    if not faltan:
        print("\nSIN NOMBRES SIN DEFINIR. La libreria esta completa.")
        return 0

    print(f"\nFALTAN {len(faltan)} NOMBRES:")
    for n in faltan:
        print(f"  {n:30s}  (primer uso: {origen.get(n, '?')}:{r.usados[n]})")
    print("\nCada uno de estos es un NameError esperando en la primera corrida.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main([Path(a) for a in sys.argv[1:]]))
