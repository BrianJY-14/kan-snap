#!/usr/bin/env python3
"""
prueba_variantes.py — compara los cuatro modos de gestion del rango de entrada.

No entrena nada ni toca FitSNAP: construye las capas KAN con la distribucion de
entrada medida el 1 de octubre a la salida del Softplus previo al bloque KAN
(media 0.724, desviacion 0.216, todos positivos) y mide el gradiente que recibe
la componente peor servida en el primer paso. Tarda segundos.

Sirve para dos cosas: comprobar en el servidor que el parche esta bien aplicado
antes de gastar GPU, y dejar por escrito el numero que justifica el modo
elegido.

Uso:
    $PY prueba_variantes.py
"""

import importlib
import os
import sys

import torch

N, NIN = 1024, 24


def entrada():
    """Reproduce la distribucion medida: positiva, media 0.724, desv 0.216."""
    torch.manual_seed(0)
    x = torch.nn.functional.softplus(torch.randn(N, NIN) * 0.5 - 0.3)
    return ((x - x.mean()) / x.std() * 0.216 + 0.724).to(torch.float64)


def cargar(modo):
    os.environ["KAN_NORMALIZAR"] = modo
    for m in [m for m in list(sys.modules) if m.startswith("kan_lib")]:
        del sys.modules[m]
    K = importlib.import_module("kan_lib")
    if K.MODO_KAN != modo:
        raise SystemExit(f"kan_lib leyo MODO_KAN={K.MODO_KAN!r}, se pidio {modo!r}. "
                         "El parche de variantes no esta aplicado.")
    return K


def gradientes(X, Y):
    print(f"{'modo':11s} {'base':11s} {'grad min':>11s} {'grad medio':>11s} "
          f"{'params':>7s}")
    print("-" * 56)
    for modo in ("no", "canal", "layernorm", "cuantiles"):
        K = cargar(modo)
        for nombre, clase in K.BASES.items():
            if modo == "cuantiles" and nombre == "chebyshev":
                print(f"{modo:11s} {nombre:11s} {'—':>11s}  base global, sin rejilla")
                continue
            torch.manual_seed(1)
            capa = clase(NIN, 1).to(torch.float64)
            capa.train()
            y = capa(X)
            ((y - Y) ** 2).mean().backward()
            # peor nudo: maximo por nudo, y de esos el menor
            g = capa.w.grad.abs().amax(dim=tuple(range(capa.w.grad.dim() - 1)))
            n = sum(p.numel() for p in capa.parameters() if p.requires_grad)
            print(f"{modo:11s} {nombre:11s} {g.min():11.3e} {g.mean():11.3e} "
                  f"{n:7d}")


def integracion(X):
    """Red de dos capas: que corra, que no recalibre en eval, que no cambie params."""
    X2 = X * 1.7 + 0.4
    print()
    print(f"{'modo':11s} {'base':11s} {'params':>7s} {'eval':>5s} {'congelada':>10s}  buffers")
    print("-" * 70)
    for modo in ("no", "canal", "layernorm", "cuantiles"):
        K = cargar(modo)
        for base in K.BASES:
            if modo == "cuantiles" and base == "chebyshev":
                continue
            torch.manual_seed(1)
            red = K.crear_red([NIN, 16, 1], salida=base).to(torch.float64)
            red.train()
            red(X).pow(2).mean().backward()
            capa = [m for m in red.modules()
                    if isinstance(m, tuple(K.BASES.values()))][0]
            antes = {k: v.clone() for k, v in capa.named_buffers()}
            red.eval()
            with torch.no_grad():
                y2 = red(X2)
            igual = all(torch.equal(antes[k], v) for k, v in capa.named_buffers())
            print(f"{modo:11s} {base:11s} {K.n_params(red):7d} "
                  f"{str(y2.shape[0] == N):>5s} {str(igual):>10s}  "
                  f"{','.join(sorted(antes)) or '(ninguno)'}")


def main():
    X = entrada()
    torch.manual_seed(2)
    Y = torch.randn(N, 1, dtype=torch.float64)
    print(f"entrada: media {X.mean():.3f}  desv {X.std():.3f}  "
          f"rango [{X.min():.3f}, {X.max():.3f}]")
    print()
    gradientes(X, Y)
    integracion(X)
    print()
    print("Lectura: 'no' es el control. La B-spline sin corregir recibe gradiente")
    print("exactamente cero —soporte compacto—; la gaussiana, del orden de 1e-09.")
    print("El Chebyshev no mejora al corregirlo: su tanh ya gestionaba el rango.")
    print("Los params deben coincidir entre modos: ninguno cambia la capacidad.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
