#!/usr/bin/env bash
# =============================================================================
#  controles70.sh — los controles que faltan tras la corrida de 70 epocas
#  del 2 de octubre de 2026.
#
#  Que se descubrio: la KAN con layernorm si entrena (98.16 % a 70 epocas,
#  contra 0.06 % a 3 epocas). Y con tasa 5e-2 termina 35 veces por debajo del
#  MLP. Pero esa comparacion cambia DOS variables a la vez, arquitectura y
#  tasa, asi que no vale. Esto cierra esa brecha.
#
#  Ocho procesos, uno por tarjeta, ~40 min de reloj:
#
#    0  MLP          lr 5e-2   <- EL CONTROL QUE FALTA
#    1  MLP          lr 2e-1
#    2  KAN gauss ln lr 2e-1
#    3  KAN gauss no lr 5e-2
#    4  KAN gauss canal lr 5e-2
#    5  KAN bspline ln  lr 5e-2
#    6  KAN cheby   ln  lr 5e-2
#    7  KAN gauss ln lr 5e-2, semilla 2222   <- repeticion del hallazgo
#
#  Uso:  bash controles70.sh
# =============================================================================
set -u

PY=${PY:-/venv/fitsnap/bin/python}
RAIZ=${RAIZ:-/workspace/kan_snap}
B="$RAIZ/MARCO_KAN/barrida.py"
EP=${EP:-70}

lanzar () {       # $1 tarjeta  $2 etiqueta  $3 modo  $4 arquitectura  $5 lr  $6 semilla
  local gpu=$1 tag=$2 modo=$3 arq=$4 lr=$5 sem=$6
  local dir="$RAIZ/c70_$tag"
  mkdir -p "$dir"
  ( cd "$dir" && CUDA_VISIBLE_DEVICES="$gpu" KAN_NORMALIZAR="$modo" \
      nohup "$PY" "$B" --que lr --raiz "$RAIZ" --arquitectura "$arq" \
        --valores "$lr" --epocas "$EP" --semilla "$sem" \
        --salida "$RAIZ/c70_$tag.json" > "$RAIZ/c70_$tag.log" 2>&1 & )
  echo "  tarjeta $gpu  ->  c70_$tag   ($arq, modo $modo, lr $lr, semilla $sem)"
}

echo "lanzando ocho controles de $EP epocas"
lanzar 0 mlp_5e2        no        linear     5e-2 1111
lanzar 1 mlp_2e1        no        linear     2e-1 1111
lanzar 2 gauss_ln_2e1   layernorm gaussiana  2e-1 1111
lanzar 3 gauss_no_5e2   no        gaussiana  5e-2 1111
lanzar 4 gauss_can_5e2  canal     gaussiana  5e-2 1111
lanzar 5 bspline_ln_5e2 layernorm bspline    5e-2 1111
lanzar 6 cheby_ln_5e2   layernorm chebyshev  5e-2 1111
lanzar 7 gauss_ln_s2222 layernorm gaussiana  5e-2 2222

sleep 5
echo
echo "procesos vivos: $(pgrep -fc 'barrida.py' || echo 0)"
echo
echo "Para la tabla, cuando 'pgrep -fc barrida.py' de 0:"
echo "  $PY $RAIZ/MARCO_KAN/tabla70.py"
