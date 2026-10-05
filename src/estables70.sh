#!/usr/bin/env bash
# =============================================================================
#  estables70.sh — comparacion MLP vs KAN a tasa igualada, en la zona donde
#  nada diverge. 2 de octubre de 2026.
#
#  Por que existe. La tanda anterior puso siete configuraciones distintas
#  —MLP, gaussiana, B-spline, Chebyshev, con y sin normalizacion, dos tasas,
#  dos semillas— en el MISMO valor final, 5.4e-2 con dos cifras iguales. Eso
#  no es un resultado, es un suelo degenerado: con tasa 5e-2 y 2e-1 el modelo
#  revienta en el primer paso (perdidas iniciales de 3.8e+08 y 6.5e+14) y
#  aterriza ahi. Todo lo medido por encima del umbral de divergencia sobra.
#
#  Aqui se barre por debajo de ese umbral, con la arquitectura como unica
#  variable a cada tasa:
#
#    0  MLP            lr 1e-3
#    1  MLP            lr 1e-2
#    2  MLP            lr 2e-2
#    3  KAN gauss ln   lr 1e-3
#    4  KAN gauss ln   lr 1e-2
#    5  KAN gauss ln   lr 2e-2
#    6  MLP            lr 5e-3, semilla 2222
#    7  KAN gauss ln   lr 5e-3, semilla 2222
#
#  Las dos ultimas replican con otra semilla el unico par limpio que ya
#  tenemos (MLP 1.905 contra KAN 936 a 5e-3), porque el ruido de semilla esta
#  declarado como del orden del efecto en cuatro fuentes de la revision.
#
#  Uso:  bash estables70.sh
# =============================================================================
set -u

PY=${PY:-/venv/fitsnap/bin/python}
RAIZ=${RAIZ:-/workspace/kan_snap}
B="$RAIZ/MARCO_KAN/barrida.py"
EP=${EP:-70}

lanzar () {       # $1 tarjeta  $2 etiqueta  $3 modo  $4 arquitectura  $5 lr  $6 semilla
  local gpu=$1 tag=$2 modo=$3 arq=$4 lr=$5 sem=$6
  local dir="$RAIZ/s70_$tag"
  mkdir -p "$dir"
  ( cd "$dir" && CUDA_VISIBLE_DEVICES="$gpu" KAN_NORMALIZAR="$modo" \
      nohup "$PY" "$B" --que lr --raiz "$RAIZ" --arquitectura "$arq" \
        --valores "$lr" --epocas "$EP" --semilla "$sem" \
        --salida "$RAIZ/s70_$tag.json" > "$RAIZ/s70_$tag.log" 2>&1 & )
  echo "  tarjeta $gpu  ->  s70_$tag   ($arq, modo $modo, lr $lr, semilla $sem)"
}

echo "lanzando ocho corridas estables de $EP epocas"
lanzar 0 mlp_1e3       no        linear    1e-3 1111
lanzar 1 mlp_1e2       no        linear    1e-2 1111
lanzar 2 mlp_2e2       no        linear    2e-2 1111
lanzar 3 gauss_1e3     layernorm gaussiana 1e-3 1111
lanzar 4 gauss_1e2     layernorm gaussiana 1e-2 1111
lanzar 5 gauss_2e2     layernorm gaussiana 2e-2 1111
lanzar 6 mlp_5e3_s2222 no        linear    5e-3 2222
lanzar 7 gauss_5e3_s2222 layernorm gaussiana 5e-3 2222

sleep 5
echo
echo "procesos vivos: $(pgrep -fc 'barrida.py' || echo 0)"
echo
echo "Donde escribe FitSNAP sus metricas, por si hacen falta:"
ls -d "$RAIZ"/s70_*/barrida_lr_tmp 2>/dev/null | head -3
echo
echo "Tabla, cuando 'pgrep -fc barrida.py' de 0:"
echo "  $PY $RAIZ/MARCO_KAN/tabla70.py"
