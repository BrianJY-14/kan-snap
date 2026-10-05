#!/usr/bin/env bash
# =============================================================================
#  referencia70.sh — la tanda con la tasa que usa el modelo de referencia.
#  2 de octubre de 2026.
#
#  Por que. `modelo/LiF-example.in`, la configuracion con la que el grupo
#  obtiene F MAE 1.108e-02 eV/A, dice:
#
#      learning_rate = 5e-6     num_epochs = 70     batch_size = 50
#
#  Toda la sesion de hoy corrio entre 1e-3 y 2e-1, de 200 a 40 000 veces esa
#  tasa, y ninguna corrida aprendio fuerzas: los MLP dieron errores de fuerza
#  5 a 10 veces PEORES que predecir cero y las KAN colapsaron a predecir cero.
#  El origen fue una barrida de tasas medida a 3 epocas, que penaliza por
#  construccion a una tasa pensada para 70.
#
#  Esto mapea la zona donde vive la referencia:
#
#    0  MLP            5e-6      3  MLP            5e-5      6  MLP          1e-4
#    1  KAN gauss ln   5e-6      4  KAN gauss ln   5e-5      7  KAN gauss ln 1e-4
#    2  KAN gauss no   5e-6      5  KAN gauss no   5e-5
#
#  El criterio de exito no es la perdida, es `mae.py`: F MAE cerca de 1.1e-02
#  eV/A y F/cero muy por debajo de 1.
#
#  Uso:  bash referencia70.sh
# =============================================================================
set -u

PY=${PY:-/venv/fitsnap/bin/python}
RAIZ=${RAIZ:-/workspace/kan_snap}
B="$RAIZ/MARCO_KAN/barrida.py"
EP=${EP:-70}

lanzar () {       # tarjeta etiqueta modo arquitectura lr semilla
  local gpu=$1 tag=$2 modo=$3 arq=$4 lr=$5 sem=$6
  local dir="$RAIZ/r70_$tag"
  mkdir -p "$dir"
  ( cd "$dir" && CUDA_VISIBLE_DEVICES="$gpu" KAN_NORMALIZAR="$modo" \
      nohup "$PY" "$B" --que lr --raiz "$RAIZ" --arquitectura "$arq" \
        --valores "$lr" --epocas "$EP" --semilla "$sem" \
        --salida "$RAIZ/r70_$tag.json" > "$RAIZ/r70_$tag.log" 2>&1 & )
  echo "  tarjeta $gpu  ->  r70_$tag   ($arq, modo $modo, lr $lr)"
}

echo "tanda con la tasa de la referencia, $EP epocas"
lanzar 0 mlp_5e6      no        linear    5e-6 1111
lanzar 1 gauss_ln_5e6 layernorm gaussiana 5e-6 1111
lanzar 2 gauss_no_5e6 no        gaussiana 5e-6 1111
lanzar 3 mlp_5e5      no        linear    5e-5 1111
lanzar 4 gauss_ln_5e5 layernorm gaussiana 5e-5 1111
lanzar 5 gauss_no_5e5 no        gaussiana 5e-5 1111
lanzar 6 mlp_1e4      no        linear    1e-4 1111
lanzar 7 gauss_ln_1e4 layernorm gaussiana 1e-4 1111

sleep 5
echo
echo "procesos vivos: $(pgrep -fc 'barrida.py' || echo 0)"
echo
echo "A los ~32 min, cuando 'pgrep -fc barrida.py' de 0, lo que importa es:"
echo "  $PY $RAIZ/MARCO_KAN/mae.py \"$RAIZ/r70_*\""
