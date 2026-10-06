#!/usr/bin/env bash
# =============================================================================
#  bajar_dataset.sh — trae las configuraciones de entrenamiento y las deja en
#  dataset/, que es donde el marco las busca.
#
#  Uso:   bash scripts/bajar_dataset.sh <usuario-hf>/<nombre-del-dataset>
#
#  El dataset es PRIVADO. Hace falta el token de lectura que acompana a la
#  hoja de ejecucion. No hace falta cuenta de Hugging Face:
#      export HF_TOKEN=hf_...
#  Con cuenta propia y acceso concedido, 'hf auth login' tambien sirve.
#
#  Son ~70 MB comprimidos y ~225 MB expandidos. Una sola transferencia.
# =============================================================================
set -euo pipefail

REPO="${1:-}"
if [ -z "$REPO" ]; then
  echo "uso: bash scripts/bajar_dataset.sh <usuario-hf>/<nombre-del-dataset>"
  exit 1
fi

cd "$(dirname "$0")/.."          # la raiz del repositorio
DESTINO="dataset"

if [ -d "$DESTINO" ] && [ "$(find "$DESTINO" -name '*.json' | wc -l)" -ge 23000 ]; then
  echo ">> $DESTINO ya esta completo, no se baja nada"
  exit 0
fi

command -v hf >/dev/null 2>&1 || {
  echo "ERROR: falta la orden 'hf'. Esta en environment.yml como huggingface_hub,"
  echo "       o se instala con: pip install -U 'huggingface_hub[cli]'"; exit 1; }
command -v zstd >/dev/null 2>&1 || { echo "ERROR: falta zstd"; exit 1; }

echo ">> bajando $REPO"
hf download "$REPO" dataset.tar.zst --repo-type dataset --local-dir .

echo ">> expandiendo"
tar --zstd -xf dataset.tar.zst
rm -f dataset.tar.zst

N=$(find "$DESTINO" -name '*.json' | wc -l)
G=$(find "$DESTINO" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo ">> $N configuraciones en $G sistemas"
if [ "$N" -lt 23000 ] || [ "$G" -ne 12 ]; then
  echo "   ERROR: se esperaban 23883 en 12 sistemas. No arrancar la campana asi."
  exit 1
fi
echo ">> listo"
