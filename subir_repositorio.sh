#!/usr/bin/env bash
# =============================================================================
#  subir_repositorio.sh — deja kan-snap publicado en GitHub.
#
#  Uso:   bash subir_repositorio.sh <usuario-de-github> [nombre-del-repo]
#  Ej:    bash subir_repositorio.sh BrianJY-14
#
#  ANTES de correrlo hacen falta dos cosas que este guion no puede hacer solo:
#
#    1. El repositorio VACIO creado en github.com/new. Sin README, sin
#       .gitignore, sin licencia: si trae algo, el primer push es rechazado.
#    2. Un token personal (github.com/settings/tokens, alcance `repo`). Git lo
#       va a pedir como contrasena; el password de la cuenta ya no sirve.
#
#  Lo demas lo hace este guion, y se puede volver a correr sin romper nada.
# =============================================================================
set -euo pipefail

USUARIO="${1:-}"
REPO="${2:-kan-snap}"
if [ -z "$USUARIO" ]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

cd "$(dirname "$0")"
command -v git >/dev/null 2>&1 || { echo "ERROR: no hay git en el PATH"; exit 1; }
[ -d src ] && [ -f run_kan_snap.md ] || {
  echo "ERROR: este guion va DENTRO de la carpeta del repositorio"; exit 1; }

# --- 1. Que el dataset NO se cuele -------------------------------------------
# Vive en Hugging Face, privado. Si alguien lo dejo aqui, .gitignore deberia
# frenarlo; esto avisa antes de que un descuido lo publique.
if [ -d dataset ] || [ -d dataset_kan ]; then
  N=$(find dataset dataset_kan -name '*.json' 2>/dev/null | wc -l)
  echo "   AVISO: hay una carpeta de datos aqui ($N archivos)."
  if git check-ignore -q dataset dataset_kan 2>/dev/null; then
    echo "   .gitignore la excluye, no se va a subir."
  else
    echo "   .gitignore NO la excluye: se publicaria. Se detiene."
    exit 1
  fi
fi

# --- 2. El marcador <user> de la documentacion -------------------------------
for f in README.md run_kan_snap.md; do
  [ -f "$f" ] || continue
  if grep -q '<user>' "$f"; then
    sed -i "s|<user>|$USUARIO|g" "$f"
    echo ">> $f: <user> -> $USUARIO"
  fi
  # si el repositorio no se llama kan-snap, la URL de ejemplo tambien cambia
  if [ "$REPO" != "kan-snap" ] && grep -q "/kan-snap.git" "$f"; then
    sed -i "s|/kan-snap.git|/$REPO.git|g" "$f"
    echo ">> $f: kan-snap -> $REPO"
  fi
done

# --- 3. Repositorio local ----------------------------------------------------
if [ -d .git ]; then
  echo ">> ya existe un repositorio local, se reutiliza"
else
  git init -q -b main
  echo ">> git init (rama main)"
fi

# Finales de linea LF. Sin esto, un clon en Windows reescribe las 23 883
# configuraciones y el repositorio aparece modificado entero sin tocarlo.
git config core.autocrlf false
git config user.name  >/dev/null 2>&1 || git config user.name  "Brian Jara"
git config user.email >/dev/null 2>&1 || git config user.email "brian.jara.y@uni.pe"

# --- 4. Commit ---------------------------------------------------------------
echo ">> indexando (son ~24 000 archivos; sobre /mnt/d de WSL tarda minutos)..."
git add -A
if git diff --cached --quiet; then
  echo ">> no hay cambios que registrar"
else
  git commit -qm "kan-snap: marco experimental para el bloque de salida KAN"
  echo ">> commit hecho: $(git log --oneline -1)"
fi

# --- 5. Remoto y envio -------------------------------------------------------
URL="https://github.com/$USUARIO/$REPO.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$URL"
else
  git remote add origin "$URL"
fi
echo ">> origin = $URL"

echo ">> tamano empaquetado:"
git count-objects -vH | grep size-pack | sed 's/^/   /'

echo ">> enviando. Usuario: $USUARIO   Contrasena: el TOKEN, no la del correo."
git push -u origin main

echo
echo "=========================================================================="
echo "  Publicado en https://github.com/$USUARIO/$REPO"
echo
echo "  Falta el dataset, que va aparte y privado:"
echo "     bash ../subir_dataset_hf.sh <usuario-hf>"
echo "=========================================================================="
