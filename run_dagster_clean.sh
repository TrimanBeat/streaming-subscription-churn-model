#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Proyecto: ${ROOT_DIR}"
cd "${ROOT_DIR}"

if ! command -v dagster >/dev/null 2>&1; then
  echo "Error: no se encontró 'dagster' en el PATH."
  echo "Activa tu entorno virtual e instala dependencias antes de ejecutar este script."
  exit 1
fi

echo ">> Conservando estado local de Dagster (.dagster/)..."
mkdir -p "${ROOT_DIR}/.dagster"

echo ">> Limpiando __pycache__..."
find "${ROOT_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +

if command -v streamlit >/dev/null 2>&1; then
  echo ">> Limpiando cache de Streamlit..."
  streamlit cache clear >/dev/null 2>&1 || true
fi

export PYTHONPATH="${ROOT_DIR}/src"
export DAGSTER_HOME="${ROOT_DIR}/.dagster"

echo ">> Iniciando Dagster..."
exec dagster dev -m churn_project.definitions "$@"
