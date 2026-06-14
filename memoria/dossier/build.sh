#!/usr/bin/env bash
# build.sh — compila main.tex → main.pdf
# Uso: ./build.sh          (compilación normal)
#      ./build.sh --clean  (limpia auxiliares y recompila desde cero)
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [[ "$1" == "--clean" ]]; then
  latexmk -pdf -CA
fi

latexmk -pdf -interaction=nonstopmode -g main.tex

echo ""
echo "OK: $(ls -lh main.pdf | awk '{print $5, $9}')"
