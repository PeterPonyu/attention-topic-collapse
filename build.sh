#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_independent
Rscript scripts/build_figures.R
export TEXINPUTS="build_independent:.:${TEXINPUTS:-}"
xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex
bibtex build_independent/paper
xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex
xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex
cp build_independent/paper.pdf paper.pdf
