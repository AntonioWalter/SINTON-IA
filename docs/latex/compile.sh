#!/bin/bash
# Compila il documento LaTeX con tutti gli output in build/
cd "$(dirname "$0")"

mkdir -p build

echo "▸ Pass 1: pdflatex..."
pdflatex -interaction=nonstopmode -output-directory=build main.tex > /dev/null 2>&1

echo "▸ Bibtex..."
cp references.bib build/
bibtex build/main > /dev/null 2>&1

echo "▸ Pass 2: pdflatex..."
pdflatex -interaction=nonstopmode -output-directory=build main.tex > /dev/null 2>&1

echo "▸ Pass 3: pdflatex..."
pdflatex -interaction=nonstopmode -output-directory=build main.tex > /dev/null 2>&1

echo "✔ Compilazione completata: build/main.pdf"
