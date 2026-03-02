# Output di compilazione nella cartella build/
$out_dir = 'build';
$aux_dir = 'build';

# Usa pdflatex
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -output-directory=build %O %S';

# Bibtex con supporto alla directory di output
$bibtex = 'bibtex build/%B';
