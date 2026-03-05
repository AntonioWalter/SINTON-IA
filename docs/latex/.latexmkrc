# Output di compilazione nella cartella build/
$out_dir = 'build';
$aux_dir = 'build';

# Usa pdflatex
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';

# Usa BibTeX (non Biber) per la bibliografia
$bibtex_use = 1;
$bibtex = 'bibtex %O %B';
