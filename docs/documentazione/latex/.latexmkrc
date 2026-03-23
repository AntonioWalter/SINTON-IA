# Nome del file finale (senza estensione)
$jobname = "SINTON-IA_report_policola-delsorbo-defusco";

# Output del PDF nella cartella superiore (docs/documentazione)
$out_dir = "..";
# File ausiliari nella cartella build/
$aux_dir = "build";

# Usa pdflatex
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';

# Usa BibTeX (non Biber) per la bibliografia
$bibtex_use = 1;
$bibtex = 'bibtex %O %B';
