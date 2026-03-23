# Nome del file finale (senza estensione)
$jobname = "SINTON-IA_report_policola-delsorbo-defusco";

# File in uscita e ausiliari nella cartella build/
$out_dir = "build";
$aux_dir = "build";

# Configurazione PDF ed engine
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';

# Usa BibTeX (non Biber) per la bibliografia
$bibtex_use = 1;
$bibtex = 'bibtex %O %B';

# Hook di pulizia automatica per il root (rimuove leak di file ausiliari)
END {
    system("rm -f main.aux main.fdb_latexmk main.fls main.log main.out main.synctex.gz main.toc main.pdf SINTON-IA_report_policola-delsorbo-defusco.aux SINTON-IA_report_policola-delsorbo-defusco.fdb_latexmk SINTON-IA_report_policola-delsorbo-defusco.fls SINTON-IA_report_policola-delsorbo-defusco.log SINTON-IA_report_policola-delsorbo-defusco.out SINTON-IA_report_policola-delsorbo-defusco.toc SINTON-IA_report_policola-delsorbo-defusco.synctex.gz");
}
