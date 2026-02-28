# Modello 1 – NLP Rischio Suicidario

Modello di Natural Language Processing per l'individuazione di segnali di rischio suicidario nei testi prodotti dal paziente (messaggi, diari, risposte ai questionari).

## Struttura

```
nlp_suicide_risk/
├── notebooks/     # Notebook di esplorazione, preprocessing e training
├── src/           # Codice Python del modello (pipeline, training, inference)
├── data/          # Dataset (gitignored – vedi istruzioni sotto)
└── README.md
```

## Notebook (ordine sequenziale)

| # | Nome | Descrizione |
|---|---|---|
| 00 | `00_eda.ipynb` | Analisi esplorativa del dataset |
| 01 | `01_preprocessing.ipynb` | Pulizia testo, tokenizzazione |
| 02 | `02_training.ipynb` | Fine-tuning del modello |
| 03 | `03_evaluation.ipynb` | Metriche, confusion matrix, report |

## Dataset

I dati **non sono versionati su git**. Per ottenere il dataset:
> ⚠️ Inserire qui le istruzioni per scaricare/accedere al dataset.

## Dipendenze specifiche

Vedi `requirements.txt` nella root. Librerie principali: `transformers`, `torch`, `datasets`.

## Metriche target

- F1-score (macro)
- Precision e Recall per la classe "a rischio"
- AUC-ROC
