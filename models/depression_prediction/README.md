# Modello 3 – Depression Prediction

Modello per il rilevamento dello stato depressivo del paziente a partire dal tracciamento giornaliero del suo stato d'animo (mood tracking), prodotto dall'app SINTONIA.

## Struttura

```
depression_detection/
├── notebooks/     # Notebook di esplorazione, preprocessing e training
├── src/           # Codice Python del modello
├── data/          # Dataset (gitignored – vedi istruzioni sotto)
└── README.md
```

## Notebook (ordine sequenziale)

| # | Nome | Descrizione |
|---|---|---|
| 00 | `00_eda.ipynb` | Analisi esplorativa delle serie temporali di umore |
| 01 | `01_preprocessing.ipynb` | Normalizzazione, gestione valori mancanti |
| 02 | `02_training.ipynb` | Training del modello |
| 03 | `03_evaluation.ipynb` | Metriche e analisi degli errori |

## Dataset

I dati **non sono versionati su git**. Per ottenere il dataset:
> ⚠️ Inserire qui le istruzioni per scaricare/accedere al dataset.

## Dipendenze specifiche

Vedi `requirements.txt` nella root. Librerie principali: `scikit-learn`, `pandas`, `numpy`.

## Metriche target

- Accuracy, F1-score (macro)
- Confusion matrix multi-classe
- MAE/RMSE se formulato come regressione
