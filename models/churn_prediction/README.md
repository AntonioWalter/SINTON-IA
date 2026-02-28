# Modello 2 – Churn Prediction

Modello di classificazione per la previsione dell'abbandono del percorso terapeutico da parte del paziente, con l'obiettivo di inviare notifiche mirate al momento giusto.

## Struttura

```
churn_prediction/
├── notebooks/     # Notebook di esplorazione, feature engineering e training
├── src/           # Codice Python del modello
├── data/          # Dataset (gitignored – vedi istruzioni sotto)
└── README.md
```

## Notebook (ordine sequenziale)

| # | Nome | Descrizione |
|---|---|---|
| 00 | `00_eda.ipynb` | Analisi esplorativa del comportamento utente |
| 01 | `01_feature_engineering.ipynb` | Costruzione delle feature predittive |
| 02 | `02_training.ipynb` | Training del modello di classificazione |
| 03 | `03_evaluation.ipynb` | Metriche e analisi degli errori |

## Dataset

I dati **non sono versionati su git**. Per ottenere il dataset:
> ⚠️ Inserire qui le istruzioni per scaricare/accedere al dataset.

## Dipendenze specifiche

Vedi `requirements.txt` nella root. Librerie principali: `scikit-learn`, `pandas`, `numpy`.

## Metriche target

- Precision e Recall (focus su recall per minimizzare i falsi negativi)
- F1-score
- AUC-ROC
