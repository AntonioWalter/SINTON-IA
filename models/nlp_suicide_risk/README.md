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

| #   | Nome                     | Descrizione                        |
| --- | ------------------------ | ---------------------------------- |
| 00  | `00_eda.ipynb`           | Analisi esplorativa del dataset    |
| 01  | `01_preprocessing.ipynb` | Pulizia testo, tokenizzazione      |
| 02  | `02_training.ipynb`      | Fine-tuning del modello            |
| 03  | `03_evaluation.ipynb`    | Metriche, confusion matrix, report |

## Dataset

I dati **non sono versionati su git** a causa della loro natura sensibile.

Il dataset è ospitato privatamente su [Hugging Face](https://huggingface.co/datasets/SINTON-IA/nlp_suicide_risk).

### Prerequisiti

Assicurati di avere le librerie necessarie e di aver effettuato il login:

```bash
pip install datasets huggingface_hub
huggingface-cli login
```

_(Inserisci il token di accesso Hugging Face autorizzato quando richiesto)._

### Scaricare il dataset

Per scaricare comodamente il dataset (che verrà salvato in automatico come CSV in `data/processed/`), spostati nella cartella del modello e lancia lo script apposito:

```bash
cd models/nlp_suicide_risk
python src/download_dataset.py
```

## Dipendenze specifiche

Vedi `requirements.txt` nella root. Librerie principali: `transformers`, `torch`, `datasets`.

## Metriche target

- F1-score (macro)
- Precision e Recall per la classe "a rischio"
- AUC-ROC
