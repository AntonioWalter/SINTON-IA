# Modello 3 – Churn Prevention (Algoritmo Genetico)

Sistema di prevenzione dell'abbandono della piattaforma SINTONIA, basato su un **Algoritmo Genetico** che evolve strategie ottimali di _nudging_ (notifiche personalizzate) per massimizzare la retention dei pazienti in lista d'attesa.

## Struttura

```
churn_prevention/
├── notebooks/
│   └── 04_GA_Experimentation.ipynb  # Sperimentazione interattiva
├── src/
│   ├── generate_synthetic_data.py   # Generatore dati sintetici
│   ├── aggregate_features.py       # Preprocessing e feature scaling
│   └── genetic_algorithm.py        # Core logic dell'Algoritmo Genetico
├── data/
│   ├── synthetic/                  # Dati generati (gitignored)
│   └── processed/                  # Dataset normalizzato per il GA
└── README.md
```

## Script Disponibili

| Script                       | Descrizione                                   | Comando                                 |
| ---------------------------- | --------------------------------------------- | --------------------------------------- |
| `generate_synthetic_data.py` | Genera dati comportamentali sintetici         | `python src/generate_synthetic_data.py` |
| `aggregate_features.py`      | Aggrega i log in feature vettoriali per il GA | `python src/aggregate_features.py`      |
| `genetic_algorithm.py`       | Esegue l'ottimizzazione delle strategie       | `python src/genetic_algorithm.py`       |

### Opzioni dello script

```bash
python src/generate_synthetic_data.py --patients 500 --days 90
```

| Parametro    | Default | Descrizione                 |
| ------------ | ------- | --------------------------- |
| `--patients` | 500     | Numero di pazienti simulati |
| `--days`     | 90      | Giorni di storico simulato  |

### Output generato

Lo script produce 5 file CSV in `data/synthetic/`:

| File                | Contenuto                                 |
| ------------------- | ----------------------------------------- |
| `patients.csv`      | Anagrafica pazienti con profilo assegnato |
| `stato_animo.csv`   | Inserimenti di stato d'animo              |
| `pagina_diario.csv` | Pagine del diario personale               |
| `questionario.csv`  | Compilazioni di questionari clinici       |
| `notifica.csv`      | Notifiche inviate con stato di lettura    |

### Profili utente simulati

| Profilo   | %   | Descrizione                                |
| --------- | --- | ------------------------------------------ |
| Engaged   | 30% | Utilizzo costante, alta compliance         |
| Moderato  | 30% | Utilizzo intermittente                     |
| A Rischio | 25% | Pattern pre-abbandono con calo progressivo |
| Ghost     | 15% | Attività cessata dopo breve periodo        |

A differenza dei modelli di Red Flag Detection (NLP) e Depression Prediction (regressione), il Churn Prevention **non utilizza dataset esterni etichettati**. Il GA opera esclusivamente su dati interni generati dalla piattaforma SINTONIA, evolendo strategie di intervento senza necessità di apprendimento supervisionato.

## Metodologia Algoritmo Genetico

L'algoritmo ottimizza una **strategia di nudging** (frequenza e timing delle notifiche) rappresentata da un cromosoma a 32 bit:

- **G1 (2 bit)**: Tipologia di notifica (Promemoria, Motivazionale, Informativa, Questionario).
- **G2 (5 bit)**: Frequenza settimanale (da 1 a 31).
- **G3 (24 bit)**: Schedule orario (bitmask per le 24 ore del giorno).
- **G4 (1 bit)**: Distribuzione (Uniforme vs Concentrata).

### Funzione di Fitness

La fitness massimizza la **retention stimata** minimizzando le penalità per:

1. **Notification Fatigue**: Eccessiva frequenza di invio.
2. **Timing**: Invio in orari notturni (23:00 - 06:00).

## Utilizzo del Notebook

Per visualizzare i risultati dell'evoluzione e generare i grafici per il documento LaTeX, utilizzare il notebook `notebooks/04_GA_Experimentation.ipynb`. Il notebook salverà automaticamente i grafici di convergenza in `docs/latex/figures/` con percorsi relativi portatili.

## Dataset

I dati per l'addestramento e la validazione del modello **non sono versionati su git** a causa delle loro dimensioni.

Il dataset è ospitato su [Hugging Face](https://huggingface.co/datasets/SINTON-IA/churn_prevention).

### Prerequisiti

Assicurati di avere le librerie necessarie e di aver effettuato il login:

```bash
pip install datasets huggingface_hub
huggingface-cli login
```

### Scaricare il dataset

Per scaricare comodamente il dataset (che verrà salvato in automatico come CSV in `data/processed/`), spostati nella cartella del modello e lancia lo script apposito:

```bash
cd models/churn_prevention
python src/download_dataset.py
```

## Dipendenze

Vedi `requirements.txt` nella root. Librerie utilizzate: `numpy`, `pandas`, `datasets`, `huggingface_hub`.
