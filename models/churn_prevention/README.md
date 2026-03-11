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
| `genetic_algorithm.py`       | Core logic dell'Algoritmo Genetico            | `python src/genetic_algorithm.py`       |
| `tune_ga.py`                 | Hyperparameter Tuning (Grid Search)           | `python src/tune_ga.py`                 |
| `validate_modeling.py`       | Validazione statistica su larga scala         | `python src/validate_modeling.py`       |
| `sensitivity_analysis.py`    | Analisi di sensibilità dei pesi fitness       | `python src/sensitivity_analysis.py`    |
| `benchmark_baselines.py`     | Benchmark comparativo GA vs Baselines         | `python src/benchmark_baselines.py`     |

### Opzioni dello script

```bash
python src/generate_synthetic_data.py --patients 500 --days 90
```

| Parametro    | Default | Descrizione                 |
| ------------ | ------- | --------------------------- |
| `--patients` | 500     | Numero di pazienti simulati |
| `--days`     | 90      | Giorni di storico simulato  |

### Output generato

Lo script produce 7 file CSV in `data/synthetic/`:

| File                    | Contenuto                                 |
| ----------------------- | ----------------------------------------- |
| `patients.csv`          | Anagrafica pazienti con profilo assegnato |
| `stato_animo.csv`       | Inserimenti di stato d'animo              |
| `pagina_diario.csv`     | Pagine del diario personale               |
| `questionario.csv`      | Compilazioni di questionari clinici       |
| `notifica.csv`          | Notifiche inviate con stato di lettura    |
| `domanda_forum.csv`     | Domande postate nel forum comunitario     |
| `acquisizione_badge.csv`| Badge della gamification acquisiti        |

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

### Tuning e Validazione Scientifica

Il modello è validato tramite:

1. **Hyperparameter Tuning**: Ottimizzazione dei parametri evolutivi tramite Grid Search (`tune_ga.py`).
2. **Validazione Statistica**: Stress-test su popolazione estesa per garantire stabilità e robustezza (`validate_modeling.py`).
3. **Analisi di Sensibilità**: Verifica dell'impatto dei pesi delle penalità sulle decisioni del GA (`sensitivity_analysis.py`).
4. **Benchmarking Comparativo**: Confronto delle performance del GA contro strategie casuali (_Random Baseline_) e regole fisse (_Heuristic Baseline_) per quantificare il valore aggiunto dall'ottimizzazione evolutiva (`benchmark_baselines.py`).

### Performance Metrics

I test di benchmarking evidenziano:

- **Gain vs Random**: Fitness ~350% superiore rispetto a scelte casuali.
- **Gain vs Heuristic**: Miglioramento del **25-30%** rispetto a regole fisse, grazie all'adattamento ai ritmi circadiani (_Night Owls_) e alla sensibilità alla fatica da notifica.

### Rationale Scelte Tecniche

- **Tournament Selection ($k=2$)**: Preferita per la sua robustezza contro i "super-individui" che dominerebbero precocemente la roulette wheel, garantendo una pressione selettiva più bilanciata.
- **Vincoli Adattivi (Dynamic Constraints)**: Il sistema non applica soglie rigide, ma modula le penalità in base alle feature del paziente:
  - **Frequenza**: La soglia di tolleranza varia in base all'ingaggio storico (gli utenti più attivi tollerano frequenze maggiori).
  - **Orario (Night Owls)**: Se il sistema rileva attività notturna spontanea (`night_activity_rate`), la penalità per l'invio tra le 23:00 e le 06:00 viene ridotta. **Nota Clinica**: Tale adattamento bilancia l'efficacia del nudging con il rischio di rinforzare involontariamente abitudini di sonno disfunzionali; è mantenuta una penalità minima cautelativa.

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
