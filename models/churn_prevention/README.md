# Modello 3 – Churn Prevention (Algoritmo Genetico)

Sistema di prevenzione dell'abbandono della piattaforma SINTONIA, basato su un **Algoritmo Genetico** che evolve strategie ottimali di _nudging_ (notifiche personalizzate) per massimizzare la retention dei pazienti in lista d'attesa.

## Struttura

```
churn_prevention/
├── notebooks/
│   └── 05_GA_Validation.ipynb  # Validazione statistica e tuning Bayesiano
├── src/
│   ├── data_pipeline.py            # Pipeline completa: Generazione + Aggregazione
│   ├── genetic_algorithm.py        # Core logic dell'Algoritmo Genetico
│   ├── download_dataset.py         # Utility download da HuggingFace
│   └── upload_dataset.py           # Utility upload su HuggingFace
├── data/
│   ├── raw/                  # Dati generati (gitignored)
│   └── processed/                  # Dataset normalizzato per il GA
└── README.md
```

## Script Disponibili

| Script                 | Descrizione                                      | Comando                           |
| ---------------------- | ------------------------------------------------ | --------------------------------- |
| `data_pipeline.py`     | Genera e aggrega dati sintetici in un unico step | `python src/data_pipeline.py`     |
| `genetic_algorithm.py` | Esegue l'evoluzione delle strategie              | `python src/genetic_algorithm.py` |
| `download_dataset.py`  | Scarica il dataset da HuggingFace Hub            | `python src/download_dataset.py`  |

> [!NOTE]
> La validazione (Benchmarking, Analisi di Sensibilità) è consolidata nel notebook `05_GA_Validation.ipynb` per garantire rigore statistico e riproducibilità.

### Opzioni della Pipeline

```bash
python src/data_pipeline.py --patients 100 --days 90
```

| Parametro    | Default | Descrizione                 |
| ------------ | ------- | --------------------------- |
| `--patients` | 100     | Numero di pazienti simulati |
| `--days`     | 90      | Giorni di storico simulato  |

### Output generato

La pipeline produce file CSV in `data/synthetic/` e il file aggregato in `data/processed/features_ga.csv`.

| File              | Contenuto                                 |
| ----------------- | ----------------------------------------- |
| `patients.csv`    | Anagrafica pazienti con profilo assegnato |
| `notifica.csv`    | Log notifiche e read rate                 |
| `features_ga.csv` | Dataset aggregato pronto per il GA        |

### Profili utente simulati (Pazienti Sintetici)

| Profilo   | %   | Descrizione                                |
| --------- | --- | ------------------------------------------ |
| Engaged   | 30% | Utilizzo costante, alta compliance         |
| Moderato  | 30% | Utilizzo intermittente                     |
| A Rischio | 25% | Pattern pre-abbandono con calo progressivo |
| Ghost     | 15% | Attività cessata dopo breve periodo        |

---

## Workflow del Modello: Step-by-Step

### 1. Generazione e Simulazione (`data_pipeline.py`)

Il sistema simula l'interazione dei pazienti basandosi su cluster comportamentali.

- **Dinamica**: `DataPipeline.run_generation()` crea lo storico di 90 giorni.
- **Aggregazione**: `DataPipeline.run_aggregation()` trasforma i log in feature vettoriali normalizzate.

### 2. Ottimizzazione Genetica (`genetic_algorithm.py`)

L'algoritmo evolve la strategia di nudging personalizzata (Cromosoma a 31 bit).

- **Esplorazione**: Tournament Selection ($k=3$) garantisce robustezza contro i minimi locali.
- **Configurazione Automatica**: La classe `GAParams` carica automaticamente i parametri ottimali da `ga_tuned_config.json` se presente, rendendo le run di produzione consistenti con la ricerca.

### 3. Analisi e Validazione Ottimizzata

Utilizzando `05_GA_Validation.ipynb`, si eseguono:

- **Tuning Bayesiano (Optuna)**: Ricerca congiunta dei parametri e degli operatori genetici.
- **Evolution Dynamics**: Monitoraggio fitness e diversità della popolazione (Hamming distance).
- **Benchmark Comparativo**: Confronto statistico contro Random e Heuristic baselines.
- **Scenario Analysis**: Stress-test dei pesi della fitness function.

## Justification delle Scelte Progettuali

| Scelta Tecnica           | Giustificazione Scientifica / Criterio                                                           |
| :----------------------- | :----------------------------------------------------------------------------------------------- |
| **Tournament Selection** | Garantisce stabilità evolutiva controllando la pressione selettiva tramite $k=3$.                |
| **Cromosoma a 31 bit**   | Copertura totale delle 24h (bitmask) e 4 tipologie di notifica per una personalizzazione spinta. |
| **Penalità Fatigue**     | Implementazione adattiva della **Notification Fatigue** per evitare il rigetto della terapia.    |
| **Vincoli Adattivi**     | Integrazione della **Cronobiologia** individuale (Night Owls) nei parametri orari.               |

## Metodologia Algoritmo Genetico

L'algoritmo evolve una strategia articolata in tre geni:

1.  **G1 - Tipologia (2 bit)**: Promemoria, Motivazionale, Informativa, Questionario.
2.  **G2 - Frequenza (5 bit)**: Volume settimanale (massimo 31 notifiche).
3.  **G3 - Schedule Orario (24 bit)**: Bitmask delle 24 ore di eleggibilità all'invio.

---

## Dataset e Hub Cloud

Il dataset è ospitato su [Hugging Face](https://huggingface.co/datasets/SINTON-IA/churn_prevention).

### Scaricare il dataset

```bash
python src/download_dataset.py
```

## Dipendenze

Vedi `requirements.txt` nella root. Librerie chiave: `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.
