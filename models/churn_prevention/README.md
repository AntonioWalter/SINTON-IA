# Modello 2 – Churn Prevention (Algoritmo Genetico)

Sistema di prevenzione dell'abbandono precoce (Churn) della piattaforma **SINTON-IA**. A differenza dei classici modelli di classificazione, questo modulo implementa un **Algoritmo Genetico (GA)** avanzato che evolve e ottimizza proattivamente le migliori strategie di _nudging_ (invio di notifiche e promemoria personalizzati) per ogni cluster di pazienti, massimizzandone la retention.

L'algoritmo non prevede solo il rischio, ma prescrive attivamente un piano d'azione minimizzando la *Notification Fatigue* (rigetto della App per troppi avvisi).

## Struttura della Cartella

```
churn_prevention/
├── notebooks/     # Notebook di Validazione e Benchmarking (Tuning Bayesiano Optuna)
├── src/           # Core Logic in Python (Pipeline, Fitness Function, Evolution)
├── data/          # Dataset e Dati Sintetici (gitignored – vedi istruzioni sotto)
└── README.md      # Questo file
```

## Guida Step-by-Step per Riprodurre i Risultati

A differenza degli altri modelli (basati in toto su Jupyter Notebooks), il cuore dell'Algoritmo Genetico è stato ingegnerizzato direttamente in script Python puro `src/` per massimizzare la velocità computazionale. Il notebook funge esclusivamente da validatore statistico. Per replicare test e risultati, esegui questi step in ordine rigoroso:

### Step 1: Generazione dello Storico Pazienti (Dati Sintetici)
Questo modulo non attinge da dataset online. Sfrutta un motore interno (`data_pipeline.py`) per simulare in maniera realistica i log comportamentali, gli orari delle notifiche e le probabilità di interazione nell'App in base a 4 profili utente stratificabili.

Per simulare localmente un nuovo storico d'uso di SINTON-IA per 100 pazienti generici nell'arco degli ultimi 90 giorni, esegui:
```bash
cd models/churn_prevention
python src/data_pipeline.py --patients 100 --days 90
```
Questo comando genererà l'anagrafica ex novo e trasformerà i log prodotti in **Feature Aggregate** vettorializzate all'interno di `data/processed/features_ga.csv`, rendendole pronte per l'ottimizzazione genetica.

### Step 2: Esecuzione dell'Algoritmo Genetico (Core)
Per far evolvere attivamente le strategie tramite il calcolo delle fitness, la selezione a Torneo (Tournament Selection) e l'incrocio (Crossover):
```bash
python src/genetic_algorithm.py
```
Questo modulo genererà interattivamente le schedule calcolate per i tre blocchi cromosomici (Tipologia Notifica, Frequenza Invio, Schedule Orario).

### Step 3: Validazione Statistica e Tuning (Notebook)
Infine, spostati nella cartella `notebooks/` per convalidare in maniera accademica la convergenza matematica dell'Algoritmo.

| Ordine | Nome Notebook | Obiettivo e Descrizione |
| :---: | :--- | :--- |
| **05** | `05_GA_Validation.ipynb` | Esecuzione del framework **Optuna** per Tuning Bayesiano. Confronto iterativo tra il nostro GA e le classiche *Random Baseline* e *Heuristic Baseline*. Generazione formale dell'analisi di convergenza e Plot della distanza di Hamming da esportare nella cartella `docs/latex/figures/` per la Relazione Tecnica. |

## Dipendenze Specifiche (Model Level)
Oltre ai requisiti basilari della repository (`numpy`, `pandas`), questo modulo matematico implementa librerie sperimentali:
- `optuna` (per il parameter tuning Bayesiano e alberi Parzen Estimator)
- `tqdm` (per barre di caricamento CLI)
- `seaborn` e `matplotlib` per i plot generati dal Validation Notebook

## Metriche Target Ottenute
Il modello viene validato sulla base di metriche non classiche (nessuna Accuracy O F1-Score), ma focalizzate sulla convergenza di ricerca locale e globale:
- **Fitness Score**: Massimizzazione del punteggio cumulativo (Engagement Atteso vs Fatigue Penalty).
- **Evolution Dynamics**: Stabilizzazione della *Hamming Distance* genotipica dopo le prime $x$ generazioni (garanzia di uscita dai minimi locali protetta dalla Tournament Selection $k=3$).
