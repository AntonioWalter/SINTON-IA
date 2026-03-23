# Modello 3 – Depression Prediction

Modello di *Machine Learning Lineare e Non-Lineare* (Regressione Tabulare) sviluppato per l'applicativo **SINTON-IA**. Il suo scopo primario è pre-analizzare ed interpolare oggettivamente il tracciamento giornaliero dello stato d'animo (Mood Tracking e Diari Emotivi) compilato dal paziente sull'App, restituendo uno score numerico predittivo coerente con una somministrazione clinica del **Questionario standard PHQ-9**.
Tale sistema permette allo psicologo dell'ASL Campania di monitorare la regressione clinica dei pazienti anche durante le finestre biologiche in cui non vi è alcun log esplicito o test somministrato fisicamente.

## Struttura della Cartella

```
depression_prediction/
├── notebooks/     # Notebook di Data Prep, Synthetic Generation, Ingegnerizzazione Features ed Estrazione Predittiva
├── src/           # Script per la generazione e manipolazione dei file grezzi generati a valle
├── data/          # Dataset (gitignored – vedi istruzioni sotto)
└── README.md      # Questo file
```

## Guida Step-by-Step per Riprodurre i Risultati

Per garantire la completa trasparenza ingegneristica e validare formalmente la pipeline per l'esame accademico, è possibile riprodurre fedelmente la generazione e il modeling della predizione umorale tramite i passi seguenti:

### Step 1: Iniezione e Scaricamento Base Dati
La pipeline fa uso di log comportamentali che includono umore, ore di sonno e costanza di registrazione raccolti dalle App mobile. Anche in questo caso i file originali **non sono versionati su Git** a causa della Privacy Policy. Per reperire dal modulo ML remoto l'architettura dei dati (che verrà depositata momentaneamente in `data/raw/`), esegui da root:

```bash
cd models/depression_prediction
python data/download_dataset.py
```
*(Nota: Assicurati sempre di aver installato le librerie `datasets` e `huggingface_hub` presenti nel `requirements.txt` principale)*

### Step 2: Esecuzione Sequenziale dei Notebook
All'interno della cartella `notebooks/`, esegui i Jupyter Notebook uno di seguito all'altro, rispettando la catena cronologica. Molti notebook scaricano output intermedi in cartelle come `data/interim` e `data/processed` necessari allo step successivo.

| Ordine | Nome Notebook | Obiettivo e Descrizione |
| :---: | :--- | :--- |
| **01** | `01_Data_Analysis_and_Mapping.ipynb` | Analisi esplorativa dei dati originali, validazione delle regole semantiche (*Valenza e Arousal* per il tracking umorale) e mappatura iniziale dei range PHQ-9 per fasce cliniche. |
| **02** | `02_Advanced_Synthetic_Generation.ipynb` | **Cruciale**: Poiché lo storico clinico è intrinsecamente carente, questo notebook scatena protocolli di Augmentation Ibrida per generare un dataset stratificato molto corposo (usato poi per il Machine Learning tabulare) garantendone l'ancoraggio medico ai valori di base reali. |
| **03** | `03_Feature_Engineering_and_Prep.ipynb` | Costruzione della finestra temporale di osservazione (14 giorni cumulativi). Operazioni matematiche e statistiche (min, media, deviazione standard del sonno e umore) per appiattire la componente temporale su una tavola sinottica singola riga-per-paziente. Splitting e Standardizzazione. |
| **04** | `04_Modeling.ipynb` | Ripartizione del Training con Cross-Validation. Grid/Random Search e test parallelo di algoritmi di regressione lineari ed ensemble (Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, ecc.) per individuare il benchmark migliore. |
| **05** | `05_Evaluation.ipynb` | **Verifica ed Estrazione**: Valutazione indipendente delle metriche sul validatore Test Set tramite $R^2$, Calcolo Residual Error, Root Mean Squared Error (RMSE) ed esportazione formale dei file scatter-plot `docs/latex/figures/`. |

## Dipendenze Specifiche (Model Level)
La struttura ha un approccio Data-Science classico per dati strutturati/tabulari:
- `scikit-learn` (Pipeline ed algoritmi Regressivi)
- `pandas` e `numpy`
- `matplotlib` e `seaborn` per i residui e scatter prediction vs actual
- *(Opzionale per ottimizzazione pipeline XGBoost)* `xgboost` / `lightgbm` 

## Metriche Target Ottenute
Questa infrastruttura valuta l'accuratezza previsionale continua piuttosto che l'affido ad una classe fissa, estrapolando da zero il rating PHQ-9 (il cui delta va da 0 a 27). 
Si fa affidamento all'esame dei diagrammi dei **Residui (Residual Error Analysis)** per appurare che il bias modellistico sia distribuito regolarmente attorno allo 'zero' error e un **Mean Absolute Error (MAE)** o RMSE possibilmente inferiore ai 2 o 3 punti, garantendo così uno sbalzo previsionale ininfluente ai fini del range clinico complessivo (laddove uno step di patologia, es. Mite vs Grave, è di distacco 5 punti).
