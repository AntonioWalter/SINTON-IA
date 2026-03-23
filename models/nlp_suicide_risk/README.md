# Modello 1 – NLP Rischio Suicidario (Red Flag)

Modello di *Natural Language Processing* sviluppato per l'applicativo **SINTON-IA**. Il suo scopo è identificare in tempo reale intenti autolesionistici o minacce di suicidio all'interno delle note di diario e dei messaggi testuali inseriti dai pazienti, per avviare una macro-segnalazione pro-attiva (Red Flag) al personale specializzato dell'ASL Campania.

Il modello di produzione eletto è un **TF-IDF Vectorizer + Logistic Regression**, affiatato da una rigorosa calibrazione della soglia per massimizzare la precisione clinica cautelativa (Recall).

## Struttura della Cartella

```
nlp_suicide_risk/
├── notebooks/     # Notebook di EDA, pulizia, addestramento, e calibrazione
├── src/           # Script per il download e la gestione dei dati
├── data/          # Dataset (gitignored – vedi istruzioni sotto)
└── README.md      # Questo file
```

## Guida Step-by-Step per Riprodurre i Risultati

Per garantire la completa trasparenza clinica e accademica del progetto, è possibile riprodurre da zero l'intero ciclo vitale del modello eseguendo in ordine sequenziale i seguenti step.

### Step 1: Scaricare il Dataset Originario
I dati testuali **non sono versionati su Git** a causa del loro peso e per rispettare le policy standard.
Il dataset alla base (ricavato da Reddit) è ospitato nel cloud accademico/Hugging Face o salvato in remoto. 
Per scaricarlo localmente e in automatico nella directory corretta (`data/raw/`), esegui:

```bash
cd models/nlp_suicide_risk
python src/download_dataset.py
```
*(Nota: Assicurati di aver installato le librerie `datasets` e `huggingface_hub` come da `requirements.txt` globale)*

### Step 2: Esecuzione Sequenziale dei Notebook
Una volta scaricati i dati grezzi, spostati nella cartella `notebooks/` ed esegui i Jupyter Notebook nel seguente ordine rigoroso. Ognuno funge da perno per lo step successivo salvando l'output temporaneo nell'apposita directory `data/processed/`.

| Ordine | Nome Notebook | Obiettivo e Descrizione |
| :---: | :--- | :--- |
| **01** | `01_RedFlag_DataExploration.ipynb` | Analisi esplorativa dei dati (EDA), bilanciamento e conteggio delle parole e caratteri predominanti. |
| **02** | `02_RedFlag_DataPreparation.ipynb` | Pulizia profonda del testo testuale (rimozione link, punteggiatura) e normalizzazione del dataset in formato addestrabile. |
| **03** | `03_RedFlag_Modelling_TFIDF.ipynb` | **Splitting** (Train: 80%, Validation: 10%, Test: 10%). Addestramento del modello TF-IDF e primo salvataggio. |
| **04** | `04_RedFlag_Baseline_Limitations.ipynb` | Test logico dei limiti operativi per i falsi negativi e valutazione dei constraint (Edge Case Analysis). |
| **05** | `05_RedFlag_Synthetic_Robustness.ipynb` | Analisi di robustezza (Stress Test) mediante l'inquinamento del Validation Set con dati sintetici GPT-4. |
| **06** | `06_RedFlag_Threshold_Tuning.ipynb` | **Calibrazione Finale**: Risoluzione analitica del Trade-Off, massimizzazione dello $F_2$-Score per identificare la nuova soglia (23.84%) ed estratto formale di tutte le metriche sul *Test Set* Intoccato. |

Al termine dell'esecuzione del notebook `06`, i grafici (Curve Precision-Recall e Matrici di Confusione) verranno esportati materialmente nella cartella globale della Relazione Tecnica per il LaTeX (`docs/documentazione/latex/figures/`).

## Dipendenze Specifiche (Model Level)
Oltre ai requisiti basilari, il modulo necessita di:
- `scikit-learn` (vettorializzazione testuale lineare)
- `seaborn` e `matplotlib` (grafici clinici)
- `pandas` e `numpy`

## Metriche Target Ottenute
Questo modulo privilegia l'identificazione precoce dei "falsi allarmi" salvaguardando in toto le omissioni critiche. Le metriche garantite in produzione basate sul tuning del **Test Set Indipendente** sono:
- **Recall (Suicide Risk)**: $> 96.8\%$
- **Accuracy Globale**: $\approx 91.7\%$
- **Soglia di Rischio Ottimizzata ($\tau$)**: $23.84\%$
