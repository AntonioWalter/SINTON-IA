# Modello 3 – Churn Prevention (Algoritmo Genetico)

Sistema di prevenzione dell'abbandono della piattaforma SINTONIA, basato su un **Algoritmo Genetico** che evolve strategie ottimali di _nudging_ (notifiche personalizzate) per massimizzare la retention dei pazienti in lista d'attesa.

## Struttura

```
churn_prevention/
├── notebooks/     # Notebook di analisi e sperimentazione
├── src/           # Script Python
│   └── generate_synthetic_data.py   # Generatore dati sintetici
├── data/
│   └── synthetic/ # Dati generati (gitignored)
└── README.md
```

## Script Disponibili

| Script                       | Descrizione                                                     | Comando                                 |
| ---------------------------- | --------------------------------------------------------------- | --------------------------------------- |
| `generate_synthetic_data.py` | Genera dati comportamentali sintetici per 500 pazienti simulati | `python src/generate_synthetic_data.py` |

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

## Differenza rispetto agli altri modelli

A differenza dei modelli di Red Flag Detection (NLP) e Depression Prediction (regressione), il Churn Prevention **non utilizza dataset esterni etichettati**. Il GA opera esclusivamente su dati interni generati dalla piattaforma SINTONIA, evolendo strategie di intervento senza necessità di apprendimento supervisionato.

## Dipendenze

Vedi `requirements.txt` nella root. Librerie utilizzate: `numpy`, `pandas`.
