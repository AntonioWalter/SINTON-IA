# SINTON-IA 🧠

Componente di Intelligenza Artificiale del progetto **SINTONIA** — sistema di monitoraggio della salute mentale per pazienti in lista d'attesa per lo psicologo con **ASL Campania**.

---

## Descrizione

SINTON-IA integra tre modelli di intelligenza artificiale all'interno del software preesistente **SINTONIA**, una piattaforma gestionale e di monitoraggio per la salute mentale. L'obiettivo primario è supportare attivamente psicologi, psichiatri e operatori dell'**ASL Campania** nella gestione dei pazienti in lista d'attesa, fornendo strumenti predittivi capaci di identificare minacce precoci, prevenire l'abbandono dell'applicativo prima della presa in carico clinica e stimare costantemente l'integrità umorale del paziente a distanza.

📥 **[Consulta e Scarica la Relazione Tecnica Completa in PDF](./docs/latex/build/main.pdf)**

### I tre modelli e le Pipeline Riproducibili

| Modello                       | Obiettivo Clinico | Riproducibilità |
| ----------------------------- | ------------------------------------------------------------------------------------- | ---------------------- |
| 🔴 **NLP Rischio Suicidario** | Analisi testuale dei log del paziente per individuare le _Red Flag_ e segnali di rischio suicidario | [Guida Integrale Modello (Step-by-Step)](./models/nlp_suicide_risk/README.md) |
| 🔔 **Churn Prevention**       | Prevenzione abbandono (*Notification Fatigue*) tramite evoluzione di strategia di Nudging N-Dimensionale con Algoritmo Genetico | [Guida Integrale Modello (Step-by-Step)](./models/churn_prevention/README.md) |
| 🌧️ **Depression Prediction**  | Rilevamento e interpolazione dello stato depressivo a partire da metriche comportamentali e *Mood Tracker* | [Guida Integrale Modello (Step-by-Step)](./models/depression_prediction/README.md) |

---

## Struttura della Repository

```
SINTON-IA/
├── .github/               # Template PR e workflow CI (da configurare)
├── docs/
│   ├── assets/            # Diagrammi architettura
│   └── latex/             # Documentazione tecnica unificata (LaTeX)
├── models/
│   ├── nlp_suicide_risk/      # Modello TF-IDF e NLP per il Rischio Suicidario
│   ├── churn_prevention/      # Algoritmo Genetico per la Prevenzione dell'Abbandono
│   └── depression_prediction/ # Modello ML di Regressione (Mood Tracking)
├── api/                   # Backend FastAPI che espone i modelli
├── integration/           # Connettore con il software SINTONIA
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Crea e attiva virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# oppure:
venv\Scripts\activate           # Windows

# Installa dipendenze
pip install -r requirements.txt
```

---

## Team SINTON-IA

Progetto Sperimentale per l'Intelligenza Artificiale Medica — Indirizzato al dominio **ASL Campania**.
Il software, l'ingegnerizzazione dati e la validazione dei tre algoritmi sono stati curati da:

- **Antonio Walter De Fusco**  | [Github Profile](https://github.com/AntonioWalter)   | Matr. `0512119006`
- **Alessio Del Sorbo**        | [Github Profile](https://github.com/aleds25)         | Matr. `0512119618`
- **Gianni Policola**          | [Github Profile](https://github.com/GiaPol)          | Matr. `0512119747`

---

## Licenza

Questo progetto accademico è rilasciato pubblicamente sotto i termini della [Licenza MIT](LICENSE).
