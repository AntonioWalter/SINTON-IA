# SINTON-IA 🧠

Componente di Intelligenza Artificiale del progetto **SINTONIA** — sistema di monitoraggio della salute mentale per pazienti in lista d'attesa per lo psicologo con **ASL Campania**.

---

## Descrizione

SINTON-IA integra tre modelli di machine learning nel software preesistente SINTONIA, con l'obiettivo di supportare clinici e operatori nell'identificazione precoce di situazioni a rischio e nel miglioramento del follow-up dei pazienti.

### I tre modelli

| Modello                       | Obiettivo                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| 🔴 **NLP Rischio Suicidario** | Analisi del testo prodotto dal paziente per individuare segnali di rischio suicidario |
| 🔔 **Churn Prevention**       | Ottimizzazione del nudging tramite Algoritmo Genetico                                 |
| 🌧️ **Depression Prediction**  | Rilevamento dello stato depressivo a partire dal diario dell'umore giornaliero        |

---

## Struttura della Repository

```
SINTON-IA/
├── .github/               # Template PR e workflow CI (da configurare)
├── docs/
│   ├── assets/            # Diagrammi architettura
│   └── latex/             # Documentazione tecnica unificata (LaTeX)
├── models/
│   ├── nlp_suicide_risk/      # Modello NLP rischio suicidario
│   ├── churn_prevention/      # Modello churn prediction
│   └── depression_detection/  # Modello rilevamento stato depressivo
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

## Team

Progetto universitario — ASL Campania.

## Licenza

MIT
