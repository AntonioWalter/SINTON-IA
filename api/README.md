# API – Backend FastAPI

Servizio che espone i tre modelli AI di SINTON-IA come endpoint REST, consumabili dal software SINTONIA preesistente.

## Struttura

```
api/
├── routes/      # Definizione degli endpoint (uno per modello)
├── services/    # Logica di inference per ciascun modello
└── README.md
```

## Endpoint previsti

| Metodo | Path | Modello |
|---|---|---|
| `POST` | `/predict/suicide-risk` | NLP Rischio Suicidario |
| `POST` | `/predict/churn` | Churn Prediction |
| `POST` | `/predict/depression` | Depression Detection |

## Avvio locale

```bash
uvicorn api.main:app --reload
```
