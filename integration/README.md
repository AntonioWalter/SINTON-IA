# Integration – Connettore SINTONIA ↔ SINTON-IA

Questo modulo gestisce il collegamento tra i modelli AI di SINTON-IA e il software preesistente SINTONIA (sistema di monitoraggio della salute mentale ASL Campania).

## Responsabilità

- Ricezione dei dati dal software SINTONIA (es. risposte ai questionari, registro umore, sessioni utente)
- Trasformazione dei dati nel formato atteso dai modelli
- Invio delle predizioni al software SINTONIA per l'utilizzo clinico

## Schema di integrazione

```
Software SINTONIA  ──(API call)──▶  SINTON-IA (api/)  ──▶  Modelli (models/)
        ◀──(risposta JSON)──────────────────────────────────────────────
```

## Note

> ⚠️ Documentare qui i dettagli dell'integrazione man mano che vengono definiti con il team SINTONIA.
