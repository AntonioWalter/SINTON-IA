# Dataset Red Flag Detection

Questa cartella è destinata alla memorizzazione locale dei dataset durante le fasi di sviluppo e manipolazione. **Nessun file CSV o di grandi dimensioni presente in questa cartella deve essere tracciato su Git**, per questo motivo è presente un file `.gitignore` dedicato.

## Struttura della Cartella

- `raw/`: Contiene il dataset originale e inalterato (es. `Suicide_Detection.csv` scaricato da Kaggle). **Non modificare MAI i file in questa cartella.**
- `interim/`: (Opzionale) Contiene dati parzialmente processati durante l'esecuzione dei notebook.
- `processed/`: Contiene i dataset finali, puliti e pronti per l'addestramento o il testing.

## Condivisione dei Dati e Cloud MLOps

Per garantire un flusso di lavoro aziendale riproducibile e non sovraccaricare il repository Git, **i dataset processati non sono presenti fisicamente in questa cartella**. 
Tutti i dati generati da questa issue sono ospitati in cloud nel repository privato dell'organizzazione [SINTON-IA su Hugging Face Hub](https://huggingface.co/SINTON-IA).

### Come scaricare il dataset pulito in locale (Per il Team)

Quando devi addestrare un modello sul tuo computer, dovrai dotarti di una copia fisica (locale) aggiornata del csv processato `red_flag_cleaned.csv`. 
Invece di scambiare file su Google Drive, abbiamo sviluppato uno script automatico universale che compie il download criptato in un colpo solo.

**Requisito Fondamentale:** 
L'Amministratore del progetto deve prima averti invitato sull'Organizzazione Hugging Face "SINTON-IA" affinchè le tue richieste non vengano rigettate.

#### Opzione A: Autenticazione Singola (Consigliata per CI/CD o One-Shot)
Apri il terminale alla radice della cartella Modello (`models/nlp_suicide_risk/`) e passa al terminale il tuo 'Access Token' (generato dal tuo [account Hugging Face](https://huggingface.co/settings/tokens) con permessi *Read*):

```bash
python3 src/download_dataset.py --token hf_LaTuaChiavePersonaleDaLettura
```

#### Opzione B: Profilo Memorizzato (Consigliata per Server Locali)
1. Fai accedere permanentemente il tuo intero sistema Mac/PC/Linux digitando:
   `huggingface-cli login`
2. Avvia lo script autonomo senza parametri:
   `python3 src/download_dataset.py`

In entrambi i casi, lo script scaricherà in cache il dataset sicuro e **genererà materialmente e automaticamente il file .csv all'interno della direttiva `data/processed/` del tuo computer**, rendendo l'ambiente testuale identico per tutti i team members.
