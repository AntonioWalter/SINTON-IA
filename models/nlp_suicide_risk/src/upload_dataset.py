import os
import argparse
import pandas as pd
from datasets import Dataset

def upload_and_save(token=None):
    print("🔄 Avvio procedura di upload per il dataset NLP Rischio Suicidario...")
    
    # Risolviamo il percorso di SINTON-IA dinamicamente
    # lo script si trova in models/nlp_suicide_risk/src, andiamo alla directory data/processed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "..", "data", "processed", "red_flag_cleaned.csv")
    
    if not os.path.exists(input_path):
        print("❌ ERRORE CRITICO: File dataset non trovato in locale.")
        print(f"   Assicurati che esista il file: {input_path}")
        return
        
    try:
        print("✅ Lettura dataset locale in corso...")
        df = pd.read_csv(input_path)
        print(f"   Trovati {len(df)} records e {len(df.columns)} colonne.")
        
        print("🔄 Conversione nel formato Hugging Face (Arrow/Parquet)...")
        hf_dataset = Dataset.from_pandas(df)
        
        repo_id = "SINTON-IA/red_flag_processed"
        print(f"🚀 Creazione connessione verso l'Hub ({repo_id})...")
        
        # Passiamo il token in Scrittura oppure assumiamo che il client sia loggato
        hf_dataset.push_to_hub(repo_id, private=True, token=token)
        
        print("✅ Upload completato con successo!")
        print(f"📂 URL del dataset: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE L'UPLOAD.\nErrore nativo: {e}")
        print("\n\n💡 ISTRUZIONI PER LA RISOLUZIONE (Data Scientist SINTON-IA):")
        print("   1. Devi avere il permesso di SCRITTURA (Write) sull'Organizzazione 'SINTON-IA'.")
        print("   2. Devi autenticarti nel terminale con: huggingface-cli login")
        print("      *OPPURE* passare il token Write esplicitamente a questo script lanciando:")
        print("      python3 upload_dataset.py --token hf_IlTuoTokenPersonaleDaScrittura")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uploader Universale SINTON-IA Dataset NLP Rischio Suicidario")
    parser.add_argument("--token", type=str, help="Il token HF Write del tuo account (Es: hf_xxxxxx)", default=None)
    
    args = parser.parse_args()
    upload_and_save(token=args.token)
