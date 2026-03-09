import os
import argparse
from datasets import load_dataset

def download_and_save(token=None):
    print("🔄 Connessione all'Hub Hugging Face sicura...")
    try:
        # Passiamo il token in Lettura oppure assumiamo che il client sia loggato
        dataset = load_dataset("SINTON-IA/red_flag_processed", token=token if token else True)
        
        print("✅ Dataset individuato e analizzato in memoria temporanea (Parquet).")
        print("🔄 Estrazione in locale in corso...")
        
        # Convertiamo l'oggetto HF Dataset nel classico DataFrame
        df = dataset['train'].to_pandas()
        
        # Risolviamo il percorso di SINTON-IA dinamicamente
        # lo script si trova in models/nlp_suicide_risk/src, andiamo alla directory data/processed
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "data", "processed")
        
        # Creiamo direttrici fantasma se macanti nel collega host
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "red_flag_cleaned.csv")
        
        df.to_csv(output_path, index=False)
        print(f"✅ Download completato: {len(df)} records salvati correttamente nel computer.")
        print(f"📂 Percorso di destinazione: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: Accesso Cloud Rifiutato o libreria mancante.\nErrore nativo: {e}")
        print("\n\n💡 ISTRUZIONI PER LA RISOLUZIONE (Data Scientist SINTON-IA):")
        print("   1. Devi avere il permesso di Lettura sull'Organizzazione 'SINTON-IA'.")
        print("   2. Devi autenticarti nel terminale con: huggingface-cli login")
        print("      *OPPURE* passare il token Read esplicitamente a questo script lanciando:")
        print("      python3 download_dataset.py --token hf_IlTuoTokenPersonaleDaLettura")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader e Unpacker Universale SINTON-IA Dataset NLP")
    parser.add_argument("--token", type=str, help="Il token HF Read del tuo account (Es: hf_xxxxxx)", default=None)
    
    args = parser.parse_args()
    download_and_save(token=args.token)
