import os
import argparse
import pandas as pd
from huggingface_hub import snapshot_download
import shutil

def download_and_save(token=None):
    print("🔄 Avvio connessione per il download del dataset Red Flag Detection...")
    
    # Determiniamo la cartella root del modello (nlp_suicide_risk)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.abspath(os.path.join(script_dir, ".."))
    output_dir = os.path.join(model_root, "data", "raw")
    
    # Creiamo la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        repo_id = "SINTON-IA/red_flag_processed"
        print(f"Inizio download dall'Hub ({repo_id})...")
        
        # Scarica i file in una cartella temporanea
        temp_dir = os.path.join(output_dir, "_temp_hf")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=temp_dir,
            allow_patterns="data/*.parquet",
            token=token
        )
        
        # Spostiamo e convertiamo i file
        hf_data_subdir = os.path.join(temp_dir, "data")
        if os.path.exists(hf_data_subdir):
            for item in os.listdir(hf_data_subdir):
                if item.endswith(".parquet"):
                    src_path = os.path.join(hf_data_subdir, item)
                    # Forziamo il nome richiesto: Suicide_Detection.csv
                    csv_filename = "Suicide_Detection.csv"
                    dst_path = os.path.join(output_dir, csv_filename)
                    
                    print(f"📦 Convertendo {item} in {csv_filename}...")
                    df = pd.read_parquet(src_path)
                    df.to_csv(dst_path, index=False)
                    print(f"✅ Salvato: {dst_path}")
        
        # Pulizia cartella temporanea
        shutil.rmtree(temp_dir)
        
        print("\n✨ Download e conversione completati con successo!")
        print(f"I file CSV sono pronti in: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO DURANTE IL DOWNLOAD O CONVERSIONE.\nErrore: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader e Converter SINTON-IA Dataset")
    parser.add_argument("--token", type=str, help="Token HF Read", default=None)
    args = parser.parse_args()
    download_and_save(token=args.token)
