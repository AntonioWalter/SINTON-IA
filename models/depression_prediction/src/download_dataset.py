import os
import pandas as pd
from datasets import load_dataset

def download_dataset():
    print("Downloading depression_prediction dataset from Hugging Face...")
    try:
        # Carica il dataset
        dataset = load_dataset("SINTON-IA/depression_prediction", split="train")
        df = dataset.to_pandas()
        
        # Percorso dove salvare il file scaricato
        # Visto che lo script è in src/, andiamo al livello genitore per trovare 'data'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(model_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        file_path = os.path.join(data_dir, "dataset.csv")
        df.to_csv(file_path, index=False)
        print(f"Dataset salvato con successo in: {file_path}")
        print(f"Righe scaricate: {len(df)}")
    except Exception as e:
        print(f"Errore durante il download: {e}")
        print("Assicurati di aver fatto il login con 'huggingface-cli login'.")

if __name__ == "__main__":
    download_dataset()
