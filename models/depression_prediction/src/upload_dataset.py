import pandas as pd
from datasets import Dataset
import os

def upload_dataset():
    print("Caricamento dei dati...")
    # Percorso del file CSV del dataset (risolto dinamicamente rispetto a questo script in src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    file_path = os.path.join(model_dir, "data", "dataset.csv")
    
    if not os.path.exists(file_path):
        print(f"Errore: File non trovato in {file_path}")
        print("Genera prima il dataset e assicuratevi che sia salvato lì.")
        return
        
    df = pd.read_csv(file_path)
    print(f"Dataset caricato: {len(df)} righe, {len(df.columns)} colonne.")
    
    # Converte in formato Dataset di Hugging Face
    print("Conversione nel formato Hugging Face Dataset...")
    hf_dataset = Dataset.from_pandas(df)
    
    repo_id = "SINTON-IA/depression_prediction" 
    print(f"Caricamento su Hugging Face Hub: {repo_id}...")
    try:
        # Imposta private=True per sicurezza sui dati sensibili
        hf_dataset.push_to_hub(repo_id, private=True)
        print(f"Dataset caricato con successo su https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Errore durante l'upload: {e}")

if __name__ == "__main__":
    upload_dataset()
