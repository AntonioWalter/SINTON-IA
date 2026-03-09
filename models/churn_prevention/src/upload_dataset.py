import pandas as pd
from datasets import Dataset
import os

def upload_dataset():
    print("Loading data...")
    # Percorso del file CSV processato (risolto dinamicamente rispetto a questo script in src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    file_path = os.path.join(model_dir, "data", "processed", "churn_features.csv")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    print(f"Dataset caricato: {len(df)} righe, {len(df.columns)} colonne.")
    
    # Converte in formato Dataset di Hugging Face
    print("Converting to Hugging Face Dataset format...")
    hf_dataset = Dataset.from_pandas(df)
    
    repo_id = "SINTON-IA/churn_prevention"
    print(f"Pushing to Hugging Face Hub: {repo_id}...")
    try:
        hf_dataset.push_to_hub(repo_id, private=True)
        print(f"Successfully pushed to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error pushing to hub: {e}")

if __name__ == "__main__":
    upload_dataset()
