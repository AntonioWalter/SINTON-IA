import os
import argparse
from huggingface_hub import HfApi

def upload_and_save(token=None):
    print("Avvio procedura di upload per il dataset Depression Prediction (Raw Data)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    input_path = os.path.join(model_dir, "data", "raw")
    
    if not os.path.exists(input_path) or not os.listdir(input_path):
        print("❌ ERRORE CRITICO: Cartella dataset non trovata o vuota in locale.")
        print(f"   Assicurati che esista e contenga i dati: {input_path}")
        return
        
    try:
        print(f"Individuata cartella dati: {input_path}")
        repo_id = "SINTON-IA/depression_prediction"
        print(f"Inizio upload della cartella verso l'Hub ({repo_id})...")
        
        api = HfApi(token=token)
        # Se il repo non dovesse esistere, crealo prima (come privato)
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception:
            pass # Ignoriamo eventuali errori se l'utente non ha i permessi per crearne di nuovi ma ha accesso

        api.upload_folder(
            folder_path=input_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="raw" # Salva tutto sotto la cartella raw nel repo
        )
        
        print("Upload completato con successo (Folder intera esportata)!")
        print(f"URL del dataset: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"\nERRORE DURANTE L'UPLOAD.\nErrore nativo: {e}")
        print("\nAssicurati di aver fatto login con 'huggingface-cli login' o usa --token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uploader per SINTON-IA Dataset Depression Prediction")
    parser.add_argument("--token", type=str, help="Token HF Write", default=None)
    args = parser.parse_args()
    upload_and_save(token=args.token)
