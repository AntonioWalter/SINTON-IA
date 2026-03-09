import os
import argparse
from huggingface_hub import snapshot_download

def download_and_save(token=None):
    print("Avvio connessione per il download del dataset Depression Prediction...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(model_dir, "data", "raw")
    
    # Creiamo la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        repo_id = "SINTON-IA/depression_prediction"
        print(f"Inizio download sincrono della cartella dall'Hub ({repo_id})...")
        print("   Questa operazione scaricherà tutti i file JSON e CSV necessari.")
        
        # Scarica solo il contenuto della cartella 'raw' dal repository
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns="raw/*",
            token=token if token else True
        )
        
        # Rinominare i percorsi interni se HF ha annidiato il folder "raw/" 
        # (allow_patterns mantiene in genere la struttura del folder)
        hf_raw_subdir = os.path.join(output_dir, "raw")
        if os.path.exists(hf_raw_subdir):
            import shutil
            for item in os.listdir(hf_raw_subdir):
                src = os.path.join(hf_raw_subdir, item)
                dst = os.path.join(output_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, output_dir)
            os.rmdir(hf_raw_subdir)
        
        print("Download completato con successo!")
        print(f"Tutti i file sono stati estratti in: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\nERRORE CRITICO DURANTE IL DOWNLOAD.\nErrore nativo: {e}")
        print("\nAssicurati di aver fatto login con 'huggingface-cli login' o usa --token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader per SINTON-IA Dataset Depression Prediction")
    parser.add_argument("--token", type=str, help="Token HF Read", default=None)
    args = parser.parse_args()
    download_and_save(token=args.token)
