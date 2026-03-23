import os
import argparse
from huggingface_hub import snapshot_download
import shutil

def download_and_save(token=None):
    print("🔄 Avvio connessione per il download del dataset raw Red Flag Detection...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "raw")
    
    # Creiamo la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        repo_id = "SINTON-IA/red_flag_processed"
        print(f"Inizio download sincrono della cartella dall'Hub ({repo_id})...")
        print("Questa operazione scaricherà i dati processati nella cartella raw.")
        
        # Scarica solo il contenuto della cartella 'data' dal repository
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns="data/*",
            token=token
        )
        
        # Rinominare i percorsi interni se HF ha annidiato il folder "data/" 
        hf_data_subdir = os.path.join(output_dir, "data")
        if os.path.exists(hf_data_subdir):
            for item in os.listdir(hf_data_subdir):
                src = os.path.join(hf_data_subdir, item)
                dst = os.path.join(output_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, output_dir)
            os.rmdir(hf_data_subdir)
        
        print("✅ Download completato con successo!")
        print(f"Tutti i dati grezzi sono stati estratti in: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO DURANTE IL DOWNLOAD.\nErrore nativo: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader e Unpacker Universale SINTON-IA Dataset NLP")
    parser.add_argument("--token", type=str, help="Token HF Read", default=None)
    args = parser.parse_args()
    download_and_save(token=args.token)
