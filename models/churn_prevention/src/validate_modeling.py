import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import List, Dict
from tqdm import tqdm

# Aggiungiamo il percorso dei sorgenti
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genetic_algorithm import GAParams, GeneticAlgorithm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_model(df: pd.DataFrame, n_samples_per_profile: int = 10):
    """
    Esegue il GA su un campione di pazienti per ogni profilo per valutare stabilità e performance.
    """
    profiles = df['profilo_assegnato'].unique()
    validation_results = []
    
    # Caricamento parametri ottimizzati (se disponibili da tune_ga.py)
    results_path = os.path.join(os.path.dirname(__file__), "..", "data", "results", "ga_tuning_results.csv")
    
    if os.path.exists(results_path):
        logger.info(f"Caricamento iperparametri ottimizzati da {results_path}")
        df_results = pd.read_csv(results_path)
        # Seleziona la config con la migliore fitness media
        best_row = df_results.sort_values(by='avg_best_fitness', ascending=False).iloc[0]
        best_params = GAParams(
            pop_size=int(best_row['pop_size']),
            generations=150, # Manteniamo generazioni elevate per validazione
            crossover_rate=float(best_row['crossover_rate']),
            mutation_rate=float(best_row['mutation_rate']),
            tournament_size=int(best_row['tournament_size'])
        )
    else:
        logger.warning("File tuning non trovato. Utilizzo default robusti.")
        best_params = GAParams(
            pop_size=100,
            generations=150,
            mutation_rate=0.05,
            tournament_size=3
        )
    
    logger.info(f"Avvio validazione statistica su {n_samples_per_profile} campioni per profilo.")
    
    for profile in profiles:
        profile_df = df[df['profilo_assegnato'] == profile].head(n_samples_per_profile)
        logger.info(f"Validazione profilo: {profile} ({len(profile_df)} pazienti)")
        
        for idx, row in profile_df.iterrows():
            ga = GeneticAlgorithm(best_params, row)
            best_ind = ga.run()
            phenotype = best_ind.decode()
            
            res = {
                'id_paziente': row['id_paziente'],
                'profilo': profile,
                'final_fitness': best_ind.fitness,
                'generations_to_converge': len(ga.history_best),
                'strategy_type': phenotype['tipologia'],
                'strategy_freq': phenotype['frequenza_settimanale']
            }
            validation_results.append(res)
            
    # Crea DataFrame dei risultati
    df_val = pd.DataFrame(validation_results)
    
    # Calcolo metriche di stabilità
    summary = df_val.groupby('profilo').agg({
        'final_fitness': ['mean', 'std'],
        'generations_to_converge': ['mean', 'std'],
        'strategy_freq': 'mean'
    }).reset_index()
    
    logger.info("Sintesi Validazione Statistica:")
    print(summary)
    
    return df_val, summary

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "churn_features.csv")
    if not os.path.exists(data_path):
        logger.error(f"Dataset non trovato in {data_path}.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # Esegui validazione (usiamo 10 campioni per profilo per velocità in demo, 50+ in real setting)
    df_val, summary = validate_model(df, n_samples_per_profile=10)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    df_val.to_csv(os.path.join(output_dir, "ga_validation_full.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "ga_validation_summary.csv"), index=False)
    
    logger.info(f"Report di validazione salvati in {output_dir}")
