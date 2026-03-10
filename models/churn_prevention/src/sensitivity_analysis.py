import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import List, Dict

# Aggiungiamo il percorso dei sorgenti
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genetic_algorithm import GAParams, GeneticAlgorithm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_sensitivity_analysis(patient_data: pd.Series, output_path: str):
    """
    Valuta come i pesi della fitness influenzano la strategia finale.
    W1: Retention weight
    W2: Frequency penalty weight
    W3: Time penalty weight
    """
    # Griglia di pesi da testare
    weights_grid = [
        (1.0, 0.1, 0.1), # Focalizzato su retention
        (1.0, 0.4, 0.2), # Default/Bilanciato
        (1.0, 0.8, 0.1), # Forte penalità frequenza
        (1.0, 0.1, 0.8), # Forte penalità orario
        (0.5, 0.5, 0.5)  # Equamente distribuito
    ]
    
    results = []
    
    logger.info("Avvio analisi di sensibilità sui pesi della fitness.")
    
    for w1, w2, w3 in weights_grid:
        logger.info(f"Testando pesi: w1={w1}, w2={w2}, w3={w3}")
        
        params = GAParams(
            generations=100,
            pop_size=100,
            w_retention=w1,
            w_penalty_freq=w2,
            w_penalty_time=w3
        )
        
        # Esegui 5 volte per stabilità
        final_freqs = []
        final_fitnesses = []
        
        for _ in range(5):
            ga = GeneticAlgorithm(params, patient_data)
            best_ind = ga.run()
            phenotype = best_ind.decode()
            final_freqs.append(phenotype['frequenza_settimanale'])
            final_fitnesses.append(best_ind.fitness)
            
        results.append({
            'w_retention': w1,
            'w_penalty_freq': w2,
            'w_penalty_time': w3,
            'avg_final_freq': np.mean(final_freqs),
            'avg_final_fitness': np.mean(final_fitnesses)
        })
        
    df_sens = pd.DataFrame(results)
    df_sens.to_csv(output_path, index=False)
    logger.info(f"Analisi di sensibilità completata. Risultati: \n{df_sens}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "churn_features.csv")
    df = pd.read_csv(data_path)
    # Usiamo un paziente "A Rischio" per il test di sensibilità
    patient = df[df['profilo_assegnato'] == 'A Rischio'].iloc[0]
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "ga_sensitivity_analysis.csv")
    
    run_sensitivity_analysis(patient, results_path)
