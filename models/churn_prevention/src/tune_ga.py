import pandas as pd
import numpy as np
import itertools
import logging
import os
import sys
from typing import List, Dict
from dataclasses import asdict

# Aggiungiamo il percorso dei sorgenti per importare il GA
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genetic_algorithm import GAParams, GeneticAlgorithm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_grid_search(patient_data: pd.Series, output_path: str):
    """
    Esegue una ricerca a griglia sugli iperparametri del GA.
    """
    # Definizione dello spazio di ricerca
    grid = {
        'pop_size': [50, 100, 200],
        'crossover_rate': [0.7, 0.8, 0.9],
        'mutation_rate': [0.01, 0.05, 0.1],
        'tournament_size': [2, 3, 5]
    }
    
    # Genera tutte le combinazioni
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    total = len(combinations)
    
    logger.info(f"Avvio Grid Search: {total} combinazioni da testare.")
    
    for i, config in enumerate(combinations):
        logger.info(f"[{i+1}/{total}] Testando config: {config}")
        
        # Inizializza parametri (usiamo un numero fisso di generazioni per confronto equo)
        params = GAParams(
            generations=100,
            pop_size=config['pop_size'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            tournament_size=config['tournament_size']
        )
        
        # Esegui GA (3 ripetizioni per mediare la componente stocastica)
        seeds = [42, 123, 999]
        run_fitnesses = []
        run_generations = []
        
        for seed in seeds:
            np.random.seed(seed)
            ga = GeneticAlgorithm(params, patient_data)
            best_ind = ga.run()
            run_fitnesses.append(best_ind.fitness)
            run_generations.append(len(ga.history_best))
            
        avg_fitness = np.mean(run_fitnesses)
        avg_gens = np.mean(run_generations)
        
        res = config.copy()
        res['avg_best_fitness'] = avg_fitness
        res['avg_generations_to_converge'] = avg_gens
        results.append(res)
        
    # Salva risultati
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    logger.info(f"Grid Search completata. Risultati salvati in: {output_path}")
    
    # Trova la migliore configurazione
    best_config = df_results.sort_values(by='avg_best_fitness', ascending=False).iloc[0]
    logger.info(f"Migliore configurazione trovata:\n{best_config}")

if __name__ == "__main__":
    # Caricamento di un paziente rappresentativo (es. "A Rischio") per il tuning
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "churn_features.csv")
    if not os.path.exists(data_path):
        logger.error(f"Dataset non trovato in {data_path}. Eseguire prima aggregate_features.py.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    # Prendiamo il primo paziente "A Rischio" come benchmark per il tuning
    benchmark_patient = df[df['profilo_assegnato'] == 'A Rischio'].iloc[0]
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "ga_tuning_results.csv")
    
    run_grid_search(benchmark_patient, results_path)
