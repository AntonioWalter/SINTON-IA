import pandas as pd
import numpy as np
import os
import sys
import logging
import random
from typing import List, Dict

# Aggiungiamo il percorso dei sorgenti
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genetic_algorithm import GAParams, GeneticAlgorithm, Chromosome, FitnessEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_benchmark(df: pd.DataFrame, n_samples: int = 20):
    """
    Confronta il GA con una baseline casuale e una euristica fissa.
    """
    params = GAParams(pop_size=100, generations=100)
    evaluator = FitnessEvaluator(params)
    
    results = []
    
    # Campionamento bilanciato tra i profili
    print(f"[*] Colonne trovate: {df.columns.tolist()}")
    profiles = df['profilo_assegnato'].unique()
    sample_df = pd.concat([df[df['profilo_assegnato'] == p].sample(min(len(df[df['profilo_assegnato'] == p]), n_samples // 4)) for p in profiles])
    sample_df = sample_df.reset_index(drop=True)
    
    logger.info(f"Avvio Benchmark su {len(sample_df)} pazienti.")
    
    for idx in range(len(sample_df)):
        patient_row = sample_df.iloc[idx]
        patient_dict = patient_row.to_dict()
        if 'profilo_assegnato' not in patient_dict:
            logger.error(f"ERRORE: 'profilo_assegnato' non in {patient_dict.keys()}")
            # Fallback se la colonna è l'indice
            profile = patient_row.name if hasattr(patient_row, 'name') else "Unknown"
        else:
            profile = patient_dict['profilo_assegnato']
            
        pid = patient_dict.get('id_paziente', f"unknown_{idx}")
        patient = pd.Series(patient_dict)
        
        # 1. Strategia GA (Ottimizzata)
        ga = GeneticAlgorithm(params, patient)
        best_ga = ga.run()
        fitness_ga = best_ga.fitness
        
        # 2. Strategia Random (Media di 5 tentativi casuali)
        random_fitnesses = []
        for _ in range(10):
            rand_chrom = Chromosome()
            random_fitnesses.append(evaluator.evaluate(rand_chrom, patient))
        fitness_random = np.mean(random_fitnesses)
        
        # 3. Strategia Heuristic (Regola fissa: Motivazionale, 3/week, h 09:00 e 18:00, Uniforme)
        # Codifichiamo manualmente i bit per questa strategia
        # G1: Motivazionale (1) -> [0, 1]
        # G2: Frequenza (3) -> [0, 0, 0, 1, 1]
        # G3: Orari (9, 18) -> bitmask con 1 a pos 9 e 18
        bits_heuristic = np.zeros(Chromosome.TOTAL_LENGTH, dtype=np.int8)
        bits_heuristic[1] = 1 # Tipologia
        bits_heuristic[5] = 1; bits_heuristic[6] = 1 # Frequenza 3
        bits_heuristic[7+9] = 1; bits_heuristic[7+18] = 1 # Orari
        # G4: Distribuzione Uniforme (0) -> bit 31 = 0
        
        heuristic_chrom = Chromosome(bits=bits_heuristic)
        fitness_heuristic = evaluator.evaluate(heuristic_chrom, patient)
        
        results.append({
            'id_paziente': pid,
            'profilo': profile,
            'fitness_ga': fitness_ga,
            'fitness_random': fitness_random,
            'fitness_heuristic': fitness_heuristic,
            'gain_vs_random': (fitness_ga - fitness_random) / (fitness_random + 1e-6),
            'gain_vs_heuristic': (fitness_ga - fitness_heuristic) / (fitness_heuristic + 1e-6)
        })
        
    df_bench = pd.DataFrame(results)
    
    # Riassunto per profilo
    summary = df_bench.groupby('profilo').agg({
        'fitness_ga': 'mean',
        'fitness_random': 'mean',
        'fitness_heuristic': 'mean',
        'gain_vs_random': 'mean',
        'gain_vs_heuristic': 'mean'
    }).reset_index()
    
    return df_bench, summary

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "churn_features.csv")
    if not os.path.exists(data_path):
        logger.error("Dataset non trovato. Esegui prima aggregate_features.py")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    df_bench, summary = run_benchmark(df)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    df_bench.to_csv(os.path.join(output_dir, "benchmark_comparison.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
    
    print("\n" + "="*50)
    print(" RISULTATI BENCHMARK COMPARATIVO")
    print("="*50)
    print(summary.to_string(index=False))
    print("="*50)
    logger.info(f"Risultati salvati in {output_dir}")
