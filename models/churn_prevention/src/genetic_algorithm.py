import os
import uuid
import logging
import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class GAParams:
    """Parametri di configurazione dell'Algoritmo Genetico ottimizzati via Grid Search."""
    pop_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.7
    mutation_rate: float = 0.01
    elitism_rate: float = 0.05
    tournament_size: int = 2
    
    # Pesi della fitness (w1 = retention, w2 = penalty_freq, w3 = penalty_time)
    w_retention: float = 1.0
    w_penalty_freq: float = 0.4
    w_penalty_time: float = 0.2

    # Soglia di notifica prima che scatti la penalità (es: 14 a settimana = 2 al giorno)
    max_freq_threshold: int = 14


class Chromosome:
    """
    Rappresenta una strategia di nudging (genotipo).
    Lunghezza totale: 32 bit
    - G1: Tipologia primaria (2 bit) -> 0=Promemoria, 1=Motivazionale, 2=Informativa, 3=Questionario
    - G2: Frequenza settimanale (5 bit) -> 0-31 notifiche/settimana
    - G3: Schedule orario (24 bit) -> 1 bit per ogni ora (00:00 - 23:59)
    - G4: Distribuzione giorni (1 bit) -> 0=Uniforme, 1=Concentrata
    """
    GENE_LENGTHS = {
        'tipologia': 2,
        'frequenza': 5,
        'orario': 24,
        'distribuzione': 1
    }
    TOTAL_LENGTH = sum(GENE_LENGTHS.values())

    def __init__(self, bits: np.ndarray = None):
        if bits is None:
            self.bits = np.random.randint(0, 2, self.TOTAL_LENGTH, dtype=np.int8)
        else:
            if len(bits) != self.TOTAL_LENGTH:
                raise ValueError(f"Dimensione cromosoma errata. Attesa: {self.TOTAL_LENGTH}, Trovata: {len(bits)}")
            self.bits = np.array(bits, dtype=np.int8)
        self._fitness = None

    def decode(self) -> Dict:
        """Decodifica il genotipo nel fenotipo (parametri reali della strategia)."""
        b = self.bits
        
        # Estrai sottovettori
        tip_bits = b[0:2]
        freq_bits = b[2:7]
        ora_bits = b[7:31]
        dist_bit = b[31]

        # Converti binario in intero (Big Endian)
        tipologia_idx = int(tip_bits.dot(1 << np.arange(tip_bits.size)[::-1]))
        frequenza = int(freq_bits.dot(1 << np.arange(freq_bits.size)[::-1]))
        
        tipologie = ["Promemoria", "Motivazionale", "Informativa", "Questionario"]
        tipologia_nome = tipologie[tipologia_idx]

        # Garantisci almeno 1 notifica per evitare strategie a frequenza zero assoluto
        frequenza = max(1, frequenza)

        return {
            'tipologia': tipologia_nome,
            'frequenza_settimanale': frequenza,
            'orari_attivi': [h for h, val in enumerate(ora_bits) if val == 1],
            'distribuzione': 'Concentrata' if dist_bit == 1 else 'Uniforme'
        }

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value


class FitnessEvaluator:
    """Calcola la fitness di una strategia simulando l'impatto sul paziente."""
    
    def __init__(self, params: GAParams):
        self.params = params

    def evaluate(self, chromosome: Chromosome, patient_features: pd.Series) -> float:
        """
        Valuta il cromosoma adattando i vincoli al comportamento del paziente.
        Fitness = w1*Retention - w2*PenaltyFreq - w3*PenaltyTime
        """
        phenotype = chromosome.decode()
        
        # 1. Calcolo Retention (Simulata tramite proxy)
        retention_score = self._simulate_retention(phenotype, patient_features)
        
        # 2. Penalty Frequenza (Adattiva)
        # Se l'utente è molto attivo (mood_freq alto), tollera più notifiche
        mood_freq = patient_features.get('mood_frequency_7d', 0.5)
        dynamic_threshold = self.params.max_freq_threshold
        if mood_freq > 0.8: dynamic_threshold += 4 # Gli engaged tollerano di più
        if mood_freq < 0.2: dynamic_threshold -= 4 # I ghost/at risk tollerano meno
        
        freq = phenotype['frequenza_settimanale']
        if freq > dynamic_threshold:
            diff = freq - dynamic_threshold
            penalty_freq = min(1.0, (np.exp(0.15 * diff) - 1) / 10) 
        else:
            penalty_freq = 0.0

        # 3. Penalty Tempo (Adattiva per Night Owls)
        # Se l'utente ha un alto night_activity_rate, penalizziamo meno l'invio notturno
        night_rate = patient_features.get('night_activity_rate', 0.0)
        night_hours = [23, 0, 1, 2, 3, 4, 5, 6]
        active_night_hours = sum(1 for h in phenotype['orari_attivi'] if h in night_hours)
        
        if len(phenotype['orari_attivi']) == 0:
             penalty_time = 1.0
        else:
             # Se night_rate > 0.3, l'utente è un "Night Owl" e tollera notifiche notturne
             time_sensitivity = max(0.1, 1.0 - night_rate) 
             penalty_time = (active_night_hours / len(night_hours)) * time_sensitivity
        
        # Calcolo finale
        fitness = (self.params.w_retention * retention_score) - \
                  (self.params.w_penalty_freq * penalty_freq) - \
                  (self.params.w_penalty_time * penalty_time)
                  
        return float(fitness)

    def _simulate_retention(self, phenotype: Dict, features: pd.Series) -> float:
        """
        Simula l'effetto della strategia S sul paziente P usando un rudimentale sistema esperto.
        L'output è in [0, 1].
        """
        # Le features sono già normalizzate in [0,1]
        mood_freq = features.get('mood_frequency_7d', 0.5)
        avg_valence = features.get('avg_mood_valence_7d', 0.5)
        read_rate = features.get('notification_read_rate', 0.5)
        
        # 1. Cluster "Ghost" (Interazioni quasi nulle, altissimo rischio abbandono)
        is_ghost = (mood_freq < 0.1) and (read_rate < 0.2)
        
        # 2. Cluster "A Rischio" (Attività in calo o umore basso, segnali di instabilità)
        is_at_risk = not is_ghost and ((mood_freq < 0.3) or (avg_valence < 0.45))
        
        # 3. Cluster "Depresso Attivo" (Engaged ma con umore molto basso)
        is_depressed_engaged = (mood_freq >= 0.6) and (avg_valence < 0.4)
        
        # 4. Cluster "Standard/Sano" (Moderato/Engaged senza forti criticità)
        is_standard = not is_ghost and not is_at_risk and not is_depressed_engaged
        
        score = 0.5  # Base
        
        tipo = phenotype['tipologia']
        freq = phenotype['frequenza_settimanale']
        distr = phenotype['distribuzione']
        
        # Regole euristiche per la simulazione
        if is_ghost:
            # Utenti quasi persi: messaggi motivazionali rari per non infastidire
            if tipo == 'Motivazionale': score += 0.3
            if freq > 3: score -= 0.4
            if freq <= 2: score += 0.1
            if distr == 'Concentrata': score += 0.1
            
        elif is_at_risk:
            # Utenti instabili: serve supporto motivazionale e promemoria
            if tipo == 'Motivazionale': score += 0.2
            if tipo == 'Promemoria': score += 0.1
            if 2 <= freq <= 5: score += 0.2
            if freq > 7: score -= 0.2
            if distr == 'Uniforme': score += 0.1
            
        elif is_depressed_engaged:
            # Utenti attivi ma fragili: supporto costante, no task pesanti
            if tipo in ['Motivazionale', 'Promemoria']: score += 0.2
            if tipo == 'Questionario': score -= 0.2
            if 4 <= freq <= 10: score += 0.2
            
        elif is_standard:
            # Utenti attivi: ricordiamo loro i task e monitoriamo con questionari
            if tipo == 'Promemoria': score += 0.1
            if tipo == 'Questionario': score += 0.2
            if 3 <= freq <= 7: score += 0.2
            if freq > 12: score -= 0.3
            
        # Adattamento all'orario: non vogliamo strategie senza orari validi
        if len(phenotype['orari_attivi']) == 0:
            score = 0.0
            
        # Capping e normalizzazione
        return max(0.0, min(1.0, score))


class GeneticAlgorithm:
    """Implementa il loop evolutivo."""
    
    def __init__(self, params: GAParams, patient_features: pd.Series):
        self.params = params
        self.patient_features = patient_features
        self.evaluator = FitnessEvaluator(params)
        self.population: List[Chromosome] = []
        
        # Log metriche
        self.history_best = []
        self.history_avg = []
        self.history_diversity = []

    def initialize_population(self):
        self.population = [Chromosome() for _ in range(self.params.pop_size)]
        self._evaluate_population()

    def _evaluate_population(self):
        for ind in self.population:
            # Ricalcola solo se necessario
            if ind.fitness is None:
                     ind.fitness = self.evaluator.evaluate(ind, self.patient_features)
        
        # Ordina per fitness decrescente (massimizzazione)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def _selection_tournament(self) -> Chromosome:
        """Selezione a torneo k=3"""
        k = self.params.tournament_size
        candidates = random.sample(self.population, k)
        best = max(candidates, key=lambda x: x.fitness)
        return Chromosome(bits=best.bits.copy())

    def _crossover_two_point(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Crossover a due punti. Taglia scambiando blocchi centrali preservando la struttura semantica vicino ai tagli."""
        if random.random() > self.params.crossover_rate:
            return Chromosome(bits=parent1.bits.copy()), Chromosome(bits=parent2.bits.copy())
            
        length = Chromosome.TOTAL_LENGTH
        pt1 = random.randint(1, length - 2)
        pt2 = random.randint(pt1 + 1, length - 1)
        
        child1_bits = np.concatenate((parent1.bits[:pt1], parent2.bits[pt1:pt2], parent1.bits[pt2:]))
        child2_bits = np.concatenate((parent2.bits[:pt1], parent1.bits[pt1:pt2], parent2.bits[pt2:]))
        
        return Chromosome(bits=child1_bits), Chromosome(bits=child2_bits)

    def _mutate(self, chromosome: Chromosome):
        """Mutazione bit-flip indipendente per ogni gene basata sul mutation_rate."""
        for i in range(Chromosome.TOTAL_LENGTH):
            if random.random() < self.params.mutation_rate:
                chromosome.bits[i] = 1 - chromosome.bits[i] # Flip: 0->1, 1->0
        chromosome.fitness = None # Invalida la fitness

    def _calculate_diversity(self) -> float:
        """Calcola la distanza di Hamming media normalizzata tra tutte le coppie possibili 
        (approssimata con un campione per popolazioni molto grandi, qui la calcoliamo calcolando 
        la varianza degli alleli alle varie posizioni).
        """
        # Creiamo una matrice PopSize x GenLength
        pop_matrix = np.array([ind.bits for ind in self.population])
        # P(allele=1) in the population
        p1 = pop_matrix.mean(axis=0)
        # La diversità genetica (avg pairwise Hamming distance) è legata alla varianza binaria 2*p*(1-p)
        # Sommiamo su tutti i loci e normalizziamo per lunghezza gene
        avg_hamming = np.sum(2 * p1 * (1 - p1))
        return float(avg_hamming / Chromosome.TOTAL_LENGTH)

    def run(self):
        """Esegue l'evoluzione."""
        logger.info("Avvio Algoritmo Genetico...")
        self.initialize_population()
        
        n_elites = int(self.params.pop_size * self.params.elitism_rate)
        
        for gen in range(self.params.generations):
            new_population = []
            
            # 1. Elitismo (preserva i migliori)
            for i in range(n_elites):
                new_population.append(Chromosome(bits=self.population[i].bits.copy()))
                new_population[-1].fitness = self.population[i].fitness
                
            # 2. Crea il resto della popolazione
            while len(new_population) < self.params.pop_size:
                # Selezione
                p1 = self._selection_tournament()
                p2 = self._selection_tournament()
                
                # Crossover
                c1, c2 = self._crossover_two_point(p1, p2)
                
                # Mutazione
                self._mutate(c1)
                self._mutate(c2)
                
                new_population.extend([c1, c2])
                
            # Trunca ad esatta pop_size (se abbiamo aggiunto 2 e superato la max size)
            self.population = new_population[:self.params.pop_size]
            
            # Valutazione
            self._evaluate_population()
            
            # Tracking
            best_fit = self.population[0].fitness
            avg_fit = sum(ind.fitness for ind in self.population) / self.params.pop_size
            diversity = self._calculate_diversity()
            
            self.history_best.append(best_fit)
            self.history_avg.append(avg_fit)
            self.history_diversity.append(diversity)
            
            # Log ogni 10 generazioni
            if (gen+1) % 10 == 0 or gen == 0:
                logger.info(f"Gen {gen+1:03d} | Best: {best_fit:.4f} | Avg: {avg_fit:.4f} | Div: {diversity:.4f}")
                
            # Early stopping (plateau)
            if gen > 15:
                recent_variance = np.var(self.history_best[-10:])
                if recent_variance < 1e-6:
                    logger.info(f"Convergenza raggiunta (plateau variance < 1e-6) alla generazione {gen+1}. Arresto anticipato.")
                    break
                    
        return self.population[0]

if __name__ == "__main__":
    # Semplice test se eseguito direttamente (verrà usato soprattutto da notebook)
    # Creiamo un paziente fittizio per il test
    test_features = pd.Series({
        'mood_frequency_7d': 0.1,  # Bassa assiduità
        'notification_read_rate': 0.1, # Bassa lettura (Profilo tipo Ghost)
        'avg_mood_valence_7d': 0.5,
        'avg_diary_length_7d': 0.2
    })
    
    params = GAParams(generations=50, pop_size=50)
    ga = GeneticAlgorithm(params, test_features)
    best_strategy = ga.run()
    
    print("\n--- Migliore Strategia Trovata per paziente Ghost ---")
    print(f"Fitness Score: {best_strategy.fitness:.4f}")
    
    phenotype = best_strategy.decode()
    for k, v in phenotype.items():
        print(f"{k}: {v}")
