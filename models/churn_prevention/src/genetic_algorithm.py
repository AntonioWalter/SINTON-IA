import os
import logging
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class GAParams:
    """Configurazione parametri GA."""
    pop_size: int = 100
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    elitism_rate: float = 0.05
    patience: int = 20
    selection_method: str = "tournament"
    crossover_method: str = "two-point"
    mutation_method: str = "adaptive"
    weights: Tuple[float, float, float] = (1.0, 0.4, 0.2)
    tournament_size: int = 3
    truncation_rate: float = 0.5
    k_points: int = 2
    max_freq_threshold: int = 16
    seed: int = 42
    load_gold_standard: bool = False # Mantenuto per compatibilità con l'API del Notebook

    # Campi derivati (Properties) per evitare stati non sincronizzati
    @property
    def w_retention(self) -> float:
        return self.weights[0]

    @property
    def w_penalty_freq(self) -> float:
        return self.weights[1]

    @property
    def w_penalty_time(self) -> float:
        return self.weights[2]

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        # Mappatura sicura per ignorare chiavi extra o mappate diversamente
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def load_gold_standard_config(cls, path: str = "ga_tuned_config.json") -> "GAParams":
        """Factory method pulito per caricare il JSON solo quando esplicitamente richiesto."""
        if not os.path.exists(path):
            logger.warning(f"File {path} non trovato. Fallback ai parametri di default.")
            return cls()
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            mapping = {
                "selection_method": config.get("best_selection", "tournament"),
                "crossover_method": config.get("best_crossover", "two-point"),
                "mutation_method": config.get("best_mutation", "adaptive"),
                "pop_size": config.get("opt_pop_size", 100),
                "mutation_rate": config.get("opt_mutation_rate", 0.01),
                "weights": tuple(config.get("weights", [1.0, 0.4, 0.2]))
            }
            return cls(**mapping)
        except Exception as e:
            logger.error(f"Errore caricamento JSON: {e}")
            return cls()


class Chromosome:
    """Rappresenta una strategia di nudging (genotipo)."""
    GENE_LENGTHS = {
        'tipologia': 2,
        'frequenza': 5,
        'orario': 24
    }
    TOTAL_LENGTH = sum(GENE_LENGTHS.values())

    def __init__(self, bits: np.ndarray = None, rng: np.random.Generator = None):
        if bits is None:
            # Blindiamo la riproducibilità usando il generatore RNG se fornito
            if rng is not None:
                self.bits = rng.integers(0, 2, self.TOTAL_LENGTH, dtype=np.int8)
            else:
                self.bits = np.random.randint(0, 2, self.TOTAL_LENGTH, dtype=np.int8)
        else:
            if len(bits) != self.TOTAL_LENGTH:
                raise ValueError(f"Dimensione errata. Attesa: {self.TOTAL_LENGTH}, Trovata: {len(bits)}")
            self.bits = np.array(bits, dtype=np.int8)
        self._fitness = None

    def decode(self) -> Dict:
        """Decodifica il genotipo nel fenotipo (parametri reali della strategia)."""
        b = self.bits
        tip_bits, freq_bits, ora_bits = b[0:2], b[2:7], b[7:31]

        tipologia_idx = int(tip_bits.dot(1 << np.arange(tip_bits.size)[::-1]))
        frequenza = int(freq_bits.dot(1 << np.arange(freq_bits.size)[::-1]))
        
        tipologie = ["Promemoria", "Motivazionale", "Informativa", "Questionario"]
        frequenza = max(1, frequenza) # Evita strategie da 0 notifiche

        return {
            'tipologia': tipologie[tipologia_idx],
            'frequenza_settimanale': frequenza,
            'orari_attivi': [h for h, val in enumerate(ora_bits) if val == 1],
            'start_hour': next((h for h, val in enumerate(ora_bits) if val == 1), 9),
            'end_hour': next((h for h, val in enumerate(reversed(ora_bits)) if val == 1), 18)
        }

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value


class FitnessEvaluator:
    """Calcola la fitness pesata a priori (Fase 0)."""
    
    def __init__(self, patient_features: pd.Series, params: GAParams = None, rng: np.random.Generator = None):
        self.patient_features = patient_features
        self.params = params if params is not None else GAParams()
        self.rng = rng if rng is not None else np.random.default_rng(self.params.seed)
        
        # --- IL CUORE DEL BENCHMARK ---
        # Questo contatore ci permette di fare un confronto "ad armi pari" con la Random Search
        self.evaluation_calls = 0 

    def evaluate(self, chromosome: Chromosome, patient_features: pd.Series = None) -> float:
        self.evaluation_calls += 1 
        
        fetch_features = patient_features if patient_features is not None else self.patient_features
        phenotype = chromosome.decode()
        
        retention_score = self._simulate_retention(phenotype, fetch_features)
        
        mood_freq = fetch_features.get('mood_frequency_7d', 0.5)
        dynamic_threshold = self.params.max_freq_threshold
        if mood_freq > 0.8: dynamic_threshold += 5 
        if mood_freq < 0.2: dynamic_threshold -= 5 
        
        freq = phenotype['frequenza_settimanale']
        # --- TASSA CONTINUA FREQUENZA ---
        base_tax_f = (freq / 31.0) * 0.1 # Ogni messaggio ha un micro-costo di attenzione
        penalty_freq = base_tax_f
        if freq > dynamic_threshold:
            diff = freq - dynamic_threshold
            penalty_freq = min(1.0, base_tax_f + (np.exp(0.2 * diff) - 1) / 50) # Muro clinico
            
        night_rate = fetch_features.get('night_activity_rate', 0.0)
        night_hours = [23, 0, 1, 2, 3, 4, 5, 6]
        active_hours = phenotype['orari_attivi']
        
        # --- TASSA CONTINUA TEMPORALE ---
        penalty_time = 0.0
        if not active_hours:
             penalty_time = 1.0  
        else:
             # Ogni ora occupata ha una 'tassa di ingombro cognitivo'
             base_tax_t = (len(active_hours) / 24.0) * 0.05
             active_night_hours = sum(1 for h in active_hours if h in night_hours)
             time_sensitivity = max(0.2, 1.0 - night_rate) 
             night_penalty = (active_night_hours / 8.0) * time_sensitivity
             penalty_time = min(1.0, base_tax_t + night_penalty)
        
        raw_fitness = (self.params.w_retention * retention_score) - \
                      (self.params.w_penalty_freq * penalty_freq) - \
                      (self.params.w_penalty_time * penalty_time)
        
        return max(0.0001, float(raw_fitness))

    def _simulate_retention(self, phenotype: Dict, features: pd.Series) -> float:
        """Simulatore del Patient Environment."""
        mood_freq = features.get('mood_frequency_7d', 0.5)
        avg_valence = features.get('avg_mood_valence_7d', 0.5)
        read_rate = features.get('notification_read_rate', 0.5)
        
        # 1. Identificazione Archetipi (Gerarchica e Coerente con Data Pipeline)
        # Engaged: Il paziente ideale (Alta attività e umore stabile/positivo)
        is_engaged = (mood_freq >= 0.6) and (avg_valence >= 0.5)
        
        # Ghost: Il paziente che ha abbandonato (attività nulla o quasi)
        is_ghost = not is_engaged and (mood_freq < 0.1) and (read_rate < 0.2)
        
        # A Rischio: Il paziente in crisi (calo attività o umore negativo/preoccupante)
        is_at_risk = not (is_engaged or is_ghost) and ((mood_freq < 0.3) or (avg_valence < 0.45))
        
        # Moderato: Il paziente stabile, uso intermittente
        is_moderato = not (is_engaged or is_ghost or is_at_risk)
        
        score = 0.5  
        tipo = phenotype['tipologia']
        freq = phenotype['frequenza_settimanale']
        
        if is_engaged:
            # Allineamento: L'engaged vuole mantenere l'abitudine (Promemoria)
            if tipo == 'Promemoria': score += 0.2
            if 7 <= freq <= 14: score += 0.2
            if freq > 25: score -= 0.3 # Anche l'engaged si stanca
        elif is_at_risk:
            # Allineamento: Chi è in crisi ha bisogno di motivazione o info
            if tipo in ['Motivazionale', 'Informativa']: score += 0.3
            if 3 <= freq <= 7: score += 0.2
            if freq > 10: score -= 0.2
        elif is_ghost:
            # Allineamento: Chi è sparito va recuperato con cautela
            if tipo == 'Motivazionale': score += 0.3
            if freq <= 2: score += 0.2
            if freq > 5: score -= 0.4 # Effetto spam garantito
        elif is_moderato:
            if tipo == 'Questionario': score += 0.2
            if tipo == 'Promemoria': score += 0.1
            if 4 <= freq <= 10: score += 0.2
            if freq > 25: score -= 0.3
            
        if len(phenotype['orari_attivi']) == 0:
            score = 0.0
            
        return max(0.0, min(1.0, score))


class GeneticAlgorithm:
    """Implementa il loop evolutivo con operatori configurabili (Fase 1 e 2)."""
    
    def __init__(self, evaluator: FitnessEvaluator, params: GAParams, rng: np.random.Generator = None):
        self.evaluator = evaluator
        self.params = params
        self.evaluator.params = self.params
        self.population: List[Chromosome] = []
        self.rng = rng if rng is not None else np.random.default_rng(self.params.seed)
        
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": []
        }
        self.current_diversity = 1.0

    def initialize_population(self):
        self.population = []
        for _ in range(self.params.pop_size):
            # Passiamo l'RNG locale per evitare leakage stocastico
            self.population.append(Chromosome(rng=self.rng))
        self._evaluate_population()

    def _evaluate_population(self):
        # Valuta solo chi non ha la fitness (Risparmio computazionale)
        for ind in self.population:
            if ind.fitness is None:
                ind.fitness = self.evaluator.evaluate(ind)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    # --- SELECTION METHODS ---
    def _select(self) -> Chromosome:
        method = self.params.selection_method.lower()
        if method == "tournament": return self._selection_tournament()
        elif method == "roulette": return self._selection_roulette()
        elif method == "ranking": return self._selection_ranking()
        elif method == "truncation": return self._selection_truncation()
        else: return self._selection_roulette()

    def _selection_tournament(self) -> Chromosome:
        candidates = self.rng.choice(self.population, size=self.params.tournament_size, replace=False)
        best = max(candidates, key=lambda x: x.fitness)
        new_ind = Chromosome(bits=best.bits.copy())
        new_ind.fitness = best.fitness
        return new_ind

    def _selection_roulette(self) -> Chromosome:
        fitnesses = np.array([max(0, ind.fitness) for ind in self.population])
        total = sum(fitnesses)
        if total == 0: 
            idx = self.rng.integers(0, len(self.population))
            new_ind = Chromosome(bits=self.population[idx].bits.copy())
            new_ind.fitness = self.population[idx].fitness
            return new_ind
        probs = fitnesses / total
        idx = self.rng.choice(len(self.population), p=probs)
        new_ind = Chromosome(bits=self.population[idx].bits.copy())
        new_ind.fitness = self.population[idx].fitness
        return new_ind

    def _selection_truncation(self) -> Chromosome:
        cutoff = max(1, int(len(self.population) * self.params.truncation_rate))
        best_set = self.population[:cutoff]
        idx = self.rng.integers(0, len(best_set))
        new_ind = Chromosome(bits=best_set[idx].bits.copy())
        new_ind.fitness = best_set[idx].fitness
        return new_ind

    def _selection_ranking(self) -> Chromosome:
        n = len(self.population)
        ranks = np.arange(n, 0, -1) 
        total_ranks = sum(ranks)
        probs = ranks / total_ranks
        idx = self.rng.choice(n, p=probs)
        new_ind = Chromosome(bits=self.population[idx].bits.copy())
        new_ind.fitness = self.population[idx].fitness
        return new_ind

    # --- CROSSOVER METHODS ---
    def _crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() > self.params.crossover_rate:
            # RISPARMIO COMPTUAZIONALE: Se non c'è crossover, passiamo la fitness in eredità intatta
            c1, c2 = Chromosome(bits=p1.bits.copy()), Chromosome(bits=p2.bits.copy())
            c1.fitness, c2.fitness = p1.fitness, p2.fitness 
            return c1, c2
            
        method = self.params.crossover_method.lower()
        if method == "single-point": return self._crossover_1point(p1, p2)
        elif method == "two-point": return self._crossover_2point(p1, p2)
        elif method == "uniform": return self._crossover_uniform(p1, p2)
        elif method == "k-point": return self._crossover_kpoint(p1, p2)
        else: return self._crossover_1point(p1, p2)

    def _crossover_1point(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pt = self.rng.integers(1, Chromosome.TOTAL_LENGTH)
        c1 = np.concatenate((p1.bits[:pt], p2.bits[pt:]))
        c2 = np.concatenate((p2.bits[:pt], p1.bits[pt:]))
        return Chromosome(bits=c1), Chromosome(bits=c2)

    def _crossover_2point(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pt1 = self.rng.integers(1, Chromosome.TOTAL_LENGTH - 1)
        pt2 = self.rng.integers(pt1 + 1, Chromosome.TOTAL_LENGTH)
        c1 = np.concatenate((p1.bits[:pt1], p2.bits[pt1:pt2], p1.bits[pt2:]))
        c2 = np.concatenate((p2.bits[:pt1], p1.bits[pt1:pt2], p2.bits[pt2:]))
        return Chromosome(bits=c1), Chromosome(bits=c2)

    def _crossover_uniform(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        mask = self.rng.integers(0, 2, Chromosome.TOTAL_LENGTH)
        c1 = np.where(mask == 1, p1.bits, p2.bits)
        c2 = np.where(mask == 1, p2.bits, p1.bits)
        return Chromosome(bits=c1), Chromosome(bits=c2)

    def _crossover_kpoint(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pts = sorted(self.rng.choice(range(1, Chromosome.TOTAL_LENGTH), size=self.params.k_points, replace=False))
        pts = [0] + list(pts) + [Chromosome.TOTAL_LENGTH]
        c1_bits, c2_bits = [], []
        for i in range(len(pts)-1):
            if i % 2 == 0:
                c1_bits.append(p1.bits[pts[i]:pts[i+1]])
                c2_bits.append(p2.bits[pts[i]:pts[i+1]])
            else:
                c1_bits.append(p2.bits[pts[i]:pts[i+1]])
                c2_bits.append(p1.bits[pts[i]:pts[i+1]])
        return Chromosome(bits=np.concatenate(c1_bits)), Chromosome(bits=np.concatenate(c2_bits))

    # --- MUTATION METHODS ---
    def _mutate(self, ind: Chromosome) -> None:
        """Esegue la mutazione. Azzera la fitness SOLO se c'è stato un reale cambiamento del DNA."""
        method = self.params.mutation_method.lower()
        mutated = False
        
        if method == "flip-bit": mutated = self._mutation_flip(ind)
        elif method == "multi-bit": mutated = self._mutation_multi(ind)
        elif method == "adaptive": mutated = self._mutation_adaptive(ind)
        else: mutated = self._mutation_flip(ind)

        if mutated:
            ind.fitness = None # Invalida la cache

    def _mutation_flip(self, ind: Chromosome) -> bool:
        mutated = False
        for i in range(Chromosome.TOTAL_LENGTH):
            if self.rng.random() < self.params.mutation_rate:
                ind.bits[i] = 1 - ind.bits[i]
                mutated = True
        return mutated

    def _mutation_multi(self, ind: Chromosome) -> bool:
        k = self.rng.integers(1, 4, endpoint=True)
        indices = self.rng.choice(range(Chromosome.TOTAL_LENGTH), size=k, replace=False)
        for idx in indices:
            ind.bits[idx] = 1 - ind.bits[idx]
        return True

    def _mutation_adaptive(self, ind: Chromosome) -> bool:
        adj_rate = self.params.mutation_rate
        if self.current_diversity < 0.1:
            adj_rate *= 2.0  
        elif self.current_diversity > 0.4:
            adj_rate *= 0.5  
            
        mutated = False
        for i in range(Chromosome.TOTAL_LENGTH):
            if self.rng.random() < adj_rate:
                ind.bits[i] = 1 - ind.bits[i]
                mutated = True
        return mutated

    def _calculate_diversity(self) -> float:
        pop_matrix = np.array([ind.bits for ind in self.population])
        p1 = pop_matrix.mean(axis=0)
        avg_hamming = np.sum(2 * p1 * (1 - p1))
        return float(avg_hamming / Chromosome.TOTAL_LENGTH)

    def run(self):
        self.initialize_population()
        n_elites = max(1, int(self.params.pop_size * self.params.elitism_rate))
        
        for gen in range(self.params.generations):
            self.current_diversity = self._calculate_diversity()
            new_population = []
            
            # Elitarismo: i migliori passano incondizionatamente con la fitness già calcolata!
            for i in range(n_elites):
                elite_ind = Chromosome(bits=self.population[i].bits.copy())
                elite_ind.fitness = self.population[i].fitness
                new_population.append(elite_ind)
                
            # Riproduzione
            while len(new_population) < self.params.pop_size:
                p1 = self._select()
                p2 = self._select()
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_population.extend([c1, c2])
                
            self.population = new_population[:self.params.pop_size]
            self._evaluate_population()
            
            # Statistiche
            best_fit = self.population[0].fitness
            avg_fit = sum(ind.fitness for ind in self.population) / self.params.pop_size
            self.history["best_fitness"].append(best_fit)
            self.history["avg_fitness"].append(avg_fit)
            self.history["diversity"].append(self.current_diversity)
            
            # Early Stopping (Fase 1.5)
            if gen >= self.params.patience:
                recent_bests = self.history["best_fitness"][-self.params.patience:]
                if (recent_bests[-1] - recent_bests[0]) < 1e-6:
                    logger.debug(f"Early Stopping alla generazione {gen} per mancanza di miglioramento.")
                    break
                    
        return self.population[0]