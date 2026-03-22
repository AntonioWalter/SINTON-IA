from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
import numpy as np

# --- IMPORTA IL TUO ALGORITMO GENETICO ---
from genetic_algorithm import GAParams, FitnessEvaluator, GeneticAlgorithm

# 1. Setup Iniziale e Download NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = FastAPI(title="SINTON-IA Multi-Model API")

# 2. Caricamento dei Modelli Addestrati
MODELS = {}

def load_models():
    try:
        # Modello 1: Rischio Suicidario (NLP)
        MODELS['suicide_vec'] = joblib.load('tfidf_vectorizer.pkl')
        MODELS['suicide_model'] = joblib.load('logreg_model.pkl')
        if not hasattr(MODELS['suicide_model'], 'multi_class'):
            MODELS['suicide_model'].multi_class = 'auto'
        
        # Modello 2: Depression Prediction (LightGBM)
        if os.path.exists('final_model_depression.pkl'):
            MODELS['depression_model'] = joblib.load('final_model_depression.pkl')
        
        print("✅ Tutti i modelli caricati con successo!")
    except Exception as e:
        print(f"❌ Errore nel caricamento dei modelli: {e}")

load_models()

# ==========================================
# MODELLO 1: RED FLAG (SUI RISK)
# ==========================================
default_stopwords = set(stopwords.words('english'))
words_to_keep = {'not', 'no', 'nor', 'don', "don't", "isn't", "wasn't", 'never'}
custom_stopwords = default_stopwords - words_to_keep
lemmatizer = WordNetLemmatizer()

def clean_text_pipeline(text: str) -> str:
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\.,!\?]', '', text)
    text = re.sub(r'([\.,!\?])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) if w not in ['.',',','!','?'] else w for w in words if w not in custom_stopwords ]
    return ' '.join(cleaned_words)

class RedFlagRequest(BaseModel):
    testo: str

@app.post("/api/red-flag")
async def analyze_red_flag(request: RedFlagRequest):
    cleaned = clean_text_pipeline(request.testo)
    X = MODELS['suicide_vec'].transform([cleaned])
    prob = MODELS['suicide_model'].predict_proba(X)[0][1]
    return {"risk_detected": bool(prob >= 0.2384), "probability": float(prob)}

# ==========================================
# MODELLO 2: PREDIZIONE DEPRESSIONE
# ==========================================
VA_MAP = {
    "Felice": 0.85, "Sereno": 0.7, "Energico": 0.5, "Neutro": 0.0,
    "Stanco": -0.2, "Triste": -0.8, "Ansioso": -0.55, "Arrabbiato": -0.7,
    "Spaventato": -0.65, "Confuso": -0.3
}

DEPRESSION_SCALER = {
    "valence_mean": {"mean": 0.1564, "std": 0.2384},
    "valence_std": {"mean": 0.3107, "std": 0.0866},
    "valence_ema_3d": {"mean": 0.1433, "std": 0.3104},
    "valence_trend_5d": {"mean": -0.0001, "std": 0.1230},
    "max_neg_streak": {"mean": 2.5505, "std": 2.0963},
    "missing_ratio": {"mean": 0.1056, "std": 0.1198},
    "intensity_mean": {"mean": 4.9645, "std": 0.7972},
    "dominant_mood_valence": {"mean": 0.0586, "std": 0.2910}
}
FEAT_ORDER = ["valence_mean", "valence_std", "valence_ema_3d", "valence_trend_5d", "max_neg_streak", "missing_ratio", "intensity_mean", "dominant_mood_valence"]

class DailyLog(BaseModel):
    mood_state: str
    valence: float
    intensity: float
    is_missing: bool = False

class DepressionRequest(BaseModel):
    logs: List[DailyLog]

def extract_depression_features(logs: List[DailyLog]):
    df = pd.DataFrame([l.dict() for l in logs])
    present = df[~df['is_missing']]
    if len(present) < 3:
        raise HTTPException(status_code=400, detail="Dati insufficienti (minimo 3 log validi)")
    valences = present['valence'].values
    intensities = present['intensity'].values
    
    f = {}
    f['valence_mean'] = np.mean(valences)
    f['valence_std'] = np.std(valences) if len(valences) > 1 else 0.0
    f['valence_ema_3d'] = valences[-1]
    
    if len(valences) >= 2:
        x = np.arange(len(valences))
        f['valence_trend_5d'], _ = np.polyfit(x, valences, 1)
    else: 
        f['valence_trend_5d'] = 0.0
        
    max_neg = 0; curr_neg = 0
    for v in valences:
        if v < 0: 
            curr_neg += 1
            max_neg = max(max_neg, curr_neg)
        else: 
            curr_neg = 0
    f['max_neg_streak'] = max_neg
    f['missing_ratio'] = (len(df) - len(present)) / len(df)
    f['intensity_mean'] = np.mean(intensities)
    f['dominant_mood_valence'] = VA_MAP.get(present['mood_state'].mode()[0], 0.0)
    
    scaled = []
    for k in FEAT_ORDER:
        val = (f[k] - DEPRESSION_SCALER[k]['mean']) / DEPRESSION_SCALER[k]['std']
        scaled.append(val)
    return np.array(scaled).reshape(1, -1)

@app.post("/api/predict-depression")
async def predict_depression(request: DepressionRequest):
    try:
        features = extract_depression_features(request.logs)
        prediction = MODELS['depression_model'].predict(features)[0]
        return {"phq9_score": float(prediction), "risk_level": "High" if prediction > 15 else "Normal"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MODELLO 3: GENETIC ALGORITHM (CHURN PREVENTION)
# ==========================================
class GARequest(BaseModel):
    mood_frequency_7d: float
    avg_mood_valence_7d: float
    notification_read_rate: float
    night_activity_rate: float
    
    # "extra = allow" permette al backend NestJS di inviare anche eventuali altre feature 
    # (es. badges_total, profil_assegnato) che il tuo script Python si aspetta, senza dare errore
    class Config:
        extra = "allow" 

@app.post("/api/genetic-algorithm")
async def run_genetic_algorithm(request: GARequest):
    try:
        # 1. Carica il Gold Standard generato da Optuna
        params = GAParams.load_gold_standard_config('ga_tuned_config.json')
        
        # 2. Trasforma il JSON in arrivo in una Serie Pandas
        patient_features = pd.Series(request.dict())
        
        # 3. Inizializza l'algoritmo usando il tuo file esterno
        rng = np.random.default_rng(params.seed)
        evaluator = FitnessEvaluator(patient_features, params, rng=rng)
        ga = GeneticAlgorithm(evaluator, params, rng=rng)
        
        # 4. Trova la strategia perfetta per la settimana!
        best_chromosome = ga.run()
        best_strategy = best_chromosome.decode()
        
        return best_strategy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ROOT / HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {"status": "SINTON-IA Multi-Model Engine (RedFlag, Depression, GA) is awake and running."}