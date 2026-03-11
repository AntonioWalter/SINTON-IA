"""
SINTON-IA – Generatore di Dati Sintetici per il Churn Prevention
=============================================================================
Questo script genera record comportamentali simulati che riproducono
l'interazione di pazienti fittizi con la piattaforma SINTONIA.

I dati prodotti servono a validare il funzionamento dell'Algoritmo Genetico
in fase prototipale, prima che siano disponibili dati reali di utilizzo.

Tabelle generate (CSV):
    - patients.csv        : anagrafica pazienti con profilo assegnato
    - stato_animo.csv     : inserimenti di stato d'animo
    - pagina_diario.csv   : pagine del diario personale
    - questionario.csv    : compilazioni di questionari clinici
    - notifica.csv        : notifiche inviate e relativo stato di lettura
    - domanda_forum.csv   : domande postate nel forum comunitario
    - acquisizione_badge.csv: badge della gamification ottenuti

Profili utente simulati:
    - Engaged   (30%) : utilizzo costante e motivato
    - Moderato  (30%) : utilizzo intermittente
    - A Rischio (25%) : pattern pre-abbandono
    - Ghost     (15%) : attività cessata

Uso:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --patients 1000 --days 120
"""

import os
import uuid
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
#  Costanti e configurazione
# ─────────────────────────────────────────────────────────────────────────────

PROFILI = {
    "Engaged":   0.30,
    "Moderato":  0.30,
    "A Rischio": 0.25,
    "Ghost":     0.15,
}

# I 10 stati emotivi definiti nell'enum 'umore' del DB SINTONIA
UMORI = [
    "Felice", "Sereno", "Energico", "Neutro", "Stanco",
    "Triste", "Ansioso", "Arrabbiato", "Spaventato", "Confuso",
]

# Valenza numerica per ciascun umore (scala bipolare [-1, +1])
VALENZA_UMORE = {
    "Felice": 1.0, "Sereno": 0.7, "Energico": 0.8,
    "Neutro": 0.0, "Stanco": -0.3,
    "Triste": -0.8, "Ansioso": -0.6, "Arrabbiato": -0.7,
    "Spaventato": -0.9, "Confuso": -0.4,
}

# Probabilità di scegliere umori positivi/negativi per profilo
MOOD_WEIGHTS = {
    "Engaged":   [0.25, 0.20, 0.15, 0.15, 0.05, 0.05, 0.05, 0.03, 0.02, 0.05],
    "Moderato":  [0.10, 0.12, 0.08, 0.20, 0.12, 0.12, 0.10, 0.06, 0.04, 0.06],
    "A Rischio": [0.03, 0.05, 0.02, 0.10, 0.15, 0.20, 0.18, 0.12, 0.08, 0.07],
    "Ghost":     [0.05, 0.05, 0.05, 0.15, 0.15, 0.18, 0.15, 0.10, 0.07, 0.05],
}

# Frequenza di somministrazione questionari (ogni N giorni)
FREQUENZA_QUESTIONARI = 14

# Tipologie di questionario presenti in SINTONIA
TIPOLOGIE_QUESTIONARIO = ["PHQ-9", "GAD-7", "PC-PTSD-5", "WHO-5"]

# Tipologie di notifica
TIPOLOGIE_NOTIFICA = ["Promemoria", "Motivazionale", "Informativa", "Questionario"]

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
#  Parametri comportamentali per ciascun profilo
# ─────────────────────────────────────────────────────────────────────────────

PROFILE_PARAMS = {
    "Engaged": {
        "mood_daily_prob":       (0.80, 0.95),   # prob. di inserire mood ogni giorno
        "diary_daily_prob":      (0.40, 0.70),   # prob. di scrivere nel diario
        "diary_length_range":    (200, 1800),     # lunghezza testo (caratteri)
        "questionnaire_compliance": (0.90, 1.00), # prob. di compilare il questionario
        "questionnaire_score":   (2, 12),         # range score PHQ-9
        "notification_read_prob": (0.70, 0.95),   # prob. di leggere la notifica
        "notifications_per_week": (2, 4),         # notifiche ricevute a settimana
        "forum_weekly_prob":     (0.15, 0.35),    # prob. di postare nel forum a settimana
        "ghost_after_day":       None,            # mai diventa ghost
        "badges_total_range":    (10, 30),        # range totale badge acquisiti
    },
    "Moderato": {
        "mood_daily_prob":       (0.40, 0.65),
        "diary_daily_prob":      (0.15, 0.35),
        "diary_length_range":    (80, 800),
        "questionnaire_compliance": (0.65, 0.85),
        "questionnaire_score":   (5, 18),
        "notification_read_prob": (0.40, 0.65),
        "notifications_per_week": (2, 5),
        "forum_weekly_prob":     (0.05, 0.15),
        "ghost_after_day":       None,
        "badges_total_range":    (5, 15),
    },
    "A Rischio": {
        "mood_daily_prob":       (0.50, 0.70),    # inizia ok, poi cala
        "diary_daily_prob":      (0.20, 0.40),
        "diary_length_range":    (30, 500),
        "questionnaire_compliance": (0.40, 0.70),
        "questionnaire_score":   (10, 24),
        "notification_read_prob": (0.15, 0.35),
        "notifications_per_week": (3, 6),
        "forum_weekly_prob":     (0.02, 0.08),
        "ghost_after_day":       (0.50, 0.80),    # diventa inattivo dopo 50-80% dei giorni
        "badges_total_range":    (2, 8),
    },
    "Ghost": {
        "mood_daily_prob":       (0.30, 0.50),    # attività iniziale breve
        "diary_daily_prob":      (0.05, 0.15),
        "diary_length_range":    (10, 200),
        "questionnaire_compliance": (0.20, 0.50),
        "questionnaire_score":   (8, 22),
        "notification_read_prob": (0.05, 0.20),
        "notifications_per_week": (2, 5),
        "forum_weekly_prob":     (0.00, 0.05),
        "ghost_after_day":       (0.10, 0.35),    # sparisce molto presto (10-35% dei giorni)
        "badges_total_range":    (0, 3),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Funzioni di generazione
# ─────────────────────────────────────────────────────────────────────────────

def _random_time(day: datetime, is_night_owl: bool = False) -> datetime:
    """Aggiunge un orario casuale alla data. Se night_owl, può essere attivo h24."""
    if is_night_owl and random.random() < 0.4:
        # 40% di probabilità di attività notturna (23-07)
        hour = random.choice([23, 0, 1, 2, 3, 4, 5, 6])
    else:
        # Altrimenti orario standard (07-22)
        hour = random.randint(7, 22)
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=second)





def generate_patients(n_patients: int, days_range: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera la tabella anagrafica dei pazienti con profilo assegnato."""
    profiles = []
    for profile, pct in PROFILI.items():
        profiles.extend([profile] * int(n_patients * pct))

    # Compensazione per arrotondamento
    while len(profiles) < n_patients:
        profiles.append(rng.choice(list(PROFILI.keys())))
    profiles = profiles[:n_patients]
    rng.shuffle(profiles)

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sessi = ["M", "F", "Altro"]
    priorita = ["Urgente", "Breve", "Differibile", "Programmabile"]

    patients = []
    for i in range(n_patients):
        # Ogni paziente ha una data_ingresso randomica entro il range
        days_ago = rng.integers(days_range // 2, days_range)
        data_ingresso = today - timedelta(days=int(days_ago))
        
        # 20% dei pazienti sono gufi notturni (night owls)
        is_night_owl = rng.random() < 0.2

        # Cutoff unificato: un singolo giorno di disengagement condiviso
        # da tutte le tabelle per garantire coerenza narrativa
        ghost_range = PROFILE_PARAMS[profiles[i]].get("ghost_after_day")
        if ghost_range is not None:
            cutoff_pct = rng.uniform(ghost_range[0], ghost_range[1])
            ghost_cutoff_day = int(int(days_ago) * cutoff_pct)
        else:
            ghost_cutoff_day = int(days_ago) + 1  # Mai cutoff

        patients.append({
            "id_paziente": str(uuid.uuid4()),
            "profilo": profiles[i],
            "data_ingresso": data_ingresso.strftime("%Y-%m-%d"),
            "sesso": rng.choice(sessi),
            "id_priorita": rng.choice(priorita, p=[0.10, 0.25, 0.35, 0.30]),
            "stato": True,
            "days_in_platform": int(days_ago),
            "is_night_owl": is_night_owl,
            "ghost_cutoff_day": ghost_cutoff_day,
        })

    return pd.DataFrame(patients)


def generate_mood_entries(patients_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera i record di stato_animo per ogni paziente."""
    records = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for _, patient in patients_df.iterrows():
        profile = patient["profilo"]
        params = PROFILE_PARAMS[profile]
        total_days = patient["days_in_platform"]

        # Prob. giornaliera di inserire il mood (sampled once per paziente)
        daily_prob = rng.uniform(*params["mood_daily_prob"])

        # Cutoff unificato dal DataFrame pazienti
        cutoff_day = patient["ghost_cutoff_day"]

        weights = MOOD_WEIGHTS[profile]

        for day_idx in range(total_days):
            # Dopo il cutoff, l'utente smette
            if day_idx >= cutoff_day:
                break

            # Decadimento graduale per "A Rischio"
            if profile == "A Rischio" and cutoff_day > 0:
                progress = day_idx / cutoff_day
                adjusted_prob = daily_prob * (1.0 - 0.6 * progress)
            else:
                adjusted_prob = daily_prob

            if rng.random() < adjusted_prob:
                day_date = today - timedelta(days=(total_days - day_idx))
                umore = rng.choice(UMORI, p=weights)
                intensita = rng.integers(1, 11)  # 1-10
                
                is_night_owl = patient.get("is_night_owl", False)

                records.append({
                    "id_stato_animo": str(uuid.uuid4()),
                    "umore": umore,
                    "intensita": int(intensita),
                    "data_inserimento": _random_time(day_date, is_night_owl).strftime("%Y-%m-%d %H:%M:%S"),
                    "id_paziente": patient["id_paziente"],
                })

    return pd.DataFrame(records)


def generate_diary_entries(patients_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera le pagine del diario per ogni paziente."""
    records = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for _, patient in patients_df.iterrows():
        profile = patient["profilo"]
        params = PROFILE_PARAMS[profile]
        total_days = patient["days_in_platform"]

        daily_prob = rng.uniform(*params["diary_daily_prob"])
        len_min, len_max = params["diary_length_range"]

        # Cutoff unificato dal DataFrame pazienti
        cutoff_day = patient["ghost_cutoff_day"]

        for day_idx in range(total_days):
            if day_idx >= cutoff_day:
                break

            if profile == "A Rischio" and cutoff_day > 0:
                progress = day_idx / cutoff_day
                adjusted_prob = daily_prob * (1.0 - 0.7 * progress)
            else:
                adjusted_prob = daily_prob

            if rng.random() < adjusted_prob:
                day_date = today - timedelta(days=(total_days - day_idx))
                text_length = int(rng.integers(len_min, len_max + 1))
                testo = "A" * text_length

                records.append({
                    "id_pagina_diario": str(uuid.uuid4()),
                    "titolo": f"Pagina del {day_date.strftime('%d/%m/%Y')}",
                    "testo": testo,
                    "data_inserimento": _random_time(day_date).strftime("%Y-%m-%d %H:%M:%S"),
                    "id_paziente": patient["id_paziente"],
                })

    return pd.DataFrame(records)


def generate_questionnaires(patients_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera le compilazioni dei questionari clinici."""
    records = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for _, patient in patients_df.iterrows():
        profile = patient["profilo"]
        params = PROFILE_PARAMS[profile]
        total_days = patient["days_in_platform"]
        compliance_prob = rng.uniform(*params["questionnaire_compliance"])
        score_min, score_max = params["questionnaire_score"]

        # Cutoff unificato dal DataFrame pazienti
        cutoff_day = patient["ghost_cutoff_day"]

        # I questionari vengono somministrati ogni FREQUENZA_QUESTIONARI giorni
        for day_idx in range(0, total_days, FREQUENZA_QUESTIONARI):
            if day_idx >= cutoff_day:
                break

            # L'utente compila o no in base alla compliance
            compiled = rng.random() < compliance_prob
            day_date = today - timedelta(days=(total_days - day_idx))
            tipologia = rng.choice(TIPOLOGIE_QUESTIONARIO)

            if compiled:
                score = float(rng.integers(score_min, score_max + 1))
                # Piccola probabilità di invalidazione
                invalidato = bool(rng.random() < 0.05) if profile in ("A Rischio", "Ghost") else False

                records.append({
                    "id_questionario": str(uuid.uuid4()),
                    "score": score,
                    "data_compilazione": _random_time(day_date).strftime("%Y-%m-%d %H:%M:%S"),
                    "revisionato": bool(rng.random() < 0.3),
                    "invalidato": invalidato,
                    "nome_tipologia": tipologia,
                    "id_paziente": patient["id_paziente"],
                })

    return pd.DataFrame(records)


def generate_notifications(patients_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera le notifiche inviate ai pazienti con stato di lettura."""
    records = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for _, patient in patients_df.iterrows():
        profile = patient["profilo"]
        params = PROFILE_PARAMS[profile]
        total_days = patient["days_in_platform"]
        read_prob = rng.uniform(*params["notification_read_prob"])
        notif_min, notif_max = params["notifications_per_week"]

        # Numero totale di settimane
        total_weeks = max(1, total_days // 7)

        for week_idx in range(total_weeks):
            n_notif = int(rng.integers(notif_min, notif_max + 1))

            for _ in range(n_notif):
                # Giorno casuale nella settimana
                day_offset = week_idx * 7 + int(rng.integers(0, 7))
                if day_offset >= total_days:
                    continue

                day_date = today - timedelta(days=(total_days - day_offset))
                tipologia = rng.choice(TIPOLOGIE_NOTIFICA)
                
                is_night_owl = patient.get("is_night_owl", False)

                records.append({
                    "id_notifica": str(uuid.uuid4()),
                    "titolo": f"Notifica {tipologia}",
                    "tipologia": tipologia,
                    "data_invio": _random_time(day_date, is_night_owl).strftime("%Y-%m-%d %H:%M:%S"),
                    "letto": bool(rng.random() < read_prob),
                    "id_paziente": patient["id_paziente"],
                })

    return pd.DataFrame(records)


def generate_forum_entries(patients_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera i post nel forum per ogni paziente."""
    records = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for _, patient in patients_df.iterrows():
        profile = patient["profilo"]
        params = PROFILE_PARAMS[profile]
        total_days = patient["days_in_platform"]
        forum_prob = rng.uniform(*params["forum_weekly_prob"])
        
        total_weeks = max(1, total_days // 7)

        for week_idx in range(total_weeks):
            # L'utente posta nel forum o no in base alla probabilità settimanale
            if rng.random() < forum_prob:
                day_offset = week_idx * 7 + int(rng.integers(0, 7))
                if day_offset >= total_days:
                    continue

                day_date = today - timedelta(days=(total_days - day_offset))

                records.append({
                    "id_domanda": str(uuid.uuid4()),
                    "titolo": f"Domanda dal paziente {profile}",
                    "testo": "Testo della domanda...",
                    "categoria": "Supporto Generale",
                    "data_inserimento": _random_time(day_date).strftime("%Y-%m-%d %H:%M:%S"),
                    "id_paziente": patient["id_paziente"],
                })

    return pd.DataFrame(records)


def generate_badges(patients_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Generazione dinamica di badge fittizi (quanti ne servono a seconda del max)
    max_possible_badges = max(params.get("badges_total_range", (0, 30))[1] for params in PROFILE_PARAMS.values())
    BADGE_NAMES = [f"Badge_Livello_{i}" for i in range(1, max_possible_badges + 1)]
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    records = []

    for _, patient in patients_df.iterrows():
        profile = patient["profilo"]
        params = PROFILE_PARAMS[profile]
        total_days = patient["days_in_platform"]
        b_min, b_max = params["badges_total_range"]
        
        n_badges = int(rng.integers(b_min, b_max + 1))
        
        # Un paziente non puo' acquisire lo stesso badge due volte
        n_badges = min(n_badges, len(BADGE_NAMES))
        acquired_badges = rng.choice(BADGE_NAMES, size=n_badges, replace=False)
        
        for badge_name in acquired_badges:
            day_offset = int(rng.integers(0, total_days))
            day_date = today - timedelta(days=(total_days - day_offset))
            
            records.append({
                "data_acquisizione": _random_time(day_date).strftime("%Y-%m-%d %H:%M:%S"),
                "nome_badge": badge_name,
                "id_paziente": patient["id_paziente"],
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_patients: int = 500, days_range: int = 90):
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # Percorso output: models/churn_prevention/data/synthetic/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  SINTON-IA — Generatore Dati Sintetici (Churn Prevention)")
    print("=" * 65)
    print(f"\n  Pazienti:  {n_patients}")
    print(f"  Giorni:    {days_range}")
    print(f"  Profili:   {', '.join(f'{k} ({v:.0%})' for k, v in PROFILI.items())}")
    print(f"  Seed:      {SEED}")
    print(f"  Output:    {os.path.abspath(output_dir)}\n")

    # 1. Genera pazienti
    print("[*] Generazione pazienti...")
    patients_df = generate_patients(n_patients, days_range, rng)
    patients_path = os.path.join(output_dir, "patients.csv")
    patients_df.to_csv(patients_path, index=False)
    counts = patients_df["profilo"].value_counts()
    for profile in PROFILI:
        print(f"   {profile:12s}: {counts.get(profile, 0):>4d} pazienti")

    # 2. Genera stato d'animo
    print("\n[*] Generazione stati d'animo (mood tracking)...")
    mood_df = generate_mood_entries(patients_df, rng)
    mood_path = os.path.join(output_dir, "stato_animo.csv")
    mood_df.to_csv(mood_path, index=False)
    print(f"   [OK] {len(mood_df):,} record generati")

    # 3. Genera pagine diario
    print("\n[*] Generazione pagine del diario...")
    diary_df = generate_diary_entries(patients_df, rng)
    diary_path = os.path.join(output_dir, "pagina_diario.csv")
    diary_df.to_csv(diary_path, index=False)
    print(f"   [OK] {len(diary_df):,} record generati")

    # 4. Genera questionari
    print("\n[*] Generazione questionari clinici...")
    quest_df = generate_questionnaires(patients_df, rng)
    quest_path = os.path.join(output_dir, "questionario.csv")
    quest_df.to_csv(quest_path, index=False)
    total = len(quest_df)
    print(f"   [OK] {total:,} somministrazioni compilate")

    # 5. Genera notifiche
    print("\n[*] Generazione notifiche...")
    notif_df = generate_notifications(patients_df, rng)
    notif_path = os.path.join(output_dir, "notifica.csv")
    notif_df.to_csv(notif_path, index=False)
    read_count = notif_df["letto"].sum()
    print(f"   [OK] {len(notif_df):,} notifiche ({read_count:,} lette, {len(notif_df) - read_count:,} ignorate)")

    # 6. Genera forum
    print("\n[*] Generazione post nel forum...")
    forum_df = generate_forum_entries(patients_df, rng)
    forum_path = os.path.join(output_dir, "domanda_forum.csv")
    forum_df.to_csv(forum_path, index=False)
    print(f"   [OK] {len(forum_df):,} domande nel forum generate")

    # 7. Genera badges
    print("\n[*] Generazione acquisizione badge...")
    badges_df = generate_badges(patients_df, rng)
    badges_path = os.path.join(output_dir, "acquisizione_badge.csv")
    badges_df.to_csv(badges_path, index=False)
    print(f"   [OK] {len(badges_df):,} badge assegnati")

    # Riepilogo
    print("\n" + "=" * 65)
    print("  [OK] GENERAZIONE COMPLETATA")
    print("=" * 65)
    print(f"\n  File salvati in: {os.path.abspath(output_dir)}")
    print(f"     - patients.csv       ({len(patients_df):>6,} record)")
    print(f"     - stato_animo.csv    ({len(mood_df):>6,} record)")
    print(f"     - pagina_diario.csv  ({len(diary_df):>6,} record)")
    print(f"     - questionario.csv   ({len(quest_df):>6,} record)")
    print(f"     - notifica.csv       ({len(notif_df):>6,} record)")
    print(f"     - domanda_forum.csv  ({len(forum_df):>6,} record)")
    print(f"     - acquisizione_badge.csv ({len(badges_df):>6,} record)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generatore di dati sintetici per il GA di Churn Prevention (SINTON-IA)"
    )
    parser.add_argument("--patients", type=int, default=500,
                        help="Numero di pazienti simulati (default: 500)")
    parser.add_argument("--days", type=int, default=90,
                        help="Giorni di storico simulato (default: 90)")

    args = parser.parse_args()
    main(n_patients=args.patients, days_range=args.days)
