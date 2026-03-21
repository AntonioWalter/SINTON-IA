"""
SINTON-IA – Data Pipeline Unificata per Churn Prevention
=============================================================================
Questo script consolida la generazione di dati sintetici e l'aggregazione di
feature in un unico workflow, garantendo coerenza e semplicità.
"""

import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
#  Costanti e Parametri Profilo (da generate_synthetic_data.py)
# ─────────────────────────────────────────────────────────────────────────────

PROFILI = {
    "Engaged":   0.30,
    "Moderato":  0.30,
    "A Rischio": 0.25,
    "Ghost":     0.15,
}

UMORI = ["Felice", "Sereno", "Energico", "Neutro", "Stanco", "Triste", "Ansioso", "Arrabbiato", "Spaventato", "Confuso"]

VALENZA_UMORE = {
    "Felice": 1.0, "Sereno": 0.7, "Energico": 0.8, "Neutro": 0.0, "Stanco": -0.3,
    "Triste": -0.8, "Ansioso": -0.6, "Arrabbiato": -0.7, "Spaventato": -0.9, "Confuso": -0.4,
}

MOOD_WEIGHTS = {
    "Engaged":   [0.25, 0.20, 0.15, 0.15, 0.05, 0.05, 0.05, 0.03, 0.02, 0.05],
    "Moderato":  [0.10, 0.12, 0.08, 0.20, 0.12, 0.12, 0.10, 0.06, 0.04, 0.06],
    "A Rischio": [0.03, 0.05, 0.02, 0.10, 0.15, 0.20, 0.18, 0.12, 0.08, 0.07],
    "Ghost":     [0.05, 0.05, 0.05, 0.15, 0.15, 0.18, 0.15, 0.10, 0.07, 0.05],
}

PROFILE_PARAMS = {
    "Engaged": {
        "mood_daily_prob": (0.80, 0.95), "diary_daily_prob": (0.40, 0.70), "diary_length_range": (200, 1800),
        "questionnaire_compliance": (0.90, 1.00), "questionnaire_score": (2, 12), "notification_read_prob": (0.70, 0.95),
        "notifications_per_week": (2, 4), "forum_weekly_prob": (0.15, 0.35), "ghost_after_day": None, "badges_total_range": (10, 30),
    },
    "Moderato": {
        "mood_daily_prob": (0.40, 0.65), "diary_daily_prob": (0.15, 0.35), "diary_length_range": (80, 800),
        "questionnaire_compliance": (0.65, 0.85), "questionnaire_score": (5, 18), "notification_read_prob": (0.40, 0.65),
        "notifications_per_week": (2, 5), "forum_weekly_prob": (0.05, 0.15), "ghost_after_day": None, "badges_total_range": (5, 15),
    },
    "A Rischio": {
        "mood_daily_prob": (0.50, 0.70), "diary_daily_prob": (0.20, 0.40), "diary_length_range": (30, 500),
        "questionnaire_compliance": (0.40, 0.70), "questionnaire_score": (10, 24), "notification_read_prob": (0.15, 0.35),
        "notifications_per_week": (3, 6), "forum_weekly_prob": (0.02, 0.08), "ghost_after_day": (0.50, 0.80), "badges_total_range": (2, 8),
    },
    "Ghost": {
        "mood_daily_prob": (0.30, 0.50), "diary_daily_prob": (0.05, 0.15), "diary_length_range": (10, 200),
        "questionnaire_compliance": (0.20, 0.50), "questionnaire_score": (8, 22), "notification_read_prob": (0.05, 0.20),
        "notifications_per_week": (2, 5), "forum_weekly_prob": (0.00, 0.05), "ghost_after_day": (0.10, 0.35), "badges_total_range": (0, 3),
    },
}

class DataPipeline:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Generazione Dati Sintetici
    # ─────────────────────────────────────────────────────────────────────────

    def _random_time(self, day: datetime, is_night_owl: bool = False) -> datetime:
        if is_night_owl and random.random() < 0.4:
            hour = random.choice([23, 0, 1, 2, 3, 4, 5, 6])
        else:
            hour = random.randint(7, 22)
        return day.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))

    def run_generation(self, n_patients: int = 500, days_range: int = 90) -> Dict[str, pd.DataFrame]:
        print(f"[*] Generazione di {n_patients} pazienti per {days_range} giorni...")
        
        # 1. Patients
        profiles = []
        for p, pct in PROFILI.items():
            profiles.extend([p] * int(n_patients * pct))
        while len(profiles) < n_patients:
            profiles.append(self.rng.choice(list(PROFILI.keys())))
        profiles = profiles[:n_patients]
        self.rng.shuffle(profiles)

        patients = []
        for i in range(n_patients):
            days_ago = self.rng.integers(days_range // 2, days_range)
            data_ingresso = self.today - timedelta(days=int(days_ago))
            is_night_owl = self.rng.random() < 0.2
            ghost_range = PROFILE_PARAMS[profiles[i]].get("ghost_after_day")
            cutoff_day = int(int(days_ago) * self.rng.uniform(*ghost_range)) if ghost_range else int(days_ago) + 1
            
            patients.append({
                "id_paziente": str(uuid.uuid4()), "profilo": profiles[i], "data_ingresso": data_ingresso,
                "is_night_owl": is_night_owl, "days_in_platform": int(days_ago), "ghost_cutoff_day": cutoff_day
            })
        patients_df = pd.DataFrame(patients)

        # 2. Mood & Entries
        mood_records, diary_records, quest_records, notif_records, forum_records, badge_records = [], [], [], [], [], []
        
        for _, p in patients_df.iterrows():
            params = PROFILE_PARAMS[p["profilo"]]
            pid, is_owl, cutoff = p["id_paziente"], p["is_night_owl"], p["ghost_cutoff_day"]
            total_days = p["days_in_platform"]

            # Mood & Diary
            mood_prob = self.rng.uniform(*params["mood_daily_prob"])
            diary_prob = self.rng.uniform(*params["diary_daily_prob"])
            for d_idx in range(total_days):
                if d_idx >= cutoff:
                    break
                dt = self.today - timedelta(days=int(total_days - d_idx))
                
                if self.rng.random() < mood_prob:
                    mood_records.append({
                        "id_paziente": pid, "umore": self.rng.choice(UMORI, p=MOOD_WEIGHTS[p["profilo"]]),
                        "data_inserimento": self._random_time(dt, is_owl)
                    })
                if self.rng.random() < diary_prob:
                    diary_records.append({
                        "id_paziente": pid, "testo": "A" * int(self.rng.integers(*params["diary_length_range"])),
                        "data_inserimento": self._random_time(dt)
                    })

            # Questionnaires (every 14 days)
            q_prob = self.rng.uniform(*params["questionnaire_compliance"])
            for d_idx in range(0, total_days, 14):
                if d_idx >= cutoff:
                    break
                if self.rng.random() < q_prob:
                    dt = self.today - timedelta(days=int(total_days - d_idx))
                    quest_records.append({
                        "id_paziente": pid, "data_compilazione": self._random_time(dt)
                    })

            # Notifications
            n_min, n_max = params["notifications_per_week"]
            r_prob = self.rng.uniform(*params["notification_read_prob"])
            for w_idx in range(max(1, total_days // 7)):
                for _ in range(int(self.rng.integers(n_min, n_max + 1))):
                    d_off = w_idx * 7 + self.rng.integers(0, 7)
                    if d_off < total_days:
                        dt = self.today - timedelta(days=int(total_days - d_off))
                        notif_records.append({
                            "id_paziente": pid, "data_invio": self._random_time(dt, is_owl),
                            "letto": self.rng.random() < r_prob
                        })

            # Forum & Badges
            f_prob = self.rng.uniform(*params["forum_weekly_prob"])
            for w_idx in range(max(1, total_days // 7)):
                if self.rng.random() < f_prob:
                    d_off = w_idx*7 + self.rng.integers(0, 7)
                    f_dt = self.today - timedelta(days=int(total_days - d_off))
                    forum_records.append({"id_paziente": pid, "data_inserimento": self._random_time(f_dt)})
            
            b_count = int(self.rng.integers(*params["badges_total_range"]))
            for _ in range(b_count):
                d_off = self.rng.integers(0, total_days)
                badge_records.append({"id_paziente": pid, "data_acquisizione": self.today - timedelta(days=int(d_off))})

        data = {
            "patients": patients_df, "moods": pd.DataFrame(mood_records), "diaries": pd.DataFrame(diary_records),
            "quests": pd.DataFrame(quest_records), "notifs": pd.DataFrame(notif_records), 
            "forums": pd.DataFrame(forum_records), "badges": pd.DataFrame(badge_records)
        }
        print("[OK] Generazione completata.")
        return data

    # ─────────────────────────────────────────────────────────────────────────
    #  Aggregazione Feature
    # ─────────────────────────────────────────────────────────────────────────

    def run_aggregation(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        print("[*] Aggregazione feature...")
        patients = data["patients"]
        moods, diaries, quests, notifs, forums, badges = data["moods"], data["diaries"], data["quests"], data["notifs"], data["forums"], data["badges"]
        
        ref_date = self.today
        start_7d = ref_date - timedelta(days=7)
        features = []

        for _, p in patients.iterrows():
            pid = p["id_paziente"]
            m_7d = moods[(moods["id_paziente"] == pid) & (moods["data_inserimento"] >= start_7d)]
            d_7d = diaries[(diaries["id_paziente"] == pid) & (diaries["data_inserimento"] >= start_7d)]
            
            # Mood metrics
            mood_freq = len(m_7d)
            avg_valence = m_7d["umore"].map(VALENZA_UMORE).mean() if mood_freq > 0 else np.nan
            
            # Diary & Forum
            diary_freq = len(d_7d)
            avg_diary_len = d_7d["testo"].str.len().mean() if diary_freq > 0 else 0.0
            forum_freq = len(forums[(forums["id_paziente"] == pid) & (forums["data_inserimento"] >= start_7d)])
            
            # Compliance & Read Rate
            total_days = (ref_date - p["data_ingresso"]).days
            expected_q = max(1, total_days // 14)
            q_compl = min(1.0, len(quests[quests["id_paziente"] == pid]) / expected_q)
            
            last_notifs = notifs[notifs["id_paziente"] == pid].sort_values("data_invio", ascending=False).head(15)
            read_rate = last_notifs["letto"].mean() if len(last_notifs) > 0 else 0.5
            
            # Night Rate
            night_hours = [23, 0, 1, 2, 3, 4, 5, 6]
            activity = pd.concat([m_7d["data_inserimento"], d_7d["data_inserimento"]])
            night_rate = activity.dt.hour.isin(night_hours).mean() if len(activity) > 0 else 0.0

            features.append({
                "id_paziente": pid, "profilo_assegnato": p["profilo"],
                "mood_frequency_7d": mood_freq, "avg_mood_valence_7d": avg_valence,
                "diary_entries_7d": diary_freq, "avg_diary_length_7d": avg_diary_len,
                "questionnaire_compliance": q_compl, "forum_activity_7d": forum_freq,
                "notification_read_rate": read_rate, "days_in_waitlist": total_days,
                "badges_total": len(badges[badges["id_paziente"] == pid]),
                "night_activity_rate": night_rate
            })

        df = pd.DataFrame(features)
        scaler = MinMaxScaler()
        cols = ["mood_frequency_7d", "avg_mood_valence_7d", "diary_entries_7d", "avg_diary_length_7d", "forum_activity_7d", "days_in_waitlist", "badges_total", "night_activity_rate"]
        df[cols] = scaler.fit_transform(df[cols].fillna(0.5))
        
        print(f"[OK] Aggregazione completata: {len(df)} pazienti processati.")
        return df

    def save_data(self, data: Dict[str, pd.DataFrame], base_path: str):
        synthetic_path = os.path.join(base_path, "data", "synthetic")
        os.makedirs(synthetic_path, exist_ok=True)
        for name, df in data.items():
            df.to_csv(os.path.join(synthetic_path, f"{name}.csv"), index=False)
        print(f"[*] Dati sintetici salvati in {synthetic_path}")

    def save_features(self, df: pd.DataFrame, base_path: str):
        processed_path = os.path.join(base_path, "data", "processed")
        os.makedirs(processed_path, exist_ok=True)
        df.to_csv(os.path.join(processed_path, "churn_features.csv"), index=False)
        print(f"[*] Feature salvate in {processed_path}/churn_features.csv")

if __name__ == "__main__":
    pipeline = DataPipeline()
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data = pipeline.run_generation(n_patients=200)
    pipeline.save_data(data, base_dir)
    features = pipeline.run_aggregation(data)
    pipeline.save_features(features, base_dir)
