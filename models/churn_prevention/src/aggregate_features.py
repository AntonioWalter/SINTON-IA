"""
SINTON-IA – Aggregazione Feature per Churn Prevention
=============================================================================
Calcola le metriche descritte in 04_Data_Preparation.tex (Sez. 11.2) partendo
dai record grezzi generati, per costruire il dataset di input all'Algoritmo Genetico.
Simula la finestra temporale degli "ultimi 7 giorni" per valutare il KPI di retention.

Metriche estratte (normalizzate min-max 0-1):
  1. mood_frequency_7d
  2. mood_consistency_7d
  3. avg_mood_valence_7d
  4. diary_entries_7d
  5. avg_diary_length_7d
  6. questionnaire_compliance
  7. forum_activity_7d
  8. notification_read_rate
  9. days_in_waitlist
 10. badges_total
"""

import os
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

VALENZA_UMORE = {
    "Felice": 1.0, "Sereno": 0.7, "Energico": 0.8,
    "Neutro": 0.0, "Stanco": -0.3,
    "Triste": -0.8, "Ansioso": -0.6, "Arrabbiato": -0.7,
    "Spaventato": -0.9, "Confuso": -0.4,
}

def load_data(input_dir: str):
    """Carica i csv dalla cartella dei dati sintetici."""
    patients = pd.read_csv(os.path.join(input_dir, "patients.csv"))
    moods = pd.read_csv(os.path.join(input_dir, "stato_animo.csv"))
    diaries = pd.read_csv(os.path.join(input_dir, "pagina_diario.csv"))
    quests = pd.read_csv(os.path.join(input_dir, "questionario.csv"))
    notifs = pd.read_csv(os.path.join(input_dir, "notifica.csv"))
    forums = pd.read_csv(os.path.join(input_dir, "domanda_forum.csv"))
    badges = pd.read_csv(os.path.join(input_dir, "acquisizione_badge.csv"))
    
    # Casting date
    patients["data_ingresso"] = pd.to_datetime(patients["data_ingresso"])
    moods["data_inserimento"] = pd.to_datetime(moods["data_inserimento"])
    diaries["data_inserimento"] = pd.to_datetime(diaries["data_inserimento"])
    quests["data_compilazione"] = pd.to_datetime(quests["data_compilazione"])
    notifs["data_invio"] = pd.to_datetime(notifs["data_invio"])
    forums["data_inserimento"] = pd.to_datetime(forums["data_inserimento"])
    badges["data_acquisizione"] = pd.to_datetime(badges["data_acquisizione"])
    
    return patients, moods, diaries, quests, notifs, forums, badges

def aggregate_features(input_dir: str, output_dir: str):
    print("[*] Caricamento dati grezzi...")
    patients, moods, diaries, quests, notifs, forums, badges = load_data(input_dir)
    
    # La "data odierna" di riferimento è la data massima presente nei log (come fosse in produzione real-time)
    # Troviamo l'ultimo log di notifica, mood, o diario, per cui useremo datetime.now() fittizio o il max
    all_dates = pd.concat([
        moods["data_inserimento"], 
        diaries["data_inserimento"], 
        notifs["data_invio"]
    ])
    reference_date = all_dates.max().replace(hour=23, minute=59, second=59)
    start_7d = reference_date - timedelta(days=7)
    
    print(f"[*] Finestra temporale di osservazione: {start_7d.date()} -> {reference_date.date()}")
    
    features = []
    
    for _, p in patients.iterrows():
        pid = p["id_paziente"]
        
        # ─────────────────────────────────────────────────────────────
        # 1. & 2. & 3. Dati Mood (7d)
        # ─────────────────────────────────────────────────────────────
        m_7d = moods[(moods["id_paziente"] == pid) & (moods["data_inserimento"] >= start_7d)]
        mood_freq = len(m_7d)
        unique_days = m_7d["data_inserimento"].dt.date.nunique()
        mood_consist = unique_days / 7.0
        # N.B. Nel dataset sintetico, mood_frequency_7d e mood_consistency_7d sono spesso
        # identiche perché la generazione prevede al massimo un mood al giorno.
        # Con dati reali (dove si possono inserire più mood/giorno) divergeranno.
        
        if mood_freq > 0:
            avg_valence = m_7d["umore"].map(VALENZA_UMORE).mean()
        else:
            avg_valence = float('nan')  # Nessun dato: sarà gestito post-normalizzazione
            
        # ─────────────────────────────────────────────────────────────
        # 4. & 5. Dati Diario (7d)
        # ─────────────────────────────────────────────────────────────
        d_7d = diaries[(diaries["id_paziente"] == pid) & (diaries["data_inserimento"] >= start_7d)]
        diary_freq = len(d_7d)
        
        # Consideriamo il campo testo per evitare TypeErrors, in dataframe simulato è una stringa
        if diary_freq > 0 and 'testo' in d_7d.columns:
            avg_diary_len = d_7d["testo"].str.len().mean()
        else:
            avg_diary_len = 0.0
        
        # ─────────────────────────────────────────────────────────────
        # 6. Questionari Compliance (Storico totale)
        # ─────────────────────────────────────────────────────────────
        total_days = (reference_date - p["data_ingresso"]).days
        expected_quests = total_days // 14  # Frequenza tipica 14 giorni
        
        q_storico = quests[quests["id_paziente"] == pid]
        filled_quests = len(q_storico)
        
        if expected_quests > 0:
            quest_compl = min(1.0, filled_quests / expected_quests)
        else:
            quest_compl = 1.0 if filled_quests > 0 else 0.5
            
        # ─────────────────────────────────────────────────────────────
        # 7. Forum (7d)
        # ─────────────────────────────────────────────────────────────
        f_7d = forums[(forums["id_paziente"] == pid) & (forums["data_inserimento"] >= start_7d)]
        forum_freq = len(f_7d)
        
        # ─────────────────────────────────────────────────────────────
        # 8. Notifiche (Ultime K)
        # ─────────────────────────────────────────────────────────────
        K_NOTIFS = 15
        notif_storico = notifs[notifs["id_paziente"] == pid].sort_values("data_invio", ascending=False).head(K_NOTIFS)
        if len(notif_storico) > 0:
            notif_read = notif_storico["letto"].sum() / len(notif_storico)
        else:
            notif_read = 0.5 # default           
        # ─────────────────────────────────────────────────────────────
        # 9. & 10. Waitlist e Badges
        # ─────────────────────────────────────────────────────────────
        days_wait = (reference_date - p["data_ingresso"]).days
        if days_wait < 0:
            days_wait = 0
        
        b_storico = badges[badges["id_paziente"] == pid]
        badges_tot = len(b_storico)
        
        # ─────────────────────────────────────────────────────────────
        # 11. Night Activity Rate (23:00 - 06:59)
        # ─────────────────────────────────────────────────────────────
        night_hours = [23, 0, 1, 2, 3, 4, 5, 6]
        all_patient_activity = pd.concat([
            m_7d["data_inserimento"],
            d_7d["data_inserimento"]
        ])
        if len(all_patient_activity) > 0:
            night_acts = all_patient_activity.dt.hour.isin(night_hours).sum()
            night_rate = night_acts / len(all_patient_activity)
        else:
            night_rate = 0.0
        
        features.append({
            "id_paziente": pid,
            "profilo_assegnato": p["profilo"],
            "mood_frequency_7d": mood_freq,
            "mood_consistency_7d": mood_consist,
            "avg_mood_valence_7d": avg_valence,
            "diary_entries_7d": diary_freq,
            "avg_diary_length_7d": avg_diary_len,
            "questionnaire_compliance": quest_compl,
            "forum_activity_7d": forum_freq,
            "notification_read_rate": notif_read,
            "days_in_waitlist": days_wait,
            "badges_total": badges_tot,
            "night_activity_rate": night_rate
        })
        
    df_features = pd.DataFrame(features)
    
    # ─────────────────────────────────────────────────────────────
    # Normalizzazione Min-Max (0-1) come da documento 04_Data_Preparation.tex
    # ─────────────────────────────────────────────────────────────
    print("[*] Normalizzazione e scalatura indicatori...")
    
    # Le colonne che sono gia [0, 1] naturali non vanno scalate (o lo facciamo su tutto per sicurezza, 
    # ma il documento dice "Gli indicatori che presentano già range naturale in [0, 1]... non necessitano".
    # Quindi scaliamo solo quelle libere.
    
    cols_to_scale = [
        "mood_frequency_7d", 
        "avg_mood_valence_7d", # range originario -1,+1, va scalato in 0,1
        "diary_entries_7d", 
        "avg_diary_length_7d", 
        "forum_activity_7d",
        "days_in_waitlist",
        "badges_total",
        "night_activity_rate"
    ]
    # Inizializziamo lo scaler standard per riportare tutto in range [0, 1]
    scaler = MinMaxScaler()
    df_features[cols_to_scale] = scaler.fit_transform(df_features[cols_to_scale])

    # Pazienti senza mood entries nella finestra 7d: assegna valore neutro (0.5)
    # post-normalizzazione, indipendente dalla distribuzione del dataset
    df_features["avg_mood_valence_7d"] = df_features["avg_mood_valence_7d"].fillna(0.5)
    
    # Salvataggio
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "churn_features.csv")
    df_features.to_csv(out_path, index=False)
    print(f"[OK] Salvato dataset aggregato e normalizzato in:\n     {os.path.abspath(out_path)}")
    print(f"     (Pazienti processati: {len(df_features)})")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "data", "synthetic")
    output_dir = os.path.join(script_dir, "..", "data", "processed")
    
    if not os.path.exists(input_dir):
        print("Errore: la cartella dati sintetici non esiste. Eseguire prima 'generate_synthetic_data.py'.")
    else:
        aggregate_features(input_dir, output_dir)
