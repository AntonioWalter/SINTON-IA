import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

base_dir = "/Users/antoniowalterdefusco/Documents/Project/SINTON-IA"
data_dir = f"{base_dir}/models/nlp_suicide_risk/data/processed"
weights_dir = f"{base_dir}/models/nlp_suicide_risk/weights"
figures_dir = f"{base_dir}/docs/documentazione/latex/figures"
os.makedirs(weights_dir, exist_ok=True)

print("Caricamento dataset...")
train_df = pd.read_csv(f"{data_dir}/train.csv")
val_df = pd.read_csv(f"{data_dir}/val.csv")

train_text = train_df['text'].fillna('')
val_text = val_df['text'].fillna('')
y_train = train_df['class']
y_val = val_df['class']

print("Costruzione del vocabolario TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)

print("Addestramento della Regressione Logistica...")
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

print("Risultati sul Validation Set:")
y_pred_val = clf.predict(X_val)
print(classification_report(y_val, y_pred_val))

# Plot CM
cm = confusion_matrix(y_val, y_pred_val, labels=['non-suicide', 'suicide'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Suicide', 'Suicide'], 
            yticklabels=['Non-Suicide', 'Suicide'])
plt.title('Baseline (LogReg) - Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(f"{figures_dir}/nlp_baseline_cm.png", dpi=300)
plt.close()

# Save models
joblib.dump(vectorizer, f"{weights_dir}/tfidf_vectorizer_baseline.pkl")
joblib.dump(clf, f"{weights_dir}/logreg_baseline.pkl")

# Update Jupyter Notebook
nb_path = f"{base_dir}/models/nlp_suicide_risk/notebooks/03_RedFlag_Modelling_Baseline.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

new_cells = [
 {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Addestramento del Modello Baseline\n",
    "\n",
    "In questa fase definiamo una baseline solida e interpretabile da utilizzare come benchmark nel corso del ciclo di esperimenti.\\\n",
    "La scelta ricade su un'architettura ibrida basata sulla vettorializzazione **TF-IDF** (Term Frequency - Inverse Document Frequency) accoppiata a un classificatore a **Regressione Logistica** (Logistic Regression)."
   ]
 },
 {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "# Assicuriamoci che non ci siano valori NaN nel testo\n",
    "train_text = train_df['text'].fillna('')\n",
    "val_text = val_df['text'].fillna('')\n",
    "\n",
    "# 1. Vettorizzazione TF-IDF\n",
    "vectorizer = TfidfVectorizer(max_features=10000)\n",
    "X_train = vectorizer.fit_transform(train_text)\n",
    "X_val = vectorizer.transform(val_text)\n",
    "\n",
    "y_train = train_df['class']\n",
    "y_val = val_df['class']\n",
    "\n",
    "print(f\"Dimensioni matrice Training (Testi processati, Feature): {X_train.shape}\")"
   ]
 },
 {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Inizializzazione e Addestramento Logistic Regression\n",
    "clf = LogisticRegression(random_state=42, max_iter=1000)\n",
    "clf.fit(X_train, y_train)\n",
    "print(\"Addestramento terminato.\")"
   ]
 },
 {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Test Rapido sul Set di Validazione\n",
    "y_pred_val = clf.predict(X_val)\n",
    "\n",
    "print(\"--- Report Classificazione (Validation) ---\")\n",
    "print(classification_report(y_val, y_pred_val))\n",
    "\n",
    "cm = confusion_matrix(y_val, y_pred_val, labels=['non-suicide', 'suicide'])\n",
    "print(\"\\nMatrice di Confusione:\")\n",
    "print(cm)"
   ]
 },
 {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.makedirs('../weights', exist_ok=True)\n",
    "joblib.dump(vectorizer, '../weights/tfidf_vectorizer_baseline.pkl')\n",
    "joblib.dump(clf, '../weights/logreg_baseline.pkl')\n",
    "print(\"Oggetti esportati in `/weights` per la fase di ottimizzazione delle soglie.\")"
   ]
 }
]

notebook['cells'].extend(new_cells)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Elaborazione completata, script aggiornato.")
