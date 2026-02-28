# 🤝 Guida al Contributo – SINTON-IA

Questo documento definisce il flusso di lavoro del team, le convenzioni per i commit e le regole per le Pull Request. Leggerlo è **obbligatorio** prima di iniziare a contribuire.

---

## 📋 Metodologia: Scrum

Il team lavora seguendo la metodologia **Scrum**. I principi fondamentali che ci guidano sono:

| Concetto | Applicazione in questo progetto |
|---|---|
| **Sprint** | Cicli di lavoro con obiettivi chiari |
| **Backlog** | Le task sono gestite come **GitHub Issues** |
| **Sprint Planning** | Si apre ogni sprint assegnando le issue ai membri |
| **Daily / Check-in** | Aggiornamento rapido sullo stato dei task aperti |
| **Sprint Review** | A fine sprint si fa la review delle PR completate |

> Ogni issue deve corrispondere a un branch e poi a una Pull Request.

---

## 🌿 Convenzione dei Branch

```
<tipo>/<descrizione-breve>
```

### Tipi di branch

| Tipo | Quando usarlo | Esempio |
|---|---|---|
| `feature/` | Nuova funzionalità o modello | `feature/nlp-training-pipeline` |
| `fix/` | Correzione di un bug | `fix/churn-preprocessing-error` |
| `notebook/` | Aggiunta/modifica notebook | `notebook/depression-eda` |
| `docs/` | Documentazione LaTeX o README | `docs/chapter-nlp-suicide-risk` |
| `refactor/` | Pulizia codice senza nuove feature | `refactor/api-services-structure` |

> ⚠️ **Non lavorare mai direttamente su `main` o `dev`.** Ogni modifica passa obbligatoriamente da un branch e una Pull Request.

---

## ✍️ Convenzione dei Commit

Utilizziamo lo standard **[Conventional Commits](https://www.conventionalcommits.org/)**.

### Formato

```
<tipo>(<scope>): <descrizione breve>
```

### Tipi e scope

| Tipo | Significato |
|---|---|
| `feat` | Nuova funzionalità |
| `fix` | Bugfix |
| `docs` | Solo documentazione |
| `notebook` | Aggiunta o modifica di un notebook |
| `refactor` | Refactoring senza cambiamenti funzionali |
| `chore` | Configurazioni, dipendenze, setup |

### Scope consigliati

`nlp`, `churn`, `depression`, `api`, `integration`, `docs`, `root`

### Esempi

```bash
feat(nlp): aggiungo pipeline di preprocessing del testo
notebook(churn): aggiungo EDA iniziale sul dataset abbandoni
fix(depression): correggo gestione valori NaN nello storico umore
docs(docs): aggiorno capitolo 2 del documento LaTeX
chore(root): aggiorno requirements.txt con torch 2.2
```

---

## 🔁 Flusso di Lavoro (Workflow)

```
main
 └── dev
      └── feature/nome-branch   ← qui lavori tu
```

### Step-by-step

1. **Prendi una issue** dal backlog dello sprint corrente su GitHub
2. **Crea un branch** da `dev` seguendo la convenzione:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/nome-branch
   ```
3. **Lavora** sul branch, facendo commit frequenti e significativi
4. **Apri una Pull Request** verso `dev` quando il lavoro è pronto
5. **Attendi le review** (vedi sezione sotto)
6. **Merge** su `dev` solo dopo l'approvazione

> Il merge da `dev` a `main` avviene a fine sprint dopo la Sprint Review.

---

## 👀 Regole per le Pull Request

### Approvazione obbligatoria

> **Sono necessarie almeno 2 approvazioni** prima che una PR possa essere mergiata.

Questo garantisce che il codice sia sempre visto da tutto il team. Non è possibile fare self-merge.

### Template PR

Quando apri una Pull Request, GitHub ti pre-compilerà il form con questo template:

---

**## Descrizione**

<!-- Breve descrizione delle modifiche introdotte da questa PR. -->

**## Tipo di modifica**

- [ ] 🆕 Nuova feature
- [ ] 🐛 Bug fix
- [ ] 📓 Notebook (EDA / training / evaluation)
- [ ] 📄 Documentazione
- [ ] 🏗️ Refactor / struttura

**## Modulo coinvolto**

- [ ] `nlp_suicide_risk`
- [ ] `churn_prediction`
- [ ] `depression_detection`
- [ ] `api`
- [ ] `integration`
- [ ] Nessuno (root / docs)

**## Checklist**

- [ ] Il codice è stato testato localmente
- [ ] I notebook girano dall'inizio alla fine senza errori
- [ ] La documentazione è stata aggiornata (se necessario)
- [ ] Approva ✅ oppure richiedi modifiche con un commento chiaro 💬

---

### Come fare una buona review

Quando ti viene assegnata una PR da revisionare:

- Leggi la descrizione e capisci cosa fa
- Controlla che i notebook girino dall'inizio alla fine
- Verifica che il codice sia leggibile e commentato
- Approva ✅ oppure richiedi modifiche con un commento chiaro 💬

---

## 🔒 Branch Protection

I branch `main` e `dev` sono protetti:

- ❌ Nessun push diretto
- ✅ Solo merge tramite Pull Request
- ✅ Minimo **2 approvazioni** richieste
- ✅ La CI deve passare

---

## 📦 Gestione dei Dati

I dataset **non vanno mai committati**. La cartella `data/` di ogni modello è già gitignored.

Le istruzioni per ottenere i dati sono nel `README.md` di ciascun modello.
