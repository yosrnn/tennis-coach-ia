# Coach IA — Tennis Performance Analytics

Système multi-agents RAG pour l'analyse de performance tennis ATP.  

---

## Architecture

```
Question du coach
        ↓
   Orchestrateur (LLM)
     ↓           ↓
Agent RAG    Agent Stats
(Qdrant)     (Pandas/CSV)
     ↓           ↓
   Réponse ancrée dans les vraies données ATP
```

### Composants

| Composant     | Rôle                                | Technologie                   |
| ------------- | ----------------------------------- | ----------------------------- |
| Orchestrateur | Route la question vers le bon agent | LangChain + Groq              |
| Agent RAG     | Recherche sémantique sur les matchs | Qdrant + SentenceTransformers |
| Agent Stats   | Analyse statistique précise         | Pandas + Groq                 |
| API           | Exposition du système               | FastAPI                       |
| Évaluation    | Mesure de la qualité                | LLM-as-a-judge                |

---

## Dataset

- **Source** : [Jeff Sackmann ATP Dataset](https://github.com/JeffSackmann/tennis_atp)
- **Période** : 2020 - 2024
- **Volume** : ~8000 matchs ATP, 2000 textes narratifs indexés
- **Format** : CSV structuré + textes narratifs vectorisés

---

## Choix techniques

**Pourquoi Qdrant ?**  
Base vectorielle locale, pas de dépendance cloud, maîtrise totale des données et des coûts.

**Pourquoi deux agents séparés ?**  
Les questions de sens ("comment joue Djokovic ?") et les questions de chiffres ("combien d'aces en 2023 ?") nécessitent des approches différentes. Le routing LLM évite de programmer des règles fixes fragiles.

**Pourquoi LLM-as-a-judge ?**  
RAGAS v1.0 ne supporte plus les LLMs open-source. On implémente notre propre évaluation plus flexible et indépendante d'OpenAI.

**Pourquoi Groq ?**  
Gratuit, rapide, API standard compatible LangChain. Facilement remplaçable par OpenAI ou Anthropic sans changer l'architecture.

---

## Résultats d'évaluation

| Métrique         | Score    |
| ---------------- | -------- |
| Faithfulness     | 1.00     |
| Answer Relevancy | 0.83     |
| **Score global** | **0.92** |

---

## Structure du projet

```
tennis-coach-ia/
├── data/
│   ├── raw/                   # CSV ATP bruts (non versionné)
│   └── processed/             # Données nettoyées (non versionné)
├── src/
│   ├── ingestion/
│   │   ├── vector_store.py    # Indexation Qdrant
│   │   └── test_search.py     # Test recherche sémantique
│   ├── agents/
│   │   ├── rag_agent.py       # Agent recherche sémantique
│   │   ├── stats_agent.py     # Agent analyse statistique
│   │   └── orchestrator.py    # Routing intelligent
│   └── api/
│       └── main.py            # API FastAPI
├── evals/
│   └── eval_rag.py            # Évaluation LLM-as-a-judge
├── notebooks/
│   └── 01_exploration.ipynb   # Exploration données
└── .github/
    └── workflows/
        └── ci.yml             # Pipeline CI/CD
```

---

## Lancer le projet

### 1. Prérequis

```bash
python 3.12+
```

### 2. Installation

```bash
git clone https://github.com/yosrnn/tennis-coach-ia.git
cd tennis-coach-ia
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Ajoute ta clé Groq dans .env
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 4. Télécharger et indexer les données

```bash
# Dans Jupyter
jupyter lab
# Ouvre notebooks/01_exploration.ipynb et lance toutes les cellules

# Puis indexe dans Qdrant
python src/ingestion/vector_store.py
```

### 5. Lancer l'API

```bash
python -m uvicorn src.api.main:app --reload
```

API disponible sur **http://127.0.0.1:8000/docs**

### 6. Lancer l'évaluation

```bash
python evals/eval_rag.py
```

---

## Exemple d'utilisation

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Djokovic perform on clay?"}'
```

```json
{
  "question": "How does Djokovic perform on clay?",
  "answer": "Djokovic performs very well on clay...",
  "agent_used": "rag"
}
```

---

## Stack technique

- **LLM** : LLaMA 3.3 70B via Groq API
- **Embeddings** : all-MiniLM-L6-v2 (SentenceTransformers / HuggingFace)
- **Vector DB** : Qdrant (local)
- **Orchestration** : LangChain + LangGraph
- **API** : FastAPI + Uvicorn
- **Évaluation** : LLM-as-a-judge custom
- **CI/CD** : GitHub Actions
