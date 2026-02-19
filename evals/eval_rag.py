# =============================================================
# eval_rag.py — Évaluation custom du système RAG
# =============================================================
# Rôle : mesurer la qualité des réponses du rag_agent
# sans dépendance OpenAI — on utilise le LLM Groq comme juge.
#
# Métriques :
#   faithfulness     → la réponse vient-elle du contexte ? (0 à 1)
#   relevancy        → la réponse répond-elle à la question ? (0 à 1)
# =============================================================

import os
import sys
import json
sys.path.append("/Users/yosrnoureddine/tennis-coach-ia")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Config ---
COLLECTION_NAME = "tennis_matches"
QDRANT_PATH = "/Users/yosrnoureddine/tennis-coach-ia/data/qdrant"
MODEL_NAME = "all-MiniLM-L6-v2"

client = QdrantClient(path=QDRANT_PATH)
embedding_model = SentenceTransformer(MODEL_NAME)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# --- Jeu de test ---
test_cases = [
    {
        "question": "How does Djokovic perform on clay?",
        "ground_truth": "Djokovic performs very well on clay, winning matches at Roland Garros and Rome Masters."
    },
    {
        "question": "Who played at Roland Garros in 2020?",
        "ground_truth": "Djokovic, Tsitsipas, and Nadal played at Roland Garros in 2020."
    },
    {
        "question": "How does Nadal play on clay surface?",
        "ground_truth": "Nadal is dominant on clay, known for his consistency and ability to win break points."
    }
]

def get_rag_response(question: str):
    """Retourne la réponse ET les contextes utilisés"""
    query_vector = embedding_model.encode(question).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5
    ).points
    contexts = [r.payload["text"] for r in results]
    context_str = "\n\n".join(contexts)
    messages = [
        SystemMessage(content="Tu es Coach IA, expert tennis. Réponds UNIQUEMENT en te basant sur le contexte fourni."),
        HumanMessage(content=f"Contexte :\n{context_str}\n\nQuestion : {question}")
    ]
    response = llm.invoke(messages).content
    return response, contexts

def evaluate_faithfulness(answer: str, contexts: list) -> float:
    """Le LLM juge si la réponse vient bien du contexte"""
    context_str = "\n\n".join(contexts)
    messages = [
        SystemMessage(content="""Tu es un évaluateur. 
Réponds UNIQUEMENT avec un JSON : {"score": X, "reason": "..."}
où X est entre 0 et 1 :
1.0 = réponse entièrement basée sur le contexte
0.5 = réponse partiellement basée sur le contexte  
0.0 = réponse inventée, pas dans le contexte"""),
        HumanMessage(content=f"Contexte :\n{context_str}\n\nRéponse à évaluer :\n{answer}")
    ]
    result = llm.invoke(messages).content
    try:
        data = json.loads(result.replace("```json", "").replace("```", "").strip())
        return data["score"], data["reason"]
    except:
        return 0.5, "Erreur de parsing"

def evaluate_relevancy(question: str, answer: str) -> float:
    """Le LLM juge si la réponse répond bien à la question"""
    messages = [
        SystemMessage(content="""Tu es un évaluateur.
Réponds UNIQUEMENT avec un JSON : {"score": X, "reason": "..."}
où X est entre 0 et 1 :
1.0 = réponse répond parfaitement à la question
0.5 = réponse partiellement pertinente
0.0 = réponse hors sujet"""),
        HumanMessage(content=f"Question : {question}\n\nRéponse : {answer}")
    ]
    result = llm.invoke(messages).content
    try:
        data = json.loads(result.replace("```json", "").replace("```", "").strip())
        return data["score"], data["reason"]
    except:
        return 0.5, "Erreur de parsing"

# --- Évaluation ---
print("📊 Évaluation du système RAG\n")
print("=" * 60)

total_faithfulness = 0
total_relevancy = 0

for test in test_cases:
    question = test["question"]
    print(f"\n🎾 Question : {question}")

    answer, contexts = get_rag_response(question)
    print(f"🤖 Réponse : {answer[:100]}...")

    faith_score, faith_reason = evaluate_faithfulness(answer, contexts)
    rel_score, rel_reason = evaluate_relevancy(question, answer)

    total_faithfulness += faith_score
    total_relevancy += rel_score

    print(f"✅ Faithfulness  : {faith_score:.2f} — {faith_reason}")
    print(f"✅ Relevancy     : {rel_score:.2f} — {rel_reason}")

# --- Scores finaux ---
n = len(test_cases)
print("\n" + "=" * 60)
print(f"📈 Score moyen Faithfulness  : {total_faithfulness/n:.2f}")
print(f"📈 Score moyen Relevancy     : {total_relevancy/n:.2f}")
print(f"📈 Score global              : {(total_faithfulness + total_relevancy) / (2*n):.2f}")