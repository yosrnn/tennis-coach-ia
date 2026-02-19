# =============================================================
# rag_agent.py — Agent de recherche sémantique
# =============================================================
# Rôle : reçoit une question en langage naturel,
# cherche les matchs pertinents dans Qdrant,
# puis demande au LLM de formuler une réponse
# basée sur ces vrais matchs (pas d'hallucination)
# =============================================================

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# --- Config ---
COLLECTION_NAME = "tennis_matches"
QDRANT_PATH = "/Users/yosrnoureddine/tennis-coach-ia/data/qdrant"
MODEL_NAME = "all-MiniLM-L6-v2"

# --- Initialisation ---
client = QdrantClient(path=QDRANT_PATH)
embedding_model = SentenceTransformer(MODEL_NAME)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def rag_agent(question: str) -> str:
    # 1. Transforme la question en vecteur
    query_vector = embedding_model.encode(question).tolist()

    # 2. Cherche les 5 matchs les plus pertinents dans Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5
    ).points

    # 3. Construit le contexte avec les vrais matchs
    context = "\n\n".join([r.payload["text"] for r in results])

    # 4. Demande au LLM de répondre basé sur ce contexte
    messages = [
        SystemMessage(content="""Tu es Coach IA, un assistant expert en tennis.
Tu réponds UNIQUEMENT en te basant sur les matchs fournis dans le contexte.
Si la réponse n'est pas dans le contexte, dis-le clairement."""),
        HumanMessage(content=f"""Contexte — matchs réels ATP :
{context}

Question : {question}""")
    ]

    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    question = "How does Djokovic perform on clay?"
    print(f"🎾 Question : {question}\n")
    print(f"🤖 Réponse :\n{rag_agent(question)}")