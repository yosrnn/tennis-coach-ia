# =============================================================
# orchestrator.py — Chef d'orchestre des agents
# =============================================================
# Rôle : analyse la question et route vers le bon agent.
# C'est le cerveau du système — il décide sans qu'on lui
# programme de règles fixes. Le LLM choisit lui-même.
#
# Routing :
#   question de sens/style/contexte → rag_agent
#   question de chiffres/stats      → stats_agent
# =============================================================

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Ajoute le dossier courant au path pour trouver rag_agent et stats_agent
sys.path.append(os.path.dirname(__file__))

from rag_agent import rag_agent
from stats_agent import stats_agent

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def orchestrator(question: str) -> str:
    # 1. Demande au LLM de choisir l'agent
    messages = [
        SystemMessage(content="""Tu es un orchestrateur d'agents tennis.
Analyse la question et réponds UNIQUEMENT par un de ces deux mots :
- RAG       → question sur le style, contexte, performance générale
- STATS     → question sur des chiffres précis, comptages, moyennes"""),
        HumanMessage(content=f"Question : {question}")
    ]

    routing = llm.invoke(messages).content.strip().upper()
    print(f"🔀 Routing → {routing}")

    # 2. Appelle le bon agent
    if "STATS" in routing:
        return stats_agent(question)
    else:
        return rag_agent(question)

if __name__ == "__main__":
    questions = [
        "How does Djokovic perform on clay?",
        "How many aces did Nadal make in 2022?",
        "Who is the best player on hard court?",
        "How many matches did Federer win in 2021?"
    ]

    for q in questions:
        print(f"\n🎾 Question : {q}")
        print(f"🤖 Réponse : {orchestrator(q)}")
        print("-" * 50)