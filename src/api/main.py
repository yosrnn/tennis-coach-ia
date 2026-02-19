# =============================================================
# main.py — API FastAPI du Coach IA
# =============================================================
# Rôle : exposer le système multi-agents comme une API REST.
# Un client envoie une question, l'API retourne une réponse.
#
# Endpoints :
#   GET  /         → health check (est-ce que l'API tourne ?)
#   POST /ask      → pose une question au Coach IA
# =============================================================

import os
import sys
sys.path.append("/Users/yosrnoureddine/tennis-coach-ia")
sys.path.append("/Users/yosrnoureddine/tennis-coach-ia/src/agents")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from orchestrator import orchestrator

load_dotenv()

app = FastAPI(
    title="Coach IA — Tennis Analytics",
    description="API multi-agents pour l'analyse de performance tennis ATP",
    version="1.0.0"
)

# --- Schéma de la requête ---
class Question(BaseModel):
    question: str  # ex: "How does Djokovic perform on clay?"

# --- Schéma de la réponse ---
class Answer(BaseModel):
    question: str
    answer: str
    agent_used: str

# --- Endpoints ---
@app.get("/")
def health_check():
    """Vérifie que l'API tourne"""
    return {"status": "ok", "message": "Coach IA is running 🎾"}

@app.post("/ask", response_model=Answer)
def ask(body: Question):
    """Pose une question au Coach IA"""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        # Détermine quel agent sera utilisé
        question_lower = body.question.lower()
        agent = "stats" if any(w in question_lower for w in [
            "how many", "combien", "count", "total", "number"
        ]) else "rag"

        answer = orchestrator(body.question)

        return Answer(
            question=body.question,
            answer=answer,
            agent_used=agent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))