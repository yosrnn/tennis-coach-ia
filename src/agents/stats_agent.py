# =============================================================
# stats_agent.py — Agent d'analyse statistique
# =============================================================
# Rôle : répond aux questions précises sur des chiffres
# en interrogeant directement le CSV avec Pandas.
# Exemple : "combien d'aces Djokovic en 2023 ?"
#
# Différence avec rag_agent :
#   rag_agent   → questions de sens, contexte, style de jeu
#   stats_agent → questions précises, chiffres, classements
# =============================================================

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# --- Config ---
DATA_PATH = "/Users/yosrnoureddine/tennis-coach-ia/data/processed/atp_clean.csv"

# --- Initialisation ---
df = pd.read_csv(DATA_PATH)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def stats_agent(question: str) -> str:
    # 1. Donne au LLM le schéma du CSV pour qu'il génère le bon code Pandas
    schema = """
    Colonnes disponibles :
    - tourney_name, surface, tourney_date, round
    - winner_name, loser_name, score
    - winner_rank, loser_rank
    - w_ace, w_df, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced
    - l_ace, l_df, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced

    Notes :
    - tourney_date est au format YYYYMMDD (ex: 20230101)
    - w_ = stats du vainqueur, l_ = stats du perdant
    """

    # 2. Demande au LLM de générer du code Pandas
    messages = [
        SystemMessage(content=f"""Tu es un expert en analyse de données tennis.
Tu génères du code Pandas pour répondre à des questions sur un DataFrame appelé 'df'.
Réponds UNIQUEMENT avec du code Python valide, sans explication, sans markdown.
{schema}"""),
        HumanMessage(content=f"Question : {question}\nGénère le code Pandas et stocke le résultat dans une variable 'result'.")
    ]

    code_response = llm.invoke(messages)

    # 3. Nettoie le code (enlève les backticks markdown)
    code = code_response.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    # 4. Exécute le code généré
    try:
        local_vars = {"df": df, "pd": pd}
        exec(code, local_vars)
        result = local_vars.get("result", "Aucun résultat")
    except Exception as e:
        return f"Erreur d'exécution : {e}\nCode généré : {code}"

    # 5. Demande au LLM de formuler une réponse lisible
    messages2 = [
        SystemMessage(content="Tu es Coach IA. Formule une réponse claire et concise basée sur ce résultat de données tennis."),
        HumanMessage(content=f"Question : {question}\nRésultat des données : {result}")
    ]

    final_response = llm.invoke(messages2)
    return final_response.content

if __name__ == "__main__":
    question = "How many aces did Djokovic make in 2023?"
    print(f"🎾 Question : {question}\n")
    print(f"🤖 Réponse :\n{stats_agent(question)}")