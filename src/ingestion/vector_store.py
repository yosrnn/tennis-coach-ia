# =============================================================
# vector_store.py — Construction de la base vectorielle
# =============================================================
# Rôle : transformer nos textes de matchs ATP en vecteurs
# et les stocker dans Qdrant pour la recherche sémantique.
#
# Flux :
#   atp_texts.json
#       ↓ SentenceTransformer (HuggingFace, tourne en local)
#       ↓ chaque texte → vecteur de 384 nombres
#       ↓ stocké dans Qdrant avec le texte original (payload)
#   Résultat : on peut chercher par sens, pas par mot exact
# =============================================================

import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# --- Config ---
COLLECTION_NAME = "tennis_matches"
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "/Users/yosrnoureddine/tennis-coach-ia/data/processed/atp_texts.json"
QDRANT_PATH = "/Users/yosrnoureddine/tennis-coach-ia/data/qdrant"

def load_texts():
    """Charge les 2000 textes narratifs générés depuis les CSV ATP"""
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def build_vector_store():
    print("📂 Chargement des textes...")
    texts = load_texts()
    print(f"✅ {len(texts)} textes chargés")

    # Télécharge le modèle depuis HuggingFace (1ère fois seulement)
    print("🤖 Chargement du modèle d'embedding...")
    model = SentenceTransformer(MODEL_NAME)

    # Transforme chaque texte en vecteur de 384 nombres
    print("🔢 Calcul des vecteurs...")
    vectors = model.encode(texts, show_progress_bar=True)
    print(f"✅ Vecteurs calculés — dimension : {vectors.shape}")

    # Crée le dossier Qdrant si il n'existe pas
    os.makedirs(QDRANT_PATH, exist_ok=True)

    # Connexion à Qdrant en local (pas de serveur, juste un dossier)
    print("💾 Stockage dans Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)

    # Crée la collection uniquement si elle n'existe pas déjà
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vectors.shape[1],  # 384 pour all-MiniLM-L6-v2
                distance=Distance.COSINE
            )
        )

    # Chaque point = 1 texte avec son vecteur + le texte original
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": texts[i]}
        )
        for i in range(len(texts))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ {len(points)} textes indexés dans Qdrant !")

if __name__ == "__main__":
    build_vector_store()