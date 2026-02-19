from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "tennis_matches"
QDRANT_PATH = "/Users/yosrnoureddine/tennis-coach-ia/data/qdrant"
MODEL_NAME = "all-MiniLM-L6-v2"

client = QdrantClient(path=QDRANT_PATH)
model = SentenceTransformer(MODEL_NAME)

question = "How does Djokovic perform on clay?"

# Transforme la question en vecteur
query_vector = model.encode(question).tolist()

# Nouvelle méthode Qdrant
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
).points

print(f"🔍 Question : {question}\n")
for i, result in enumerate(results):
    print(f"--- Résultat {i+1} (score: {result.score:.3f}) ---")
    print(result.payload["text"])
    print()