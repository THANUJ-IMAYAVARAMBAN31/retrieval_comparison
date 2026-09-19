from sentence_transformers import SentenceTransformer
import torch
from utils import load_data

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

data = load_data()

documents = [
    item["document"]
    for item in data
]

print("Loading dense retrieval model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

print("Creating document embeddings...")

document_embeddings = model.encode(
    documents,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

def search_dpr(query, top_k=5):
    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = document_embeddings @ query_embedding

    values, indices = torch.topk(
        scores,
        k=min(top_k, len(data))
    )

    results = []

    for score, idx in zip(values, indices):
        idx = int(idx)

        results.append({
            "doc_id": data[idx]["doc_id"],
            "document": data[idx]["document"],
            "score": float(score)
        })

    return results