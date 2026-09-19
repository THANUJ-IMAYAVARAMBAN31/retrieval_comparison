import faiss
import numpy as np

def build_hnsw(embeddings,M=32,ef_construction=200):
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        M
    )

    index.hnsw.efConstruction = (
        ef_construction
    )

    index.add(embeddings)

    return index


def search_hnsw(
    index,
    query_embedding,
    top_k=5,
    ef_search=64
):

    index.hnsw.efSearch = ef_search

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(
            1, -1
        )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return distances[0], indices[0]