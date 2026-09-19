import faiss
import numpy as np


def build_ivf( embeddings,nlist=100):

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    quantizer = faiss.IndexFlatL2(
        dimension
    )

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_L2
    )

    index.train(embeddings)

    index.add(embeddings)

    return index


def search_ivf(
    index,
    query_embedding,
    top_k=5,
    nprobe=10
):

    index.nprobe = nprobe

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