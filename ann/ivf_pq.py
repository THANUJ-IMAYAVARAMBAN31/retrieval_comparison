import faiss
import numpy as np


def build_ivf_pq(
    embeddings,
    nlist=100,
    m=16,
    nbits=8
):

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    quantizer = faiss.IndexFlatL2(
        dimension
    )

    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        m,
        nbits
    )

    index.train(embeddings)

    index.add(embeddings)

    return index


def search_ivf_pq(
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