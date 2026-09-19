from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_data

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
N_QUERIES = 50

N_LIST = 8
N_PROBE = 4

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 32

PQ_M = 4
PQ_BITS = 4

def create_embeddings(model, texts):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.astype("float32")

def measure_latency(search_function, query_embeddings, top_k):
    for query in query_embeddings[:5]:
        search_function(query, top_k)

    latencies = []

    for query in query_embeddings:
        start = time.perf_counter()

        search_function(
            query,
            top_k
        )

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    latencies = np.array(latencies)

    return {
        "Mean_ms": np.mean(latencies),
        "P50_ms": np.percentile(latencies, 50),
        "P95_ms": np.percentile(latencies, 95),
        "P99_ms": np.percentile(latencies, 99)
    }

def calculate_recall(ground_truth, retrieved):
    ground_truth = set(ground_truth)
    retrieved = set(retrieved)

    return len(
        ground_truth & retrieved
    ) / len(ground_truth)

def main():
    print("=" * 90)
    print("ANN BENCHMARK")
    print("=" * 90)

    data = load_data()

    documents = [
        item["document"]
        for item in data
    ]

    print("\nDocuments:", len(documents))

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("\nCreating document embeddings...")

    embeddings = create_embeddings(
        model,
        documents
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    dimension = embeddings.shape[1]

    queries = [
        item["query"]
        for item in data
    ]

    queries = queries[
        :min(N_QUERIES, len(queries))
    ]

    print("\nCreating query embeddings...")

    query_embeddings = create_embeddings(
        model,
        queries
    )

    # ========================================================
    # FLAT
    # ========================================================

    print("\nBuilding Flat index...")

    start = time.perf_counter()

    flat_index = faiss.IndexFlatIP(
        dimension
    )

    flat_index.add(
        embeddings
    )

    flat_build_time = (
        time.perf_counter() - start
    )

    def flat_search(query, top_k):
        return flat_index.search(
            query.reshape(1, -1),
            top_k
        )

    ground_truth = []

    for query in query_embeddings:
        _, indices = flat_search(
            query,
            TOP_K
        )

        ground_truth.append(
            indices[0]
        )

    flat_latency = measure_latency(
        flat_search,
        query_embeddings,
        TOP_K
    )

    # ========================================================
    # HNSW
    # ========================================================

    print("\nBuilding HNSW index...")

    start = time.perf_counter()

    hnsw_index = faiss.IndexHNSWFlat(
        dimension,
        HNSW_M,
        faiss.METRIC_INNER_PRODUCT
    )

    hnsw_index.hnsw.efConstruction = (
        HNSW_EF_CONSTRUCTION
    )

    hnsw_index.add(
        embeddings
    )

    hnsw_build_time = (
        time.perf_counter() - start
    )

    hnsw_index.hnsw.efSearch = (
        HNSW_EF_SEARCH
    )

    def hnsw_search(query, top_k):
        return hnsw_index.search(
            query.reshape(1, -1),
            top_k
        )

    hnsw_recalls = []

    for query, truth in zip(
        query_embeddings,
        ground_truth
    ):
        _, indices = hnsw_search(
            query,
            TOP_K
        )

        hnsw_recalls.append(
            calculate_recall(
                truth,
                indices[0]
            )
        )

    hnsw_recall = np.mean(
        hnsw_recalls
    )

    hnsw_latency = measure_latency(
        hnsw_search,
        query_embeddings,
        TOP_K
    )

    # ========================================================
    # IVF
    # ========================================================

    print("\nBuilding IVF index...")

    quantizer = faiss.IndexFlatIP(
        dimension
    )

    start = time.perf_counter()

    ivf_index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        N_LIST,
        faiss.METRIC_INNER_PRODUCT
    )

    ivf_index.train(
        embeddings
    )

    ivf_index.add(
        embeddings
    )

    ivf_build_time = (
        time.perf_counter() - start
    )

    ivf_index.nprobe = N_PROBE

    def ivf_search(query, top_k):
        return ivf_index.search(
            query.reshape(1, -1),
            top_k
        )

    ivf_recalls = []

    for query, truth in zip(
        query_embeddings,
        ground_truth
    ):
        _, indices = ivf_search(
            query,
            TOP_K
        )

        ivf_recalls.append(
            calculate_recall(
                truth,
                indices[0]
            )
        )

    ivf_recall = np.mean(
        ivf_recalls
    )

    ivf_latency = measure_latency(
        ivf_search,
        query_embeddings,
        TOP_K
    )

    # ========================================================
    # IVF-PQ
    # ========================================================

    print("\nBuilding IVF-PQ index...")

    ivf_pq_recall = None
    ivf_pq_latency = None
    ivf_pq_build_time = None

    try:
        quantizer_pq = faiss.IndexFlatIP(
            dimension
        )

        start = time.perf_counter()

        ivf_pq_index = faiss.IndexIVFPQ(
            quantizer_pq,
            dimension,
            N_LIST,
            PQ_M,
            PQ_BITS,
            faiss.METRIC_INNER_PRODUCT
        )

        ivf_pq_index.train(
            embeddings
        )

        ivf_pq_index.add(
            embeddings
        )

        ivf_pq_build_time = (
            time.perf_counter() - start
        )

        ivf_pq_index.nprobe = N_PROBE

        def ivf_pq_search(query, top_k):
            return ivf_pq_index.search(
                query.reshape(1, -1),
                top_k
            )

        recalls = []

        for query, truth in zip(
            query_embeddings,
            ground_truth
        ):
            _, indices = ivf_pq_search(
                query,
                TOP_K
            )

            recalls.append(
                calculate_recall(
                    truth,
                    indices[0]
                )
            )

        ivf_pq_recall = np.mean(
            recalls
        )

        ivf_pq_latency = measure_latency(
            ivf_pq_search,
            query_embeddings,
            TOP_K
        )

    except Exception as e:
        print("\nIVF-PQ failed:")
        print(e)

    print("\nANN PERFORMANCE COMPARISON")

    print(
        f"{'Index':<12}"
        f"{'Recall@5':<14}"
        f"{'Mean(ms)':<14}"
        f"{'P50(ms)':<14}"
        f"{'P95(ms)':<14}"
        f"{'P99(ms)':<14}"
        f"{'Build(s)':<14}"
    )

    print(
        f"{'Flat':<12}"
        f"{1.0000:<14.4f}"
        f"{flat_latency['Mean_ms']:<14.4f}"
        f"{flat_latency['P50_ms']:<14.4f}"
        f"{flat_latency['P95_ms']:<14.4f}"
        f"{flat_latency['P99_ms']:<14.4f}"
        f"{flat_build_time:<14.4f}"
    )

    print(
        f"{'HNSW':<12}"
        f"{hnsw_recall:<14.4f}"
        f"{hnsw_latency['Mean_ms']:<14.4f}"
        f"{hnsw_latency['P50_ms']:<14.4f}"
        f"{hnsw_latency['P95_ms']:<14.4f}"
        f"{hnsw_latency['P99_ms']:<14.4f}"
        f"{hnsw_build_time:<14.4f}"
    )

    print(
        f"{'IVF':<12}"
        f"{ivf_recall:<14.4f}"
        f"{ivf_latency['Mean_ms']:<14.4f}"
        f"{ivf_latency['P50_ms']:<14.4f}"
        f"{ivf_latency['P95_ms']:<14.4f}"
        f"{ivf_latency['P99_ms']:<14.4f}"
        f"{ivf_build_time:<14.4f}"
    )

    if ivf_pq_recall is not None:
        print(
            f"{'IVF-PQ':<12}"
            f"{ivf_pq_recall:<14.4f}"
            f"{ivf_pq_latency['Mean_ms']:<14.4f}"
            f"{ivf_pq_latency['P50_ms']:<14.4f}"
            f"{ivf_pq_latency['P95_ms']:<14.4f}"
            f"{ivf_pq_latency['P99_ms']:<14.4f}"
            f"{ivf_pq_build_time:<14.4f}"
        )

if __name__ == "__main__":
    main()