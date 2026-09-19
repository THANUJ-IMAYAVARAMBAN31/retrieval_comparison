import time
import numpy as np


def measure_latency(retriever, query_embeddings, top_k=5, warmup=5):
    
    for query_embedding in query_embeddings[:warmup]:
        retriever(query_embedding, top_k)

    latencies = []

    for query_embedding in query_embeddings:

        start = time.perf_counter()

        retriever(query_embedding, top_k)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    latencies = np.array(latencies)

    return {
        "Mean_ms": np.mean(latencies),
        "P50_ms": np.percentile(latencies, 50),
        "P95_ms": np.percentile(latencies, 95),
        "P99_ms": np.percentile(latencies, 99),
        "QPS": 1000 / np.mean(latencies)
    }