import sys
from pathlib import Path

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0,str(PROJECT_ROOT))

sys.path.insert(0,str(PROJECT_ROOT / "Evaluation"))

from utils import load_data
from preprocess import preprocess_bm25_tfidf
from tfidf import (tfidf,search_tfidf)
from bm25 import (bm25,search_bm25)
from dpr import search_dpr
from hybrid.hybrid import hybrid_search
from evaluator import evaluate

data = load_data()

tokenized_docs = preprocess_bm25_tfidf(data)

tfidf_matrix, idf, word2idx, vocab = tfidf(tokenized_docs)

bm25_matrix, idf_bm25, word2idx_bm25, vocab_bm25 = bm25(
    tokenized_docs
)

models = {
    "TF-IDF": lambda query, k:
        search_tfidf(
            query,
            tfidf_matrix,
            vocab,
            word2idx,
            idf,
            data,
            top_k=k
        ),

    "BM25": lambda query, k:
        search_bm25(
            query,
            bm25_matrix,
            vocab_bm25,
            word2idx_bm25,
            idf_bm25,
            data,
            top_k=k
        ),

    "DPR": lambda query, k:
        search_dpr(
            query,
            top_k=k
        ),

    "Hybrid-BM25+DPR": lambda query, k:
        hybrid_search(
            query,
            bm25_matrix,
            vocab_bm25,
            word2idx_bm25,
            idf_bm25,
            data,
            top_k=k,
            retrieval_k=20
        )
}

results = []
K = 5

for name, retriever in models.items():
    print(f"\nEvaluating {name}...")

    metrics = evaluate(
        retriever,
        data,
        k=K
    )
    metrics["Model"] = name
    results.append(metrics)

df = pd.DataFrame(results)
columns = [
    "Model",
    f"Hit@{K}",
    f"Precision@{K}",
    f"Recall@{K}",
    "MRR",
    "MAP"
]

df = df[columns]

print("RETRIEVAL BENCHMARK")

print(df.to_string(index=False))