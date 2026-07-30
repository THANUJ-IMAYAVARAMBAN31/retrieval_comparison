import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_data
from preprocess import preprocess_bm25_tfidf

from tfidf import tfidf, search_tfidf
from bm25 import bm25, search_bm25
from dpr import search_dpr

from evaluator import evaluate

import pandas as pd

data = load_data()

tokenized_docs = preprocess_bm25_tfidf(data)

tfidf_matrix, idf, word2idx, vocab = tfidf(tokenized_docs)

bm25_matrix, idf_bm25, word2idx_bm25, vocab_bm25 = bm25(tokenized_docs)

models = {
    "TF-IDF": lambda q, k: search_tfidf(
        q,
        tfidf_matrix,
        vocab,
        word2idx,
        idf,
        data,
        top_k=k
    ),

    "BM25": lambda q, k: search_bm25(
        q,
        bm25_matrix,
        vocab_bm25,
        word2idx_bm25,
        idf_bm25,
        data,
        top_k=k
    ),

    "DPR": lambda q, k: search_dpr(
        q,
        top_k=k
    )
}

results = []

for name, retriever in models.items():

    metrics = evaluate(
        retriever,
        data,
        k=5
    )

    metrics["Model"] = name

    results.append(metrics)

df = pd.DataFrame(results)

print(df.to_string(index=False))