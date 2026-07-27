from utils import load_data
from preprocess import preprocess_bm25_tfidf

from tfidf import tfidf, search_tfidf
from bm25 import bm25, search_bm25
from dpr import search_dpr

data = load_data()

docs = [x["document"] for x in data]

tokenized_docs = preprocess_bm25_tfidf(data)

tfidf_matrix, idf, word2idx, vocab = tfidf(tokenized_docs)

bm25_matrix, idf_bm25, word2idx_bm25, vocab_bm25 = bm25(tokenized_docs)

queries = [

    # Exact lexical match
    "what is nlp",

    # Semantic paraphrase
    "how do computers understand human language",

    # Exact lexical match
    "what is machine learning",

    # Semantic paraphrase
    "how can computers learn from data",

    # Exact lexical match
    "what is encryption",

    # Semantic paraphrase
    "how is information kept secret",

    # Exact lexical match
    "what is api",

    # Semantic paraphrase
    "how do two software applications communicate"

]

for query in queries:

    print("=" * 80)
    print("QUERY :", query)
    print("=" * 80)

    print("\nTF-IDF")
    results = search_tfidf(
        query,
        tfidf_matrix,
        vocab,
        word2idx,
        idf,
        data
    )

    for r in results:
        print(f"Score : {r['score']:.4f}")
        print(r["document"])
        print()

    print("-" * 80)

    print("BM25")
    results = search_bm25(
        query,
        bm25_matrix,
        vocab_bm25,
        word2idx_bm25,
        idf_bm25,
        data
    )

    for r in results:
        print(f"Score : {r['score']:.4f}")
        print(r["document"])
        print()

    print("-" * 80)

    print("DPR")
    results = search_dpr(
        query,
        top_k=3
    )

    for r in results:
        print(f"Score : {r['score']:.4f}")
        print(r["document"])
        print()

    print("\n\n")