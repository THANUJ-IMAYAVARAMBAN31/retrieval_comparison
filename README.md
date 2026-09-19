# Retrieval Comparison: Lexical, Dense, Hybrid & ANN Search

A practical Information Retrieval benchmark comparing **lexical retrieval, dense semantic retrieval, hybrid retrieval, and approximate nearest-neighbor (ANN) search** on a small custom retrieval dataset.

The project is designed to demonstrate how different retrieval strategies behave in terms of:

* Retrieval quality
* Semantic vs lexical matching
* Hybrid retrieval
* Ranking quality
* ANN recall
* Query latency
* Index build time
---

## Implemented

* TF-IDF retrieval
* BM25 retrieval
* Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`
* Hybrid BM25 + Dense retrieval using Reciprocal Rank Fusion (RRF)
* Hit@K
* Precision@K
* Recall@K
* MRR
* MAP
* Exact nearest-neighbor search using FAISS Flat
* HNSW
* IVF
* IVF-PQ
* Mean latency
* P50 latency
* P95 latency
* P99 latency
* Index build-time benchmarking

# Results 

### Retrieval benchmark results

---

Current benchmark results on the 179-sample dataset:

| Model           |    Hit@5 | Precision@5 | Recall@5 |      MRR |      MAP |
| --------------- | -------: | ----------: | -------: | -------: | -------: |
| TF-IDF          | 0.955556 |    0.191111 | 0.955556 | 0.726574 | 0.726574 |
| BM25            | 0.955556 |    0.191111 | 0.955556 | 0.683704 | 0.683704 |
| Dense           | 1.000000 |    0.200000 | 1.000000 | 0.991852 | 0.991852 |
| Hybrid-BM25+DPR | 0.972222 |    0.194444 | 0.972222 | 0.952037 | 0.952037 |

### ANN benchmark results
---

| Index  | Recall@5 | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Build (s) |
| ------ | -------: | --------: | -------: | -------: | -------: | --------: |
| Flat   |   1.0000 |    0.0157 |   0.0142 |   0.0159 |   0.0530 |    0.0003 |
| HNSW   |   0.9960 |    0.0303 |   0.0276 |   0.0395 |   0.0749 |    0.0071 |
| IVF    |   0.9680 |    0.0181 |   0.0180 |   0.0211 |   0.0255 |    0.0100 |
| IVF-PQ |   0.4680 |    0.0493 |   0.0448 |   0.0561 |   0.1369 |    0.0068 |

### Dataset

The current project uses a custom dataset containing **179 query-document pairs**.

Each sample contains:

```text
{
    "doc_id": 178,
    "query": "what is retrieval augmented generation",
    "document": "Retrieval-Augmented Generation combines information retrieval with language generation to produce more accurate responses."
}
```

The dataset is intentionally small and is currently used to validate the retrieval implementations and compare their behavior.

### Current limitation

Each query currently has **one relevant document**.

Therefore:

```text
Hit@K = Recall@K
```

for this dataset.

Similarly, because there is only one relevant document per query:

```text
MAP = MRR
```

This is a limitation of the current evaluation dataset, not a property of the metrics themselves.

---
# Architecture

The project currently compares four retrieval approaches:

```text
                         Query
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       TF-IDF            BM25           Dense
          |                |                |
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Hybrid Retrieval
                       BM25 + Dense
                           |
                           v
                         RRF
                           |
                           v
                       Top-K Docs
```

Dense retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Document embeddings are created once when the dense retriever is loaded.

---

# Retrieval Methods

## 1. TF-IDF

TF-IDF provides a classical sparse lexical retrieval baseline.

It primarily matches query terms with terms appearing in documents.

General intuition:

```text
Query
  |
  v
Term frequency
  +
Inverse document frequency
  |
  v
Document score
```

Implemented in:

```text
tfidf.py
```

---

## 2. BM25

BM25 is a probabilistic lexical retrieval method and improves upon simple TF-IDF-style scoring by accounting for:

* Term frequency saturation
* Document length normalization
* Inverse document frequency

Implemented in:

```text
bm25.py
```

---

## 3. Dense Semantic Retrieval

Dense retrieval converts queries and documents into vector representations.

The project currently uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The retrieval process is:

```text
Document
   |
   v
Embedding Model
   |
   v
Document Vector

Query
   |
   v
Embedding Model
   |
   v
Query Vector
   |
   v
Similarity Search
   |
   v
Top-K Documents
```

Embeddings are normalized, allowing the dot product to represent cosine similarity.

Implemented in:

```text
dpr.py
```

> Note: The file is currently named `dpr.py` for project continuity, but the implementation uses `all-MiniLM-L6-v2`, which is a Sentence-Transformer bi-encoder rather than the original Facebook/DPR architecture. A future cleanup can rename this module to `dense.py`.

---

# 4. Hybrid Retrieval

Hybrid retrieval combines:

```text
BM25
+
Dense Semantic Retrieval
```

The project uses **Reciprocal Rank Fusion (RRF)** rather than directly adding BM25 and dense scores.

This is important because the raw score scales of BM25 and dense similarity are different.

The pipeline is:

```text
                  Query
                    |
          +---------+---------+
          |                   |
          v                   v
        BM25               Dense
          |                   |
       Top-20              Top-20
          |                   |
          +---------+---------+
                    |
                    v
                   RRF
                    |
                    v
                 Top-K
```

RRF score:

```text
RRF(d) = Σ 1 / (k + rank)
```

Implemented in:

```text
hybrid/
└── hybrid.py
```

---

# Evaluation

The project evaluates retrieval quality using:

### Hit@K

Checks whether the relevant document appears anywhere in the top K results.

```text
Hit@K = 1 if relevant document ∈ top K
        0 otherwise
```

### Precision@K

Measures how much of the retrieved top-K set is relevant.

For the current one-relevant-document dataset:

```text
Precision@K = Hit@K / K
```

### Recall@K

Measures whether the relevant document was retrieved.

Because the current dataset contains one relevant document:

```text
Recall@K = Hit@K
```

### MRR

Mean Reciprocal Rank measures how early the relevant document appears.

```text
RR = 1 / rank
```

### MAP

Mean Average Precision is implemented for the current relevance setup.

Since each query currently has only one relevant document:

```text
MAP = MRR
```

for this dataset.

Evaluation code:

```text
Evaluation/
├── metrics.py
├── evaluator.py
└── benchmark.py
```

### Observations

The current dataset shows:

* TF-IDF retrieves the relevant document in the top 5 for about 95.56% of queries.
* BM25 has the same Hit@5 and Recall@5 as TF-IDF on this dataset.
* Dense retrieval achieves 100% Hit@5 and Recall@5.
* Dense retrieval also places the relevant document very early in the ranking, reflected by its MRR of approximately 0.992.
* Hybrid BM25 + Dense retrieval achieves approximately 97.22% Hit@5.
* On this particular dataset, the hybrid approach does not outperform the dense retriever.

These results should not be generalized to larger retrieval workloads because the current dataset is small and has only one relevant document per query.

---

# ANN Benchmark

The project also compares several FAISS vector indexes.

```text
Dense Embeddings
       |
       +----------------+
       |                |
       v                v
     Exact             ANN
     Search             |
       |          +-----+-----+------+
       |          |           |      |
       v          v           v      v
     Flat       HNSW        IVF    IVF-PQ
```

The benchmark measures:

* Recall@5
* Mean latency
* P50 latency
* P95 latency
* P99 latency
* Index build time

### Important benchmark limitation

The current corpus contains only **179 documents**.

This is far too small to demonstrate the real production advantages of ANN indexing.

For a small corpus, exact Flat search is already extremely cheap.

The ANN benchmark is therefore primarily an **implementation and behavior experiment**, rather than a realistic large-scale performance comparison.

With millions of vectors, the trade-offs between:

```text
Recall
Latency
Memory
Index construction time
Search parameters
```

become much more meaningful.

Implemented in:

```text
benchmarks/
├── ann_benchmark.py
└── latency.py
```

---

# ANN Techniques

## Flat

FAISS exact inner-product search.

```text
Query
  |
  v
Compare against every vector
  |
  v
Exact Top-K
```

Provides the reference result used to calculate ANN recall.

---

## HNSW

Hierarchical Navigable Small World graph.

The index builds a graph connecting nearby vectors.

Search navigates the graph instead of exhaustively comparing against every vector.

Important parameters:

```text
M
efConstruction
efSearch
```

Implemented in:

```text
ann/hnsw.py
```

---

## IVF

Inverted File index.

The vector space is partitioned into clusters.

At query time, only selected clusters are searched.

Important parameters:

```text
nlist
nprobe
```

Implemented in:

```text
ann/ivf.py
```

---

## IVF-PQ

Combines:

```text
IVF
+
Product Quantization
```

IVF reduces the search space while PQ compresses vectors.

This creates a memory/accuracy trade-off.

Implemented in:

```text
ann/ivf_pq.py
```

---

# Project Structure

```text
retrieval_comparison/
│
├── .gitignore
├── bm25.py
├── dpr.py
├── embeddings.py
├── main.py
├── preprocess.py
├── README.md
├── requirements.txt
├── tfidf.py
├── utils.py
│
├── ann/
│   ├── __init__.py
│   ├── hnsw.py
│   ├── ivf.py
│   └── ivf_pq.py
│
├── benchmarks/
│   ├── ann_benchmark.py
│   └── latency.py
│
├── docs/
│   └── documents.json
│
├── Evaluation/
│   ├── benchmark.py
│   ├── evaluator.py
│   └── metrics.py
│
└── hybrid/
    ├── __init__.py
    └── hybrid.py
```

`__pycache__/` directories are generated automatically by Python and should not be committed to Git.

---

# Current Retrieval Pipeline

The complete current pipeline is:

```text
                    Dataset
                       |
                       v
                 Preprocessing
                       |
          +------------+------------+
          |            |            |
          v            v            v
       TF-IDF        BM25        Dense
          |            |            |
          |            |            |
          +------------+------------+
                       |
                       v
                    Hybrid
                       |
                      RRF
                       |
                       v
                    Top-K
                       |
                       v
                   Evaluation
```

The vector-search branch additionally supports:

```text
Dense Embeddings
       |
       +------> Flat
       |
       +------> HNSW
       |
       +------> IVF
       |
       +------> IVF-PQ
```

# Goal of the Project

The goal is not simply to implement several retrieval algorithms.

The project is intended to demonstrate the engineering trade-offs involved in building a retrieval system:

```text
Lexical Retrieval
      ↓
Semantic Retrieval
      ↓
Hybrid Retrieval
      ↓
Approximate Nearest Neighbor
      ↓
Evaluation
      ↓
Latency / Recall / Memory Trade-offs
```

This provides a practical foundation for building production-oriented **RAG and LLM retrieval systems**.
