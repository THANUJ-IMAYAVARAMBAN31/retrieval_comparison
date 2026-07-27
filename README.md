# Information Retrieval System Comparison

A Python implementation comparing three popular information retrieval techniques:

- **TF-IDF (Term Frequency - Inverse Document Frequency)**
- **BM25 (Best Matching 25)**
- **DPR (Dense Passage Retrieval using BERT)**

The project demonstrates the difference between **sparse retrieval** (TF-IDF, BM25) and **dense retrieval** (DPR), particularly on semantic search queries.

---

# Project Structure

```
project/
│
├── data/
│   └── documents.json
│
├── embeddings/
│   └── document_embeddings.pt
│
├── models/
│   ├── bert-base-uncased/
│   ├── query_encoder/
│   └── doc_encoder/
│
├── utils.py
├── preprocess.py
├── tfidf.py
├── bm25.py
├── embeddings.py
├── dpr.py
├── main.py
│
└── README.md
```

---

# Retrieval Methods

## 1. TF-IDF

TF-IDF is a sparse retrieval technique that represents each document as a vector of weighted terms.

### Pipeline

- Load dataset
- Preprocess text
- Build vocabulary
- Compute Term Frequency (TF)
- Compute Inverse Document Frequency (IDF)
- Create TF-IDF vectors
- Compute cosine similarity between query and documents
- Return Top-K documents

Similarity:

```
Cosine(Query, Document)
```

---

## 2. BM25

BM25 is an improved probabilistic ranking function based on TF-IDF.

It considers

- Term Frequency
- Document Length
- Inverse Document Frequency

instead of relying purely on cosine similarity.

Pipeline

- Preprocess documents
- Build vocabulary
- Compute document frequencies
- Compute BM25 scores
- Score every document for the query
- Return Top-K documents

---

## 3. DPR (Dense Passage Retrieval)

DPR is a dense retrieval approach that uses BERT embeddings instead of keyword matching.

Architecture

```
                 Query
                   │
                   ▼
           Query Encoder (BERT)
                   │
                   ▼
            Query Embedding
                   │
          Dot Product Similarity
                   │
                   ▼
      Document Embeddings (Offline)
                   │
                   ▼
            Top-K Documents
```

Unlike TF-IDF and BM25, DPR can retrieve semantically similar documents even when the wording is different.

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- BERT Base Uncased

---

# Dataset Format

```
[
    {
        "query": "what is nlp",
        "document": "Natural Language Processing enables computers to understand human language."
    },
    {
        "query": "what is machine learning",
        "document": "Machine learning enables computers to learn patterns from data."
    }
]
```

---

# Workflow

```
Documents
    │
    ▼
Preprocessing
    │
    ├──────────────┐
    │              │
    ▼              ▼
 TF-IDF          BM25
    │              │
    ▼              ▼
 Sparse Search  Sparse Search

Documents
    │
    ▼
DPR Training
    │
    ▼
Document Embeddings
    │
    ▼
Dense Search
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/THANUJ-IMAYAVARAMBAN31/retrieval_comparison
cd project
```

Install dependencies

```bash
pip install torch transformers
```

---

# Running the Project

## Step 1 — Train DPR

Run

```bash
python embeddings.py
```

This will

- Download BERT (first run only)
- Fine-tune the DPR encoders
- Save the trained query encoder
- Save the trained document encoder
- Generate document embeddings

Generated files

```
models/query_encoder/

models/doc_encoder/

embeddings/document_embeddings.pt
```

---

## Step 2 — Compare Retrieval Methods

Run

```bash
python main.py
```

This performs

- TF-IDF Retrieval
- BM25 Retrieval
- DPR Retrieval

on both exact keyword queries and semantic queries.

---

# Example Queries

Exact keyword

```
what is nlp
```

Semantic query

```
how do computers understand human language
```

Exact keyword

```
what is machine learning
```

Semantic query

```
how can computers learn from data
```

Exact keyword

```
what is api
```

Semantic query

```
how do software applications communicate
```

---

# Expected Results

| Query Type | TF-IDF | BM25 | DPR |
|------------|---------|------|------|
| Exact Keyword Match | Excellent | Excellent | Excellent |
| Partial Keyword Match | Good | Excellent | Excellent |
| Semantic Query | Poor | Poor | Excellent |

---

# Output Example

```
==================================================
Query:
what is nlp
==================================================

TF-IDF

Score : 0.84

Natural Language Processing enables computers to understand human language.

---------------------------------------

BM25

Score : 4.21

Natural Language Processing enables computers to understand human language.

---------------------------------------

DPR

Score : 0.97

Natural Language Processing enables computers to understand human language.
```

---

# Components

## utils.py

- Load dataset

---

## preprocess.py

- Lowercase conversion
- Remove punctuation
- Tokenization
- Shared preprocessing for TF-IDF and BM25

---

## tfidf.py

- Build TF-IDF matrix
- Cosine similarity search

---

## bm25.py

- Build BM25 matrix
- BM25 document ranking

---

## embeddings.py

- DPR Dataset
- DPR Model
- Training loop
- InfoNCE Loss
- Save trained models
- Generate document embeddings

---

## dpr.py

- Load trained query encoder
- Load document embeddings
- Encode query
- Dense retrieval

---

## main.py

Runs and compares

- TF-IDF
- BM25
- DPR

using the same queries.

---

# Future Improvements

- Use the official Facebook DPR models
- Train with positive and hard negative passages
- Use FAISS for fast dense retrieval
- Support large document collections
- Add Precision@K, Recall@K, and MRR evaluation
- Add a Streamlit or Gradio web interface

---

# References

- Robertson, S. and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
- HuggingFace Transformers Documentation
- PyTorch Documentation

---

# Author

Information Retrieval System Comparison Project

- Sparse Retrieval: TF-IDF, BM25
- Dense Retrieval: DPR (BERT)