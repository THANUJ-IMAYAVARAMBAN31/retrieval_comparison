from preprocess import preprocess_bm25_tfidf
import torch
import re

def tfidf(tokenized):
    vocab = sorted(set(word for doc in tokenized for word in doc)) 
    word2idx = {word: i for i, word in enumerate(vocab)}

    # Count matrix
    count_matrix = torch.zeros(
        (len(tokenized), len(vocab)),
        dtype=torch.float32
    )

    for i, doc in enumerate(tokenized):
        for word in doc:
            count_matrix[i, word2idx[word]] += 1

    # TF
    doc_lengths = count_matrix.sum(dim=1, keepdim=True)
    tf = count_matrix / doc_lengths

    # DF
    df = (count_matrix > 0).float().sum(dim=0)

    # IDF
    N = len(tokenized)
    idf = torch.log((N + 1) / (df + 1)) + 1

    # TF-IDF
    tfidf = tf * idf

    return tfidf, idf, word2idx, vocab


def search_tfidf(query, tfidf, vocab, word2idx, idf, docs, top_k=3):

    # Preprocess query
    query = query.lower()
    query = re.sub(r"[^\w\s]", "", query)
    tokens = query.split()

    # Count vector
    query_count = torch.zeros(len(vocab))

    for word in tokens:
        if word in word2idx:
            query_count[word2idx[word]] += 1

    # TF
    if query_count.sum() == 0:
        return []

    query_tf = query_count / query_count.sum()

    # TF-IDF
    query_tfidf = query_tf * idf

    # Cosine similarity
    query_norm = torch.norm(query_tfidf)
    doc_norms = torch.norm(tfidf, dim=1)

    similarities = torch.matmul(tfidf, query_tfidf) / (doc_norms * query_norm + 1e-8)

    # Top-k
    scores, indices = torch.topk(similarities, top_k)

    results = []

    for score, idx in zip(scores, indices):
        results.append({
            "document": docs[idx]["document"],
            "score": float(score)
        })

    return results

