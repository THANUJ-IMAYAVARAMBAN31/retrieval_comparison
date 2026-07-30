import re
import torch

def bm25(tokenized):
    vocab = sorted(set(word for doc in tokenized for word in doc))  
    word2idx = {word: i for i, word in enumerate(vocab)}

    # Count matrix
    count_matrix = torch.zeros((len(tokenized), len(vocab)), dtype=torch.float32)

    for i, doc in enumerate(tokenized):
        for word in doc:
            count_matrix[i, word2idx[word]] += 1

    # TF
    doc_lengths = count_matrix.sum(dim=1, keepdim=True)
    tf = count_matrix 

    # DF
    df = (count_matrix > 0).float().sum(dim=0)

    # IDF
    N = len(tokenized)
    idf = torch.log((N -df + 0.5) / (df + 0.5) +1)   

    k1,b = 1.2,0.75
    bm25 = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lengths / doc_lengths.mean())) 

    return bm25, idf, word2idx, vocab

def search_bm25(query, bm25, vocab, word2idx, idf, docs, top_k=3):
    query = query.lower()
    query = re.sub(r"[^\w\s]", "", query)
    token = query.split()

    scores = torch.zeros(bm25.shape[0])

    for word in token:
        if word in word2idx:
            scores += bm25[:,word2idx[word]]

    values,indices = torch.topk(scores, top_k)

    results = []

    for score, idx in zip(values, indices):
        results.append({
            "doc_id": docs[idx]["doc_id"],
            "document": docs[idx]["document"],
            "score": float(score)
        })

    return results
