from utils import load_data
from bm25 import search_bm25
from dpr import search_dpr


def reciprocal_rank_fusion(
    ranked_lists,
    k=60
):
    """
    Reciprocal Rank Fusion (RRF).

    ranked_lists:
        List of ranked document-id lists.

    Example:
        [
            [10, 5, 7, 2],
            [5, 8, 10, 3]
        ]

    RRF score:

        score(d) = sum(1 / (k + rank))

    where rank starts from 1.
    """

    scores = {}

    for ranked_list in ranked_lists:

        for rank, doc_id in enumerate(
            ranked_list,
            start=1
        ):

            if doc_id not in scores:
                scores[doc_id] = 0.0

            scores[doc_id] += 1 / (k + rank)

    ranked_documents = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_documents


def hybrid_search(
    query,
    bm25_matrix,
    vocab_bm25,
    word2idx_bm25,
    idf_bm25,
    data,
    top_k=5,
    retrieval_k=20
):
    """
    Hybrid BM25 + DPR retrieval using RRF.

    BM25:
        lexical / sparse retrieval

    DPR:
        semantic / dense retrieval

    RRF:
        combines their rankings.
    """

    # ---------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------

    bm25_results = search_bm25(
        query,
        bm25_matrix,
        vocab_bm25,
        word2idx_bm25,
        idf_bm25,
        data,
        top_k=retrieval_k
    )

    # ---------------------------------------------------------
    # DPR retrieval
    # ---------------------------------------------------------

    dpr_results = search_dpr(
        query,
        top_k=retrieval_k
    )

    # ---------------------------------------------------------
    # Extract document IDs
    # ---------------------------------------------------------

    bm25_ids = [
        result["doc_id"]
        for result in bm25_results
    ]

    dpr_ids = [
        result["doc_id"]
        for result in dpr_results
    ]

    # ---------------------------------------------------------
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        [
            bm25_ids,
            dpr_ids
        ]
    )

    # ---------------------------------------------------------
    # Map doc_id -> original document
    # ---------------------------------------------------------

    documents_by_id = {
        item["doc_id"]: item
        for item in data
    }

    results = []

    for doc_id, rrf_score in fused_results[:top_k]:

        document = documents_by_id[doc_id]

        results.append({
            "doc_id": doc_id,
            "query": document["query"],
            "document": document["document"],
            "score": rrf_score
        })

    return results