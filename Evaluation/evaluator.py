from metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    reciprocal_rank,
    average_precision
)


def evaluate(retriever, dataset, k):

    precision = 0
    recall = 0
    hit = 0
    mrr = 0
    map_score = 0

    n = len(dataset)

    for sample in dataset:

        query = sample["query"]

        ground_truth = sample["doc_id"]

        retrieved = retriever(query, k)

        retrieved_ids = [
            doc["doc_id"]
            for doc in retrieved
        ]

        precision += precision_at_k(
            retrieved_ids,
            ground_truth,
            k
        )

        recall += recall_at_k(
            retrieved_ids,
            ground_truth,
            k
        )

        hit += hit_at_k(
            retrieved_ids,
            ground_truth,
            k
        )

        mrr += reciprocal_rank(
            retrieved_ids,
            ground_truth
        )

        map_score += average_precision(
            retrieved_ids,
            ground_truth
        )

    return {

        f"Precision@{k}":
            precision / n,

        f"Recall@{k}":
            recall / n,

        f"Hit@{k}":
            hit / n,

        f"HitRate@{k}":
            hit / n,

        "MRR":
            mrr / n,

        "MAP":
            map_score / n
    }