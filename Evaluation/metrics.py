def precision_at_k(retrieved, ground_truth, k):

    hits = int(ground_truth in retrieved[:k])

    return hits / k


def recall_at_k(retrieved, ground_truth, k):

    return int(ground_truth in retrieved[:k])


def reciprocal_rank(retrieved, ground_truth):

    for rank, doc in enumerate(retrieved, start=1):

        if doc == ground_truth:

            return 1 / rank

    return 0


def average_precision(retrieved, ground_truth):

    for rank, doc in enumerate(retrieved, start=1):

        if doc == ground_truth:

            return 1 / rank

    return 0

def hit_at_k(retrieved, ground_truth, k):
    return int(ground_truth in retrieved[:k])


def hit_rate_at_k(retrieved_results, ground_truths, k):
    hits = 0

    for retrieved, ground_truth in zip(retrieved_results,ground_truths):
        hits += hit_at_k(retrieved,ground_truth,k)

    return hits / len(ground_truths)