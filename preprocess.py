import re
import json

def preprocess_bm25_tfidf(docs):
    
    document = [x["document"] for x in docs]

    tokenized = []

    for doc in document:
        doc = doc.lower()
        doc = re.sub(r"[^\w\s]", "", doc)
        tokenized.append(doc.split())

    return tokenized

def preprocess_dpr(tokenizer,docs):

    with open("project\docs\documents.json") as f:
        docs = json.load(f)

    queries = [x["query"] for x in docs]
    document = [x["document"] for x in docs]

    query_emb = tokenizer(queries,padding=True,truncation=True,return_tensors="pt")
    doc_emb = tokenizer(document,padding=True,truncation=True,return_tensors="pt")

    return query_emb,doc_emb
