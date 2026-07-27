from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import re
from utils import load_data

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

QUERY_MODEL_PATH = "project/models/query_encoder"

EMBEDDING_PATH = "project/embeddings/document_embeddings.pt"

data = load_data()

def load_query_encoder():
    model = AutoModel.from_pretrained(QUERY_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_PATH)

    model.to(device)
    model.eval()

    return model, tokenizer

def load_document_embeddings():
    embeddings = torch.load(EMBEDDING_PATH,map_location=device)
    return embeddings

def encode_query(query, model, tokenizer):
    inputs = tokenizer(query,padding=True,truncation=True,return_tensors="pt")
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        embedding = mean_pooling(outputs, inputs["attention_mask"])

        embedding = F.normalize(embedding,dim=1)

        return embedding

def mean_pooling(outputs, attention_mask):

    token_embeddings = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).float()

    pooled = (token_embeddings * mask).sum(dim=1)
    pooled = pooled / mask.sum(dim=1)

    return pooled

def search_dpr(query,top_k=2):
    model, tokenizer = load_query_encoder()

    document_emb = load_document_embeddings()

    query_embedding = encode_query(query, model, tokenizer)

    scores = query_embedding @ document_emb.T

    values, indices = torch.topk(
        scores,
        top_k
    )

    results = []

    for score, idx in zip(values[0], indices[0]):

        results.append({

            "document": data[idx]["document"],

            "score": float(score)

        })

    return results