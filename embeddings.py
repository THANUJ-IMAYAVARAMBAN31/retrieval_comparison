from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "project/models/bert-base-uncased"

def load_encoder():

    if not os.path.exists(MODEL_PATH):

        print("Downloading pretrained BERT...")

        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        os.makedirs(MODEL_PATH, exist_ok=True)

        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)

    else:

        print("Loading pretrained BERT...")

        model = AutoModel.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    return model, tokenizer

class DPRDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        return {
            "query": sample["query"],
            "document": sample["document"]
        }


class DPRModel(nn.Module):

    def __init__(self, model_name=MODEL_PATH):

        super().__init__()

        self.encoder_query = AutoModel.from_pretrained(model_name)
        self.encoder_doc = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, outputs, attention_mask):

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (token_embeddings * mask).sum(dim=1)

        pooled = pooled / mask.sum(dim=1)

        return pooled

    def encode_query(self, inputs):

        outputs = self.encoder_query(**inputs)

        embeddings = self.mean_pooling(
            outputs,
            inputs["attention_mask"]
        )

        embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    def encode_doc(self, inputs):

        outputs = self.encoder_doc(**inputs)

        embeddings = self.mean_pooling(
            outputs,
            inputs["attention_mask"]
        )

        embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    def forward(self, query_inputs, doc_inputs):

        q = self.encode_query(query_inputs)

        d = self.encode_doc(doc_inputs)

        return q, d

def info_nce_loss(q_emb, d_emb, temperature=0.07):

    similarity = q_emb @ d_emb.T

    similarity = similarity / temperature

    labels = torch.arange(
        similarity.size(0),
        device=similarity.device
    )

    loss = F.cross_entropy(similarity, labels)

    return loss

def train(model, tokenizer, dataloader, optimizer, epochs):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch in dataloader:

            query_inputs = tokenizer(
                batch["query"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            doc_inputs = tokenizer(
                batch["document"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            query_inputs = {
                k: v.to(device)
                for k, v in query_inputs.items()
            }

            doc_inputs = {
                k: v.to(device)
                for k, v in doc_inputs.items()
            }

            optimizer.zero_grad()

            q_emb, d_emb = model(
                query_inputs,
                doc_inputs
            )

            loss = info_nce_loss(
                q_emb,
                d_emb
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}"
        )

def save_models(model, tokenizer):

    query_path = "project/models/query_encoder"
    doc_path = "project/models/doc_encoder"

    os.makedirs(query_path, exist_ok=True)
    os.makedirs(doc_path, exist_ok=True)

    model.encoder_query.save_pretrained(query_path)
    model.encoder_doc.save_pretrained(doc_path)

    tokenizer.save_pretrained(query_path)
    tokenizer.save_pretrained(doc_path)

def create_document_embeddings(model, tokenizer, dataloader):

    model.eval()

    all_embeddings = []

    with torch.no_grad():

        for batch in dataloader:

            doc_inputs = tokenizer(
                batch["document"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            doc_inputs = {
                k: v.to(device)
                for k, v in doc_inputs.items()
            }

            embeddings = model.encode_doc(doc_inputs)

            all_embeddings.append(
                embeddings.cpu()
            )

    all_embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    os.makedirs(
        "project/embeddings",
        exist_ok=True
    )

    torch.save(
        all_embeddings,
        "project/embeddings/document_embeddings.pt"
    )

    print("Document embeddings saved.")

def main():

    data = load_data()

    _, tokenizer = load_encoder()

    dataset = DPRDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    model = DPRModel(MODEL_PATH)

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-5
    )

    train(
        model,
        tokenizer,
        dataloader,
        optimizer,
        epochs=5
    )

    save_models(
        model,
        tokenizer
    )

    create_document_embeddings(
        model,
        tokenizer,
        dataloader
    )


if __name__ == "__main__":
    main()