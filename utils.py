import json

def load_data():
    
    with open("project\docs\documents.json") as f:
        docs = json.load(f)

    return docs