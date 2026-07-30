from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent

def load_data():
    file_path = BASE_DIR / "docs" / "documents.json"

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)