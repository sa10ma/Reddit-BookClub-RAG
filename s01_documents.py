# 01_documents.py
import json
import os

def load_documents(file_path="data/sparknotes_bluesteye.json"):
    """
    Loads raw SparkNotes literature documents from the local data folder.
    Returns a list of document dictionaries.
    """
    file_path = file_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Run scrape_books.py first!")
        
    with open(file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
        
    return documents

if __name__ == "__main__":
    docs = load_documents()
    print(docs[4])
    print(f"Loaded {len(docs)} documents successfully from local storage.")