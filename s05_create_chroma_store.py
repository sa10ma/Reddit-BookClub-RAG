# s05_chroma.py
import chromadb
from s03_chunking import create_chunks
# Import your vectorization module / class
from s04_vector_representation import TextRetriever

def get_chroma_client(db_path="./chroma_db"):
    """Initializes and returns a persistent local ChromaDB client."""
    return chromadb.PersistentClient(path=db_path)


def populate_chroma_store(
    db_path="./chroma_db", 
    collection_name="bluest_eye_rag", 
    strategy="sentence", 
    chunk_size=60, #for when staregy is "fixed"
    max_words=60, #for when strategy is "sentence"
    overlap=15, 
    overlap_sentences = 1,
    extract_ner=True,
    reset_db=True
):
    """
    Runs the chunking pipeline, embeds texts using the shared vectorization module, 
    and upserts them into a persistent ChromaDB collection with full metadata.
    """
    client = get_chroma_client(db_path)

    if reset_db:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection '{collection_name}' to apply new chunking strategy.")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)
    
    print("Generating chunks from preprocessing pipeline...")
    chunks = create_chunks(
        strategy=strategy, 
        chunk_size=chunk_size, 
        overlap=overlap, 
        max_words= max_words,
        overlap_sentences=overlap_sentences,
        extract_ner=extract_ner
    )
    
    if not chunks:
        print("No chunks found to store in ChromaDB.")
        return collection
        
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        
        # Format metadata (ChromaDB requires primitive types: str, int, float, bool)
        meta = {
            "original_doc_id": chunk["original_doc_id"],
            "chapter": chunk["chapter"],
            "type": chunk["type"],
            "entities": ", ".join(chunk.get("entities", [])) if chunk.get("entities") else ""
        }
        metadatas.append(meta)
        
    print("Initializing vectorizer and encoding chunks...")
    # Instantiate the TextRetriever using our chunk documents as the corpus
    retriever = TextRetriever(documents)
    
    # Generate dense embeddings using the encoder from the vectorization class
    embeddings = retriever.embedding_model.encode(
        documents, 
        normalize_embeddings=True, 
        show_progress_bar=True
    ).tolist()
    
    print(f"Upserting records into ChromaDB collection '{collection_name}'...")
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Successfully stored {len(ids)} chunks in ChromaDB!")
    return collection


if __name__ == "__main__":
    # Test populating the database
    col = populate_chroma_store(strategy="fixed", chunk_size=60, overlap=15, extract_ner=True)
    print(f"Total items in collection: {col.count()}")