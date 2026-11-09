# s05_retrieval.py
import numpy as np
from s04_vector_representation import TextRetriever
from s05_create_chroma_store import get_chroma_client


class LiteraturePipelineRetriever:
    """
    Handles hybrid retrieval (BM25 + MiniLM Embeddings) by pulling 
    documents and metadata directly from the persistent ChromaDB store.
    """
    def __init__(self, collection_name="bluest_eye_rag", alpha=0.9):
        self.alpha = alpha
        print("Connecting to ChromaDB and fetching stored chunks...")
        
        # 1. Connect to ChromaDB and pull all stored documents and metadata
        self.chroma_client = get_chroma_client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        stored_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        
        self.chunk_ids = stored_data["ids"]
        self.chunk_texts = stored_data["documents"]
        self.metadatas = stored_data["metadatas"]
        
        if not self.chunk_texts:
            raise ValueError("No documents found in ChromaDB! Please run s04_chroma.py first.")
            
        # 2. Initialize the TextRetriever using the texts pulled from Chroma
        print("Building lexical (BM25) and vector indices from stored corpus...")
        self.retriever = TextRetriever(self.chunk_texts)

    def search_hybrid(self, query, top_k=3):
        """
        Executes hybrid search (BM25 + MiniLM) using score fusion and min-max normalization.
        """
        print(f"\nExecuting Hybrid Search for: '{query}' (alpha={self.alpha})")

        scores = self.retriever.hybrid_scores(query, alpha=self.alpha)
        ranked_indices = np.argsort(-scores)
        #should have threshold for score to filter out low scores

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            meta = self.metadatas[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": self.chunk_ids[idx],
                "chapter": meta.get("chapter"),
                "type": meta.get("type"),
                "entities": meta.get("entities"),
                "text": self.chunk_texts[idx]
            })
        return results

    def search_chroma_semantic(self, query, top_k=3):
        """
        Executes pure dense semantic retrieval directly via ChromaDB.
        """
        print(f"\nExecuting ChromaDB Semantic Search for: '{query}'")
        
        query_embedding = self.retriever.embedding_model.encode([query], normalize_embeddings=True).tolist()
        
        response = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        results = []
        for rank, (chunk_id, doc, meta, dist) in enumerate(zip(
            response['ids'][0],
            response['documents'][0], 
            response['metadatas'][0], 
            response['distances'][0]
        ), start=1):
            results.append({
                "rank": rank,
                "distance": float(dist),
                "chunk_id": chunk_id,
                "chapter": meta.get("chapter"),
                "type": meta.get("type"),
                "entities": meta.get("entities"),
                "text": doc
            })
        return results

# want one more function for building context


if __name__ == "__main__":
    # Test the clean, database-backed retrieval pipeline
    #should take b paramater as well, no?
    pipeline = LiteraturePipelineRetriever(alpha=0.9)
    
    test_query = "What does Pecola feel about her eyes?"
    
    # Test Hybrid Search
    hybrid_results = pipeline.search_hybrid(test_query, top_k=2)
    print("\n--- Hybrid Search Results ---")
    for res in hybrid_results:
        print(f"Rank {res['rank']} (Score: {res['score']:.4f}) | Chapter: {res['chapter']} ({res['type']})")
        print(f"Entities: {res['entities']}")
        print(f"Text: \"{res['text']}\"\n")
        
    # Test Chroma Semantic Search
    chroma_results = pipeline.search_chroma_semantic(test_query, top_k=2)
    print("--- ChromaDB Semantic Search Results ---")
    for res in chroma_results:
        print(f"Rank {res['rank']} (Distance: {res['distance']:.4f}) | Chapter: {res['chapter']} ({res['type']})")
        print(f"Entities: {res['entities']}")
        print(f"Text: \"{res['text']}\"\n")