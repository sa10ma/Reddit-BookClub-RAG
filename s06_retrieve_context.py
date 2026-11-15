# s05_retrieval.py
import numpy as np
from s04_vector_representation import TextRetriever
from s05_create_chroma_store import get_chroma_client


class LiteraturePipelineRetriever:
    """
    Handles hybrid retrieval (BM25 + MiniLM Embeddings) by pulling 
    documents and metadata directly from the persistent ChromaDB store.
    """
    def __init__(self, collection_name="bluest_eye_rag", alpha=0.9, bm25_k1=1.5, bm25_b=0.75):

        self.alpha = alpha
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        print("Connecting to ChromaDB and fetching stored chunks...")
        
        # 1. Connect to ChromaDB and pull all stored documents and metadata
        self.chroma_client = get_chroma_client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        stored_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        
        self.chunk_ids = stored_data["ids"]
        self.chunk_texts = stored_data["documents"]
        self.metadatas = stored_data["metadatas"]
        
        if not self.chunk_texts:
            raise ValueError("No documents found in ChromaDB! Please run the chunking and embedding pipeline first.")
            
        # 2. Initialize the TextRetriever using the texts pulled from Chroma
        print("Building lexical (BM25) and vector indices from stored corpus...")
        self.retriever = TextRetriever(self.chunk_texts, embedding_model_name="all-MiniLM-L6-v2", bm25_k1=self.bm25_k1, bm25_b=self.bm25_b)

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
                "doc_id": meta.get("original_doc_id"),
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
                "doc_id": meta.get("original_doc_id"),
                "chapter": meta.get("chapter"),
                "type": meta.get("type"),
                "entities": meta.get("entities"),
                "text": doc
            })
        return results

    
    def build_context(
        self,
        retrieved_chunks,
        max_context_chunks=3,
        max_chunks_per_document=1,
        word_budget=500,
        min_score_ratio=0.40,
        min_absolute_score=0.05,
    ):
        """
        Builds an LLM-ready context from retrieved chunks.

        Accepts results from either search_hybrid()
        or search_chroma_semantic().
        """

        # 1. Sort candidates
        # Hybrid: higher score is better.
        # Chroma: lower distance is better.
        is_semantic = (
            len(retrieved_chunks) > 0
            and "distance" in retrieved_chunks[0]
        )

        if is_semantic:
            candidates = sorted(
                retrieved_chunks,
                key=lambda x: x["distance"]
            )
        else:
            candidates = sorted(
                retrieved_chunks,
                key=lambda x: x["score"],
                reverse=True
            )

        # 2. Initialize selection trackers
        selected_chunks = []
        seen_texts = set()
        per_document_counts = {}
        used_words = 0

        # 3. Calculate score threshold
        if candidates:
            if is_semantic:
                best_score = candidates[0]["distance"]
            else:
                best_score = candidates[0]["score"]
        else:
            best_score = 0.0

        # 4. Filter and select chunks
        for chunk in candidates:

            # Score filtering
            if is_semantic:
                if chunk["distance"] > best_score + min_absolute_score:
                    continue
                if (
                    best_score >= 0
                    and chunk["distance"] > best_score + min_score_ratio
                ):
                    continue
            else:
                score = chunk["score"]

                if score < min_absolute_score:
                    continue

                if (
                    best_score > 0
                    and score < best_score * min_score_ratio
                ):
                    continue

            # Deduplication
            normalized = " ".join(
                chunk["text"].lower().split()
            )

            if normalized in seen_texts:
                continue

            # Document-level limit
            document_id = chunk.get(
                "document_id",
                chunk["chunk_id"]
            )

            doc_count = per_document_counts.get(document_id, 0)

            if doc_count >= max_chunks_per_document:
                continue

            # Word budget
            chunk_words = len(chunk["text"].split())

            if used_words + chunk_words > word_budget:
                continue

            # Accept chunk
            selected_chunks.append(chunk)
            seen_texts.add(normalized)

            per_document_counts[document_id] = doc_count + 1
            used_words += chunk_words

            if len(selected_chunks) >= max_context_chunks:
                break

        # 5. Construct labeled context
        blocks = []

        for position, chunk in enumerate(selected_chunks, start=1):
            blocks.append(
                f"[Source {position}] "
                f"Chapter: {chunk.get('chapter', 'Unknown')} | "
                f"Type: {chunk.get('type', 'Unknown')}\n"
                f"{chunk['text']}"
            )

        context_text = "\n\n".join(blocks)

        # 6. Return context package
        return {
            "selected_chunks": selected_chunks,
            "context_text": context_text,
            "used_words": used_words,
            "num_sources": len(selected_chunks),
        }


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

    package = pipeline.build_context(
        retrieved_chunks=hybrid_results,
        max_context_chunks=3,
        word_budget=500
    )

    print(package["context_text"])
    
    # Test Chroma Semantic Search
    chroma_results = pipeline.search_chroma_semantic(test_query, top_k=2)
    print("--- ChromaDB Semantic Search Results ---")
    for res in chroma_results:
        print(f"Rank {res['rank']} (Distance: {res['distance']:.4f}) | Chapter: {res['chapter']} ({res['type']})")
        print(f"Entities: {res['entities']}")
        print(f"Text: \"{res['text']}\"\n")