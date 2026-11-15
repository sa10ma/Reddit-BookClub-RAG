# vectorization.py
import re
import numpy as np
#from sklearn.feature_extraction.text.tfidf import TfidfVectorizer # or sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text):
    """Tokenizes text into lowercase alphanumeric words for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TextRetriever:
    """
    Encapsulates TF-IDF, BM25, MiniLM Embeddings, and Hybrid Retrieval 
    over a given corpus of documents/chunks.
    """
    def __init__(self, corpus_texts, embedding_model_name="all-MiniLM-L6-v2"):
        self.corpus_texts = corpus_texts
        
        # 1. Initialize TF-IDF
        print("Initializing TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer().fit(self.corpus_texts)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(self.corpus_texts)
        
        # 2. Initialize BM25
        print("Initializing BM25 index...")
        tokenized_corpus = [tokenize(doc) for doc in self.corpus_texts]
        #initializer takes b and k1 
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 3. Initialize Sentence Transformer (MiniLM)
        print(f"Loading embedding model ({embedding_model_name})...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        # Pre-encode corpus normalized for fast cosine similarity matching
        self.emb_matrix = self.embedding_model.encode(
            self.corpus_texts, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )

    # -------------------------------------------------------------------
    # 1. TF-IDF Scoring Function
    # -------------------------------------------------------------------
    def tfidf_scores(self, query):
        """Computes TF-IDF cosine similarity scores across the corpus."""
        q_vec = self.tfidf_vectorizer.transform([query])
        return cosine_similarity(q_vec, self.tfidf_matrix).flatten()

    # -------------------------------------------------------------------
    # 2. BM25 Scoring Function
    # -------------------------------------------------------------------

    #if i remember correctly, bm25 had some parameters like k1 and b that can be tuned for better performance. 

    def bm25_scores(self, query):
        """Computes BM25 lexical scores across the corpus."""
        q_tokens = tokenize(query)
        return np.array(self.bm25_index.get_scores(q_tokens), dtype=float)

    # -------------------------------------------------------------------
    # 3. MiniLM Embedding Scoring Function
    # -------------------------------------------------------------------
    def embedding_scores(self, query):
        """Computes dense vector semantic similarity using all-MiniLM-L6-v2."""
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        return cosine_similarity(q_emb, self.emb_matrix).flatten()

    # -------------------------------------------------------------------
    # Helper: Min-Max Normalization
    # -------------------------------------------------------------------
    @staticmethod
    def _min_max(scores):
        """Rescales scores between 0 and 1 per query."""
        scores = np.asarray(scores, dtype=float)
        span = scores.max() - scores.min()
        return np.zeros_like(scores) if span == 0 else (scores - scores.min()) / span

    # -------------------------------------------------------------------
    # 4. Hybrid Scoring Function
    # -------------------------------------------------------------------
    def hybrid_scores(self, query, alpha=0.9):
        """
        Computes weighted hybrid scores combining normalized embeddings and BM25:
        hybrid = alpha * norm(embeddings) + (1 - alpha) * norm(bm25)
        """
        norm_emb = self._min_max(self.embedding_scores(query))
        norm_bm25 = self._min_max(self.bm25_scores(query))
        return alpha * norm_emb + (1 - alpha) * norm_bm25


if __name__ == "__main__":
    # Quick Test Execution
    sample_docs = [
        "Students can request a refund within 14 days of payment by submitting the online form.",
        "Students can request a refund within 7 days of payment by submitting the online form.",
        "Technical support can help students reset account passwords and recover portal access."
    ]
    
    retriever = TextRetriever(sample_docs)
    test_query = "How do I get my money back?"
    
    print(f"\nQuery: '{test_query}'")
    print("TF-IDF Scores:", retriever.tfidf_scores(test_query).round(3))
    print("BM25 Scores:  ", retriever.bm25_scores(test_query).round(3))
    print("Embedding Scores:", retriever.embedding_scores(test_query).round(3))
    print("Hybrid Scores (alpha=0.9):", retriever.hybrid_scores(test_query, alpha=0.9).round(3))