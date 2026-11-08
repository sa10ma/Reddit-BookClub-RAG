# 03_chunking.py
import re
import importlib
import spacy

# Lazy load spaCy English language model only if NER is enabled
nlp = None

def get_spacy_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp

from s02_preprocessing import preprocess_documents

# -------------------------------------------------------------------
# NER Extraction Function (Optional)
# -------------------------------------------------------------------
def extract_people_entities(text):
    """
    Runs spaCy NER on a text chunk and returns a sorted, deduplicated 
    list of detected PERSON entities (character names).
    """
    model = get_spacy_model()
    doc = model(text)
    people = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip().rstrip("'s")
            if len(name) > 2:  # Filter out single-letter noise
                people.add(name)
    return sorted(list(people))


# -------------------------------------------------------------------
# Strategy 1: Word-Based Fixed Size Chunker (Your Custom Function)
# -------------------------------------------------------------------
def chunk_fixed(text, chunk_size=50, overlap=10):
    """
    Splits text into fixed-size chunks based on word count with a word overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")
        
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        
        if end >= len(words):
            break
            
        start += chunk_size - overlap
        
    return chunks


# -------------------------------------------------------------------
# Strategy 2: Hybrid Sentence Chunker with Word Caps & Sentence Overlap (Fixed)
# -------------------------------------------------------------------
# def chunk_sentences_with_overlap(text, max_words=50, overlap_sentences=1):
#     """
#     Splits text into sentence groups up to a max word limit, 
#     maintaining a sentence-level overlap buffer across chunks.
#     """
#     sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
#     if not sentences:
#         return [text] if text.strip() else []
        
#     chunks = []
#     i = 0
    
#     while i < len(sentences):
#         current = []
#         words_count = 0
#         j = i
        
#         while j < len(sentences):
#             sentence_words = len(sentences[j].split())
#             if current and (words_count + sentence_words > max_words):
#                 break
#             current.append(sentences[j])
#             words_count += sentence_words
#             j += 1
            
#         chunks.append(" ".join(current))
        
#         sentences_added = len(current)
#         advance = max(1, sentences_added - overlap_sentences)
#         i += advance
        
#     return chunks
def chunk_sentences_with_overlap(text, max_words=300, overlap_sentences=1):
    """
    Splits text into sentence groups up to a max word limit, 
    maintaining a sentence-level overlap buffer across chunks.
    """
    # Clean up internal newlines so sentences split reliably
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if s.strip()]
    
    if not sentences:
        return [text] if text.strip() else []
        
    chunks = []
    i = 0
    
    while i < len(sentences):
        current = []
        words_count = 0
        j = i
        
        # Keep adding sentences until we hit max_words or run out of sentences
        while j < len(sentences):
            sentence_words = len(sentences[j].split())
            
            # If even the very first sentence exceeds max_words, take it to prevent infinite loop
            if not current and sentence_words > max_words:
                current.append(sentences[j])
                j += 1
                break
                
            # If adding this sentence exceeds the word cap, stop expanding this chunk
            if current and (words_count + sentence_words > max_words):
                break
                
            current.append(sentences[j])
            words_count += sentence_words
            j += 1
            
        chunks.append(" ".join(current))
        
        # If we reached the end of the sentences, we are done
        if j >= len(sentences):
            break
            
        # Calculate next start index using overlap, ensuring we always move forward at least 1 step
        advance = max(1, len(current) - overlap_sentences)
        i += min(advance, len(current)) # Prevent getting stuck
        
    return chunks
# -------------------------------------------------------------------
# Main Pipeline Function with Optional NER
# -------------------------------------------------------------------
def create_chunks(strategy="fixed", chunk_size=50, overlap=10, max_words=50, overlap_sentences=1, extract_ner=False):
    """
    Loads preprocessed documents, splits them into chunks, and optionally 
    extracts PERSON entities via spaCy NER.
    
    Parameters:
      - strategy: 'fixed' or 'sentence'
      - extract_ner: Boolean (True/False) to toggle character name extraction
    """
    preprocessed_docs = preprocess_documents()
    all_chunks = []
    chunk_counter = 1
    
    for doc in preprocessed_docs:
        text = doc.get("text", "")
        if not text:
            continue

        # Support both the newer data schema (chapter/type) and older raw JSON files
        doc_id = doc.get("id", f"doc_{chunk_counter}")
        chapter = doc.get("chapter", "Unknown Chapter")
        doc_type = doc.get("type", "unknown")
        
        # Select requested chunking strategy
        if strategy == "sentence":
            text_chunks = chunk_sentences_with_overlap(
                text, 
                max_words=max_words, 
                overlap_sentences=overlap_sentences
            )
        else:
            text_chunks = chunk_fixed(
                text, 
                chunk_size=chunk_size, 
                overlap=overlap
            )
            
        # Wrap each chunk into a dictionary preserving metadata
        for chunk_text in text_chunks:
            chunk_data = {
                "chunk_id": f"chunk_{chunk_counter}",
                "original_doc_id": doc_id,
                "chapter": chapter,
                "type": doc_type,
                "text": chunk_text
            }
            
            # Conditionally extract and add NER entities
            if extract_ner:
                chunk_data["entities"] = extract_people_entities(chunk_text)
                
            all_chunks.append(chunk_data)
            chunk_counter += 1
            
    return all_chunks


if __name__ == "__main__":
    # Test chunks generation with NER enabled
    chunks = create_chunks(strategy="sentence", chunk_size=50, overlap=10, max_words=50, overlap_sentences=1, extract_ner=True)
    
    print(f"--- Chunks with Optional NER Metadata ---")
    print(f"Total Chunks Created: {len(chunks)}")
    if chunks:
        sample_index = 30
        sample = chunks[sample_index] if sample_index < len(chunks) else chunks[-1]
        print(f"Sample [{sample['chunk_id']} | {sample['chapter']} ({sample['type'].upper()})]:")
        if 'entities' in sample:
            print(f"Detected Entities: {sample['entities']}")
        print(f"Text: '{sample['text']}'")