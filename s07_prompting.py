# s06_prompting.py
import os
from s06_retrieve_context import LiteraturePipelineRetriever

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_retriever = None


def get_retriever(collection_name="bluest_eye_rag", alpha=0.9):
    """Singleton helper to avoid re-connecting to ChromaDB repeatedly."""
    global _retriever
    if _retriever is None:
        _retriever = LiteraturePipelineRetriever(
            collection_name=collection_name, 
            alpha=alpha
        )
    return _retriever


def format_context(retrieved_chunks):
    #this is what should be used to print out sources in app I think
    """Formats retrieved chunks into a structured context string."""
    formatted_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        chapter = chunk.get("chapter", "Unknown Chapter")
        chunk_type = chunk.get("type", "text")
        entities = chunk.get("entities", "")
        
        block = (
            f"[Source {i} | Chapter: {chapter} | Type: {chunk_type}]\n"
            f"Entities: {entities if entities else 'None'}\n"
            f"Excerpt: \"{chunk['text']}\""
        )
        formatted_blocks.append(block)
        
    return "\n\n".join(formatted_blocks)


def build_strict_prompt(query, context_text):
    """Builds the strictly grounded system and user prompt."""
    return f"""You are an expert literary assistant specializing in Toni Morrison's novel 'The Bluest Eye'.

Rules:
1. Ground your answer strictly in the provided context. Do not use outside knowledge or extrapolate beyond the text.
2. If the answer cannot be determined from the context, state: "The provided sources do not contain enough information to answer this question."
3. Output exactly two sections:
   Answer: [your grounded answer]
   Sources: [list the source numbers you used, e.g. Source 1, Source 2]

Question:
{query}

Context:
{context_text}
"""


def ask_llm(prompt):
    """
    Executes the LLM call via OpenRouter API.
    Falls back to a printed mock response if OPENROUTER_API_KEY is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("\n[INFO] OPENROUTER_API_KEY not found in environment. Running in mock/offline mode.\n")
        return (
            "Answer: [Mock Mode] Set your OPENROUTER_API_KEY environment variable to generate live LLM responses.\n"
            "Sources: [Source 1]"
        )

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(question, top_k=3, alpha=0.9):
    """
    End-to-end RAG pipeline function:
    1. Fetches top hybrid chunks from ChromaDB.
    2. Formats context & builds strict prompt.
    3. Calls the LLM and returns response + source metadata.
    """
    retriever = get_retriever(alpha=alpha)
    chunks = retriever.search_hybrid(question, top_k=top_k)
    
    if not chunks:
        return "The provided sources do not contain enough information to answer this question.", []
        
    context_text = format_context(chunks)
    prompt = build_strict_prompt(question, context_text)
    
    llm_response = ask_llm(prompt)
    return llm_response, chunks


if __name__ == "__main__":
    test_query = "What does Pecola feel about her eyes?"
    
    answer, sources = answer_question(test_query, top_k=3)
    
    print("\n================ LLM RESPONSE ================")
    print(answer)
    print("==============================================")
    print("\nSource Chunk IDs:", [s["chunk_id"] for s in sources])
    print("Source Chapters: ", [s["chapter"] for s in sources])