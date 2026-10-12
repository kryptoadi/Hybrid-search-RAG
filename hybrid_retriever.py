
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()


# 1. Create BM25 keyword index
def create_bm25(chunks):
    texts = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(texts)


# 2. Search using BM25
def keyword_search(query, chunks, bm25, k=4):
    scores = bm25.get_scores(query.lower().split())

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [chunks[i] for i in top_indices]


# 3. Combine rankings using RRF
def reciprocal_rank_function(results, k=60):
    scores = {}
    chunks = {}

    for result_list in results:
        for rank, chunk in enumerate(result_list, start=1):
            text = chunk.page_content

            scores[text] = scores.get(text, 0) + 1 / (k + rank)
            chunks[text] = chunk

    return sorted(
        chunks.values(),
        key=lambda chunk: scores[chunk.page_content],
        reverse=True
    )


# 4. Hybrid search: BM25 + FAISS + RRF
def hybrid_search(query, chunks, bm25, vectorstore, k=4):
    bm25_results = keyword_search(query, chunks, bm25, k)

    faiss_results = vectorstore.similarity_search(query, k=k)

    results = reciprocal_rank_function(
        [bm25_results, faiss_results]
    )

    return results[:k]


# 5. Generate answer using Groq
def generate_answer(query, chunks):
    context = "\n\n".join(
        chunk.page_content for chunk in chunks
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    prompt = f"""
    Answer the question using only the context below.
    If the answer is not present, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content


# 6. Complete RAG pipeline



def ask_question(query, chunks, bm25, vectorstore):
    relevant_chunks = hybrid_search(
        query, chunks, bm25, vectorstore
    )

    answer = generate_answer(query, relevant_chunks)

    sources = []
    seen = set()

    for chunk in relevant_chunks:
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "Unknown")

        key = (source, page)

        if key not in seen:
            seen.add(key)

            sources.append({
                "source": source,
                "page": page,
                "text": chunk.page_content
            })

    return answer, sources
