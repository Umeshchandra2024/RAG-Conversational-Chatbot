from langchain_core.prompts import PromptTemplate

def create_query_rewrite_prompt():
    template = """
Rewrite the user's question into a standalone question.

Chat History:
{chat_history}

Follow-up Question:
{question}

Rewritten Question:
"""
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )

def create_rag_prompt():
    template = """
You are a helpful AI assistant.

Use the provided context to answer the user's question.
If the answer is not available in the context, say "I don't know."

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template
    )

def rerank_documents(docs, query, embeddings):
    query_embedding = embeddings.embed_query(query)

    scored_docs = []
    for doc in docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        score = sum([a*b for a, b in zip(query_embedding, doc_embedding)])
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:3]]  # top 3