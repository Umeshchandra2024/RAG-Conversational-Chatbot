from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from RAG.memory import create_memory
from RAG.pipeline import create_query_rewrite_prompt,rerank_documents

def build_rag_pipeline(vector_store):

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(model="llama-3.1-8b-instant")

    memory = create_memory()

    rewrite_prompt = create_query_rewrite_prompt()
    rag_prompt = create_rag_prompt()

    parser = StrOutputParser()

    def rag_chain(query):
        chat_history = memory.load_memory_variables({})["chat_history"]

        # 🔥 Step 1: Rewrite query
        rewritten_query = (rewrite_prompt | llm | parser).invoke({
            "chat_history": chat_history,
            "question": query
        })

        # 🔥 Step 2: Retrieve docs
        docs = retriever.invoke(rewritten_query)

        # 🔥 Step 3: Re-rank docs
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs = rerank_documents(docs, rewritten_query, embeddings)

        context = "\n\n".join(doc.page_content for doc in docs)

        # 🔥 Step 4: Final answer
        answer = (rag_prompt | llm | parser).invoke({
            "chat_history": chat_history,
            "context": context,
            "question": query
        })

        # 🔥 Step 5: Save memory
        memory.save_context({"input": query}, {"output": answer})

        return answer

    return rag_chain

def run_chatbot(text, pdf_paths):
    docs = load_documents(text, pdf_paths)
    chunks = split_docs(docs)
    vector_store = build_vector_store(chunks)

    rag_chain = build_rag_pipeline(vector_store)

    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        if is_small_talk(query):
            print("AI:", "Hello! How can I help you?")
            continue

        answer = rag_chain(query)
        print("AI:", answer, "\n")