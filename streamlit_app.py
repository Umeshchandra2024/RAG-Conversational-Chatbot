import streamlit as st
from loaders.loader import load_documents
from splitters.splitter import split_docs
from vectorstore.vectorstore import build_vector_store
# from RAG.memory import create_memory
from RAG.pipeline import create_rag_prompt
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()
SMALL_TALK = [
    "hi", "hello", "hey", "how are you",
    "who are you", "what can you do",
    "good morning", "good evening"
    ]

def is_small_talk(query: str) -> bool:
    return query.lower().strip() in SMALL_TALK

@st.cache_resource
def initialize_rag(text, pdf_paths):
    docs = load_documents(text, pdf_paths)
    chunks = split_docs(docs)
    vector_store = build_vector_store(chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    prompt = create_rag_prompt()

    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = prompt | llm

    return retriever, chain, llm

st.set_page_config(page_title="RAG Chatbot",layout="wide")
st.title("📄 Text + PDF RAG Chatbot")


with st.sidebar:
    st.header("📂 Input Data")

    text_input=st.text_area(
        "Optional Text"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    
    start = st.button("Start Chatbot")

#session state
if "memory" not in st.session_state:
    st.session_state.memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "llm" not in st.session_state:
    st.session_state.llm = None


#building rag pipeline
if start:
    if not uploaded_files and not text_input.strip():
        st.error("Please provide text or upload PDFs")
        st.stop()
    with st.spinner("Building knowledge base..."):
        pdf_paths = []

        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())
            pdf_paths.append(file.name)

        retriever, chain, llm = initialize_rag(text_input, pdf_paths)

        st.session_state.retriever = retriever
        st.session_state.chain = chain
        st.session_state.llm = llm

        st.success("Chatbot ready! Ask your questions 👇")



#chat history
for msg in st.session_state.memory:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)


query = st.chat_input("Ask something...")

if query and st.session_state.chain:
    st.session_state.memory.append(HumanMessage(query))

    with st.chat_message("user"):
        st.write(query)

    if is_small_talk(query):
        result = st.session_state.llm.invoke(query)
        answer = result.content
    else:
        docs = st.session_state.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        result = st.session_state.chain.invoke({
            "question": query,
            "chat_history": st.session_state.memory,
            "context": context
        })

        answer = result.content

    st.session_state.memory.append(AIMessage(content=answer))

    with st.chat_message("assistant"):
        st.write(answer)
