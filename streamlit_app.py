import streamlit as st
from loaders.loader import load_documents
from splitters.splitter import split_docs
from vectorstore.vectorstore import build_vector_store
from RAG.pipeline import (
    create_rag_prompt,
    create_query_rewrite_prompt
)
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 900px;}
        .status-pill {
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-ready {background-color: #16532b; color: #b8f2c9;}
        .status-idle {background-color: #3a3a3a; color: #cfcfcf;}
        div[data-testid="stChatInput"] textarea {
            border-radius: 12px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SMALL_TALK = [
    "hi", "hello", "hey", "how are you",
    "who are you", "what can you do",
    "good morning", "good evening",
]


def is_small_talk(query: str) -> bool:
    return query.lower().strip() in SMALL_TALK


@st.cache_resource(show_spinner=False)
def initialize_rag(text, pdf_paths):
    docs = load_documents(text, pdf_paths)
    if not docs:
        raise ValueError(
            "No content was loaded from the provided text/PDFs. "
            "Check that the PDF(s) contain extractable text."
        )

    chunks = split_docs(docs)
    if not chunks:
        raise ValueError("Documents were loaded but produced zero chunks after splitting.")

    vector_store = build_vector_store(chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    prompt = create_rag_prompt()

    llm = ChatGroq(model="openai/gpt-oss-20b")
    chain = prompt | llm
    rewrite_chain = create_query_rewrite_prompt() | llm

    return retriever, chain, llm, rewrite_chain, len(chunks)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "memory": [],
    "retriever": None,
    "chain": None,
    "llm": None,
    "rewrite_chain": None,
    "ready": False,
    "num_chunks": 0,
    "build_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Input Data")

    text_input = st.text_area("Optional text", height=120, placeholder="Paste any extra context here…")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    start = st.button("🚀 Start Chatbot", use_container_width=True, type="primary")

    st.divider()

    if st.session_state.ready:
        st.markdown(
            f'<span class="status-pill status-ready">● Ready</span>&nbsp;'
            f'<span style="opacity:0.7;font-size:0.8rem;">{st.session_state.num_chunks} chunks indexed</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-pill status-idle">● Not started</span>', unsafe_allow_html=True)

    if st.session_state.memory:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.memory = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📄 Text + PDF RAG Chatbot")

if start:
    if not uploaded_files and not text_input.strip():
        st.error("Please provide text or upload at least one PDF.")
        st.stop()

    with st.status("Building knowledge base…", expanded=True) as status:
        try:
            st.write("Saving uploaded files…")
            pdf_paths = []
            for file in uploaded_files or []:
                with open(file.name, "wb") as f:
                    f.write(file.read())
                pdf_paths.append(file.name)

            st.write("Loading and chunking documents…")
            st.write("Building vector store (this can take a moment)…")

            retriever, chain, llm, rewrite_chain, num_chunks = initialize_rag(
                text_input, pdf_paths
            )

            st.session_state.retriever = retriever
            st.session_state.chain = chain
            st.session_state.llm = llm
            st.session_state.rewrite_chain = rewrite_chain
            st.session_state.ready = True
            st.session_state.num_chunks = num_chunks
            st.session_state.build_error = None

            status.update(label="Knowledge base ready ✅", state="complete", expanded=False)

        except Exception as e:
            st.session_state.ready = False
            st.session_state.build_error = str(e)
            status.update(label="Build failed ❌", state="error", expanded=True)
            st.exception(e)

if st.session_state.build_error and not st.session_state.ready:
    st.error(f"Last build failed: {st.session_state.build_error}")

if not st.session_state.ready and not st.session_state.memory:
    st.info("Upload a PDF or paste some text in the sidebar, then click **Start Chatbot** to begin.")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.memory:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.write(msg.content)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
query = st.chat_input(
    "Ask something…" if st.session_state.ready else "Start the chatbot first →",
    disabled=not st.session_state.ready,
)

if query and st.session_state.chain:
    st.session_state.memory.append(HumanMessage(query))
    with st.chat_message("user", avatar="🧑"):
        st.write(query)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        try:
            if is_small_talk(query):
                result = st.session_state.llm.invoke(query)
                answer = result.content
            else:
                rewritten_query = st.session_state.rewrite_chain.invoke(
                    {
                        "chat_history": st.session_state.memory[:-1],
                        "question": query,
                    }
                ).content

                docs = st.session_state.retriever.invoke(rewritten_query)
                context = "\n\n".join(doc.page_content for doc in docs)

                result = st.session_state.chain.invoke(
                    {
                        "question": query,
                        "chat_history": st.session_state.memory,
                        "context": context,
                    }
                )
                answer = result.content

            placeholder.markdown(answer)
            st.session_state.memory.append(AIMessage(content=answer))

        except Exception as e:
            placeholder.empty()
            st.error(f"Something went wrong while generating the answer: {e}")
            st.exception(e)