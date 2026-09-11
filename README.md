# 🧠 Conversational Hybrid RAG Chatbot with Memory

## 🚀 Overview
This project is a **Conversational Hybrid Retrieval-Augmented Generation (RAG) chatbot** designed to answer user queries using multiple knowledge sources, including:

- 📄 Uploaded **PDF documents**
- ✍️ **Direct user-provided text**
- 🧠 **Conversation history and memory**

Unlike traditional RAG systems that depend solely on vector-based retrieval, this chatbot follows a **hybrid approach** by injecting user-provided text directly into the prompt context. This enables the system to generate more accurate responses, especially when dealing with short statements or information that may not benefit from vector retrieval alone.

---

## ✨ Key Features

- 📂 Upload and process multiple PDF documents
- ✍️ Integrate custom user text as an additional knowledge source
- 🔍 Perform semantic retrieval using a vector database
- 🧠 Maintain conversational context through chat memory
- 💬 Interactive chat interface built with Streamlit, with live build status and clear error reporting
- 🎯 Generate context-aware responses grounded in retrieved information
- 🚫 Minimize hallucinations by restricting answers to available context

---

## 🏗️ System Architecture

```text
User PDFs + User Text
          ↓
   Document Loading
          ↓
     Text Splitting
          ↓
 Embeddings (MiniLM)
          ↓
   Chroma Vector Store
          ↓
      Retriever
          ↓
Prompt (Context + Memory)
          ↓
   LLM (Groq - GPT-OSS 20B)
          ↓
     Streamlit UI
````

---

## 🛠️ Tech Stack

* **Python**
* **LangChain** (`langchain`, `langchain-core`, `langchain-community`, `langchain-groq`, `langchain-huggingface`)
* **Groq API** — `openai/gpt-oss-20b`
* **ChromaDB** — vector store
* **HuggingFace Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
* **Streamlit** — chat UI

> **Note:** This project originally used `llama-3.1-8b-instant` via Groq. That model was deprecated by Groq on **August 16, 2026**, and has been replaced with `openai/gpt-oss-20b`, Groq's recommended migration target.

---

## 🎯 Use Cases

* Document-based Question Answering
* Conversational Knowledge Assistants
* Research Paper Exploration
* Interactive Learning Applications
* Enterprise Knowledge Retrieval Systems

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd RAG-Conversational-Chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```dotenv
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key from [console.groq.com/keys](https://console.groq.com/keys).

> ⚠️ `.env` is git-ignored and should never be committed. Each environment (local, deployed) needs its own key configured separately.

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

---

## 📌 Project Highlights

This chatbot combines the strengths of **Retrieval-Augmented Generation (RAG)** and **conversational memory** to provide an enhanced user experience. by leveraging both retrieved doc context and direct user input, the system delivers more reliable, context-aware, and personalized responses. The UI surfaces build progress and errors directly in the app, making it easy to diagnose issues like missing API keys or empty document uploads without digging through logs.