
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
- 💬 Interactive chat interface built with Streamlit
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
   LLM (Groq - LLaMA 3)
          ↓
     Streamlit UI
````

---

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **Groq API (LLaMA 3.1-8B)**
* **CHROMA**
* **HuggingFace Embeddings**
* **Streamlit**

---

## 🎯 Use Cases

* Document-based Question Answering
* Conversational Knowledge Assistants
* Research Paper Exploration
* Interactive Learning Applications
* Enterprise Knowledge Retrieval Systems

---

## 📌 Project Highlights

This chatbot combines the strengths of **Retrieval-Augmented Generation (RAG)** and **conversational memory** to provide an enhanced user experience. By leveraging both retrieved document context and direct user input, the system delivers more reliable, context-aware, and personalized responses.

---
