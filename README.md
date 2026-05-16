# Local RAG Chatbot with FAISS and LLaMA 3.2

A lightweight Retrieval-Augmented Generation (RAG) pipeline that allows users to upload PDF documents and have context-aware conversations with them completely locally. This application uses **Streamlit** for the frontend, **LangChain** for orchestration, **FAISS** for vector storage, and **Ollama (LLaMA 3.2)** for local inference.

---

## 🚀 Features

- **100% Privacy:** Your documents never leave your local machine. Text extraction, embedding generation, vector storage, and LLM inference happen entirely locally.
- **Dynamic Ingestion:** Extracts and processes text from uploaded PDFs on the fly using `PyPDF2` and `RecursiveCharacterTextSplitter`.
- **Fast Vector Search:** Utilizes Facebook AI Similarity Search (`FAISS`) with `all-MiniLM-L6-v2` embeddings for sub-millisecond document retrieval.
- **State-of-the-Art Local LLM:** Integrates with Ollama to leverage the lightweight and powerful `llama3.2` model for generating precise answers.

---

## 🛠️ System Architecture

1. **Document Ingestion:** PDF is uploaded -> Text extracted -> Split into overlapping chunks ($chunk\_size=1000$, $chunk\_overlap=200$).
2. **Vector Indexing:** Chunks are embedded using a Hugging Face sentence transformer and saved to a local FAISS index.
3. **Retrieval & Generation (RAG):** User asks a question -> FAISS retrieves top similar chunks -> Chunks + Question are passed via a LangChain `RetrievalQA` chain -> Local LLaMA 3.2 generates the final answer.

---

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.10+**
2. **Ollama:** Download and install it from [ollama.com](https://ollama.com).
   - Once installed, open your terminal and pull the model:
     ```bash
     ollama pull llama3.2
