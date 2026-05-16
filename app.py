import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)
    return vector_store

# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Build QA Chain
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )
    return qa_chain

# Streamlit App
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = os.path.join("uploaded", uploaded_file.name)
    os.makedirs("uploaded", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        st.error("The uploaded PDF does not contain readable text.")
    else:
        st.info("Creating FAISS vector store...")
        create_faiss_vector_store(text)

        st.info("Initializing chatbot...")
        qa_chain = build_qa_chain()
        st.success("Chatbot is ready!")

        question = st.text_input("Ask a question about the uploaded PDF:")
        if question:
            st.info("Querying the document...")
            answer = qa_chain.run(question)
            st.success(f"Answer: {answer}")