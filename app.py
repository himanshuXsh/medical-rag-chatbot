import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("🩺 Upload Multiple PDFs & Ask Questions")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256,
        temperature=0.5
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = ChatPromptTemplate.from_template(
        """Answer based only on context:

Context:
{context}

Question:
{question}
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
        st.write(response)

