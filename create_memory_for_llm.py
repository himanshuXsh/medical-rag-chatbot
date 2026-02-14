# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# DATA_PATH = "data/"
# DB_FAISS_PATH = "vectorstore/db_faiss"

# def create_memory_for_llm():
#     # Load PDF documents
#     documents = []
#     for filename in os.listdir(DATA_PATH):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(DATA_PATH, filename))
#             documents.extend(loader.load())

#     print("PDF loader")

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_documents(documents)

#     print("Text splitter")

#     # Create embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(chunks, embeddings)

#     print("Embeddings created and vectorstore initialized")

#     # Save the vectorstore to disk
#     vectorstore.save_local(DB_FAISS_PATH)

# if __name__ == "__main__":
#     create_memory_for_llm()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    all_documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            all_documents.extend(docs)
            print(f"{file} loaded")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    print("Multi-PDF Vector DB Created Successfully!")

if __name__ == "__main__":
    create_vector_db()
