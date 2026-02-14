
# 🩺 Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) based AI chatbot that answers medical questions using uploaded PDF documents.

This system retrieves relevant content from medical PDFs and generates context-aware answers using a lightweight Hugging Face language model.

---

## 🚀 Features

- 📄 Multi-PDF support
- 🔍 Semantic search using FAISS vector database
- 🧠 Context-based response generation
- ⚡ Lightweight LLM (flan-t5-small)
- 🌐 Streamlit interactive UI
- 📚 Embeddings using sentence-transformers

---

## 🏗️ Architecture

1. PDFs are loaded and processed.
2. Text is split into smaller chunks.
3. Chunks are converted into embeddings.
4. FAISS stores embeddings for similarity search.
5. User question is embedded and matched with relevant chunks.
6. Retrieved context is passed to the LLM.
7. Final answer is generated based on retrieved context.

---

## 🛠️ Tech Stack

- Python
- LangChain
- FAISS
- Hugging Face Transformers
- Sentence-Transformers
- Streamlit

---

## 📂 Project Structure

```

medical_chatbot/
│
├── data/                  # Medical PDFs
├── vectorstore/           # FAISS index files
├── create_memory_for_llm.py  # Builds vector database
├── app.py                 # Streamlit application
├── requirements.txt

```

---

Create virtual environment:

```

python -m venv .venv

```

Activate environment:

Windows:
```

.venv\Scripts\activate

```

Install dependencies:

```

pip install -r requirements.txt

```

---

## 🧠 Create Vector Database (Static Version)

Place your PDFs inside the `data/` folder and run:

```

python create_memory_for_llm.py

```

This will generate the FAISS vector database.

---

## ▶️ Run the Application

```

streamlit run app.py

```

---

## 🌍 Deployment

To deploy on Streamlit Community Cloud:

1. Push this project to a GitHub repository.
2. Connect the repository to Streamlit Cloud.
3. Select `app.py` as the main file.
4. Deploy.

---


