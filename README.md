# 🎥 YouTube AI Assistant (RAG-Based)

An AI-powered web application that allows users to ask questions from any YouTube video using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🔍 Ask questions from YouTube videos
- 🌍 Supports Hindi & English transcripts
- ⚡ Fast responses using Groq LLM
- 🧠 Semantic search using FAISS
- 🎯 Context-aware answers (RAG pipeline)
- 💬 Simple chat interface

---

## 🛠 Tech Stack

- Python (Flask)
- LangChain
- FAISS (Vector Database)
- Hugging Face (Embeddings)
- Groq API (LLM)
- HTML, CSS  (Frontend)

---

## ⚙️ How It Works

1. Extracts transcript from YouTube video  
2. Splits text into chunks  
3. Converts text into embeddings  
4. Stores embeddings in FAISS  
5. Retrieves relevant context  
6. Sends context to LLM (Groq)  
7. Generates accurate answer  

---
