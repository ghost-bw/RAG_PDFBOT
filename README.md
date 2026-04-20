# 📄 PDF RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that allows you to have intelligent conversations with your PDF documents. Built with Streamlit, LangChain, and Groq's LLM.

[![Live Demo](https://img.shields.io/badge/Live_Demo-here-green?style=for-the-badge&logo=streamlit)]([https://your-app-link-here.streamlit.app](https://ragpdfbot-2.streamlit.app/))
[![GitHub](https://img.shields.io/badge/GitHub-ghost--bw-blue?style=for-the-badge&logo=github)](https://github.com/ghost-bw/RAG_PDFBOT)

## ✨ Features

- 🧠 **Intelligent Retrieval** - Uses MMR (Maximum Marginal Relevance) for better context retrieval
- 💬 **Chat History** - Maintains conversation memory for contextual responses
- 🔍 **Semantic Search** - Leverages sentence transformers for accurate document understanding
- ⚡ **Fast Processing** - Efficient chunking and embedding generation
- 🎯 **Accurate Responses** - Grounded answers based on your documents only

## 🏗️ Architecture
  PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store
  ↓
  User Query ← Response ← LLM (Groq) ← Context Retrieval (MMR)
