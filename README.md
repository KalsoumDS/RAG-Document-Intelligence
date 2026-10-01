# RAG Document Intelligence — Structured Document Analysis Pipeline

> Semantic search and 100% sourced Q&A over ESG reports, legal texts and regulatory documents using LangChain, Mistral AI, FAISS and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://langchain.com)
[![Mistral](https://img.shields.io/badge/Mistral_AI-API-orange)](https://mistral.ai)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://rag-document-intelligence-2dkrcn85yperhuxoqg6p6g.streamlit.app/)

**Live application:** https://rag-document-intelligence-2dkrcn85yperhuxoqg6p6g.streamlit.app/

---

## Overview

NGOs, citizens and SMEs struggle to navigate thousands of pages of legal documents, ESG impact reports, and complex administrative regulations. This RAG pipeline allows users to query large document corpora in natural language while guaranteeing 100% sourced answers with exact page and paragraph citations.

---

## Architecture

```
Input Documents (PDF, DOCX, TXT)
    |
RecursiveTextSplitter (chunks: 1000 chars, overlap: 200)
    |
Mistral Embeddings (mistral-embed)
    |
FAISS Vector Store (persistent index)
    |
MMR Re-ranking (Maximum Marginal Relevance)
    |
Mistral LLM (mistral-medium) — sourced answer generation
    |
Streamlit Interface — Q&A with page citations
```

---

## Key Features

- **Zero hallucination** — Strict MMR filtering ensures answers are grounded in the source document.
- **Source citation** — Every answer includes the exact page number and paragraph origin.
- **Multi-format ingestion** — PDF, DOCX, TXT supported.
- **Persistent vector index** — FAISS index is saved locally; no re-embedding required between sessions.

---

## Installation

```bash
git clone https://github.com/KalsoumDS/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add your Mistral API key to a .env file: MISTRAL_API_KEY=your_key
streamlit run app.py
```

---

## Technologies

- Python 3.10+, LangChain 0.2+, Mistral AI API
- FAISS, Streamlit, PyPDF2, python-docx

---

## Author

Oumou Kaltoum Sall — Data Scientist & ML Engineer  
[Portfolio](https://luxury-sunshine-073627.netlify.app) · [LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall)
