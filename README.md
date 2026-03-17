# 🏏 SportRAG
**A Multimodal Retrieval-Augmented Generation System for Cricket Knowledge**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)
[![Postgres](https://img.shields.io/badge/Database-PostgreSQL-blue)](https://www.postgresql.org/)
[![PGVector](https://img.shields.io/badge/VectorDB-PGVector-purple)](https://github.com/pgvector/pgvector)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-black)](https://openai.com)


SportRAG is designed as a **cricket-specialized multimodal RAG demo** built with **Streamlit, LangChain, PGVector, and OpenAI**.

It combines **PDF ingestion, vector retrieval, LLM-powered chat, tool calling, and automated RAG evaluation** into a single interactive system.

---

## 🚀 Features

### 📄 Multimodal Document Ingestion
- Extracts **text and figures** from PDFs (tables not extracted)
- Captures **figure captions and surrounding context** (instead of using a vision model that is computationally expensive)
- Stores both text chunks and figure metadata as **LangChain `Document` objects**
- Implemented in `data_ingestion/ingest.py`

---

### 🧠 Vector Retrieval with PGVector
- **PostgreSQL + PGVector** backend for embedding storage
- Embeddings generated with **OpenAI `text-embedding-3-small`**
- Manages:
  - document collections
  - evaluation datasets
  - metric history

Implemented in:
- `vector_db/store_to_pgvector.py`
- `postgres/connection.py`

---

### 💬 Streamlit Chat Interface
Interactive UI including:

- Document upload & collection selection (`app.py`)
- Streaming chat responses (`pages/chat.py`)
- Retrieved image display
- Embedding visualization via **t-SNE + Plotly**
- Evaluation dashboard (`pages/evaluation.py`)

---

### 🤖 Agent + Tool Calling
SportRAG uses a **LangChain agent powered by GPT-4o-mini**.

The agent can call external tools for live cricket information:

| Tool | Purpose |
|-----|-----|
| `get_series_update()` | Current tournament updates |
| `get_match_update()` | Match results |
| `get_iconic_cricket_stadiums()` | Information about famous stadiums |

Implemented in:
- `llm_model/model.py`
- `tool_calling/tools.py`

---

### 📊 Built-in RAG Evaluation
The system includes an evaluation pipeline with **synthetic QA generation and RAGAS metrics**.

Metrics measured:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Implemented in:
- `evaluation/RAGAS.py`
- `evaluation/synthetic_data_generation.py`


---

# 🏗 Architecture

## 1️⃣ Document Ingestion

Users upload a PDF via `app.py`.

Pipeline:

1. Upload a PDF or select an exisiting collection
2. Text extraction  **(pdfplumber)**
3. Figure extraction **fitz (pymupdf)**
4. Chunking **RecursiveCharacterTextSplitter**
5. Embedding generation
6. Storage in PGVector


**Images are stored locally and referenced via metadata.**

---

## 2️⃣ Retrieval + Chat

When a collection is selected:

1. Documents are retrieved via **vector similarity search**
2. Results are **Re-ranked with OpenAI scoring**
3. Top contexts are passed to a **LangChain agent**
4. GPT-4o-mini streams the final response

**Retrieved images are displayed directly in the chat interface.**

---

## 3️⃣ Tool Calling

The agent dynamically calls tools when needed.

Examples:

| Query | Tool Used |
|-----|-----|
| "Latest IPL results" | `get_match_update` |
| "Tell me about MCG stadium" | `get_iconic_cricket_stadiums` |
| "Current series standings" | `get_series_update` |

Tools rely on:
- **CricAPI** for match data
- **Web scraping** for stadium information

---

## 4️⃣ Evaluation Pipeline

Evaluation consists of three steps.

### Step 1 — Synthetic Dataset Generation

`generate_synthetic_dataset()`

- Use the existing data from the `langchain_pg_embeddings` table (original chunks)
- Uses GPT-4o-mini to generate **100 QA pairs**
- Stores them in `evaluation_dataset` table
- The table consists of the columns: questions, ground_truth, context

---

### Step 2 — Run RAG Pipeline

`run_rag_evaluation()`

- Runs questions through the RAG pipeline that was built
- Saves responses in `evaluation_dataset.rag_response`
- The table consists of the columns: questions, ground_truth, context, rag_response

---

### Step 3 — RAGAS Scoring

`run_ragas_evaluation()` calculates:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Results are stored in `ragas_result` and displayed in the UI.

---

# 🛠 Getting Started

## Prerequisites

- Python **3.11+**
- **PostgreSQL** with pgvector extension
- **OpenAI API key**
- **CricAPI key** 

Enable pgvector in Postgres:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
