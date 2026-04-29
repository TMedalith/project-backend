# Raccly — Backend

RAG API for scientific papers. Upload PDFs → retrieve relevant chunks → generate cited answers.

> Part of the [Raccly](https://github.com/TMedalith/Raccly) project (frontend + backend).

---

## Running

```bash
uvicorn app.main:app --reload
```

---

## Stack

- **API**: FastAPI
- **Vector DB**: PostgreSQL + pgvector
- **Embeddings**: OpenAI text-embedding-3-large
- **LLM**: GPT-4o
- **PDF Parser**: Docling
- **Reranker**: Cohere v3.5
- **Framework**: LlamaIndex

---

## Features

- Streaming SSE responses
- Automatic inline citations [1][2][3]
- Metadata preservation (title, authors, year, doi)
- Similarity filtering (0.50 threshold)
- Cohere reranking
- Chunk deduplication
