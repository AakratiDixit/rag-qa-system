# 🤖 RAG-Based Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents (PDF/TXT) and ask questions based on their content using semantic search and AI-powered answer generation.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Features

- ✅ **Document Upload**: Support for PDF and TXT file formats
- ✅ **Intelligent Chunking**: Text split into 500-character chunks with overlap
- ✅ **Vector Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- ✅ **Fast Retrieval**: FAISS vector database for similarity search
- ✅ **Free LLM**: Google Flan-T5 for answer generation (no API key required)
- ✅ **Async Processing**: Background document processing
- ✅ **Rate Limiting**: 5 uploads/min, 10 queries/min
- ✅ **API Documentation**: Auto-generated Swagger UI
- ✅ **Source Tracking**: Returns source documents for each answer
- ✅ **Similarity Scores**: Tracks retrieval quality metrics

---

## 🏗️ Architecture

```
User → FastAPI → Document Parser → Chunker → Embeddings → FAISS
                                                              ↓
User ← LLM ← Context ← Retriever ← Query Embeddings ← User Query
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended)
- Internet connection (for first-time model downloads)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AakratiDixit2510/rag-qa-system.git
cd rag-qa-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the server:**
```bash
python main.py
```

Server will start on: **http://localhost:8000**

4. **Open API documentation:**
```
http://localhost:8000/docs
```

---

## 📖 API Usage

### 1. Upload a Document

**Endpoint:** `POST /upload`

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

**Response:**
```json
{
  "message": "File uploaded successfully! Processing in background...",
  "filename": "your-document.pdf",
  "path": "uploads/your-document.pdf",
  "size": 123456,
  "status": "processing"
}
```

**Supported formats:** PDF, TXT

---

### 2. Ask a Question

**Endpoint:** `POST /query`

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

**Response:**
```json
{
  "question": "What is the main topic of the document?",
  "answer": "The document discusses daily routines for managing psoriasis, including morning hydration, sunlight exposure, and dietary recommendations.",
  "sources": [
    "Psoriasis_Daily_Routine.pdf",
    "Psoriasis_Daily_Routine.pdf"
  ],
  "similarity_scores": [0.73, 0.453, 0.442],
  "num_chunks_retrieved": 3
}
```

---

### 3. Check System Health

**Endpoint:** `GET /health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "total_chunks": 12
}
```

---

## 🧪 Interactive Testing

Visit **http://localhost:8000/docs** for the interactive Swagger UI where you can:
- Upload documents via web interface
- Test queries with live responses
- See request/response schemas
- Try different endpoints

---

## 📂 Project Structure

```
rag-qa-system/
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── explanations.md            # Technical design decisions
├── ARCHITECTURE.md            # System architecture diagram
│
├── api/
│   └── __init__.py
│
├── models/
│   ├── __init__.py
│   ├── embedding.py           # DocumentEmbedder class
│   ├── llm.py                 # LLM integration (Flan-T5)
│   └── retriever.py           # (Reserved for future use)
│
├── utils/
│   ├── __init__.py
│   ├── chunker.py             # Text chunking logic
│   ├── pdf_parser.py          # PDF/TXT parsing
│   └── rate_limit.py          # Rate limiting utilities
│
├── vector_store/
│   ├── __init__.py
│   └── faiss_db.py            # FAISS vector database
│
└── uploads/                   # Uploaded documents storage
```

---

## 🔧 Configuration

### Chunking Settings

Located in `utils/chunker.py`:
```python
chunk_size = 500      # Characters per chunk
overlap = 50          # Overlap between chunks
```

### Rate Limits

Located in `main.py`:
```python
@limiter.limit("5/minute")   # Upload endpoint
@limiter.limit("10/minute")  # Query endpoint
```

### Embedding Model

Located in `models/embedding.py`:
```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"
dimensions = 384
```

---

## 🧠 How It Works

### 1. Document Upload & Processing

1. User uploads PDF or TXT file
2. File is validated and saved to `uploads/` directory
3. Background task extracts text from document
4. Text is split into 500-character chunks with 50-char overlap
5. Each chunk is converted to 384-dimensional embedding vector
6. Embeddings stored in FAISS vector database with metadata

### 2. Question Answering

1. User submits a question
2. Question is converted to embedding vector
3. FAISS performs similarity search to find top 3 relevant chunks
4. Retrieved chunks are combined as context
5. Context + Question sent to Flan-T5 LLM
6. LLM generates answer based on retrieved context
7. Response includes answer, source documents, and similarity scores

---

## 📊 Technical Details

| Component | Technology | Details |
|-----------|-----------|---------|
| **API Framework** | FastAPI 0.104 | Async, auto-docs, validation |
| **Embedding Model** | all-MiniLM-L6-v2 | 384 dimensions, 80MB |
| **Vector Database** | FAISS (CPU) | Flat index, L2 distance |
| **LLM** | Google Flan-T5-Base | Free, CPU-based, 250MB |
| **PDF Parsing** | PyPDF2 | Lightweight, offline |
| **Rate Limiting** | SlowAPI | IP-based limits |
| **Validation** | Pydantic | Request/response schemas |

### Performance Metrics

- **Upload Processing**: 2-3 seconds per document
- **Query Latency**: 200-500ms (including LLM)
- **Memory Usage**: ~1GB (with models loaded)
- **Disk Space**: ~500MB (models + documents)

---

## ⚠️ Limitations & Known Issues

1. **Large PDFs**: Documents over 100 pages may take longer to process
2. **Scanned PDFs**: OCR not supported; text must be extractable
3. **Complex Formatting**: Tables and images are not parsed
4. **Language**: Currently optimized for English text
5. **Context Window**: LLM limited to 1500 characters of context

---

## 🔒 Rate Limits

To prevent abuse and ensure fair usage:

- **Upload**: 5 requests per minute per IP
- **Query**: 10 requests per minute per IP

Exceeding limits returns HTTP 429 (Too Many Requests).

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use different port
uvicorn main:app --port 8001
```

### Models downloading slowly
First run downloads ~400MB of models. This is normal and happens once.

### Out of memory
Reduce batch size or use smaller models. Current setup requires minimum 2GB RAM.

### PDF parsing fails
Ensure PDF has extractable text (not scanned image). Test with `pdftotext` command.

---

## 📈 Future Enhancements

- [ ] Support for DOCX, HTML files
- [ ] Hybrid search (BM25 + semantic)
- [ ] Conversational memory (multi-turn Q&A)
- [ ] Document summarization
- [ ] Better chunk boundaries (sentence-aware)
- [ ] Query expansion for better retrieval
- [ ] User authentication
- [ ] Response caching (Redis)
- [ ] Monitoring dashboard

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**AakratiDixit2510**

GitHub: [@AakratiDixit2510](https://github.com/AakratiDixit2510)

---

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) by UKPLab
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook AI Research
- [Google Flan-T5](https://huggingface.co/google/flan-t5-base) model
- [FastAPI](https://fastapi.tiangolo.com/) framework

---

## 📚 Additional Documentation

- [Technical Explanations](explanations.md) - Design decisions and metrics
- [Architecture](ARCHITECTURE.md) - Detailed system design
- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI (when server is running)

---

## 🎯 Project Goals Achieved

✅ Built functional RAG system  
✅ Implemented document upload & processing  
✅ Created semantic search with FAISS  
✅ Integrated free LLM (no API costs)  
✅ Added background processing  
✅ Implemented rate limiting  
✅ Created comprehensive documentation  
✅ Tracked performance metrics  

---

**⭐ If you find this project useful, please star the repository!**

---

*Last updated: January 28, 2026*