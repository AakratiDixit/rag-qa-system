# 🏗️ System Architecture

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                              │
│                                                                   │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ Rate Limiter │────────▶│  Pydantic    │                      │
│  │  (SlowAPI)   │         │  Validation  │                      │
│  └──────────────┘         └──────────────┘                      │
│                                                                   │
│  Endpoints:                                                       │
│  • POST /upload  (5 req/min)                                    │
│  • POST /query   (10 req/min)                                   │
│  • GET  /docs    (Swagger UI)                                   │
└────────┬─────────────────────────────┬──────────────────────────┘
         │                             │
         │ UPLOAD                      │ QUERY
         ▼                             ▼
┌─────────────────────┐        ┌──────────────────────┐
│  Document Upload    │        │   Query Handler      │
│                     │        │                      │
│  1. Validate file   │        │  1. Embed query      │
│  2. Save to disk    │        │  2. Search FAISS     │
│  3. Trigger bg task │        │  3. Retrieve chunks  │
└─────────┬───────────┘        └──────────┬───────────┘
          │                               │
          ▼                               │
┌─────────────────────────────────┐      │
│   Background Processing         │      │
│                                  │      │
│   ┌──────────────────┐          │      │
│   │  PDF/TXT Parser  │          │      │
│   │  (PyPDF2)        │          │      │
│   └────────┬─────────┘          │      │
│            ▼                     │      │
│   ┌──────────────────┐          │      │
│   │  Text Chunker    │          │      │
│   │  (500 chars)     │          │      │
│   └────────┬─────────┘          │      │
│            ▼                     │      │
│   ┌──────────────────┐          │      │
│   │  Embedder        │          │      │
│   │  (all-MiniLM)    │          │      │
│   │  384 dimensions  │          │      │
│   └────────┬─────────┘          │      │
└────────────┼──────────────────────┘     │
             │                            │
             ▼                            ▼
    ┌────────────────────────────────────────┐
    │       FAISS Vector Store               │
    │                                        │
    │  • Index: Flat (L2 distance)          │
    │  • Dimensions: 384                    │
    │  • Stores: Embeddings + Metadata      │
    │  • Search: Top-K similarity           │
    └──────────────────┬─────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Retriever    │
              │                │
              │  • Top-K = 3   │
              │  • Similarity  │
              │    scoring     │
              └────────┬───────┘
                       │
                       ▼
               ┌───────────────────┐
               │  Context Builder  │
               │                   │
               │  Combine top-3    │
               │  chunks           │
               └─────────┬─────────┘
                         │
                         ▼
                ┌─────────────────────┐
                │   LLM Generator     │
                │                     │
                │  Google Flan-T5     │
                │  Text2Text Gen      │
                │  CPU-based          │
                └──────────┬──────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │   Response    │
                   │               │
                   │  • Answer     │
                   │  • Sources    │
                   │  • Scores     │
                   └───────────────┘
```

---

## 🔄 Data Flow Examples

### Example 1: Document Upload Flow

```
User uploads "psoriasis.pdf"
         │
         ▼
FastAPI receives file
         │
         ├─▶ Validate: Is it PDF/TXT? ✅
         ├─▶ Save to: uploads/psoriasis.pdf
         └─▶ Trigger background task
                    │
                    ▼
         Extract text from PDF
         "Daily Routine for Psoriasis..."
                    │
                    ▼
         Chunk text (500 chars, 50 overlap)
         → Chunk 1: "Morning Wake up & Hydration..."
         → Chunk 2: "Sunlight: 10-15 mins..."
         → Chunk 3: "Breakfast: Vegetable upma..."
         → Chunk 4: "Midday Bath & Skin Care..."
                    │
                    ▼
         Generate embeddings
         → Chunk 1: [0.23, -0.45, 0.12, ..., 0.67] (384 dims)
         → Chunk 2: [0.11, 0.33, -0.22, ..., 0.44]
         → ... (4 total)
                    │
                    ▼
         Store in FAISS with metadata
         {
           "vector": [0.23, -0.45, ...],
           "text": "Morning Wake up...",
           "source": "psoriasis.pdf"
         }
                    │
                    ▼
         Return: "Processing complete! ✅"
```

---

### Example 2: Query Flow

```
User asks: "What is the morning routine?"
         │
         ▼
FastAPI receives query
         │
         ▼
Embed query using all-MiniLM-L6-v2
Query vector: [0.18, -0.39, 0.27, ..., 0.51]
         │
         ▼
FAISS similarity search (Top-K=3)
         │
         ├─▶ Chunk 1: Distance=0.12 → Similarity=0.893
         ├─▶ Chunk 2: Distance=0.45 → Similarity=0.689
         └─▶ Chunk 3: Distance=0.52 → Similarity=0.658
         │
         ▼
Retrieve chunk texts:
  1. "Morning Wake up & Hydration: A glass of warm water..."
  2. "Sunlight: 10–15 mins early morning sunlight..."
  3. "Breakfast: Vegetable upma / oats with veggies..."
         │
         ▼
Combine into context
Context = Chunk1 + "\n\n" + Chunk2 + "\n\n" + Chunk3
         │
         ▼
Send to LLM (Flan-T5)
Prompt: "Answer based on context: [context]\nQ: What is the morning routine?"
         │
         ▼
LLM generates answer
"The morning routine includes waking up with warm water and lemon, 
flax or chia seeds, 10-15 minutes of sunlight, and breakfast 
with vegetable upma or oats with turmeric."
         │
         ▼
Return JSON response
{
  "answer": "...",
  "sources": ["psoriasis.pdf"],
  "similarity_scores": [0.893, 0.689, 0.658]
}
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI | REST API, async support, auto-docs |
| **Embedding Model** | Sentence Transformers (all-MiniLM-L6-v2) | Convert text → 384-dim vectors |
| **Vector Store** | FAISS (CPU) | Fast similarity search, local storage |
| **LLM** | Google Flan-T5-Base | Answer generation (free, CPU) |
| **Document Parsing** | PyPDF2 | PDF text extraction |
| **Rate Limiting** | SlowAPI | Prevent API abuse |
| **Validation** | Pydantic | Request/response validation |
| **Background Tasks** | FastAPI BackgroundTasks | Async document processing |

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Document Processing** | 2-3 seconds | Per document, background |
| **Query Latency** | 200-500ms | Including LLM generation |
| **Memory Usage** | ~1GB | With models loaded |
| **Disk Space** | ~500MB | Models + uploads |
| **Embedding Speed** | ~100ms | Per document |
| **Chunk Size** | 500 chars | With 50 char overlap |
| **Vector Dimensions** | 384 | all-MiniLM-L6-v2 |

---

## 🔒 Rate Limits

```
POST /upload:  5 requests per minute
POST /query:   10 requests per minute
```

**Why?**
- Prevents abuse
- Protects server resources
- Ensures fair usage

---

## 📁 File Structure Mapping

```
rag-qa-system/
├── main.py                    # FastAPI app + endpoints
├── requirements.txt           # Python dependencies
├── README.md                  # Setup guide
├── explanations.md            # Technical decisions
├── ARCHITECTURE.md            # This file
│
├── api/                       # (Reserved for future modularization)
├── models/
│   ├── embedding.py           # DocumentEmbedder class
│   ├── llm.py                 # SimpleLLM class (Flan-T5)
│   └── retriever.py           # (Future: retrieval logic)
│
├── utils/
│   ├── chunker.py             # chunk_text() function
│   ├── pdf_parser.py          # PDF/TXT extraction
│   └── rate_limit.py          # (Reserved)
│
├── vector_store/
│   └── faiss_db.py            # FAISSVectorStore class
│
└── uploads/                   # Uploaded documents storage
```

---

## 🔍 Key Design Decisions

### 1. **Why FAISS (Local) vs Cloud Vector DB?**
- ✅ No API costs
- ✅ Works offline
- ✅ Fast for small-medium datasets
- ✅ Easy setup

### 2. **Why Sentence Transformers?**
- ✅ Pre-trained, high quality
- ✅ Fast on CPU
- ✅ Small model size (80MB)
- ✅ 384 dims = good balance

### 3. **Why Flan-T5 vs GPT?**
- ✅ Completely free
- ✅ Runs on CPU
- ✅ No API key needed
- ✅ Good quality for RAG tasks

### 4. **Why 500 Character Chunks?**
- ✅ Balances context vs precision
- ✅ Fits in embedding window
- ✅ Not too small (loses context)
- ✅ Not too large (irrelevant info)

---

## 🚀 Future Enhancements

1. **Hybrid Search:** BM25 +
