from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import shutil
import time

# Import custom modules
from utils.pdf_parser import extract_text_from_pdf, extract_text_from_txt
from utils.chunker import chunk_text
from models.embedding import DocumentEmbedder
from vector_store.faiss_db import FAISSVectorStore
from models.llm import llm_instance

# FastAPI app initialize
app = FastAPI(
    title="RAG Question Answering System",
    description="Upload documents and ask questions using RAG",
    version="1.0.0"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize global instances
print("Initializing embedder and vector store...")
embedder = DocumentEmbedder()
vector_store = FAISSVectorStore(dimension=384)
print("✅ System ready!")

# Uploads folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic model for query
class QueryRequest(BaseModel):
    question: str

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "RAG QA System is running!",
        "endpoints": {
            "upload": "/upload - POST request to upload documents",
            "query": "/query - POST request to ask questions",
            "docs": "/docs - Interactive API documentation"
        }
    }

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "total_chunks": vector_store.get_total_chunks()
    }

# Background processing function
def process_document_background(file_path: str, filename: str):
    """Background task to process document"""
    try:
        print(f"🔄 Background processing started for: {filename}")
        start_time = time.time()
        
        # Extract text
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_txt(file_path)
        
        # Chunk text
        chunks = chunk_text(text)
        print(f"📄 Created {len(chunks)} chunks from {filename}")
        
        # Generate embeddings
        embeddings = embedder.embed_documents(chunks)
        
        # Add to vector store with metadata
        metadata = [{"source": filename} for _ in chunks]
        vector_store.add_documents(chunks, embeddings, metadata)
        
        elapsed = time.time() - start_time
        print(f"✅ Background processing completed for: {filename} in {elapsed:.2f}s")
        
    except Exception as e:
        print(f"❌ Background processing failed for {filename}: {e}")

# Document upload endpoint
@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(request: Request, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload PDF or TXT files
    """
    # Check file type
    allowed_types = [".pdf", ".txt"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Only PDF and TXT files allowed. You uploaded: {file_ext}"
        )
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add background task for processing
        background_tasks.add_task(process_document_background, file_path, file.filename)
        
        return {
            "message": "File uploaded successfully! Processing in background...",
            "filename": file.filename,
            "path": file_path,
            "size": os.path.getsize(file_path),
            "status": "processing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Query Endpoint
@app.post("/query")
@limiter.limit("10/minute")
async def query_documents(request: Request, query_request: QueryRequest):
    """
    Ask questions based on uploaded documents using RAG
    """
    try:
        # Check if vector store has any documents
        if vector_store.get_total_chunks() == 0:
            return {
                "question": query_request.question,
                "answer": "No documents uploaded yet. Please upload documents first.",
                "sources": [],
                "similarity_scores": []
            }
        
        # Get embeddings for question
        question_embedding = embedder.embed_query(query_request.question)
        
        # Search in vector store
        distances, indices = vector_store.search(question_embedding, k=3)
        
        # Get retrieved chunks
        retrieved_chunks = []
        sources = []
        similarity_scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(vector_store.chunks):
                chunk_text = vector_store.chunks[idx]
                source_doc = vector_store.metadata[idx].get("source", "unknown")
                similarity_score = 1 / (1 + float(distances[0][i]))  # Convert to Python float
                
                retrieved_chunks.append(chunk_text)
                sources.append(source_doc)
                similarity_scores.append(round(float(similarity_score), 3))  # Ensure Python float
        
        # Combine context
        context = "\n\n".join(retrieved_chunks)
        
        # Generate answer using LLM
        answer = llm_instance.generate_answer(query_request.question, context)
        
        return {
            "question": query_request.question,
            "answer": str(answer),  # Ensure string
            "sources": list(sources),  # Ensure list
            "similarity_scores": similarity_scores,
            "num_chunks_retrieved": len(retrieved_chunks)
        }
    
    except Exception as e:
        import traceback
        print(f"❌ Query failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")