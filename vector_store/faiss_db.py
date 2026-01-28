import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        """Initialize FAISS vector store"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.metadata = []
        print(f"✅ FAISS vector store initialized (dimension: {dimension})")
    
    def add_documents(self, chunks: list, embeddings: list, metadata: list):
        """Add documents to the vector store"""
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        print(f"✅ Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: list, k: int = 3):
        """Search for similar documents"""
        # Convert to numpy array
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_np, k)
        
        return distances, indices
    
    def get_total_chunks(self):
        """Get total number of chunks stored"""
        return len(self.chunks)