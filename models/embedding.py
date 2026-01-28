from sentence_transformers import SentenceTransformer

class DocumentEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded!")
    
    def embed_documents(self, texts: list) -> list:
        """Generate embeddings for a list of documents"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list:
        """Generate embedding for a single query"""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding.tolist()[0]