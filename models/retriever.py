from models.embedding import EmbeddingModel
from vector_store import VectorStore

class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant documents for a query."""
        query_embedding = self.embedding_model.encode_single(query)
        results = self.vector_store.search(query_embedding, k)
        return results
