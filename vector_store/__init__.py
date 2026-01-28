import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dimension: int = 384, index_file: str = "vector_store/index.faiss", metadata_file: str = "vector_store/metadata.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata = []  # List of dicts with 'text', 'source', etc.
        self.load_or_create_index()
    
    def load_or_create_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_vectors(self, vectors: np.ndarray, metadata: list[dict]):
        """Add vectors and their metadata to the index."""
        self.index.add(vectors)
        self.metadata.extend(metadata)
        self.save_index()
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """Search for the k nearest neighbors."""
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = distances[0][i]
                results.append(result)
        return results
    
    def save_index(self):
        """Save the index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)