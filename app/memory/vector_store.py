import faiss
import numpy as np
import os
from app.config import settings
from app.core.logger import logger

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = []  # FAISS index to DB entry ID
        
        if os.path.exists(settings.FAISS_INDEX_PATH):
            self.load()
            
    def add(self, entry_id: int, embedding: np.ndarray):
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        self.index.add(embedding.astype('float32'))
        self.id_map.append(entry_id)
        self.save()
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[int]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        entry_ids = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.id_map):
                entry_ids.append(self.id_map[idx])
        return entry_ids

    def save(self):
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        # In a production app, we'd also persist id_map to a file (e.g. JSON/Pickle)
        # For simplicity in this local version, we'll keep it in memory or extend below
        with open(f"{settings.FAISS_INDEX_PATH}.ids", "w") as f:
            f.write(",".join(map(str, self.id_map)))

    def load(self):
        logger.info("Loading FAISS index from {}", settings.FAISS_INDEX_PATH)
        self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        id_path = f"{settings.FAISS_INDEX_PATH}.ids"
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                content = f.read()
                if content:
                    self.id_map = [int(x) for x in content.split(",")]

# Singleton instance
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
