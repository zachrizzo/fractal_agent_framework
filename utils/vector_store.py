# framework/utils/vector_store.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import faiss
import hashlib
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import re
from openai import OpenAI

# Initialize logging
logger = logging.getLogger(__name__)

@dataclass
class CodeEmbedding:
    """Represents a code embedding with metadata"""
    code_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float

class CodeVectorizer:
    """Code vectorizer using OpenAI's embeddings."""

    def __init__(self, model_name: str = 'text-embedding-3-small', api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key) if api_key else None

    def vectorize(self, text: str) -> np.ndarray:
        """Convert text or code to vector embedding using OpenAI's API."""
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized. API key required.")

            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error vectorizing text: {str(e)}")
            # Determine embedding size based on model
            embedding_size = 1536 if 'small' in self.model_name else 3072
            return np.zeros(embedding_size, dtype=np.float32)

class LocalVectorStore:
    """Vector store using OpenAI's embeddings."""

    def __init__(self,
                 model_name: str = 'text-embedding-3-small',
                 dimension: Optional[int] = None,
                 api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.vectorizer = CodeVectorizer(model_name=self.model_name, api_key=self.api_key)

        # Determine embedding dimension based on model
        if dimension is None:
            if 'small' in self.model_name:
                self.dimension = 1536
            elif 'large' in self.model_name:
                self.dimension = 3072
            else:
                self.dimension = dimension or 1536
        else:
            self.dimension = dimension

        # Initialize FAISS index for Inner Product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        # Lock for thread-safe operations
        self.lock = threading.Lock()

    def add_embedding(
        self,
        code_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Add a code embedding with metadata."""
        try:
            # Generate embedding using OpenAI
            vector = self.vectorizer.vectorize(text)

            # Normalize the vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            with self.lock:
                # Add to FAISS index
                self.index.add(vector.reshape(1, -1))
                index = self.index.ntotal - 1

                # Update mappings
                self.id_to_index[code_id] = index
                self.index_to_id[index] = code_id
                self.metadata[code_id] = metadata

            logger.info(f"Successfully added embedding for code_id: {code_id}")

        except Exception as e:
            logger.error(f"Error adding embedding for code_id {code_id}: {str(e)}")

    def search_similar(
        self,
        query_text: str,
        k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code embeddings based on a query."""
        try:
            # Generate embedding for the query
            query_vector = self.vectorizer.vectorize(query_text)
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

            # Search in FAISS index
            distances, indices = self.index.search(query_vector.reshape(1, -1), k)

            # Process results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    code_id = self.index_to_id.get(idx)
                    if code_id:
                        similarity = float(dist)
                        if similarity >= threshold:
                            metadata = self.metadata.get(code_id, {})
                            results.append((code_id, similarity, metadata))

            return sorted(results, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            return []

    def save_store(self, file_path: str) -> None:
        """Save the FAISS index and metadata to disk."""
        try:
            with self.lock:
                faiss.write_index(self.index, f"{file_path}.faiss")
                with open(f"{file_path}_metadata.pkl", "wb") as f:
                    pickle.dump({
                        'id_to_index': self.id_to_index,
                        'index_to_id': self.index_to_id,
                        'metadata': self.metadata
                    }, f)
            logger.info(f"Vector store saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")

    def load_store(self, file_path: str) -> None:
        """Load the FAISS index and metadata from disk."""
        try:
            with self.lock:
                self.index = faiss.read_index(f"{file_path}.faiss")
                with open(f"{file_path}_metadata.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = data['index_to_id']
                    self.metadata = data['metadata']
            logger.info(f"Vector store loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
