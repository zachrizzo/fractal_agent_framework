# framework/utils/vector_store.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
import faiss
import pickle
import logging
from dataclasses import dataclass
import threading
from pathlib import Path
from datetime import datetime
from openai import OpenAI


logger = logging.getLogger(__name__)

@dataclass
class CodeData:
    """Represents structured code data for indexing"""
    id: str
    code: str
    file_path: Optional[str] = None
    label: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None

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
            embedding_size = 1536 if 'small' in self.model_name else 3072
            return np.zeros(embedding_size, dtype=np.float32)

class LocalVectorStore:
    """Enhanced vector store for code embeddings with bulk indexing capabilities"""

    def __init__(self,
                 model_name: str = 'text-embedding-3-small',
                 dimension: Optional[int] = None,
                 api_key: Optional[str] = None,
                 auto_save: bool = False,
                 save_path: Optional[str] = None):

        self.model_name = model_name
        self.api_key = api_key
        self.vectorizer = CodeVectorizer(model_name=self.model_name, api_key=self.api_key)
        self.auto_save = auto_save
        self.save_path = save_path

        # Determine embedding dimension based on model
        self.dimension = dimension or (1536 if 'small' in model_name else 3072)

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.lock = threading.Lock()

        # Statistics
        self.total_indexed = 0
        self.failed_indices = 0
        self.last_save_time = None

    def bulk_index_code(self, code_data_iterator: Iterator[Dict[str, Any]],
                       batch_size: int = 100) -> Dict[str, Any]:
        """
        Bulk index code data from an iterator of dictionaries.

        Args:
            code_data_iterator: Iterator yielding code data dictionaries
            batch_size: Number of items to process before auto-saving (if enabled)

        Returns:
            Dict containing indexing statistics
        """
        indexed_count = 0
        failed_count = 0
        current_batch = 0

        for data in code_data_iterator:
            try:
                if not self._is_valid_code_data(data):
                    continue

                code_data = self._extract_code_data(data)
                if not code_data or not code_data.code.strip():
                    continue

                self._index_single_code(code_data)
                indexed_count += 1
                current_batch += 1

                # Auto-save handling
                if self.auto_save and self.save_path and current_batch >= batch_size:
                    self.save_store(self.save_path)
                    current_batch = 0
                    self.last_save_time = datetime.now()

            except Exception as e:
                failed_count += 1
                logger.error(f"Error indexing code: {str(e)}")

        # Final save if auto_save is enabled
        if self.auto_save and self.save_path and current_batch > 0:
            self.save_store(self.save_path)
            self.last_save_time = datetime.now()

        # Update statistics
        self.total_indexed += indexed_count
        self.failed_indices += failed_count

        return {
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "total_embeddings": self.index.ntotal,
            "last_save_time": self.last_save_time
        }

    def _is_valid_code_data(self, data: Dict[str, Any]) -> bool:
        """Validate if the data dictionary has required fields"""
        return (isinstance(data, dict) and
                'data' in data and
                isinstance(data['data'], dict) and
                'code' in data['data'] and
                'id' in data)

    def _extract_code_data(self, data: Dict[str, Any]) -> Optional[CodeData]:
        """Extract structured code data from input dictionary"""
        if not data['data']['code']:
            return None

        return CodeData(
            id=data['id'],
            code=data['data']['code'],
            file_path=data['data'].get('filePath', ''),
            label=data['data'].get('label', ''),
            additional_metadata=data['data'].get('metadata', {})
        )

    def _index_single_code(self, code_data: CodeData) -> None:
        """Index a single code entry"""
        metadata = {
            'filePath': code_data.file_path,
            'label': code_data.label,
            'id': code_data.id,
            'code': code_data.code,
            'indexed_at': datetime.now().isoformat()
        }
        if code_data.additional_metadata:
            metadata.update(code_data.additional_metadata)

        self.add_embedding(code_data.id, code_data.code, metadata)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics of the vector store"""
        return {
            "total_indexed": self.total_indexed,
            "failed_indices": self.failed_indices,
            "current_index_size": self.index.ntotal,
            "last_save_time": self.last_save_time,
            "dimension": self.dimension,
            "model_name": self.model_name
        }

    # Original methods remain the same but with enhanced error handling
    def add_embedding(self, code_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Add a code embedding with metadata. Returns success status."""
        try:
            vector = self.vectorizer.vectorize(text)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            with self.lock:
                self.index.add(vector.reshape(1, -1))
                index = self.index.ntotal - 1
                self.id_to_index[code_id] = index
                self.index_to_id[index] = code_id
                self.metadata[code_id] = metadata

            logger.info(f"Successfully added embedding for code_id: {code_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding embedding for code_id {code_id}: {str(e)}")
            return False

    def save_store(self, file_path: str) -> bool:
        """Save the vector store to disk. Returns success status."""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with self.lock:
                faiss.write_index(self.index, f"{file_path}.faiss")
                with open(f"{file_path}_metadata.pkl", "wb") as f:
                    pickle.dump({
                        'id_to_index': self.id_to_index,
                        'index_to_id': self.index_to_id,
                        'metadata': self.metadata,
                        'statistics': self.get_statistics()
                    }, f)
            logger.info(f"Vector store saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    def remove_embedding(self, code_id: str) -> bool:
        """Remove an embedding by its code_id"""
        with self.lock:
            if code_id in self.id_to_index:
                index = self.id_to_index[code_id]
                # Would need to handle FAISS index reconstruction
                del self.id_to_index[code_id]
                del self.index_to_id[index]
                del self.metadata[code_id]
                return True
        return False

    def cleanup_old_embeddings(self, max_age: float) -> int:
        """Remove embeddings older than max_age"""
        # Would need to add timestamp tracking first
        pass


    def simple_search_similar(self, text: str, k: int = 5, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Search for similar code embeddings based on a query."""
        try:
            # Generate embedding for the query
            query_vector = self.vectorizer.vectorize(text)
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

