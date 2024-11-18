#framework/utils/vector_store.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import faiss
import ast
import hashlib
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import re
from .language_processors import LanguageProcessorRegistry


logger = logging.getLogger(__name__)

@dataclass
class CodeEmbedding:
    """Represents a code embedding with metadata"""
    code_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float


class CodeVectorizer:
    """Enhanced code vectorizer with language support"""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension  # Fixed dimension for vectors
        self.processor_registry = LanguageProcessorRegistry()

    def vectorize(self, code: str) -> np.ndarray:
        """Convert code to vector embedding"""
        try:
            # Get appropriate processor
            processor = self.processor_registry.get_processor(code)
            if not processor:
                # Fallback to basic tokenization
                tokens = re.findall(r'\w+', code)
            else:
                # Preprocess and analyze code
                processed_code = processor.preprocess(code)
                analysis = processor.analyze(code)

                # Combine tokens from different sources
                tokens = (
                    analysis.tokens +
                    analysis.functions +
                    analysis.classes +
                    analysis.variables +
                    analysis.patterns
                )

            # Create feature vector
            return self._tokens_to_vector(tokens)
        except Exception as e:
            logger.error(f"Error vectorizing code: {str(e)}")
            return np.zeros(self.dimension)

    def _tokens_to_vector(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to fixed-dimension vector"""
        vector = np.zeros(self.dimension)

        for token in tokens:
            # Hash the token to get an index
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            index = hash_val % self.dimension

            # Add to vector
            vector[index] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

class LocalVectorStore:
    """Enhanced vector store with language support"""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vectorizer = CodeVectorizer(dimension=dimension)  # Pass dimension to vectorizer
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.processor_registry = LanguageProcessorRegistry()

    def add_embedding(
        self,
        code_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        """Add a code embedding with enhanced metadata"""
        try:
            # Ensure vector has correct dimension
            if vector.shape[0] != self.dimension:
                logger.warning(f"Vector dimension mismatch. Expected {self.dimension}, got {vector.shape[0]}")
                vector = np.resize(vector, (self.dimension,))

            # Get code analysis
            code = metadata.get('code', '')
            processor = self.processor_registry.get_processor(code)
            if processor:
                analysis = processor.analyze(code)
                metadata['analysis'] = {
                    'language': analysis.language.value,
                    'imports': analysis.imports,
                    'functions': analysis.functions,
                    'classes': analysis.classes,
                    'variables': analysis.variables,
                    'patterns': analysis.patterns,
                    'documentation': processor.extract_documentation(code)
                }

            # Add to FAISS index
            index = self.index.ntotal
            self.index.add(vector.reshape(1, -1))

            # Update mappings
            self.id_to_index[code_id] = index
            self.index_to_id[index] = code_id
            self.metadata[code_id] = metadata

        except Exception as e:
            logger.error(f"Error adding embedding: {str(e)}")

    def search_similar(
        self,
        vector: np.ndarray,
        k: int = 5,
        threshold: float = 0.8
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code with enhanced results"""
        try:
            # Ensure vector has correct dimension
            if vector.shape[0] != self.dimension:
                vector = np.resize(vector, (self.dimension,))

            # Search in FAISS index
            distances, indices = self.index.search(vector.reshape(1, -1), k)

            # Process results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:  # Valid index
                    code_id = self.index_to_id[idx]
                    similarity = 1.0 - (dist / 2.0)

                    if similarity >= threshold:
                        results.append((
                            code_id,
                            similarity,
                            self.metadata[code_id]
                        ))

            return results

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            return []
