#fractal_framework/agents/vector_search_agent.py
from typing import List, Dict, Any, Optional
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from core.fractal_task import FractalTask, TaskType
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from agents.fractal_agent import FractalAgent
from collections import defaultdict

logger = logging.getLogger(__name__)

class VectorSearchAgent(FractalAgent):
    """Agent for vectorizing and searching code similarities"""

    def __init__(self, name: str):
        super().__init__(name, TaskType.ANALYZE.value)
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._code_tokenizer,
            stop_words=None,
            lowercase=False,
            max_features=10000
        )
        self.vectors = None
        self.node_ids = []
        self.indexed = False

    def _code_tokenizer(self, code: str) -> List[str]:
        """Tokenize code into meaningful tokens"""
        tokens = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Function tokens
                if isinstance(node, ast.FunctionDef):
                    tokens.append(f"func_{node.name}")
                    for arg in node.args.args:
                        tokens.append(f"param_{arg.arg}")

                # Class tokens
                elif isinstance(node, ast.ClassDef):
                    tokens.append(f"class_{node.name}")
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            tokens.append(f"inherits_{base.id}")

                # Variables and assignments
                elif isinstance(node, ast.Name):
                    tokens.append(f"var_{node.id}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            tokens.append(f"assign_{target.id}")

                # Method calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        tokens.append(f"call_{node.func.id}")
                    elif isinstance(node.func, ast.Attribute):
                        tokens.append(f"method_{node.func.attr}")

        except Exception as e:
            logger.warning(f"Error during tokenization: {str(e)}, using simple tokenization")
            tokens = code.split()

        return tokens

    async def _process_task(self, task: FractalTask, context: FractalContext,
                          subtask_results: List[FractalResult]) -> FractalResult:
        """Process the task based on operation type"""
        operation = task.data.get('operation')

        if operation == 'index':
            return await self._handle_indexing(context)
        elif operation == 'search':
            return await self._handle_search(task, context)
        else:
            return FractalResult(task.id, False, error=f"Unknown operation: {operation}")

    async def _handle_indexing(self, context: FractalContext) -> FractalResult:
        """Index all code in the graph"""
        try:
            # Collect code from all nodes
            code_snippets = []
            self.node_ids = []

            for node_id in context.graph.nodes():
                code = context.graph.nodes[node_id].get('code', '')
                if code:
                    code_snippets.append(code)
                    self.node_ids.append(node_id)

            if not code_snippets:
                return FractalResult("index", False, error="No code found to index")

            # Create vectors
            self.vectors = self.vectorizer.fit_transform(code_snippets)
            self.indexed = True

            return FractalResult("index", True, result={
                'nodes_indexed': len(code_snippets),
                'features': len(self.vectorizer.get_feature_names_out())
            })

        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            return FractalResult("index", False, error=str(e))

    async def _handle_search(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Search for similar code"""
        try:
            # Index first if not already done
            if not self.indexed:
                index_result = await self._handle_indexing(context)
                if not index_result.success:
                    return index_result

            node_id = task.data.get('node_id')
            if not node_id or node_id not in context.graph:
                return FractalResult(task.id, False, error="Invalid node ID")

            # Get query code
            query_code = context.graph.nodes[node_id].get('code', '')
            if not query_code:
                return FractalResult(task.id, False, error="No code found in query node")

            # Transform query
            query_vector = self.vectorizer.transform([query_code])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()

            # Get top k results
            k = task.data.get('k', 5)
            top_indices = similarities.argsort()[-k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    similar_node_id = self.node_ids[idx]
                    results.append({
                        'node_id': similar_node_id,
                        'similarity': float(similarities[idx]),
                        'code': context.graph.nodes[similar_node_id].get('code', '')
                    })

            return FractalResult(task.id, True, result={
                'query_node': node_id,
                'similar_nodes': results
            })

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    def _calculate_stats(self, similarities: np.ndarray) -> Dict[str, float]:
        """Calculate statistics about similarities"""
        return {
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'max': float(np.max(similarities)),
            'min': float(np.min(similarities))
        }
