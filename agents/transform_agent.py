# fractal_framework/agents/transform_agent.py

import logging
from typing import List, Dict, Any, Optional

from .fractal_agent import FractalAgent
from core.fractal_task import FractalTask
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from patterns.pattern_registry import PatternRegistry
import ast
import astunparse

logger = logging.getLogger(__name__)

class TransformAgent(FractalAgent):
    """Agent for transforming code
    This agent can apply direct transformations to code or apply pattern-based transformations
    examples:
    >>> task = FractalTask('transform', data={'node_id': 'node1', 'transformation': {'language': 'python3'}})
    """

    def __init__(self, name: str):
        super().__init__(name, 'transform')
        self.pattern_registry = PatternRegistry()

    async def _process_task(self, task: FractalTask, context: FractalContext,
                            subtask_results: List[FractalResult]) -> FractalResult:
        try:
            node_id = task.data.get('node_id')
            transformation = task.data.get('transformation', {})
            pattern_name = task.data.get('pattern_name')

            if not node_id or node_id not in context.graph:
                return FractalResult(task.id, False, error="Invalid node ID")

            node_data = context.graph.nodes[node_id]
            code = node_data.get('code', '')

            if pattern_name:
                # Apply pattern transformation
                transformed_code = self.apply_pattern_transformation(pattern_name, code)
                if transformed_code is None:
                    return FractalResult(task.id, False, error="Pattern transformation failed")
                node_data['code'] = transformed_code
            else:
                # Apply direct transformation
                transformed_data = self._apply_transformation(node_data, transformation)
                node_data.update(transformed_data)

            # Update the graph
            context.graph.nodes[node_id] = node_data

            return FractalResult(task.id, True, result=node_data)
        except Exception as e:
            logger.error(f"Error in TransformAgent: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    def apply_pattern_transformation(self, pattern_name: str, code: str) -> Optional[str]:
        """Apply a pattern transformation to code"""
        try:
            tree = ast.parse(code)
            matches = self.pattern_registry.find_patterns(code)
            for name, node in matches:
                if name == pattern_name:
                    transformed_code = self.pattern_registry.apply_transformation(pattern_name, node)
                    return transformed_code
            return None
        except Exception as e:
            logger.error(f"Error applying pattern transformation: {str(e)}")
            return None

    def _apply_transformation(self, node_data: Dict[str, Any], transformation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a transformation to node data"""
        result = node_data.copy()
        for key, value in transformation.items():
            if isinstance(value, dict) and key in result:
                result[key] = {**result[key], **value}
            else:
                result[key] = value
        return result
