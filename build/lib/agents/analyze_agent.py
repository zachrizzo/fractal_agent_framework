#fractal/agents/analyze_agent.py
from typing import List, Dict, Any
import ast
import logging
from core.fractal_task import FractalTask, TaskType
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from agents.fractal_agent import FractalAgent

logger = logging.getLogger(__name__)

class AnalyzeAgent(FractalAgent):
    """Agent for analyzing code patterns and structure"""

    def __init__(self, name: str):
        super().__init__(name, TaskType.ANALYZE.value)
        self.pattern_matchers = {
            "singleton": self._analyze_singleton_pattern,
            "factory": self._analyze_factory_pattern,
            "repository": self._analyze_repository_pattern
        }

    async def _process_task(self, task: FractalTask, context: FractalContext,
                          subtask_results: List[FractalResult]) -> FractalResult:
        """Process analysis tasks"""
        try:
            node_id = task.data.get('node_id')
            analysis_type = task.data.get('analysis_type')

            if not node_id or node_id not in context.graph:
                return FractalResult(task.id, False, error="Invalid node ID")

            code = context.graph.nodes[node_id].get('code', '')
            if not code:
                return FractalResult(task.id, False, error="No code found in node")

            if analysis_type == 'patterns':
                return await self._analyze_patterns(task.id, code)
            else:
                return FractalResult(task.id, False, error=f"Unknown analysis type: {analysis_type}")

        except Exception as e:
            logger.error(f"Error in AnalyzeAgent: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _analyze_patterns(self, task_id: str, code: str) -> FractalResult:
        """Analyze code for common patterns"""
        try:
            tree = ast.parse(code)
            patterns_found = []

            # Analyze for each pattern
            for pattern_name, matcher in self.pattern_matchers.items():
                if matcher(tree):
                    patterns_found.append({
                        "name": pattern_name,
                        "confidence": self._calculate_pattern_confidence(pattern_name, tree)
                    })

            return FractalResult(task_id, True, result={
                "patterns": patterns_found,
                "metrics": self._calculate_code_metrics(tree)
            })

        except Exception as e:
            return FractalResult(task_id, False, error=f"Pattern analysis failed: {str(e)}")

    def _analyze_singleton_pattern(self, tree: ast.AST) -> bool:
        """Detect singleton pattern"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for private instance variable
                has_private_instance = False
                has_instance_method = False

                for item in node.body:
                    # Check for private instance variable
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id.startswith('_'):
                                has_private_instance = True

                    # Check for instance getter method
                    elif isinstance(item, ast.FunctionDef):
                        if item.name.startswith('get_instance'):
                            has_instance_method = True

                if has_private_instance and has_instance_method:
                    return True
        return False

    def _analyze_factory_pattern(self, tree: ast.AST) -> bool:
        """Detect factory pattern"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for create/make/build methods
                factory_methods = 0
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        name = item.name.lower()
                        if any(keyword in name for keyword in ['create', 'make', 'build']):
                            factory_methods += 1

                if factory_methods > 0:
                    return True
        return False

    def _analyze_repository_pattern(self, tree: ast.AST) -> bool:
        """Detect repository pattern"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for CRUD-like methods
                crud_methods = set()
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        name = item.name.lower()
                        if 'add' in name or 'create' in name:
                            crud_methods.add('create')
                        elif 'get' in name or 'find' in name:
                            crud_methods.add('read')
                        elif 'update' in name:
                            crud_methods.add('update')
                        elif 'delete' in name or 'remove' in name:
                            crud_methods.add('delete')

                if len(crud_methods) >= 2:  # If implements at least 2 CRUD operations
                    return True
        return False

    def _calculate_pattern_confidence(self, pattern_name: str, tree: ast.AST) -> float:
        """Calculate confidence score for pattern detection"""
        # Implementation would vary based on pattern
        # This is a simplified version
        return 0.8

    def _calculate_code_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate various code metrics"""
        metrics = {
            "num_classes": 0,
            "num_methods": 0,
            "num_attributes": 0,
            "complexity": 0
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics["num_classes"] += 1
            elif isinstance(node, ast.FunctionDef):
                metrics["num_methods"] += 1
            elif isinstance(node, ast.Assign):
                metrics["num_attributes"] += len(node.targets)

        return metrics
