#fractal_framework/patterns/pattern_analyzer.py
from typing import Dict, Set, List, Tuple, Any
import ast
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict
import logging
from .pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)

@dataclass
class PatternMetrics:
    """Metrics for code patterns"""
    pattern_count: Dict[str, int]
    pattern_complexity: Dict[str, float]
    pattern_relationships: nx.DiGraph
    pattern_quality: Dict[str, float]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pattern metrics"""
        return {
            'total_patterns': sum(self.pattern_count.values()),
            'pattern_distribution': self.pattern_count,
            'average_complexity': sum(self.pattern_complexity.values()) / len(self.pattern_complexity) if self.pattern_complexity else 0,
            'average_quality': sum(self.pattern_quality.values()) / len(self.pattern_quality) if self.pattern_quality else 0,
            'relationship_density': nx.density(self.pattern_relationships)
        }

class PatternAnalyzer:
    """Analyzes code patterns and calculates metrics"""

    def __init__(self, registry: PatternRegistry):
        self.registry = registry
        self.pattern_graph = nx.DiGraph()

    def analyze_code(self, code: str) -> PatternMetrics:
        """Analyze code and calculate pattern metrics"""
        try:
            tree = ast.parse(code)
            pattern_count = defaultdict(int)
            pattern_complexity = defaultdict(float)
            pattern_quality = defaultdict(float)

            # Find patterns and calculate metrics
            matches = self.registry.find_patterns(code)
            for pattern_name, node in matches:
                pattern_count[pattern_name] += 1
                complexity = self._calculate_complexity(node)
                pattern_complexity[pattern_name] = complexity
                quality = self._calculate_quality(node, pattern_name)
                pattern_quality[pattern_name] = quality

            # Build pattern relationship graph
            self._build_pattern_graph(matches)

            return PatternMetrics(
                pattern_count=dict(pattern_count),
                pattern_complexity=dict(pattern_complexity),
                pattern_relationships=self.pattern_graph,
                pattern_quality=dict(pattern_quality)
            )

        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return PatternMetrics({}, {}, nx.DiGraph(), {})

    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate complexity score for a pattern implementation"""
        complexity = 0.0

        for child in ast.walk(node):
            # Count control flow statements
            if isinstance(child, (ast.If, ast.For, ast.While)):
                complexity += 1.0
            # Count function definitions
            elif isinstance(child, ast.FunctionDef):
                complexity += 0.5
                # Add complexity for parameters
                complexity += len(child.args.args) * 0.1
            # Count class definitions
            elif isinstance(child, ast.ClassDef):
                complexity += 1.0
                # Add complexity for base classes
                complexity += len(child.bases) * 0.2
            # Count try-except blocks
            elif isinstance(child, ast.Try):
                complexity += 1.0 + len(child.handlers) * 0.5
            # Count attribute access
            elif isinstance(child, ast.Attribute):
                complexity += 0.1

        return complexity

    def _calculate_quality(self, node: ast.AST, pattern_name: str) -> float:
        """Calculate quality score for a pattern implementation"""
        quality = 1.0  # Base quality score

        # Check for docstrings
        docstring = ast.get_docstring(node)
        if docstring:
            quality += 1.0

        # Check for proper naming conventions
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if not (node.name.startswith('_') and not node.name.startswith('__')):
                quality += 0.5

        # Check for comments
        comment_count = sum(1 for child in ast.walk(node)
                          if isinstance(child, ast.Expr)
                          and isinstance(child.value, ast.Str))
        quality += min(comment_count * 0.2, 1.0)

        # Pattern-specific quality checks
        if pattern_name == "singleton":
            quality += self._check_singleton_quality(node)
        elif pattern_name == "factory":
            quality += self._check_factory_quality(node)
        elif pattern_name == "observer":
            quality += self._check_observer_quality(node)

        return min(5.0, quality)  # Cap quality at 5.0

    def _check_singleton_quality(self, node: ast.AST) -> float:
        """Check quality of singleton pattern implementation"""
        quality = 0.0

        # Check for private instance variable
        if any(isinstance(child, ast.Name) and child.id.startswith('_instance')
               for child in ast.walk(node)):
            quality += 1.0

        # Check for instance check in getter
        has_instance_check = False
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                if any('instance' in ast.unparse(cond).lower()
                      for cond in [child.test]):
                    has_instance_check = True
                    break
        if has_instance_check:
            quality += 1.0

        return quality

    def _check_factory_quality(self, node: ast.AST) -> float:
        """Check quality of factory pattern implementation"""
        quality = 0.0

        # Check for create method
        has_create = False
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef) and 'create' in child.name.lower():
                has_create = True
                # Check for type parameter
                if any(isinstance(arg, ast.arg) and 'type' in arg.arg.lower()
                      for arg in child.args.args):
                    quality += 1.0
                break
        if has_create:
            quality += 1.0

        return quality

    def _check_observer_quality(self, node: ast.AST) -> float:
        """Check quality of observer pattern implementation"""
        quality = 0.0

        # Check for observer list
        if any(isinstance(child, ast.Name) and 'observer' in child.id.lower()
               for child in ast.walk(node)):
            quality += 1.0

        # Check for notify method
        if any(isinstance(child, ast.FunctionDef) and 'notify' in child.name.lower()
               for child in ast.walk(node)):
            quality += 1.0

        return quality

    def _build_pattern_graph(self, matches: List[Tuple[str, ast.AST]]):
        """Enhanced pattern relationship detection"""
        self.pattern_graph.clear()
        relationship_strengths = {}

        # Add pattern nodes
        for pattern_name, _ in matches:
            self.pattern_graph.add_node(pattern_name)

        # Analyze relationships between patterns
        for i, (pattern1, node1) in enumerate(matches):
            for pattern2, node2 in matches[i+1:]:
                # Skip self-relationships
                if pattern1 == pattern2:
                    continue

                relationship_score = self._calculate_relationship_strength(node1, node2)
                if relationship_score > 0:
                    self.pattern_graph.add_edge(
                        pattern1,
                        pattern2,
                        weight=relationship_score
                    )
                    relationship_strengths[f"{pattern1}-{pattern2}"] = relationship_score

    def _calculate_relationship_strength(self, node1: ast.AST, node2: ast.AST) -> float:
        """Calculate the strength of relationship between two pattern implementations"""
        score = 0.0

        # Get all names from both nodes
        names1 = self._get_node_names(node1)
        names2 = self._get_node_names(node2)

        # Check direct references
        shared_names = names1.intersection(names2)
        score += len(shared_names) * 0.3

        # Check inheritance relationships
        for node in ast.walk(node1):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in names2:
                        score += 0.5

        # Check composition/aggregation
        compositions = self._find_compositions(node1, names2)
        score += compositions * 0.4

        # Check method parameter types
        parameter_relationships = self._check_parameter_relationships(node1, names2)
        score += parameter_relationships * 0.2

        # Check method return types
        return_relationships = self._check_return_relationships(node1, names2)
        score += return_relationships * 0.2

        return min(1.0, score)  # Normalize to 0-1 range

    def _find_compositions(self, node: ast.AST, target_names: Set[str]) -> int:
        """Count composition relationships"""
        count = 0
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Assign):
                try:
                    if isinstance(subnode.value, ast.Call):
                        if isinstance(subnode.value.func, ast.Name):
                            if subnode.value.func.id in target_names:
                                count += 1
                except Exception:
                    continue
        return count

    def _check_parameter_relationships(self, node: ast.AST, target_names: Set[str]) -> int:
        """Check method parameter type relationships"""
        count = 0
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.FunctionDef):
                for arg in subnode.args.args:
                    if isinstance(arg.annotation, ast.Name):
                        if arg.annotation.id in target_names:
                            count += 1
        return count

    def _check_return_relationships(self, node: ast.AST, target_names: Set[str]) -> int:
        """Check method return type relationships"""
        count = 0
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.FunctionDef):
                if isinstance(subnode.returns, ast.Name):
                    if subnode.returns.id in target_names:
                        count += 1
        return count

    def _is_tightly_coupled(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two nodes are tightly coupled"""
        names1 = self._get_node_names(node1)
        names2 = self._get_node_names(node2)

        # Check for direct references
        references = names1.intersection(names2)
        if len(references) > 0:
            return True

        # Check for inheritance relationships
        for node in ast.walk(node1):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in names2:
                        return True

        # Check for composition relationships
        for node in ast.walk(node1):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names2:
                        return True

        return False

    def _get_node_names(self, node: ast.AST) -> Set[str]:
        """Get all names defined or used in a node"""
        names = set()

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                names.add(subnode.id)
            elif isinstance(subnode, ast.ClassDef):
                names.add(subnode.name)
                # Add base class names
                for base in subnode.bases:
                    if isinstance(base, ast.Name):
                        names.add(base.id)
            elif isinstance(subnode, ast.FunctionDef):
                names.add(subnode.name)

        return names

    def analyze_code(self, code: str) -> PatternMetrics:
        """Enhanced code analysis with improved relationship detection"""
        try:
            tree = ast.parse(code)
            pattern_count = defaultdict(int)
            pattern_complexity = defaultdict(float)
            pattern_quality = defaultdict(float)

            # Find and analyze patterns
            matches = self.registry.find_patterns(code)

            for pattern_name, node in matches:
                # Update pattern count
                pattern_count[pattern_name] += 1

                # Calculate metrics
                pattern_complexity[pattern_name] = self._calculate_complexity(node)
                pattern_quality[pattern_name] = self._calculate_quality(node, pattern_name)

            # Build and analyze pattern relationships
            self._build_pattern_graph(matches)

            # Create metrics object
            metrics = PatternMetrics(
                pattern_count=dict(pattern_count),
                pattern_complexity=dict(pattern_complexity),
                pattern_relationships=self.pattern_graph,
                pattern_quality=dict(pattern_quality)
            )

            # Add relationship analysis to suggestions
            self._analyze_relationships(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return PatternMetrics({}, {}, nx.DiGraph(), {})
    def _analyze_relationships(self, metrics: PatternMetrics):
        """Analyze pattern relationships and add insights"""
        try:
            # Analyze pattern interaction complexity
            for edge in metrics.pattern_relationships.edges(data=True):
                pattern1, pattern2, data = edge
                relationship_strength = data.get('weight', 0)

                if relationship_strength > 0.7:
                    logger.info(f"Strong coupling detected between {pattern1} and {pattern2}")

            # Check for circular dependencies
            try:
                cycles = list(nx.simple_cycles(metrics.pattern_relationships))
                if cycles:
                    logger.warning(f"Circular dependencies detected: {cycles}")
            except Exception:
                pass

            # Analyze pattern distribution
            if len(metrics.pattern_count) > 2:
                density = nx.density(metrics.pattern_relationships)
                if density > 0.7:
                    logger.warning("High pattern coupling detected")

        except Exception as e:
            logger.error(f"Error in relationship analysis: {str(e)}")

    def _patterns_are_related(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two pattern implementations are related"""
        # Get names used in both nodes
        names1 = set()
        names2 = set()

        for node in ast.walk(node1):
            if isinstance(node, ast.Name):
                names1.add(node.id)
            elif isinstance(node, ast.ClassDef):
                names1.add(node.name)

        for node in ast.walk(node2):
            if isinstance(node, ast.Name):
                names2.add(node.id)
            elif isinstance(node, ast.ClassDef):
                names2.add(node.name)

        # Check for common names
        return bool(names1.intersection(names2))

    def get_pattern_suggestions(self, metrics: PatternMetrics) -> List[str]:
        """Generate improvement suggestions based on pattern metrics"""
        suggestions = []

        # Check pattern complexity
        for pattern, complexity in metrics.pattern_complexity.items():
            if complexity > 5.0:
                suggestions.append(
                    f"Pattern '{pattern}' has high complexity ({complexity:.2f}). "
                    "Consider breaking it down into smaller components."
                )

        # Check pattern quality
        for pattern, quality in metrics.pattern_quality.items():
            if quality < 2.5:
                suggestions.append(
                    f"Pattern '{pattern}' has low quality score ({quality:.2f}). "
                    "Consider adding documentation and improving encapsulation."
                )

        # Check pattern relationships
        if metrics.pattern_relationships.number_of_edges() > 0:
            density = nx.density(metrics.pattern_relationships)
            if density > 0.7:
                suggestions.append(
                    "High coupling between patterns detected. "
                    "Consider reducing dependencies for better maintainability."
                )

        return suggestions
