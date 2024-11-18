# fractal_framework/patterns/pattern_registry.py

import ast
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class PatternRegistry:
    """Enhanced pattern registry with improved detection"""

    def find_patterns(self, code: str) -> List[Tuple[str, ast.AST]]:
        """Find all patterns in code with comprehensive detection"""
        try:
            tree = ast.parse(code)
            matches = []

            # Analyze each class independently
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check for each pattern in this class
                    if self._is_singleton_candidate(node):
                        matches.append(("singleton", node))
                    if self._is_factory_candidate(node):
                        matches.append(("factory", node))
                    if self._is_repository_candidate(node):
                        matches.append(("repository", node))

            return matches

        except Exception as e:
            logger.error(f"Error finding patterns: {str(e)}")
            return []

    def _is_singleton_candidate(self, node: ast.AST) -> bool:
        """Enhanced singleton pattern detection"""
        if not isinstance(node, ast.ClassDef):
            return False

        has_instance_var = False
        has_instance_method = False
        has_instance_check = False

        for item in ast.walk(node):
            # Check for _instance class variable
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == '_instance':
                        has_instance_var = True
                        break

            # Check for get_instance method
            elif isinstance(item, ast.FunctionDef):
                if 'get_instance' in item.name:
                    has_instance_method = True
                    # Look for instance check in method body
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.If):
                            if isinstance(stmt.test, ast.Compare):
                                if isinstance(stmt.test.left, ast.Name) and 'instance' in stmt.test.left.id:
                                    has_instance_check = True
                                    break

        return has_instance_var and (has_instance_method or has_instance_check)

    def _is_factory_candidate(self, node: ast.AST) -> bool:
        """Enhanced factory pattern detection"""
        if not isinstance(node, ast.ClassDef):
            return False

        has_create_method = False
        creates_different_types = False

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Check for create/make/build methods
                if any(keyword in item.name.lower() for keyword in ['create', 'make', 'build']):
                    has_create_method = True

                    # Check method body for different return types
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                            creates_different_types = True
                            break

        return has_create_method and creates_different_types

    def _is_repository_candidate(self, node: ast.AST) -> bool:
        """Enhanced repository pattern detection"""
        if not isinstance(node, ast.ClassDef):
            return False

        crud_operations = set()
        has_storage = False

        for item in node.body:
            # Check for storage initialization
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and any(
                                name in target.attr.lower()
                                for name in ['storage', 'data', 'items', 'records', 'db', 'users']
                            ):
                                has_storage = True

            # Check for CRUD methods
            if isinstance(item, ast.FunctionDef):
                name = item.name.lower()
                if any(word in name for word in ['add', 'create', 'insert', 'new']):
                    crud_operations.add('create')
                elif any(word in name for word in ['get', 'find', 'read', 'load']):
                    crud_operations.add('read')
                elif any(word in name for word in ['update', 'modify', 'save']):
                    crud_operations.add('update')
                elif any(word in name for word in ['delete', 'remove']):
                    crud_operations.add('delete')

        # Consider it a repository if it has storage and at least 2 CRUD operations
        return has_storage and len(crud_operations) >= 2
