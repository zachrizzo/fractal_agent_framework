# fractal_framework/patterns/pattern.py

from dataclasses import dataclass
from typing import Optional, Callable
import ast

@dataclass
class Pattern:
    """Represents a code pattern to match"""
    name: str
    description: str
    matcher: Callable[[ast.AST], bool]
    transformer: Optional[Callable[[ast.AST], ast.AST]] = None
