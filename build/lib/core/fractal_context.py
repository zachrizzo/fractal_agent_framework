# fractal_framework/core/fractal_context.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import networkx as nx

@dataclass
class FractalContext:
    """Context object that gets passed down through the fractal structure"""
    graph: nx.DiGraph
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    parent_id: Optional[str] = None

    def create_child(self, parent_id: str) -> 'FractalContext':
        """Create a new context for child operations"""
        return FractalContext(
            graph=self.graph,
            metadata=self.metadata.copy(),
            depth=self.depth + 1,
            parent_id=parent_id
        )
