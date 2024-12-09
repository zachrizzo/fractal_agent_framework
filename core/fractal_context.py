from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import networkx as nx
from core.agent_pool import AgentPool
import logging

logger = logging.getLogger(__name__)

@dataclass
class FractalContext:
    """Context object that gets passed down through the fractal structure"""
    graph: nx.DiGraph
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    parent_id: Optional[str] = None

    agent_pool: AgentPool = field(init=False)
    data: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.agent_pool = AgentPool()
        logger.debug("Initialized FractalContext with AgentPool")

    def create_child(self, description: str = "") -> 'FractalContext':
        """Create a child context that inherits the agent pool"""
        child_graph = self.graph.copy()
        child_metadata = self.metadata.copy()
        child = FractalContext(
            graph=child_graph,
            metadata=child_metadata,
            depth=self.depth + 1,
            parent_id=self.get_id()  # Assuming there's a method to get the current context's ID
        )
        child.agent_pool = self.agent_pool  # Share the same agent pool
        child.data = self.data.copy()  # Copy parent data
        logger.debug(f"Created child context with description: {description}")
        return child

    def get_id(self) -> str:
        """Method to retrieve the current context's ID.
        Implement this method based on how IDs are managed in your framework.
        """
        # Placeholder implementation
        return str(id(self))
