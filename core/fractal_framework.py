# fractal_framework/core/fractal_framework.py

import logging
from typing import Dict, List
import networkx as nx

from .fractal_task import FractalTask
from .fractal_result import FractalResult
from .fractal_context import FractalContext
from agents.fractal_agent import FractalAgent

logger = logging.getLogger(__name__)

class FractalFramework:
    """Main framework for managing fractal agents and task execution"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_agents: List[FractalAgent] = []
        self.results: Dict[str, FractalResult] = {}

    def add_root_agent(self, agent: FractalAgent):
        """Add a root-level agent to the framework
        Root agents are the entry points for task execution
        """
        self.root_agents.append(agent)
        logger.info(f"Added root agent: {agent.name}")

    def add_node(self, node_id: str, **attrs):
        """Add a node to the graph
        example:
        >>> add_node('node_id', code='print("Hello, World !")', type='code', path='/path/to/file.py')

        Args:
            node_id (str): The ID of the node
            **attrs: Additional attributes for the node

        """
        self.graph.add_node(node_id, **attrs)

    def add_edge(self, source: str, target: str, **attrs):
        """Add an edge to the graph"""
        self.graph.add_edge(source, target, **attrs)

    async def execute_task(self, task: FractalTask) -> FractalResult:
        """Execute a single task"""
        context = FractalContext(self.graph)

        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.results or not self.results[dep_id].success:
                return FractalResult(task.id, False, error=f"Dependency {dep_id} not met")

        # Find suitable agent
        for agent in self.root_agents:
            if agent.can_handle_task(task):
                result = await agent.execute_task(task, context)
                self.results[task.id] = result
                return result

        return FractalResult(task.id, False, error="No suitable agent found")

    async def run(self, tasks: List[FractalTask]) -> Dict[str, FractalResult]:
        """Run multiple tasks"""
        results = {}
        for task in tasks:
            result = await self.execute_task(task)
            results[task.id] = result
        return results
