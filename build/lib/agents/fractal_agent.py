# fractal_framework/agents/fractal_agent.py

from abc import ABC, abstractmethod
import logging
from typing import List, Any
from core.fractal_task import FractalTask
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult

logger = logging.getLogger(__name__)

class FractalAgent(ABC):
    """Base class for fractal agents that can handle tasks recursively"""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.sub_agents: List[FractalAgent] = []
        self.tasks_completed = 0

    def add_sub_agent(self, agent: 'FractalAgent'):
        """Add a sub-agent to this agent"""
        self.sub_agents.append(agent)
        logger.info(f"Added sub-agent {agent.name} to {self.name}")

    def can_handle_task(self, task: FractalTask) -> bool:
        """Check if this agent can handle the given task"""
        return task.type.value == self.agent_type

    async def execute_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Execute a task, potentially delegating to sub-agents"""
        logger.info(f"Agent {self.name} executing task {task.id} at depth {context.depth}")

        # First handle any subtasks
        subtask_results = []
        for subtask in task.subtasks:
            result = await self._handle_subtask(subtask, context.create_child(task.id))
            subtask_results.append(result)

        # Then process the main task
        try:
            result = await self._process_task(task, context, subtask_results)
            self.tasks_completed += 1
            return result
        except Exception as e:
            error_msg = f"Error in agent {self.name}: {str(e)}"
            logger.error(error_msg)
            return FractalResult(task.id, False, error=error_msg)

    async def _handle_subtask(self, subtask: FractalTask, context: FractalContext) -> FractalResult:
        """Handle a subtask by finding an appropriate agent"""
        # Try sub-agents first
        for agent in self.sub_agents:
            if agent.can_handle_task(subtask):
                return await agent.execute_task(subtask, context)

        # If no sub-agent can handle it, try to handle it directly
        if self.can_handle_task(subtask):
            return await self._process_task(subtask, context, [])

        return FractalResult(subtask.id, False, error=f"No agent found to handle task {subtask.id}")

    @abstractmethod
    async def _process_task(self, task: FractalTask, context: FractalContext,
                            subtask_results: List[FractalResult]) -> FractalResult:
        """Process a task directly (to be implemented by specific agents)"""
        pass
