# fractal_framework/agents/enhanced_fractal_agent.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
import asyncio
import logging

from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from core.fractal_task import FractalTask


logger = logging.getLogger(__name__)

class AgentCapability(Enum):
    """Enhanced capabilities system for agents"""
    CODE_ANALYSIS = "code_analysis"
    PATTERN_MATCHING = "pattern_matching"
    CODE_GENERATION = "code_generation"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    TESTING = "testing"

@dataclass
class AgentMetadata:
    """Enhanced metadata for agents"""
    capabilities: List[AgentCapability]
    specializations: Dict[str, float]  # Pattern name -> expertise level
    performance_metrics: Dict[str, float]
    load: float = 0.0

class EnhancedFractalAgent:
    """Enhanced base agent with sophisticated capabilities"""

    def __init__(self, name: str, capabilities: List[AgentCapability]):
        self.name = name
        self.metadata = AgentMetadata(
            capabilities=capabilities,
            specializations={},
            performance_metrics={
                "success_rate": 1.0,
                "complexity_handled": 0.0
            }
        )
        self.task_history: List[Dict[str, Any]] = []
        self.sub_agents: List['EnhancedFractalAgent'] = []  # Add this line


    # Update execute_task to match FractalAgent interface
    async def execute_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        try:
            result = await self._process_task(task, context)

            # Update metrics
            self._update_metrics({
                "success": True
            })
            return result
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _process_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _process_task")


    # In EnhancedFractalAgent class
    async def learn_from_execution(self, execution_result: Dict[str, Any]):
        """Enhanced learning from execution results"""
        # Learn about task decomposition effectiveness
        if "subtasks" in execution_result:
            await self._update_decomposition_patterns(
                execution_result["subtasks"],
                execution_result["success"]
            )

        # Learn about agent collaboration patterns
        if "agent_interactions" in execution_result:
            await self._update_collaboration_patterns(
                execution_result["agent_interactions"]
            )

        # Update capability weights
        if "capabilities_used" in execution_result:
            await self._update_capability_weights(
                execution_result["capabilities_used"],
                execution_result["success"]
            )

    def _update_metrics(self, task_result: Dict[str, Any]):
        """Update agent performance metrics"""
        success = task_result.get("success", False)

        # Update success rate
        current_rate = self.metadata.performance_metrics["success_rate"]
        self.metadata.performance_metrics["success_rate"] = (
            current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        )

    async def _adapt_specializations(self, task_result: Dict[str, Any]):
        """Adapt agent specializations based on task results"""
        pattern = task_result.get("pattern")
        if pattern and task_result.get("success", False):
            current_level = self.metadata.specializations.get(pattern, 0.0)
            self.metadata.specializations[pattern] = min(1.0, current_level + 0.1)

    @property
    def primary_capability(self) -> str:
        """Get the primary capability of this agent"""
        if not self.metadata.capabilities:
            return "unknown"
        return self.metadata.capabilities[0].value
