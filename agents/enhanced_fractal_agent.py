# fractal_framework/agents/enhanced_fractal_agent.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
import asyncio
import logging

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
                "avg_execution_time": 0.0,
                "complexity_handled": 0.0
            }
        )
        self.task_history: List[Dict[str, Any]] = []

    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task (to be implemented by subclasses)"""
        pass  # Should be implemented by subclasses

    # In EnhancedFractalAgent class
    async def learn_from_execution(self, task_result: Dict[str, Any]):
        """Learn from task execution results"""
        for pattern_name in task_result.get("patterns", []):
            current_level = self.metadata.specializations.get(pattern_name, 0.0)
            self.metadata.specializations[pattern_name] = min(1.0, current_level + 0.1)

    def _update_metrics(self, task_result: Dict[str, Any]):
        """Update agent performance metrics"""
        success = task_result.get("success", False)
        execution_time = task_result.get("execution_time", 0.0)

        # Update success rate
        current_rate = self.metadata.performance_metrics["success_rate"]
        self.metadata.performance_metrics["success_rate"] = (
            current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        )

        # Update average execution time
        current_avg = self.metadata.performance_metrics["avg_execution_time"]
        self.metadata.performance_metrics["avg_execution_time"] = (
            current_avg * 0.9 + execution_time * 0.1
        )

    async def _adapt_specializations(self, task_result: Dict[str, Any]):
        """Adapt agent specializations based on task results"""
        pattern = task_result.get("pattern")
        if pattern and task_result.get("success", False):
            current_level = self.metadata.specializations.get(pattern, 0.0)
            self.metadata.specializations[pattern] = min(1.0, current_level + 0.1)
