# fractal_framework/core/load_balancer.py

from typing import Dict, List, Optional, Any
from agents.enhanced_fractal_agent import EnhancedFractalAgent, AgentCapability
import asyncio
import logging

logger = logging.getLogger(__name__)

class LoadBalancer:
    """Enhanced load balancer for distributing tasks among agents"""

    def __init__(self):
        self.agent_pool: Dict[str, EnhancedFractalAgent] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

    def add_agent(self, agent: EnhancedFractalAgent):
        self.agent_pool[agent.name] = agent

    async def assign_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Assign task to the most suitable agent"""
        required_capabilities = task.get("required_capabilities", [])
        pattern = task.get("pattern")

        best_agent = None
        best_score = -1.0

        for agent in self.agent_pool.values():
            score = self._calculate_assignment_score(agent, required_capabilities, pattern)
            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent:
            await self.task_queue.put((best_score, best_agent.name, task))
            return best_agent.name
        return None

    def _calculate_assignment_score(
        self,
        agent: EnhancedFractalAgent,
        required_capabilities: List[AgentCapability],
        pattern: Optional[str]
    ) -> float:
        """Calculate assignment score for an agent"""
        score = 0.0

        # Check capabilities
        capability_score = sum(
            1.0 for cap in required_capabilities
            if cap in agent.metadata.capabilities
        ) / max(1, len(required_capabilities))
        score += capability_score * 0.4

        # Check specialization
        if pattern:
            specialization_score = agent.metadata.specializations.get(pattern, 0.0)
            score += specialization_score * 0.3

        # Check performance
        performance_score = agent.metadata.performance_metrics["success_rate"]
        score += performance_score * 0.2

        # Check load
        load_score = 1.0 - agent.metadata.load
        score += load_score * 0.1

        return score
