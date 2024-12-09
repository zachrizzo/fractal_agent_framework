# fractal/framework/core/agent_pool.py

import logging
from typing import Dict, Optional
from agents.enhanced_fractal_agent import EnhancedFractalAgent

logger = logging.getLogger(__name__)

class AgentPool:
    """Manages a pool of sub-agents with lifecycle management"""
    def __init__(self):
        self.active_agents: Dict[str, EnhancedFractalAgent] = {}
        self.idle_agents: Dict[str, EnhancedFractalAgent] = {}
        logger.debug("AgentPool initialized with empty active and idle agent pools.")

    async def get_agent(self, capability: str) -> Optional[EnhancedFractalAgent]:
        logger.debug(f"Requesting agent with capability: {capability}")
        # First check idle agents
        if capability in self.idle_agents:
            agent = self.idle_agents.pop(capability)
            self.active_agents[capability] = agent
            logger.debug(f"Reusing idle agent '{agent.name}' for capability '{capability}'.")
            return agent

        # Then check active agents
        agent = self.active_agents.get(capability)
        if agent:
            logger.debug(f"Agent '{agent.name}' is already active for capability '{capability}'.")
        else:
            logger.debug(f"No active agent found for capability '{capability}'.")
        return agent

    async def release_agent(self, agent: EnhancedFractalAgent):
        """Move agent back to idle pool"""
        capability = agent.primary_capability
        if capability in self.active_agents:
            self.active_agents.pop(capability)
            self.idle_agents[capability] = agent
            logger.debug(f"Released agent '{agent.name}' back to idle pool for capability '{capability}'.")
        else:
            logger.warning(f"Attempted to release agent '{agent.name}' which is not in active agents.")

