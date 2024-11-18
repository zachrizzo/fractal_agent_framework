# fractal_framework/core/adaptive_scheduler.py

from typing import Dict, List, Any
from .load_balancer import LoadBalancer
import asyncio
import logging

logger = logging.getLogger(__name__)

class AdaptiveScheduler:
    """Enhanced scheduler with adaptive execution strategies"""

    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.execution_strategies: Dict[str, Any] = {
            "parallel": self._execute_parallel,
            "sequential": self._execute_sequential,
            "priority": self._execute_priority
        }

    async def schedule_tasks(
        self,
        tasks: List[Dict[str, Any]],
        strategy: str = "parallel"
    ) -> Dict[str, Any]:
        """Schedule tasks using the specified strategy"""
        if strategy not in self.execution_strategies:
            strategy = "parallel"

        return await self.execution_strategies[strategy](tasks)

    async def _execute_parallel(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute tasks in parallel"""
        assignments = []
        for task in tasks:
            agent_name = await self.load_balancer.assign_task(task)
            if agent_name:
                assignments.append((agent_name, task))

        results = await asyncio.gather(
            *[self._execute_task(agent_name, task)
              for agent_name, task in assignments]
        )
        return dict(zip([t["id"] for t in tasks], results))

    async def _execute_sequential(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        results = {}
        for task in tasks:
            agent_name = await self.load_balancer.assign_task(task)
            if agent_name:
                result = await self._execute_task(agent_name, task)
                results[task["id"]] = result
        return results

    async def _execute_priority(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute tasks based on priority"""
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.get("priority", 0),
            reverse=True
        )
        return await self._execute_sequential(sorted_tasks)

    async def _execute_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task"""
        agent = self.load_balancer.agent_pool[agent_name]
        start_time = asyncio.get_event_loop().time()

        try:
            # Execute task
            result = await agent.execute_task(task)

            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time

            # Update agent metrics
            await agent.learn_from_execution({
                "success": True,
                "execution_time": execution_time,
                "pattern": task.get("pattern"),
                "result": result
            })

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            await agent.learn_from_execution({
                "success": False,
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "pattern": task.get("pattern"),
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
