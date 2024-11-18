# fractal_framework/examples/adaptive_example.py

import asyncio
from core.fractal_framework import FractalFramework
from core.fractal_task import FractalTask, TaskType
from core.load_balancer import LoadBalancer
from core.adaptive_scheduler import AdaptiveScheduler
from agents.enhanced_fractal_agent import EnhancedFractalAgent, AgentCapability
from agents.analyze_agent import AnalyzeAgent
from agents.transform_agent import TransformAgent

async def main():
    # Initialize load balancer and adaptive scheduler
    load_balancer = LoadBalancer()
    scheduler = AdaptiveScheduler(load_balancer)

    # Create agents with enhanced capabilities
    analyze_agent = AnalyzeAgent("AnalyzerAgent")
    transform_agent = TransformAgent("TransformerAgent")

    # Wrap them in EnhancedFractalAgents
    enhanced_analyze_agent = EnhancedFractalAgent(
        name="EnhancedAnalyzerAgent",
        capabilities=[AgentCapability.CODE_ANALYSIS, AgentCapability.PATTERN_MATCHING]
    )
    enhanced_transform_agent = EnhancedFractalAgent(
        name="EnhancedTransformerAgent",
        capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.OPTIMIZATION]
    )

    # Add agents to the load balancer
    load_balancer.add_agent(enhanced_analyze_agent)
    load_balancer.add_agent(enhanced_transform_agent)

    # Define tasks
    tasks = [
        {
            "id": "task1",
            "type": TaskType.ANALYZE,
            "required_capabilities": [AgentCapability.CODE_ANALYSIS],
            "data": {"node_id": "user1_code"},
            "pattern": "singleton"
        },
        {
            "id": "task2",
            "type": TaskType.TRANSFORM,
            "required_capabilities": [AgentCapability.CODE_GENERATION],
            "data": {"node_id": "user2_code"},
            "pattern": "factory"
        }
    ]

    # Schedule tasks
    results = await scheduler.schedule_tasks(tasks, strategy="priority")

    # Display results
    for task_id, result in results.items():
        if result["success"]:
            print(f"Result for {task_id}: {result['result']}")
        else:
            print(f"Error in {task_id}: {result['error']}")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
