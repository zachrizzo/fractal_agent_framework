# fractal_framework

The `fractal_framework` is a versatile, modular, and agent-based framework for code analysis, pattern recognition, vector search, and code manipulation. It is designed to operate independently of specific AI language models like ChatGPT, providing robust capabilities without requiring external dependencies. This makes it a highly flexible solution for developers looking to improve code quality and consistency, without relying on third-party AI services.

## Overview

The framework is designed with the following key features:

- **Task-Oriented Architecture**: Tasks such as code analysis, transformation, and optimization are assigned to agents that perform their roles autonomously. These tasks are handled without needing to interact with any external language model, ensuring that core functionality remains independent and modular. This architecture allows users to efficiently allocate agents to specific parts of their codebase, leading to streamlined workflows and enhanced scalability.

- **Built-In Analysis and Transformation**: The framework includes tools like `PatternAnalyzer` and `PatternRegistry` that identify common patterns (e.g., Singleton, Factory) in the codebase and apply transformations based on predefined rules. These tools are implemented as Python classes and functions, ensuring complete independence from external AI models. By utilizing built-in analysis tools, developers can quickly identify opportunities for code improvement, enforce best practices, and maintain a clean codebase.

- **Optional AI Integration for Vector Search**: While vector search is supported through embeddings of code snippets, this can be done with or without AI-based embeddings. For more advanced search, you can use embeddings from models like OpenAI's, but this is completely optional. This flexibility makes it suitable for projects of varying complexity, where users can choose between lightweight representations and more sophisticated AI-enhanced searches.

- **Adaptive Task Management and Scheduling**: Agents are managed by a load balancer and adaptive scheduler, distributing tasks effectively without requiring AI model integration. Agents can adapt to various tasks based on historical performance, maintaining efficiency even as demands shift. This adaptive approach helps in optimizing resources and ensuring agents focus on tasks that align with their capabilities, improving overall performance.

## Framework Features

### Task-Oriented Agents

- Each agent is assigned specific tasks, such as code analysis, transformation, or optimization. This modularity allows users to build specialized workflows that can be easily scaled.
- Agents perform their tasks independently, ensuring modularity and ease of scaling. This autonomy helps in building more complex workflows where multiple agents can work in parallel to achieve the desired outcomes without bottlenecks.

### Built-In Pattern Analysis

- The framework includes `PatternAnalyzer` and `PatternRegistry` for identifying design patterns like Singleton and Factory in code. This helps developers maintain design consistency throughout their codebase, ensuring best practices are consistently applied.
- Transformations are defined as Python functions that can be executed without AI model support. This allows for fast, reliable code transformations that can be easily understood and modified by developers, making it a practical tool for day-to-day use.

### Vector Search

- **Optional AI Integration**: Embeddings for vector search can be generated using advanced AI models or replaced by simpler representations, offering flexibility. This feature allows users to choose between advanced, AI-driven analysis and a simpler, more traditional approach, depending on the needs of their project and the available resources.

### Adaptive Scheduler and Load Balancer

- Task distribution is managed dynamically to balance the workload among agents. This dynamic allocation ensures that resources are utilized effectively and that no agent is overwhelmed, which is crucial in large-scale projects.
- Adaptive task assignment based on the performance and specialization of agents. This means that tasks are always handled by the most capable agent available, leading to faster completion times and higher quality outcomes.

## Optional Integration with AI Language Models

While the `fractal_framework` does not require AI language models, integrating them can add value for some use cases:

- **Natural Language Code Understanding**: AI models can enhance the ability to understand comments or documentation within the code, making pattern recognition more robust. This can be particularly helpful in large codebases where maintaining documentation quality is critical.
- **Refinement and Code Generation**: AI models can help generate alternative implementations or optimized solutions, especially for complex code. This integration can make the development process more efficient, especially when dealing with challenging algorithms or unfamiliar codebases.
- **Advanced Vector Search**: Using AI-generated embeddings may improve search relevance for large codebases. This can be useful when navigating large projects where traditional keyword-based search methods might fall short.

## Example Usage Without ChatGPT

The following example demonstrates how to use the `fractal_framework` without relying on any AI language models:

```python
import asyncio
from fractal_framework.core.fractal_framework import FractalFramework
from fractal_framework.core.fractal_task import FractalTask, TaskType
from fractal_framework.agents.analyze_agent import AnalyzeAgent
from fractal_framework.agents.transform_agent import TransformAgent

async def main():
    # Initialize the framework and agents
    framework = FractalFramework()
    analyze_agent = AnalyzeAgent("Analyzer")
    transform_agent = TransformAgent("Transformer")

    # Add agents to the framework
    framework.add_root_agent(analyze_agent)
    framework.add_root_agent(transform_agent)

    # Define tasks
    analyze_task = FractalTask(
        id="analyze_code",
        type=TaskType.ANALYZE,
        data={"node_id": "function_1"}
    )
    transform_task = FractalTask(
        id="transform_code",
        type=TaskType.TRANSFORM,
        data={"node_id": "function_1", "pattern_name": "singleton"}
    )

    # Run tasks
    results = await framework.run([analyze_task, transform_task])
    for task_id, result in results.items():
        if result.success:
            print(f"Task {task_id} succeeded with result: {result.result}")
        else:
            print(f"Task {task_id} failed with error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

In this example, the `AnalyzeAgent` and `TransformAgent` are used to perform their respective tasks autonomously, demonstrating how the framework's modular design enables seamless task execution without requiring AI-driven inputs. This example highlights the core strength of the `fractal_framework`: it allows users to define and execute complex workflows in a straightforward manner, utilizing the built-in agent capabilities.

## Summary

The `fractal_framework` is designed to operate independently, allowing you to:

- Perform code manipulation, pattern analysis, and transformation autonomously, without relying on external AI services.
- Integrate AI language models optionally, for enhanced capabilities like natural language understanding, advanced vector search, and code refinement.
- Adapt the framework to various scenarios, thanks to its modular, agent-based structure that allows different agents to be added, removed, or customized as needed.
- Leverage adaptive task management and scheduling, ensuring tasks are allocated efficiently and completed effectively based on agent specialization and historical performance.

For many use cases, the core functionality will work effectively without any AI model, making it a flexible and powerful tool for code analysis, transformation, and optimization tasks. The framework’s agent-based approach allows for highly customizable workflows, enabling users to tailor the solution to fit their exact needs, whether it’s small-scale code optimization or large-scale codebase maintenance.
