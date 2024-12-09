# Fractal Agent Framework

A sophisticated framework for building hierarchical, context-aware AI agents that can collaborate and adapt to solve complex tasks.

## Overview

The Fractal Agent Framework provides a flexible and powerful system for creating AI agents that can:

- Break down complex tasks into manageable subtasks
- Collaborate through a hierarchical agent structure
- Share context and learn from execution results
- Adapt strategies based on performance metrics
- Process and analyze code across multiple programming languages
- Store and search vector embeddings of code
- Handle conversations with memory and storage capabilities

## Core Components

### Fractal Framework (`core/fractal_framework.py`)

The main framework that manages agent hierarchies and task execution. It provides:

- Task distribution and execution
- Agent management
- Context propagation
- Result handling

### Fractal Context (`core/fractal_context.py`)

Maintains execution context throughout the agent hierarchy, including:

- Shared graph structure
- Metadata propagation
- Depth tracking
- Parent-child relationships

### Adaptive Scheduler (`core/adaptive_scheduler.py`)

Intelligent task scheduling with:

- Multiple execution strategies (parallel, sequential, priority)
- Load balancing
- Performance optimization
- Dynamic resource allocation

### Load Balancer (`core/load_balancer.py`)

Distributes tasks among agents based on:

- Agent capabilities
- Current load
- Performance metrics
- Task requirements

## Agent Types

### Enhanced Fractal Agent (`agents/enhanced_fractal_agent.py`)

Base agent class with sophisticated capabilities:

- Pattern recognition
- Code analysis
- Documentation generation
- Performance tracking
- Learning from execution

### Analyze Agent (`agents/analyze_agent.py`)

Specialized in code analysis:

- Pattern detection
- Code structure analysis
- Metrics calculation
- Quality assessment

### Transform Agent (`agents/transform_agent.py`)

Handles code transformations:

- Pattern-based transformations
- Code refactoring
- Language conversion
- Style enforcement

### Vector Search Agent (`agents/vector_search_agent.py`)

Enables semantic code search:

- Code vectorization
- Similarity search
- Pattern matching
- Context-aware results

### Graph Search Agent (`agents/graph_search_agent.py`)

Advanced code navigation:

- Graph-based code analysis
- Relationship discovery
- Path finding
- Community detection

### Chatbot Agent (`agents/chatbot_agent.py`)

Handles interactive conversations:

- Context tracking
- Memory management
- Response generation
- Storage integration

## Utilities

### Language Processors (`utils/language_processors.py`)

Multi-language code processing:

- Language detection
- Code analysis
- Tokenization
- Documentation extraction
- Currently supports Python and Julia

### Vector Store (`utils/vector_store.py`)

Efficient code embedding storage:

- OpenAI embeddings integration
- FAISS indexing
- Bulk operations
- Metadata management

### Storage (`utils/storage.py`)

Conversation and data persistence:

- Local file storage
- Firebase integration
- Conversation management
- Message tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fractal_agent_framework.git

# Install dependencies
pip install -e .
```

## Usage Examples

### Basic Agent Usage

```python
from core.fractal_framework import FractalFramework
from core.fractal_task import FractalTask, TaskType
from agents.analyze_agent import AnalyzeAgent

# Initialize framework
framework = FractalFramework()

# Create and add agent
agent = AnalyzeAgent("code_analyzer")
framework.add_root_agent(agent)

# Create and execute task
task = FractalTask("analyze_code", TaskType.ANALYZE, {
    "code": "def hello(): print('Hello, World!')",
    "analysis_type": "patterns"
})

result = await framework.execute_task(task)
```

### Vector Search Example

```python
from agents.vector_search_agent import VectorSearchAgent
from core.fractal_task import FractalTask

# Initialize agent
agent = VectorSearchAgent("code_searcher")

# Index code
index_task = FractalTask("index", TaskType.ANALYZE, {
    "operation": "index",
    "code": "your_code_here"
})

# Search similar code
search_task = FractalTask("search", TaskType.ANALYZE, {
    "operation": "search",
    "query": "function to calculate fibonacci"
})

results = await agent.execute_task(search_task)
```

### Chatbot Integration

```python
from agents.chatbot_agent import ChatbotAgent
from utils.storage import LocalStorage

# Initialize agent with storage
storage = LocalStorage()
agent = ChatbotAgent("chat_assistant", "openai", "your_api_key", storage=storage)

# Start conversation
conversation_id = await agent.memory.start_new_conversation()

# Process messages
result = await agent.process_message("Tell me about Python decorators")
```

## Architecture

The framework follows a hierarchical architecture where:

1. The FractalFramework acts as the main coordinator
2. Agents are organized in a tree structure
3. Tasks flow down through the hierarchy
4. Context is propagated and maintained
5. Results flow back up
6. Agents can learn and adapt

### Key Features

- **Fractal Structure**: Tasks can be recursively broken down
- **Context Awareness**: Rich context propagation
- **Adaptive Execution**: Dynamic strategy selection
- **Pattern Recognition**: Built-in pattern detection
- **Vector Search**: Semantic code understanding
- **Storage Integration**: Flexible data persistence
- **Language Support**: Multi-language processing

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
