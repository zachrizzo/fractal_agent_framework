# fractal_framework/core/fractal_task.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set

class TaskType(Enum):
    """Core task types focused on context-aware processing"""
    PROCESS = "process"           # General task processing
    UNDERSTAND = "understand"     # Comprehend context/requirements
    EXECUTE = "execute"          # Execute specific actions
    LEARN = "learn"             # Learn from interactions/results
    RESPOND = "respond"         # Generate responses
    CONTEXT = "context"         # Handle context management

@dataclass
class FractalTask:
    """Represents a task in the fractal structure with enhanced context awareness"""
    id: str
    type: TaskType
    data: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)  # Added context field
    subtasks: List['FractalTask'] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)

    def add_subtask(self, task: 'FractalTask'):
        """Add a subtask to this task"""
        self.subtasks.append(task)

    def add_dependency(self, task_id: str):
        """Add a dependency for this task"""
        self.dependencies.add(task_id)

    def update_context(self, new_context: Dict[str, Any]):
        """Update task context with new information"""
        self.context.update(new_context)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Safely get context value"""
        return self.context.get(key, default)
