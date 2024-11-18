# fractal_framework/core/fractal_task.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set

class TaskType(Enum):
    ANALYZE = "analyze"
    TRANSFORM = "transform"
    GENERATE = "generate"
    OPTIMIZE = "optimize"
    EXPLAIN = "explain"
    COMPOSITE = "composite"
    VECTOR_SEARCH = "vector_search"

@dataclass
class FractalTask:
    """Represents a task in the fractal structure"""
    id: str
    type: TaskType
    data: Dict[str, Any]
    subtasks: List['FractalTask'] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)

    def add_subtask(self, task: 'FractalTask'):
        """Add a subtask to this task"""
        self.subtasks.append(task)

    def add_dependency(self, task_id: str):
        """Add a dependency for this task"""
        self.dependencies.add(task_id)
