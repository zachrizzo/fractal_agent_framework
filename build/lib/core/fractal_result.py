# fractal_framework/core/fractal_result.py

from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class FractalResult:
    """Represents the result of a fractal task execution"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    subtask_results: List['FractalResult'] = field(default_factory=list)

    def add_subtask_result(self, result: 'FractalResult'):
        """Add a subtask result"""
        self.subtask_results.append(result)
