from dataclasses import dataclass
from typing import Callable, Any, List, Dict, Optional

@dataclass
class FilterStep:
    """Configuration for a single filter step"""
    name: str
    func: Callable
    params: Dict[str, Any]
    param_keys_for_metrics: Optional[List[str]] = None
    