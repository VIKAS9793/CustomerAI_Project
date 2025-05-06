"""
Core type definitions for the CustomerAI project.
"""

from typing import Any, Dict, TypeVar, Union

# Type aliases
JSON = Dict[str, Any]
ConfigDict = Dict[str, Union[str, int, float, bool, None, Dict[str, Any]]]
MetricsDict = Dict[str, float]

# Generic types
T = TypeVar("T")
KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


# Custom type definitions
class TypedDict(Dict[KeyT, ValueT]):
    """Base class for typed dictionaries"""

    pass


class MetricsResult(TypedDict):
    """Type for metrics results"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
