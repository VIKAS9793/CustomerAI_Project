"""
Type definitions for the fairness module.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Union


class FairnessMetrics(TypedDict, total=False):
    """Fairness metrics type definition"""

    demographic_parity: float
    equal_opportunity: float
    disparate_impact: float
    custom_metrics: Optional[Dict[str, float]]


@dataclass
class FairnessConfig:
    """Fairness configuration type"""

    protected_attributes: List[str]
    metrics: List[str]
    thresholds: Dict[str, float]
    mitigation_methods: List[str]
    custom_metrics: Optional[Dict[str, float]] = None


class BiasMetrics(TypedDict, total=False):
    """Bias detection metrics"""

    bias_score: float
    confidence: float
    affected_groups: List[str]
    mitigation_suggestions: Optional[List[str]]
    severity: Optional[str]


# Type aliases
MetricName = str
GroupName = str
FairnessResult = Dict[MetricName, Union[FairnessMetrics, BiasMetrics]]
MitigationResult = Dict[GroupName, Dict[str, Union[float, str, List[str]]]]
