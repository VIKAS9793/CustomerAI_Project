"""
Type definitions for the fairness module.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, ValidationError, validator


class MetricType(str, Enum):
    """Enum for different types of fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_ODDS = "equal_odds"
    EQUAL_OPPO = "equal_oppo"
    PREDICTIVE_PARITY = "predictive_parity"

    @classmethod
    def validate(cls, v: str) -> str:
        """Validate metric type."""
        if v not in cls.__members__:
            raise ValueError(f"Invalid metric type: {v}")
        return v

    def __str__(self) -> str:
        return self.value


class MitigationStrategy(str, Enum):
    """Enum for different mitigation strategies."""

    REWEIGHING = "reweighing"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    ADVERSARIAL = "adversarial"

    @classmethod
    def validate(cls, v: str) -> str:
        """Validate mitigation strategy."""
        if v not in cls.__members__:
            raise ValueError(f"Invalid mitigation strategy: {v}")
        return v

    def __str__(self) -> str:
        return self.value


class FairnessMetric(BaseModel):
    """Base class for fairness metrics."""

    name: str
    description: str
    type: MetricType
    threshold: float = Field(gt=0, le=1)

    @validator("threshold")
    def validate_threshold(cls, v):
        """Validate threshold value."""
        if v <= 0 or v > 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return json.loads(self.json())


class FairnessConfig(BaseModel):
    """Configuration for fairness analysis."""

    metrics: List[FairnessMetric]
    sensitive_attributes: List[str]
    mitigation_strategy: MitigationStrategy
    confidence_level: float = Field(gt=0.9, le=1)

    @validator("confidence_level")
    def validate_confidence_level(cls, v):
        """Validate confidence level."""
        if v <= 0.9 or v > 1:
            raise ValueError("Confidence level must be between 0.9 and 1")
        return v

    def validate_metrics(self) -> None:
        """Validate metrics configuration."""
        if not self.metrics:
            raise ValueError("At least one metric must be specified")

        metric_types = set(m.type for m in self.metrics)
        if len(metric_types) > len(self.metrics):
            raise ValueError("Duplicate metrics detected")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return json.loads(self.json())


class MitigationConfig(BaseModel):
    """Configuration for fairness mitigation."""

    strategy: MitigationStrategy
    parameters: Dict[str, Any]
    max_iterations: int = Field(gt=0)
    convergence_threshold: float = Field(gt=0, le=1)

    @validator("max_iterations")
    def validate_max_iterations(cls, v):
        """Validate max iterations."""
        if v <= 0:
            raise ValueError("Max iterations must be greater than 0")
        return v

    def validate_parameters(self) -> None:
        """Validate mitigation parameters."""
        required_params = {
            MitigationStrategy.REWEIGHING: ["sample_weight"],
            MitigationStrategy.PREPROCESSING: ["processing_steps"],
            MitigationStrategy.POSTPROCESSING: ["adjustment_factors"],
            MitigationStrategy.ADVERSARIAL: ["adversary_steps"],
        }

        if not all(param in self.parameters for param in required_params[self.strategy]):
            raise ValueError(f"Missing required parameters for {self.strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return json.loads(self.json())


class FairnessResult(BaseModel):
    """Results of fairness analysis."""

    metric_results: Dict[str, float]
    mitigation_applied: bool
    mitigation_effectiveness: float = Field(gt=0, le=1)
    recommendations: List[str]

    @validator("mitigation_effectiveness")
    def validate_mitigation_effectiveness(cls, v):
        """Validate mitigation effectiveness."""
        if v <= 0 or v > 1:
            raise ValueError("Mitigation effectiveness must be between 0 and 1")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return json.loads(self.json())


class FairnessAnalysis(BaseModel):
    """Complete fairness analysis object."""

    config: FairnessConfig
    results: FairnessResult
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("Invalid timestamp format")
        return v

    def validate_analysis(self) -> None:
        """Validate entire analysis configuration."""
        self.config.validate_metrics()
        self.results.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return json.loads(self.json())


T = TypeVar("T")


class FairnessStrategy(Generic[T]):
    """Generic fairness strategy interface."""

    def __init__(self, config: T):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        try:
            if isinstance(self.config, BaseModel):
                self.config.validate()
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

    def execute(self, data: Any) -> Any:
        """Execute the fairness strategy."""
        raise NotImplementedError

    def get_results(self) -> Dict[str, Any]:
        """Get the results of the fairness analysis."""
        raise NotImplementedError


class FairnessDashboardConfig(BaseModel):
    """Configuration for fairness dashboard."""

    metrics_to_display: List[MetricType]
    update_interval: int = Field(default=60, gt=0)  # seconds
    max_history: int = Field(default=1000, gt=0)
    show_trends: bool = True
    show_recommendations: bool = True


class FairnessAnalysisInput(BaseModel):
    """Input data for fairness analysis."""

    predictions: List[Any]
    actuals: List[Any]
    sensitive_attributes: Dict[str, List[Any]]
    config: FairnessConfig


class FairnessAnalysisOutput(BaseModel):
    """Output of fairness analysis."""

    results: FairnessResult
    dashboard_config: FairnessDashboardConfig
    mitigation_config: Optional[MitigationConfig] = None
