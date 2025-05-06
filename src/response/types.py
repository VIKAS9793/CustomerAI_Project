"""
Type definitions for the response generation module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from src.sentiment.types import SentimentLabel


class ResponseType(str, Enum):
    """Types of customer responses"""

    INQUIRY = "inquiry"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    SUPPORT = "support"
    GENERAL = "general"


class ResponseTone(str, Enum):
    """Tone of the response"""

    FORMAL = "formal"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"


class ResponseContext(TypedDict):
    """Context for response generation"""

    customer_history: Optional[List[Dict[str, str]]]
    previous_interactions: Optional[List[Dict[str, str]]]
    sentiment: Optional[SentimentLabel]
    priority: Optional[int]
    custom_data: Optional[Dict[str, Any]]


@dataclass
class ResponseTemplate:
    """Template for response generation"""

    type: ResponseType
    tone: ResponseTone
    template: str
    variables: List[str]
    conditions: Optional[Dict[str, str]] = None


class ResponseMetadata(TypedDict):
    """Metadata for generated response"""

    response_type: ResponseType
    tone: ResponseTone
    template_id: str
    context_used: Dict[str, bool]
    generation_time: float


@dataclass
class GeneratedResponse:
    """Complete response generation result"""

    text: str
    metadata: ResponseMetadata
    confidence: float
    alternatives: Optional[List[str]] = None


# Type aliases
TemplateVariables = Dict[str, str]
ResponseRules = Dict[ResponseType, List[ResponseTemplate]]
