"""CustomerAI project root module.

This module provides a centralized interface to all major components
of the CustomerAI project. Each component is properly type-hinted and
documented for better IDE support and code maintainability.
"""

from typing import Type, TypeVar, cast

from src.fairness.bias_detector import BiasDetector
from src.human_review.review_manager import ReviewManager
from src.privacy.anonymizer import DataAnonymizer
from src.response_generator import ResponseGenerator
from src.sentiment_analyzer import SentimentAnalyzer
from src.utils.date_provider import DateProvider

# Type variable for singleton instances
T = TypeVar("T")


def get_singleton(cls: Type[T]) -> T:
    """Get singleton instance of a class with proper type checking."""
    instance = cls.get_instance()
    return cast(T, instance)


__version__ = "1.0.0"
__all__ = [
    "BiasDetector",
    "ReviewManager",
    "DataAnonymizer",
    "ResponseGenerator",
    "SentimentAnalyzer",
    "DateProvider",
    "get_singleton",
]
