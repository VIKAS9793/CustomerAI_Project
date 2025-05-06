"""
Type definitions for the sentiment analysis module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TypedDict


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Emotion(str, Enum):
    """Emotion classification labels"""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"


class SentimentScore(TypedDict):
    """Sentiment analysis scores"""

    positive: float
    negative: float
    neutral: float
    compound: float


class EmotionScores(TypedDict):
    """Emotion detection scores"""

    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float


@dataclass
class SentimentResult:
    """Complete sentiment analysis result"""

    text: str
    sentiment: SentimentLabel
    scores: SentimentScore
    emotions: Optional[EmotionScores] = None
    aspects: Optional[Dict[str, SentimentScore]] = None


# Type aliases
AspectSentiments = Dict[str, SentimentScore]
BatchResults = List[SentimentResult]
