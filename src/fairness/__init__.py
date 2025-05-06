"""
Fairness Module for CustomerAI Platform

This module provides comprehensive tools for detecting, visualizing, and mitigating bias in AI systems.
It includes components for bias detection, fairness visualization, and mitigation strategies.
"""

from src.fairness.bias_detector import BiasDetector
from src.fairness.dashboard import FairnessDashboard
from src.fairness.mitigation import FairnessMitigation

__all__ = ["BiasDetector", "FairnessDashboard", "FairnessMitigation"]
