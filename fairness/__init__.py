"""
Fairness Module for CustomerAI Platform

This module provides comprehensive tools for detecting, visualizing, and mitigating bias in AI systems.
It includes components for bias detection, fairness visualization, and mitigation strategies.
"""

from fairness.bias_detector import BiasDetector
from fairness.dashboard import FairnessDashboard
from fairness.mitigation import FairnessMitigation

__all__ = ['BiasDetector', 'FairnessDashboard', 'FairnessMitigation']
