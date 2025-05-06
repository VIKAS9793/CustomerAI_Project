"""
Fairness Framework Configuration

This module provides centralized configuration for the fairness framework components.
Organizations can modify these settings to align with their specific policies,
business requirements, and regulatory needs.

Copyright (c) 2025 Vikas Sahani
"""

import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.types import ConfigDict

if TYPE_CHECKING:
    from src.fairness.types import FairnessConfig

# Default configuration values
DEFAULT_CONFIG: ConfigDict = {
    # Fairness thresholds - organizations can adjust these based on their policies
    "thresholds": {
        "disparate_impact": 0.8,  # Standard 80% rule
        "statistical_parity_difference": 0.1,
        "equal_opportunity_difference": 0.1,
        "predictive_parity_difference": 0.1,
    },
    # Statistical significance settings
    "significance": {
        "pvalue_threshold": 0.05,  # Standard significance level
        "confidence_interval": 0.95,
    },
    # Reporting settings
    "reporting": {
        "severity_levels": {
            "high": 0.2,  # Difference > 0.2 is high severity
            "medium": 0.1,  # Difference > 0.1 is medium severity
            "low": 0.05,  # Difference > 0.05 is low severity
        },
        "include_recommendations": True,
        "max_results_per_page": 1000,
    },
    # Mitigation strategy settings
    "mitigation": {
        "default_strategy": "reweighing",
        "available_strategies": [
            "reweighing",
            "disparate_impact_remover",
            "equalized_odds",
            "calibrated_equalized_odds",
            "reject_option_classification",
            "balanced_sampling",
        ],
    },
    # Visualization settings
    "visualization": {
        "color_palette": "colorblind",  # Accessible color palette
        "chart_types": ["bar", "heatmap", "scatter", "line"],
        "default_chart": "bar",
        "decimal_places": 3,
    },
    # API settings
    "api": {
        "rate_limit": 100,  # Requests per minute
        "max_payload_size_mb": 10,
        "cache_ttl_seconds": 300,
    },
}


class FairnessConfigManager:
    """
    Configuration manager for the fairness framework.

    This class loads and provides access to configuration settings for the
    fairness framework. It supports loading from environment variables,
    configuration files, and provides sensible defaults.
    """

    _instance: Optional["FairnessConfigManager"] = None
    _config: ConfigDict = {}

    @classmethod
    def get_instance(cls) -> "FairnessConfigManager":
        """Get the singleton instance of FairnessConfig."""
        if cls._instance is None:
            cls._instance = FairnessConfigManager()
        return cls._instance

    def __init__(self):
        """Initialize the configuration manager."""
        if FairnessConfigManager._instance is not None:
            raise RuntimeError("Use FairnessConfig.get_instance() instead")

        self._config = DEFAULT_CONFIG.copy()
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from various sources in order of precedence:
        1. Environment variables
        2. Configuration file
        3. Default values
        """
        # Load from configuration file if it exists
        config_path = os.environ.get("FAIRNESS_CONFIG_PATH", "config/fairness_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    self._update_config(file_config)
            except Exception as e:
                print(f"Warning: Failed to load configuration from {config_path}: {str(e)}")

        # Load from environment variables
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Example: FAIRNESS_THRESHOLDS_DISPARATE_IMPACT=0.85
        for key in os.environ:
            if key.startswith("FAIRNESS_"):
                parts = key.lower().split("_")
                if len(parts) >= 3:
                    # Skip the 'FAIRNESS_' prefix
                    parts = parts[1:]
                    self._set_nested_config(parts, os.environ[key])

    def _set_nested_config(self, keys: List[str], value: str) -> None:
        """Set a nested configuration value from a list of keys."""
        current: Dict[str, Any] = self._config
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                # Convert value to appropriate type
                if value.lower() in ("true", "yes", "1"):
                    current[key] = True
                elif value.lower() in ("false", "no", "0"):
                    current[key] = False
                elif value.replace(".", "", 1).isdigit():
                    if "." in value:
                        current[key] = float(value)
                    else:
                        current[key] = int(value)
                else:
                    current[key] = value
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values, preserving nested structure."""

        def update_nested(config: Dict[str, Any], updates: Dict[str, Any]) -> None:
            """Update nested dictionary with new values"""
            if not isinstance(config, dict) or not isinstance(updates, dict):
                return

            for key, value in updates.items():
                if isinstance(value, dict):
                    if key not in config:
                        config[key] = {}
                    elif not isinstance(config[key], dict):
                        config[key] = {}
                    update_nested(config[key], value)
                else:
                    config[key] = value

        update_nested(self._config, new_config)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a value from nested dictionary using dot notation
        """
        config = self._config
        for key in keys:
            if not isinstance(config, dict):
                return default
            config = config.get(key, default)
            if config is None:
                return default
        return config

    def set(self, value: Any, *keys: str) -> None:
        """
        Set a configuration value by key path.

        Args:
            value: The value to set
            *keys: The key path to the configuration value
        """
        if not keys:
            return

        current = self._config
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                current[key] = value
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

    def export_config(self, path: Optional[str] = None) -> Dict:
        """
        Export the current configuration to a file or return as a dictionary.

        Args:
            path: Optional file path to export the configuration to

        Returns:
            The current configuration as a dictionary
        """
        if path:
            with open(path, "w") as f:
                json.dump(self._config, f, indent=2)

        return self._config.copy()


# Convenience function to get configuration
def get_fairness_config() -> 'FairnessConfig':
    """Get the fairness configuration instance."""
    from src.fairness.types import FairnessConfig
    return FairnessConfig.get_instance()
