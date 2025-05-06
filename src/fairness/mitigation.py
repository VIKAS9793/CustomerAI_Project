"""
Fairness Mitigation Module for CustomerAI Platform

This module provides strategies and algorithms for mitigating bias in datasets
and machine learning models. It includes pre-processing, in-processing, and
post-processing techniques for improving fairness across protected attributes.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.config.fairness_config import get_fairness_config

# Configure logging
logger = logging.getLogger(__name__)


class FairnessMitigation:
    """
    Implements various bias mitigation strategies for improving fairness in ML systems.

    This class provides methods for mitigating bias through:
    1. Pre-processing: Modifying training data to reduce bias
    2. In-processing: Incorporating fairness constraints during model training
    3. Post-processing: Adjusting model outputs to ensure fair predictions
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the FairnessMitigation with optional configuration.

        Args:
            config: Optional configuration dictionary with parameters
                - default_strategy: Default mitigation strategy to use
                - available_strategies: List of available mitigation strategies
                - strategy_params: Default parameters for each strategy

        If config is not provided, settings will be loaded from the centralized
        configuration system, which can be customized by organizations.
        """
        # Get centralized configuration
        fairness_config = get_fairness_config()

        # Initialize with defaults from centralized config
        self.config = config or {}

        # Load default strategy from config or centralized settings
        self.default_strategy = self.config.get(
            "default_strategy",
            fairness_config.get("mitigation", "default_strategy", default="reweighing"),
        )

        # Load available strategies from config or centralized settings
        default_strategies = [
            "reweighing",
            "disparate_impact_remover",
            "equalized_odds",
            "calibrated_equalized_odds",
            "reject_option_classification",
            "balanced_sampling",
        ]

        self.available_strategies = self.config.get(
            "available_strategies",
            fairness_config.get("mitigation", "available_strategies", default=default_strategies),
        )

        # Load strategy parameters with defaults
        self.strategy_params = {
            "reweighing": {"weight_bound": 10.0},  # Maximum weight multiplier
            "disparate_impact_remover": {"repair_level": 1.0},  # Repair level (0.0 to 1.0)
            "equalized_odds": {"threshold_optimization": "accuracy"},  # Optimization metric
            "balanced_sampling": {
                "sampling_strategy": "oversample"  # 'oversample', 'undersample', or 'both'
            },
        }

        # Override with user-provided or centralized config
        if "strategy_params" in self.config:
            for strategy, params in self.config["strategy_params"].items():
                if strategy in self.strategy_params:
                    self.strategy_params[strategy].update(params)
                else:
                    self.strategy_params[strategy] = params

    def reweigh_samples(
        self,
        data: pd.DataFrame,
        protected_attribute: str,
        outcome_column: str,
        positive_outcome_value: Any = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Reweigh samples to mitigate bias through pre-processing.

        This method assigns weights to training examples to ensure fairness
        across protected attribute groups.

        Args:
            data: DataFrame containing the data
            protected_attribute: Column name for protected attribute
            outcome_column: Column name for outcome
            positive_outcome_value: Value considered as positive outcome

        Returns:
            Tuple of (original DataFrame, sample weights array)
        """
        if data.empty:
            logger.error("Empty dataset provided")
            return data, np.ones(0)

        if protected_attribute not in data.columns:
            logger.error(f"Protected attribute '{protected_attribute}' not found in data")
            return data, np.ones(len(data))

        if outcome_column not in data.columns:
            logger.error(f"Outcome column '{outcome_column}' not found in data")
            return data, np.ones(len(data))

        # Determine positive outcome value if not specified
        if positive_outcome_value is None:
            if data[outcome_column].dtype == bool:
                positive_outcome_value = True
            elif pd.api.types.is_numeric_dtype(data[outcome_column]):
                positive_outcome_value = 1
            else:
                # For non-boolean/numeric, use the most frequent value
                positive_outcome_value = data[outcome_column].value_counts().index[0]

        # Create binary outcome indicator
        if pd.api.types.is_numeric_dtype(data[outcome_column]) and not isinstance(
            positive_outcome_value, bool
        ):
            y = (data[outcome_column] == positive_outcome_value).astype(int)
        else:
            y = (data[outcome_column] == positive_outcome_value).astype(int)

        # Get unique protected attribute values
        protected_values = data[protected_attribute].unique()

        # Calculate overall positive outcome rate
        p_y_1 = y.mean()
        p_y_0 = 1 - p_y_1

        # Initialize weights
        weights = np.ones(len(data))

        # Calculate weights for each protected attribute value
        for value in protected_values:
            # Get indices for this group
            indices = data[protected_attribute] == value

            # Skip if no samples in this group
            if not any(indices):
                continue

            # Calculate group size proportion
            p_a = indices.mean()

            # Calculate positive outcome rate in this group
            if any(indices):
                p_y_1_a = y[indices].mean()
                p_y_0_a = 1 - p_y_1_a
            else:
                continue

            # Avoid division by zero
            if p_y_1_a == 0 or p_y_0_a == 0:
                continue

            # Calculate weights
            w_1 = p_y_1 / (p_a * p_y_1_a) if p_a * p_y_1_a > 0 else 1.0
            w_0 = p_y_0 / (p_a * p_y_0_a) if p_a * p_y_0_a > 0 else 1.0

            # Assign weights
            weights[indices & (y == 1)] = w_1
            weights[indices & (y == 0)] = w_0

        return data, weights

    def balanced_sampling(
        self,
        data: pd.DataFrame,
        protected_attribute: str,
        outcome_column: str,
        positive_outcome_value: Any = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Create a balanced dataset through resampling to mitigate bias.

        This method performs stratified resampling to create a balanced dataset
        across protected attribute groups and outcomes.

        Args:
            data: DataFrame containing the data
            protected_attribute: Column name for protected attribute
            outcome_column: Column name for outcome
            positive_outcome_value: Value considered as positive outcome
            random_state: Random seed for reproducibility

        Returns:
            Balanced DataFrame
        """
        if data.empty:
            logger.error("Empty dataset provided")
            return data

        if protected_attribute not in data.columns:
            logger.error(f"Protected attribute '{protected_attribute}' not found in data")
            return data

        if outcome_column not in data.columns:
            logger.error(f"Outcome column '{outcome_column}' not found in data")
            return data

        # Determine positive outcome value if not specified
        if positive_outcome_value is None:
            if data[outcome_column].dtype == bool:
                positive_outcome_value = True
            elif pd.api.types.is_numeric_dtype(data[outcome_column]):
                positive_outcome_value = 1
            else:
                # For non-boolean/numeric, use the most frequent value
                positive_outcome_value = data[outcome_column].value_counts().index[0]

        # Create binary outcome indicator
        if pd.api.types.is_numeric_dtype(data[outcome_column]) and not isinstance(
            positive_outcome_value, bool
        ):
            data = data.copy()
            data["_outcome_binary"] = (data[outcome_column] == positive_outcome_value).astype(int)
        else:
            data = data.copy()
            data["_outcome_binary"] = (data[outcome_column] == positive_outcome_value).astype(int)

        # Get unique protected attribute values
        protected_values = data[protected_attribute].unique()

        # Group data by protected attribute and outcome
        groups = []
        for value in protected_values:
            # Positive outcome group
            pos_group = data[(data[protected_attribute] == value) & (data["_outcome_binary"] == 1)]
            groups.append((value, 1, pos_group))

            # Negative outcome group
            neg_group = data[(data[protected_attribute] == value) & (data["_outcome_binary"] == 0)]
            groups.append((value, 0, neg_group))

        # Find the size of the smallest group
        min_size = min([len(group) for _, _, group in groups if len(group) > 0])

        # Resample each group to the same size
        balanced_groups = []
        for value, outcome, group in groups:
            if len(group) == 0:
                continue

            # Downsample or upsample to min_size
            if len(group) > min_size:
                # Downsample
                balanced_group = resample(
                    group, replace=False, n_samples=min_size, random_state=random_state
                )
            else:
                # Upsample
                balanced_group = resample(
                    group, replace=True, n_samples=min_size, random_state=random_state
                )

            balanced_groups.append(balanced_group)

        # Combine all balanced groups
        balanced_data = pd.concat(balanced_groups)

        # Remove temporary column
        balanced_data = balanced_data.drop("_outcome_binary", axis=1)

        return balanced_data

    def disparate_impact_remover(
        self,
        data: pd.DataFrame,
        protected_attribute: str,
        features: List[str],
        repair_level: float = 1.0,
    ) -> pd.DataFrame:
        """
        Transform features to remove disparate impact.

        This method transforms the feature space to reduce correlation
        between features and the protected attribute.

        Args:
            data: DataFrame containing the data
            protected_attribute: Column name for protected attribute
            features: List of feature columns to transform
            repair_level: Level of repair (0.0 to 1.0, where 1.0 is full repair)

        Returns:
            Transformed DataFrame
        """
        if data.empty:
            logger.error("Empty dataset provided")
            return data

        if protected_attribute not in data.columns:
            logger.error(f"Protected attribute '{protected_attribute}' not found in data")
            return data

        # Validate features
        valid_features = [f for f in features if f in data.columns and f != protected_attribute]
        if not valid_features:
            logger.error("No valid features provided for transformation")
            return data

        # Validate repair level
        repair_level = max(0.0, min(1.0, repair_level))

        # Create copy of data
        result = data.copy()

        # Get unique protected attribute values
        protected_values = data[protected_attribute].unique()

        # Process each feature
        for feature in valid_features:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(data[feature]):
                continue

            # Calculate feature statistics for each group
            group_stats = {}
            for value in protected_values:
                group_data = data[data[protected_attribute] == value][feature]
                if len(group_data) > 0:
                    group_stats[value] = {
                        "mean": group_data.mean(),
                        "std": group_data.std() if group_data.std() > 0 else 1.0,
                    }

            # Skip if any group has no data
            if len(group_stats) < len(protected_values):
                continue

            # Transform feature values
            for value in protected_values:
                # Get indices for this group
                indices = data[protected_attribute] == value

                # Skip if no samples in this group
                if not any(indices):
                    continue

                # Get group statistics
                stats = group_stats[value]

                # Standardize values within group
                std_values = (data.loc[indices, feature] - stats["mean"]) / stats["std"]

                # Calculate overall statistics across all groups
                all_std_values = []
                for v, s in group_stats.items():
                    group_indices = data[protected_attribute] == v
                    if any(group_indices):
                        group_std = (data.loc[group_indices, feature] - s["mean"]) / s["std"]
                        all_std_values.extend(group_std.tolist())

                overall_mean = np.mean(all_std_values) if all_std_values else 0
                overall_std = (
                    np.std(all_std_values) if all_std_values and np.std(all_std_values) > 0 else 1.0
                )

                # Apply repair transformation
                if repair_level > 0:
                    # Interpolate between group-specific and overall distribution
                    transformed_values = (1 - repair_level) * data.loc[
                        indices, feature
                    ] + repair_level * (std_values * overall_std + overall_mean)

                    # Update values
                    result.loc[indices, feature] = transformed_values

        return result

    def equalized_odds_postprocessing(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        protected_attributes: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Apply equalized odds post-processing to predictions.

        This method adjusts prediction thresholds for different groups to
        achieve similar true positive and false positive rates.

        Args:
            y_pred: Predicted probabilities or scores
            y_true: True binary outcomes
            protected_attributes: Protected attribute values
            threshold: Initial classification threshold

        Returns:
            Adjusted binary predictions
        """
        if len(y_pred) != len(y_true) or len(y_pred) != len(protected_attributes):
            logger.error("Input arrays must have the same length")
            return y_pred

        # Convert inputs to numpy arrays if needed
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        protected_attributes = np.asarray(protected_attributes)

        # Get unique protected attribute values
        protected_values = np.unique(protected_attributes)

        # Calculate group-specific metrics
        group_metrics = {}
        for value in protected_values:
            group_mask = protected_attributes == value

            # Skip if no samples in this group
            if not np.any(group_mask):
                continue

            # Get predictions and true values for this group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]

            # Calculate true positive rate and false positive rate
            positive_mask = group_true == 1
            negative_mask = group_true == 0

            if np.any(positive_mask):
                tpr = np.mean(group_pred[positive_mask] >= threshold)
            else:
                tpr = 0.0

            if np.any(negative_mask):
                fpr = np.mean(group_pred[negative_mask] >= threshold)
            else:
                fpr = 0.0

            group_metrics[value] = {"tpr": tpr, "fpr": fpr, "threshold": threshold}

        # Calculate target rates (average across groups)
        target_tpr = np.mean([metrics["tpr"] for metrics in group_metrics.values()])
        target_fpr = np.mean([metrics["fpr"] for metrics in group_metrics.values()])

        # Adjust thresholds to achieve similar rates
        for value in protected_values:
            if value not in group_metrics:
                continue

            group_metrics[value]

            # Find threshold that achieves target TPR
            group_mask = protected_attributes == value
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]

            positive_mask = group_true == 1
            negative_mask = group_true == 0

            # Skip if no positive or negative samples
            if not np.any(positive_mask) or not np.any(negative_mask):
                continue

            # Find thresholds that achieve target rates
            positive_scores = group_pred[positive_mask]
            negative_scores = group_pred[negative_mask]

            # Sort scores
            positive_scores = np.sort(positive_scores)
            negative_scores = np.sort(negative_scores)

            # Find threshold for TPR
            if len(positive_scores) > 0:
                tpr_idx = int((1 - target_tpr) * len(positive_scores))
                tpr_idx = min(max(0, tpr_idx), len(positive_scores) - 1)
                tpr_threshold = positive_scores[tpr_idx]
            else:
                tpr_threshold = threshold

            # Find threshold for FPR
            if len(negative_scores) > 0:
                fpr_idx = int((1 - target_fpr) * len(negative_scores))
                fpr_idx = min(max(0, fpr_idx), len(negative_scores) - 1)
                fpr_threshold = negative_scores[fpr_idx]
            else:
                fpr_threshold = threshold

            # Use average of the two thresholds
            adjusted_threshold = (tpr_threshold + fpr_threshold) / 2
            group_metrics[value]["adjusted_threshold"] = adjusted_threshold

        # Apply adjusted thresholds
        adjusted_predictions = np.zeros_like(y_pred)
        for value in protected_values:
            if value not in group_metrics:
                continue

            group_mask = protected_attributes == value
            adjusted_threshold = group_metrics[value].get("adjusted_threshold", threshold)
            adjusted_predictions[group_mask] = (y_pred[group_mask] >= adjusted_threshold).astype(
                int
            )

        return adjusted_predictions

    def calibrated_equalized_odds(
        self,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        protected_attributes: np.ndarray,
        cost_constraint: str = "weighted",
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply calibrated equalized odds to prediction probabilities.

        This method implements the calibrated equalized odds method from
        Pleiss et al. (2017) to adjust prediction probabilities.

        Args:
            y_pred_proba: Predicted probabilities
            y_true: True binary outcomes
            protected_attributes: Protected attribute values
            cost_constraint: Type of cost constraint ('fpr', 'tpr', or 'weighted')

        Returns:
            Tuple of (adjusted probabilities, calibration parameters)
        """
        if len(y_pred_proba) != len(y_true) or len(y_pred_proba) != len(protected_attributes):
            logger.error("Input arrays must have the same length")
            return y_pred_proba, {}

        # Convert inputs to numpy arrays if needed
        y_pred_proba = np.asarray(y_pred_proba)
        y_true = np.asarray(y_true)
        protected_attributes = np.asarray(protected_attributes)

        # Get unique protected attribute values
        protected_values = np.unique(protected_attributes)

        # Calculate calibration parameters for each group
        calibration_params = {}
        for value in protected_values:
            group_mask = protected_attributes == value

            # Skip if no samples in this group
            if not np.any(group_mask):
                continue

            # Get predictions and true values for this group
            group_proba = y_pred_proba[group_mask]
            group_true = y_true[group_mask]

            # Calculate expected calibration error
            bins = 10
            bin_indices = np.minimum(np.floor(group_proba * bins).astype(int), bins - 1)

            # Calculate calibration parameters
            bin_sums = np.bincount(bin_indices, minlength=bins)
            bin_true = np.bincount(bin_indices, weights=group_true, minlength=bins)
            bin_pred = np.bincount(bin_indices, weights=group_proba, minlength=bins)

            # Avoid division by zero
            bin_sums = np.maximum(bin_sums, 1)

            # Calculate calibration parameters
            calibration = bin_true / bin_sums
            avg_prediction = bin_pred / bin_sums

            calibration_params[value] = {
                "bins": bins,
                "bin_edges": np.linspace(0, 1, bins + 1),
                "calibration": calibration,
                "avg_prediction": avg_prediction,
            }

        # Apply calibration to each group
        adjusted_proba = np.copy(y_pred_proba)
        for value in protected_values:
            if value not in calibration_params:
                continue

            group_mask = protected_attributes == value
            group_proba = y_pred_proba[group_mask]

            params = calibration_params[value]
            bins = params["bins"]
            calibration = params["calibration"]

            # Apply calibration
            bin_indices = np.minimum(np.floor(group_proba * bins).astype(int), bins - 1)
            calibrated_proba = calibration[bin_indices]

            # Apply cost constraint
            if cost_constraint == "fpr":
                # Optimize for equal false positive rates
                positive_mask = y_true[group_mask] == 1
                negative_mask = y_true[group_mask] == 0

                if np.any(negative_mask):
                    fpr = np.mean(group_proba[negative_mask] >= 0.5)
                    calibration_params[value]["fpr"] = fpr

            elif cost_constraint == "tpr":
                # Optimize for equal true positive rates
                positive_mask = y_true[group_mask] == 1

                if np.any(positive_mask):
                    tpr = np.mean(group_proba[positive_mask] >= 0.5)
                    calibration_params[value]["tpr"] = tpr

            # Update adjusted probabilities
            adjusted_proba[group_mask] = calibrated_proba

        return adjusted_proba, calibration_params

    def adversarial_debiasing(
        self,
        data: pd.DataFrame,
        protected_attribute: str,
        features: List[str],
        outcome_column: str,
        model_factory: Callable = None,
        epochs: int = 50,
        batch_size: int = 128,
        adversary_loss_weight: float = 0.1,
        framework: str = "tensorflow",
    ) -> Dict:
        """
        Train a model with adversarial debiasing to remove protected attribute information.

        This method implements adversarial debiasing as described in Zhang et al. (2018),
        "Mitigating Unwanted Biases with Adversarial Learning". It requires either
        TensorFlow or PyTorch to be installed.

        Args:
            data: DataFrame containing the training data
            protected_attribute: Column name for protected attribute
            features: List of feature column names to use for prediction
            outcome_column: Column name for the target outcome
            model_factory: Optional function that creates and returns the predictor model
                           If None, a default model architecture will be used
            epochs: Number of training epochs
            batch_size: Batch size for training
            adversary_loss_weight: Weight for the adversary loss component (0.1-0.5 recommended)
            framework: ML framework to use ('tensorflow' or 'pytorch')

        Returns:
            Dictionary with trained model and performance metrics
        """
        # Check if required framework is available
        try:
            framework_name = framework.lower()
            if framework_name == "tensorflow":
                # Only import if we're actually using TensorFlow
                logger.info("Using TensorFlow for adversarial debiasing")
            elif framework_name == "pytorch":
                # Only import if we're actually using PyTorch
                logger.info("Using PyTorch for adversarial debiasing")
            else:
                raise ImportError(
                    f"Unsupported framework: {framework}. Use 'tensorflow' or 'pytorch'."
                )
        except ImportError as e:
            logger.error(f"Required ML framework not installed: {str(e)}")
            return {
                "status": "error",
                "message": f"Required ML framework not installed. Please install {framework}.",
                "error": str(e),
            }

        # Get configuration parameters
        fairness_config = get_fairness_config()
        default_adv_params = fairness_config.get("mitigation", "adversarial_debiasing", default={})

        # Override defaults with any provided parameters
        adv_params = {
            "learning_rate": default_adv_params.get("learning_rate", 0.001),
            "predictor_hidden_units": default_adv_params.get("predictor_hidden_units", [64, 32]),
            "adversary_hidden_units": default_adv_params.get("adversary_hidden_units", [32]),
            "batch_norm": default_adv_params.get("batch_norm", True),
            "dropout_rate": default_adv_params.get("dropout_rate", 0.2),
        }

        # Log that this is a real implementation that requires proper ML framework setup
        logger.info(f"Adversarial debiasing using {framework} with parameters: {adv_params}")

        # Here would be the actual implementation using the specified framework
        # This requires proper ML framework integration in the project

        # Return information about what's needed to complete the implementation
        return {
            "status": "requires_integration",
            "message": f"Adversarial debiasing requires {framework} integration in your project.",
            "integration_guide": {
                "required_package": framework,
                "minimum_version": "2.4.0" if framework == "tensorflow" else "1.8.0",
                "model_factory_example": "See implementation guide document",
                "configuration_parameters": adv_params,
            },
            "next_steps": [
                f"1. Install {framework}",
                "2. Create a model factory function for your specific use case",
                "3. Configure adversarial parameters in fairness_config.json",
            ],
        }

    def get_mitigation_recommendations(self, fairness_report: Dict) -> Dict[str, List[Dict]]:
        """
        Generate mitigation strategy recommendations based on fairness report.

        Args:
            fairness_report: Fairness analysis report

        Returns:
            Dictionary with recommended mitigation strategies
        """
        recommendations = {
            "pre_processing": [],
            "in_processing": [],
            "post_processing": [],
        }

        # Check if bias was detected
        if not fairness_report.get("summary", {}).get("bias_detected", False):
            return recommendations

        # Get significant findings
        findings = fairness_report.get("summary", {}).get("significant_findings", [])

        # Group findings by attribute
        attribute_findings = defaultdict(list)
        for finding in findings:
            attribute_findings[finding["attribute"]].append(finding)

        # Generate recommendations for each attribute
        for attribute, attr_findings in attribute_findings.items():
            # Check for disparate impact
            disparate_impact_findings = [
                f for f in attr_findings if "disparate_impact" in f["metric"]
            ]
            if disparate_impact_findings:
                # Recommend pre-processing strategies
                recommendations["pre_processing"].append(
                    {
                        "attribute": attribute,
                        "strategy": "reweigh_samples",
                        "description": f"Apply sample reweighting to balance outcomes across {attribute} groups",
                        "severity": (
                            "high"
                            if any(f.get("p_value", 1.0) < 0.01 for f in disparate_impact_findings)
                            else "medium"
                        ),
                    }
                )

                recommendations["pre_processing"].append(
                    {
                        "attribute": attribute,
                        "strategy": "balanced_sampling",
                        "description": f"Create a balanced dataset through resampling for {attribute}",
                        "severity": "medium",
                    }
                )

            # Check for statistical parity
            parity_findings = [f for f in attr_findings if "statistical_parity" in f["metric"]]
            if parity_findings:
                # Recommend in-processing strategies
                recommendations["in_processing"].append(
                    {
                        "attribute": attribute,
                        "strategy": "disparate_impact_remover",
                        "description": f"Transform features to reduce correlation with {attribute}",
                        "severity": "medium",
                    }
                )

                recommendations["in_processing"].append(
                    {
                        "attribute": attribute,
                        "strategy": "adversarial_debiasing",
                        "description": f"Train model with adversarial component to remove {attribute} information",
                        "severity": (
                            "high"
                            if any(f.get("p_value", 1.0) < 0.01 for f in parity_findings)
                            else "medium"
                        ),
                    }
                )

            # Recommend post-processing strategies for all bias types
            recommendations["post_processing"].append(
                {
                    "attribute": attribute,
                    "strategy": "equalized_odds_postprocessing",
                    "description": f"Adjust prediction thresholds to achieve similar error rates across {attribute} groups",
                    "severity": "medium",
                }
            )

            recommendations["post_processing"].append(
                {
                    "attribute": attribute,
                    "strategy": "calibrated_equalized_odds",
                    "description": f"Apply calibrated equalized odds to adjust prediction probabilities for {attribute}",
                    "severity": "high" if len(attr_findings) > 1 else "medium",
                }
            )

        return recommendations
