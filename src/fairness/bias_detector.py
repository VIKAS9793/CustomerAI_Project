"""
BiasDetector Module - Enhanced Fairness Framework for CustomerAI

This module provides comprehensive bias detection and fairness analysis capabilities
for machine learning models and datasets. It implements multiple fairness metrics,
statistical significance testing, and generates detailed reports with mitigation
recommendations.
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config.fairness_config import get_fairness_config
from src.utils.date_provider import DateProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BiasDetector:
    """
    Comprehensive bias detection and fairness analysis for AI systems.

    This class provides methods to detect bias in datasets and model outcomes
    across protected attributes (such as age, gender, race, etc.) and generates
    detailed fairness reports with mitigation recommendations.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the BiasDetector with optional configuration.

        Args:
            config: Optional configuration dictionary with parameters:
                - significance_level: Statistical significance threshold
                - fairness_threshold: Threshold for fairness metrics
                - metrics: List of fairness metrics to compute

        If config is not provided, settings will be loaded from the centralized
        configuration system, which can be customized by organizations.
        """
        # Get centralized configuration
        fairness_config = get_fairness_config()

        # Initialize with defaults from centralized config
        self.config = config or {}

        # Load significance level from config or centralized settings
        self.significance_level = self.config.get(
            "significance_level",
            fairness_config.get("significance", "pvalue_threshold", default=0.05),
        )

        # Load fairness threshold from config or centralized settings
        self.fairness_threshold = self.config.get(
            "fairness_threshold",
            fairness_config.get("thresholds", "disparate_impact", default=0.8),
        )

        # Load metrics from config or centralized settings
        default_metrics = [
            "disparate_impact",
            "statistical_parity",
            "equal_opportunity",
            "predictive_parity",
        ]
        self.metrics = self.config.get("metrics", default_metrics)

        # Load reporting settings
        self.severity_levels = self.config.get(
            "severity_levels",
            fairness_config.get(
                "reporting",
                "severity_levels",
                default={"high": 0.2, "medium": 0.1, "low": 0.05},
            ),
        )

        # Maximum results per page for memory efficiency
        self.max_results_per_page = self.config.get(
            "max_results_per_page",
            fairness_config.get("reporting", "max_results_per_page", default=1000),
        )

        # Initialize results storage
        self.last_results = None

    def detect_outcome_bias(
        self,
        data: pd.DataFrame,
        attributes: List[str],
        outcome_columns: List[str],
        positive_outcome_value: Any = None,
    ) -> Dict:
        """
        Detect bias in outcomes across protected attributes.

        Args:
            data: DataFrame containing the data to analyze
            attributes: List of column names for protected attributes
            outcome_columns: List of column names for outcomes to analyze
            positive_outcome_value: Value considered as positive outcome for binary outcomes
                                   (default: True for boolean, 1 for numeric)

        Returns:
            Dictionary with bias detection results
        """
        results = {
            "timestamp": DateProvider.get_instance().iso_format(),
            "dataset_size": len(data),
            "attributes_analyzed": attributes,
            "outcomes_analyzed": outcome_columns,
            "metrics_used": self.metrics,
            "attribute_results": {},
            "summary": {},
        }

        # Validate inputs
        if data.empty:
            results["error"] = "Empty dataset provided"
            return results

        if not outcome_columns:
            results["error"] = "No outcome columns provided"
            return results

        # Check if attributes exist in data
        valid_attributes = [attr for attr in attributes if attr in data.columns]
        if not valid_attributes:
            results["error"] = f"None of the provided attributes {attributes} found in dataset"
            return results

        # Analyze each attribute
        for attribute in valid_attributes:
            attribute_values = data[attribute].unique()
            if len(attribute_values) < 2:
                continue  # Skip attributes with only one value

            results["attribute_results"][attribute] = {
                "values": list(attribute_values),
                "distribution": {
                    str(val): (data[attribute] == val).mean() for val in attribute_values
                },
                "outcome_metrics": {},
            }

            # Analyze each outcome for this attribute
            for outcome in outcome_columns:
                if outcome not in data.columns:
                    continue

                # Determine positive outcome value if not specified
                if positive_outcome_value is None:
                    if data[outcome].dtype == bool:
                        pos_val = True
                    elif pd.api.types.is_numeric_dtype(data[outcome]):
                        pos_val = 1
                    else:
                        # For non-boolean/numeric, use the most frequent value
                        pos_val = data[outcome].value_counts().index[0]
                else:
                    pos_val = positive_outcome_value

                outcome_results = self._analyze_outcome(data, attribute, outcome, pos_val)

                results["attribute_results"][attribute]["outcome_metrics"][
                    outcome
                ] = outcome_results

        # Generate summary statistics
        results["summary"] = self._generate_summary(results["attribute_results"])

        # Store results for later use
        self.last_results = results

        return results

    def _analyze_outcome(
        self, data: pd.DataFrame, attribute: str, outcome: str, positive_value: Any
    ) -> Dict:
        """
        Analyze a single outcome for a single attribute.

        Args:
            data: DataFrame with the data
            attribute: Protected attribute column name
            outcome: Outcome column name
            positive_value: Value considered as positive outcome

        Returns:
            Dictionary with outcome analysis results
        """
        result = {
            "positive_outcome_value": positive_value,
            "metrics": {},
            "statistical_significance": {},
        }

        # Get unique attribute values
        attribute_values = data[attribute].unique()

        # Calculate metrics for each attribute value
        metrics_by_group = {}
        for value in attribute_values:
            group_data = data[data[attribute] == value]

            # Skip groups with no data
            if len(group_data) == 0:
                continue

            # Calculate positive outcome rate
            if pd.api.types.is_numeric_dtype(data[outcome]):
                # For numeric outcomes, calculate mean
                positive_rate = group_data[outcome].mean()
            else:
                # For categorical outcomes, calculate proportion of positive value
                positive_rate = (group_data[outcome] == positive_value).mean()

            metrics_by_group[str(value)] = {
                "sample_size": len(group_data),
                "positive_rate": positive_rate,
            }

        # Calculate fairness metrics across groups
        if len(metrics_by_group) >= 2:
            # Find reference group (highest positive rate)
            reference_group = max(metrics_by_group.items(), key=lambda x: x[1]["positive_rate"])
            reference_name = reference_group[0]
            reference_metrics = reference_group[1]

            for group_name, group_metrics in metrics_by_group.items():
                if group_name == reference_name:
                    continue

                # Calculate disparate impact
                if "disparate_impact" in self.metrics:
                    if reference_metrics["positive_rate"] > 0:
                        disparate_impact = (
                            group_metrics["positive_rate"] / reference_metrics["positive_rate"]
                        )
                    else:
                        disparate_impact = (
                            float("inf") if group_metrics["positive_rate"] > 0 else 1.0
                        )

                    result["metrics"][
                        f"disparate_impact_{group_name}_vs_{reference_name}"
                    ] = disparate_impact

                # Calculate statistical parity difference
                if "statistical_parity" in self.metrics:
                    stat_parity = (
                        group_metrics["positive_rate"] - reference_metrics["positive_rate"]
                    )
                    result["metrics"][
                        f"statistical_parity_{group_name}_vs_{reference_name}"
                    ] = stat_parity

                # Calculate statistical significance
                p_value = self._calculate_significance(
                    group_metrics["positive_rate"],
                    reference_metrics["positive_rate"],
                    group_metrics["sample_size"],
                    reference_metrics["sample_size"],
                )
                result["statistical_significance"][f"{group_name}_vs_{reference_name}"] = {
                    "p_value": p_value,
                    "is_significant": p_value < self.significance_level,
                }

        return result

    def _calculate_significance(self, rate1: float, rate2: float, n1: int, n2: int) -> float:
        """
        Calculate statistical significance between two proportions.

        Args:
            rate1: Positive rate for group 1
            rate2: Positive rate for group 2
            n1: Sample size for group 1
            n2: Sample size for group 2

        Returns:
            p-value for the difference
        """
        # Use two-proportion z-test
        count1 = rate1 * n1
        count2 = rate2 * n2

        # Handle edge cases
        if n1 == 0 or n2 == 0:
            return 1.0

        try:
            # Use statsmodels proportions_ztest if available
            from statsmodels.stats.proportion import proportions_ztest

            counts = np.array([count1, count2])
            nobs = np.array([n1, n2])
            stat, p_value = proportions_ztest(counts, nobs)
            return p_value
        except ImportError:
            # Fallback to scipy implementation
            z_score = (rate1 - rate2) / np.sqrt(
                (rate1 * (1 - rate1) / n1) + (rate2 * (1 - rate2) / n2)
            )
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            return p_value

    def _calculate_disparate_impact(self, rate1: float, rate2: float) -> float:
        """
        Calculate disparate impact between two rates.

        Args:
            rate1: Rate for group 1 (typically unprivileged group)
            rate2: Rate for group 2 (typically privileged group)

        Returns:
            Disparate impact ratio
        """
        if rate2 == 0:
            return float("inf") if rate1 > 0 else 1.0
        return rate1 / rate2

    def _generate_summary(self, attribute_results: Dict) -> Dict:
        """
        Generate summary statistics from detailed results.

        Args:
            attribute_results: Dictionary with detailed attribute results

        Returns:
            Summary dictionary
        """
        summary = {
            "fairness_score": 1.0,  # Start with perfect score
            "bias_detected": False,
            "significant_findings": [],
            "recommendations": [],
        }

        total_metrics = 0
        fair_metrics = 0

        # Analyze each attribute's metrics
        for attr_name, attr_data in attribute_results.items():
            for outcome_name, outcome_data in attr_data.get("outcome_metrics", {}).items():
                for metric_name, metric_value in outcome_data.get("metrics", {}).items():
                    total_metrics += 1

                    # Check if metric indicates bias
                    is_fair = True
                    if "disparate_impact" in metric_name:
                        # Disparate impact should be between threshold and 1/threshold
                        is_fair = (
                            self.fairness_threshold <= metric_value <= (1 / self.fairness_threshold)
                        )
                    elif "statistical_parity" in metric_name:
                        # Statistical parity difference should be close to zero
                        is_fair = abs(metric_value) <= (1 - self.fairness_threshold)

                    if is_fair:
                        fair_metrics += 1
                    else:
                        # Check if statistically significant
                        groups = metric_name.split("_")[-1]  # Extract group comparison
                        if groups in outcome_data.get("statistical_significance", {}):
                            if outcome_data["statistical_significance"][groups]["is_significant"]:
                                # Add to significant findings
                                summary["bias_detected"] = True
                                finding = {
                                    "attribute": attr_name,
                                    "outcome": outcome_name,
                                    "metric": metric_name,
                                    "value": metric_value,
                                    "threshold": self.fairness_threshold,
                                    "p_value": outcome_data["statistical_significance"][groups][
                                        "p_value"
                                    ],
                                }
                                summary["significant_findings"].append(finding)

        # Calculate overall fairness score
        if total_metrics > 0:
            summary["fairness_score"] = fair_metrics / total_metrics

        # Generate recommendations based on findings
        if summary["bias_detected"]:
            summary["recommendations"] = self._generate_recommendations(
                summary["significant_findings"]
            )

        return summary

    def _generate_recommendations(self, findings: List[Dict]) -> List[str]:
        """
        Generate recommendations based on bias findings.

        Args:
            findings: List of significant bias findings

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not findings:
            return recommendations

        # Group findings by attribute
        attr_findings = {}
        for finding in findings:
            attr = finding["attribute"]
            if attr not in attr_findings:
                attr_findings[attr] = []
            attr_findings[attr].append(finding)

        # Generate recommendations for each attribute
        for attr, attr_findings_list in attr_findings.items():
            # Basic recommendation
            recommendations.append(
                f"Review data collection and processing for '{attr}' to identify potential sources of bias."
            )

            # Specific recommendations based on metric type
            disparate_findings = [
                f for f in attr_findings_list if "disparate_impact" in f["metric"]
            ]
            if disparate_findings:
                recommendations.append(
                    f"Consider rebalancing training data for '{attr}' to address disparate impact."
                )

            parity_findings = [f for f in attr_findings_list if "statistical_parity" in f["metric"]]
            if parity_findings:
                recommendations.append(
                    f"Implement post-processing adjustments to equalize outcomes across '{attr}' groups."
                )

        # General recommendations
        recommendations.append(
            "Implement regular fairness monitoring to track bias metrics over time."
        )

        recommendations.append(
            "Consider using fairness constraints during model training to minimize bias."
        )

        return recommendations

    def generate_fairness_report(self, results: Dict = None, threshold: float = None) -> Dict:
        """
        Generate a comprehensive fairness report from bias detection results.

        Args:
            results: Bias detection results (if None, uses last results)
            threshold: Optional override for fairness threshold

        Returns:
            Dictionary with fairness report
        """
        if results is None:
            if self.last_results is None:
                return {"error": "No results available. Run detect_outcome_bias first."}
            results = self.last_results

        # Use provided threshold or default
        threshold = threshold or self.fairness_threshold

        # Create report structure
        report = {
            "timestamp": DateProvider.get_instance().iso_format(),
            "dataset_info": {
                "size": results.get("dataset_size", 0),
                "attributes_analyzed": results.get("attributes_analyzed", []),
                "outcomes_analyzed": results.get("outcomes_analyzed", []),
            },
            "summary": results.get("summary", {}),
            "detailed_findings": [],
        }

        # Add threshold to summary
        report["summary"]["threshold"] = threshold

        # Process significant findings
        for finding in report["summary"].get("significant_findings", []):
            detail = {
                "attribute": finding["attribute"],
                "outcome": finding["outcome"],
                "metric": finding["metric"],
                "value": finding["value"],
                "threshold": finding["threshold"],
                "p_value": finding["p_value"],
                "severity": "high" if finding["p_value"] < 0.01 else "medium",
            }
            report["detailed_findings"].append(detail)

        return report

    def visualize_fairness_metrics(self, results: Dict = None, save_path: str = None) -> None:
        """
        Generate visualizations for fairness metrics.

        Args:
            results: Bias detection results (if None, uses last results)
            save_path: Path to save visualizations (if None, displays them)
        """
        if results is None:
            if self.last_results is None:
                logger.error("No results available. Run detect_outcome_bias first.")
                return
            results = self.last_results

        # Extract metrics for visualization
        metrics_data = []

        for attr, attr_data in results.get("attribute_results", {}).items():
            for outcome, outcome_data in attr_data.get("outcome_metrics", {}).items():
                for metric, value in outcome_data.get("metrics", {}).items():
                    metrics_data.append(
                        {
                            "attribute": attr,
                            "outcome": outcome,
                            "metric": metric,
                            "value": value,
                        }
                    )

        if not metrics_data:
            logger.warning("No metrics data available for visualization.")
            return

        # Convert to DataFrame for easier plotting
        metrics_df = pd.DataFrame(metrics_data)

        # Create visualizations
        plt.figure(figsize=(12, 8))

        # Plot disparate impact metrics
        disparate_df = metrics_df[metrics_df["metric"].str.contains("disparate_impact")]
        if not disparate_df.empty:
            plt.subplot(2, 1, 1)
            sns.barplot(x="attribute", y="value", hue="metric", data=disparate_df)
            plt.axhline(y=self.fairness_threshold, color="r", linestyle="--", alpha=0.7)
            plt.axhline(y=1.0, color="g", linestyle="-", alpha=0.7)
            plt.axhline(y=1 / self.fairness_threshold, color="r", linestyle="--", alpha=0.7)
            plt.title("Disparate Impact by Attribute")
            plt.ylabel("Disparate Impact Ratio")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot statistical parity metrics
        parity_df = metrics_df[metrics_df["metric"].str.contains("statistical_parity")]
        if not parity_df.empty:
            plt.subplot(2, 1, 2)
            sns.barplot(x="attribute", y="value", hue="metric", data=parity_df)
            plt.axhline(y=0, color="g", linestyle="-", alpha=0.7)
            plt.title("Statistical Parity Difference by Attribute")
            plt.ylabel("Statistical Parity Difference")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Fairness visualization saved to {save_path}")
        else:
            plt.show()
