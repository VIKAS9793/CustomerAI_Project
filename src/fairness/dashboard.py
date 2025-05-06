"""
Fairness Dashboard Module for CustomerAI Platform

This module provides visualization and dashboard capabilities for fairness metrics,
making bias detection results more accessible and actionable for business users.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.fairness_config import get_fairness_config
from src.fairness.bias_detector import BiasDetector

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FairnessDashboard:
    """
    Interactive dashboard for visualizing and exploring fairness metrics.

    This class provides methods to create an interactive Streamlit dashboard
    for exploring bias detection results, with visualizations and actionable
    recommendations.
    """

    def __init__(self, bias_detector: Optional[BiasDetector] = None, config: Dict = None):
        """
        Initialize the FairnessDashboard.

        Args:
            bias_detector: Optional BiasDetector instance
            config: Optional configuration dictionary with visualization parameters
                - color_palette: Color palette for visualizations
                - chart_types: Available chart types
                - default_chart: Default chart type
                - decimal_places: Number of decimal places to display
                - max_items: Maximum number of items to display per page

        If config is not provided, settings will be loaded from the centralized
        configuration system, which can be customized by organizations.
        """
        # Get centralized configuration
        fairness_config = get_fairness_config()

        # Initialize with defaults from centralized config
        self.config = config or {}
        self.bias_detector = bias_detector or BiasDetector()
        self.results = None

        # Load visualization settings from config or centralized settings
        self.color_palette = self.config.get(
            "color_palette",
            fairness_config.get("visualization", "color_palette", default="colorblind"),
        )

        self.chart_types = self.config.get(
            "chart_types",
            fairness_config.get(
                "visualization",
                "chart_types",
                default=["bar", "heatmap", "scatter", "line"],
            ),
        )

        self.default_chart = self.config.get(
            "default_chart",
            fairness_config.get("visualization", "default_chart", default="bar"),
        )

        self.decimal_places = self.config.get(
            "decimal_places",
            fairness_config.get("visualization", "decimal_places", default=3),
        )

        # Maximum items to display per page for memory efficiency
        self.max_items = self.config.get(
            "max_items",
            fairness_config.get("reporting", "max_results_per_page", default=1000),
        )

    def load_results(self, results_dict: Dict = None, results_file: str = None) -> None:
        """
        Load bias detection results from dictionary or file.

        Args:
            results_dict: Dictionary containing bias detection results
            results_file: Path to JSON file containing results
        """
        if results_dict:
            self.results = results_dict
        elif results_file and os.path.exists(results_file):
            try:
                # Use streaming JSON parser for large files
                with open(results_file, "r") as f:
                    self.results = json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"Error parsing results file: {str(e)}")
                self.results = None
            except MemoryError:
                st.error(
                    "The results file is too large to load into memory. Please use a smaller file."
                )
                self.results = None
        else:
            # Use last results from bias detector if available
            self.results = self.bias_detector.last_results

    def run_dashboard(self) -> None:
        """
        Run the Streamlit dashboard application.
        """
        st.set_page_config(
            page_title="Fairness Analysis Dashboard",
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("⚖️ AI Fairness Analysis Dashboard")
        st.write("Analyze and visualize fairness metrics across protected attributes")

        # Sidebar for controls
        with st.sidebar:
            st.header("Controls")

            # Option to upload results file
            uploaded_file = st.file_uploader("Upload fairness results (JSON)", type=["json"])
            if uploaded_file:
                self.results = json.loads(uploaded_file.read())
                st.success("Results loaded successfully!")

            # Option to run analysis on new data
            st.subheader("Or analyze new data")
            data_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

            if data_file:
                try:
                    data = pd.read_csv(data_file)
                    st.write(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

                    # Select attributes and outcomes
                    st.subheader("Analysis Configuration")
                    attributes = st.multiselect(
                        "Select protected attributes",
                        options=data.columns,
                        help="Columns representing protected characteristics (e.g., age, gender)",
                    )

                    outcomes = st.multiselect(
                        "Select outcome columns",
                        options=data.columns,
                        help="Columns representing outcomes to analyze for fairness",
                    )

                    # Run analysis button
                    if st.button("Run Fairness Analysis") and attributes and outcomes:
                        with st.spinner("Running analysis..."):
                            self.results = self.bias_detector.detect_outcome_bias(
                                data, attributes, outcomes
                            )
                            st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error loading or processing data: {str(e)}")

        # Main content area
        if not self.results:
            st.info("Please upload fairness results or run a new analysis to begin.")
            return

        # Display summary
        self._display_summary()

        # Display detailed metrics
        self._display_detailed_metrics()

        # Display recommendations
        self._display_recommendations()

    def _display_summary(self) -> None:
        """Display summary metrics and overview."""
        st.header("Fairness Summary")

        summary = self.results.get("summary", {})

        # Create metrics row
        col1, col2, col3 = st.columns(3)

        fairness_score = summary.get("fairness_score", 0) * 100
        with col1:
            st.metric(
                "Overall Fairness Score",
                f"{fairness_score:.1f}%",
                delta=None,
                delta_color="normal",
            )

        with col2:
            bias_detected = summary.get("bias_detected", False)
            st.metric(
                "Bias Status",
                "Bias Detected" if bias_detected else "No Significant Bias",
                delta=None,
                delta_color="off",
            )

        with col3:
            findings_count = len(summary.get("significant_findings", []))
            st.metric("Significant Findings", findings_count, delta=None, delta_color="off")

        # Dataset info
        st.subheader("Dataset Information")
        dataset_info = self.results.get("dataset_info", {})
        st.write(f"**Size:** {dataset_info.get('size', 'N/A')} records")
        st.write(
            f"**Attributes Analyzed:** {', '.join(dataset_info.get('attributes_analyzed', []))}"
        )
        st.write(f"**Outcomes Analyzed:** {', '.join(dataset_info.get('outcomes_analyzed', []))}")

        # Fairness threshold
        threshold = summary.get("threshold", 0.8)
        st.write(f"**Fairness Threshold:** {threshold}")

        # Timestamp
        if "timestamp" in self.results:
            try:
                timestamp = datetime.fromisoformat(self.results["timestamp"])
                st.write(f"**Analysis Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse timestamp as ISO format: {e}")
                st.write(f"**Analysis Date:** {self.results['timestamp']}")

    def _process_metrics_data(self, max_items: int = 1000) -> pd.DataFrame:
        """Process metrics data with pagination to handle large datasets.

        Args:
            max_items: Maximum number of items to process at once

        Returns:
            DataFrame containing processed metrics data
        """
        metrics_data = []
        processed_count = 0

        attribute_results = self.results.get("attribute_results", {})
        for attr, attr_data in attribute_results.items():
            for outcome, outcome_data in attr_data.get("outcome_metrics", {}).items():
                for metric, value in outcome_data.get("metrics", {}).items():
                    # Check if we've reached the maximum items to process
                    if processed_count >= max_items:
                        st.warning(
                            f"Only showing first {max_items} metrics. Use filters to narrow down results."
                        )
                        break

                    # Parse group information from metric name
                    if "_vs_" in metric:
                        metric_type, groups = metric.split("_", 1)
                        group_comparison = groups.split("_vs_")
                        group1, group2 = group_comparison[0], group_comparison[1]
                    else:
                        metric_type = metric
                        group1, group2 = "N/A", "N/A"

                    # Get statistical significance if available
                    significance = "Unknown"
                    p_value = None
                    if groups in outcome_data.get("statistical_significance", {}):
                        p_value = outcome_data["statistical_significance"][groups]["p_value"]
                        is_significant = outcome_data["statistical_significance"][groups][
                            "is_significant"
                        ]
                        significance = "Significant" if is_significant else "Not Significant"

                    metrics_data.append(
                        {
                            "attribute": attr,
                            "outcome": outcome,
                            "metric_type": metric_type,
                            "group1": group1,
                            "group2": group2,
                            "value": value,
                            "significance": significance,
                            "p_value": p_value,
                        }
                    )

                    processed_count += 1

                if processed_count >= max_items:
                    break

            if processed_count >= max_items:
                break

        # Convert to DataFrame
        if metrics_data:
            return pd.DataFrame(metrics_data)
        else:
            return pd.DataFrame()

    def _display_detailed_metrics(self) -> None:
        """Display detailed fairness metrics with visualizations."""
        st.header("Detailed Fairness Metrics")

        # Process metrics data with pagination
        metrics_df = self._process_metrics_data(max_items=1000)

        if metrics_df.empty:
            st.info("No detailed metrics available.")
            return

        # Convert to DataFrame for easier visualization
        metrics_df = pd.DataFrame(self.results.get("metrics", []))

        # Create tabs for different visualization types
        tab1, tab2, tab3 = st.tabs(["Metric Comparison", "Attribute Analysis", "Raw Data"])

        with tab1:
            # Filter controls
            st.subheader("Metric Comparison")

            metric_types = metrics_df["metric_type"].unique()
            selected_metric = st.selectbox(
                "Select metric type",
                options=metric_types,
                index=0 if "disparate_impact" in metric_types else 0,
            )

            filtered_df = metrics_df[metrics_df["metric_type"] == selected_metric]

            if not filtered_df.empty:
                # Create visualization based on metric type
                if "disparate_impact" in selected_metric:
                    fig = px.bar(
                        filtered_df,
                        x="attribute",
                        y="value",
                        color="outcome",
                        barmode="group",
                        labels={
                            "value": "Disparate Impact Ratio",
                            "attribute": "Protected Attribute",
                        },
                        hover_data=["group1", "group2", "significance", "p_value"],
                    )

                    # Add threshold lines
                    threshold = self.results.get("summary", {}).get("threshold", 0.8)
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(filtered_df["attribute"].unique()) - 0.5,
                        y0=threshold,
                        y1=threshold,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(filtered_df["attribute"].unique()) - 0.5,
                        y0=1.0,
                        y1=1.0,
                        line=dict(color="green", width=2),
                    )
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(filtered_df["attribute"].unique()) - 0.5,
                        y0=1 / threshold,
                        y1=1 / threshold,
                        line=dict(color="red", width=2, dash="dash"),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.write(
                        """
                    **Interpretation:**
                    - Values close to 1.0 (green line) indicate fairness
                    - Values below the lower threshold or above the upper threshold (red lines) indicate potential bias
                    - Statistical significance is shown in the hover data
                    """
                    )

                elif "statistical_parity" in selected_metric:
                    fig = px.bar(
                        filtered_df,
                        x="attribute",
                        y="value",
                        color="outcome",
                        barmode="group",
                        labels={
                            "value": "Statistical Parity Difference",
                            "attribute": "Protected Attribute",
                        },
                        hover_data=["group1", "group2", "significance", "p_value"],
                    )

                    # Add zero line
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(filtered_df["attribute"].unique()) - 0.5,
                        y0=0,
                        y1=0,
                        line=dict(color="green", width=2),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.write(
                        """
                    **Interpretation:**
                    - Values close to 0 (green line) indicate fairness
                    - Larger absolute values indicate potential bias
                    - Statistical significance is shown in the hover data
                    """
                    )
                else:
                    # Generic visualization for other metrics
                    fig = px.bar(
                        filtered_df,
                        x="attribute",
                        y="value",
                        color="outcome",
                        barmode="group",
                        labels={
                            "value": selected_metric,
                            "attribute": "Protected Attribute",
                        },
                        hover_data=["group1", "group2", "significance", "p_value"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No data available for metric: {selected_metric}")

        with tab2:
            st.subheader("Attribute Analysis")

            # Select attribute to analyze
            attributes = metrics_df["attribute"].unique()
            selected_attr = st.selectbox("Select attribute to analyze", options=attributes)

            attr_df = metrics_df[metrics_df["attribute"] == selected_attr]

            if not attr_df.empty:
                # Show attribute distribution if available
                attr_data = self.results.get("attribute_results", {}).get(selected_attr, {})
                if "distribution" in attr_data:
                    st.write("**Attribute Distribution:**")

                    dist_data = attr_data["distribution"]
                    dist_df = pd.DataFrame(
                        {
                            "Group": list(dist_data.keys()),
                            "Proportion": list(dist_data.values()),
                        }
                    )

                    fig = px.pie(
                        dist_df,
                        values="Proportion",
                        names="Group",
                        title=f"Distribution of {selected_attr}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show metrics for this attribute
                st.write("**Fairness Metrics:**")

                # Pivot table for better visualization
                pivot_df = attr_df.pivot_table(
                    index=["outcome", "group1", "group2"],
                    columns="metric_type",
                    values="value",
                    aggfunc="first",
                ).reset_index()

                st.dataframe(pivot_df)
            else:
                st.info(f"No data available for attribute: {selected_attr}")

        with tab3:
            st.subheader("Raw Metrics Data")
            st.dataframe(metrics_df)

    def _display_recommendations(self) -> None:
        """Display recommendations based on fairness analysis."""
        st.header("Recommendations")

        recommendations = self.results.get("summary", {}).get("recommendations", [])

        if not recommendations:
            st.info("No recommendations available.")
            return

        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")

        # Add detailed findings
        st.subheader("Detailed Findings")

        findings = self.results.get("detailed_findings", [])
        if not findings:
            st.info("No detailed findings available.")
            return

        # Convert to DataFrame
        findings_df = pd.DataFrame(findings)

        # Add severity color
        if "severity" in findings_df.columns:

            def color_severity(val):
                if val == "high":
                    return "background-color: #ffcccc"
                elif val == "medium":
                    return "background-color: #ffffcc"
                else:
                    return ""

            st.dataframe(findings_df.style.applymap(color_severity, subset=["severity"]))
        else:
            st.dataframe(findings_df)


def main():
    """Run the fairness dashboard as a standalone application."""
    dashboard = FairnessDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
