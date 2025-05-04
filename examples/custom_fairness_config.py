"""
Custom Fairness Configuration Example

This example demonstrates how to customize the fairness framework
for different organizational requirements and industry-specific needs.

Usage:
    python examples/custom_fairness_config.py

Copyright (c) 2025 Vikas Sahani
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.fairness.bias_detector import BiasDetector
from src.fairness.mitigation import FairnessMitigation
from src.config.fairness_config import FairnessConfig, get_fairness_config

# Create a sample dataset
def create_sample_dataset(size=1000, bias_level=0.2):
    """Create a sample dataset with configurable bias level."""
    np.random.seed(42)  # For reproducibility
    
    # Create demographic attributes
    gender = np.random.choice(['male', 'female'], size=size)
    age_group = np.random.choice(['18-25', '26-35', '36-45', '46+'], size=size)
    
    # Create features with some correlation to demographics
    income = np.zeros(size)
    for i in range(size):
        # Base income with gender bias
        if gender[i] == 'male':
            income[i] = np.random.normal(70000, 20000)
        else:
            income[i] = np.random.normal(60000, 15000)
        
        # Age adjustment
        if age_group[i] == '18-25':
            income[i] *= 0.7
        elif age_group[i] == '26-35':
            income[i] *= 0.9
        elif age_group[i] == '36-45':
            income[i] *= 1.1
        else:  # 46+
            income[i] *= 1.2
    
    # Create outcome with bias
    loan_approved = np.zeros(size, dtype=int)
    for i in range(size):
        # Base approval probability
        p_approve = 0.5
        
        # Add income effect
        p_approve += (income[i] - 50000) / 100000
        
        # Add bias
        if gender[i] == 'female':
            p_approve -= bias_level
        
        # Ensure probability is between 0 and 1
        p_approve = max(0.1, min(0.9, p_approve))
        
        # Determine approval
        loan_approved[i] = np.random.choice([0, 1], p=[1-p_approve, p_approve])
    
    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'age_group': age_group,
        'income': income,
        'loan_approved': loan_approved
    })
    
    return df

def example_financial_services_config():
    """Example configuration for financial services industry."""
    # Create a custom configuration for financial services
    config = {
        "thresholds": {
            "disparate_impact": 0.9,  # Stricter than the standard 80% rule
            "statistical_parity_difference": 0.05,
            "equal_opportunity_difference": 0.05,
            "predictive_parity_difference": 0.05
        },
        "significance": {
            "pvalue_threshold": 0.01,  # More stringent significance level
            "confidence_interval": 0.99
        },
        "reporting": {
            "severity_levels": {
                "high": 0.1,    # Stricter severity thresholds
                "medium": 0.05,
                "low": 0.02
            },
            "include_recommendations": True
        },
        "mitigation": {
            "default_strategy": "reweighing",
            "available_strategies": [
                "reweighing", 
                "disparate_impact_remover", 
                "equalized_odds"
            ]
        }
    }
    
    # Save the configuration to a file
    os.makedirs('examples/configs', exist_ok=True)
    with open('examples/configs/financial_services_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Financial services configuration saved to examples/configs/financial_services_config.json")
    return config

def example_healthcare_config():
    """Example configuration for healthcare industry."""
    # Create a custom configuration for healthcare
    config = {
        "thresholds": {
            "disparate_impact": 0.85,
            "statistical_parity_difference": 0.08,
            "equal_opportunity_difference": 0.05,  # Focus on equal opportunity
            "predictive_parity_difference": 0.08
        },
        "significance": {
            "pvalue_threshold": 0.05,
            "confidence_interval": 0.95
        },
        "reporting": {
            "severity_levels": {
                "high": 0.15,
                "medium": 0.08,
                "low": 0.03
            },
            "include_recommendations": True
        },
        "mitigation": {
            "default_strategy": "calibrated_equalized_odds",  # Focus on balanced error rates
            "available_strategies": [
                "calibrated_equalized_odds",
                "equalized_odds",
                "reweighing"
            ],
            "strategy_params": {
                "reweighing": {
                    "weight_bound": 5.0  # Limit weight adjustments for healthcare data
                },
                "calibrated_equalized_odds": {
                    "cost_constraint": "fnr"  # Prioritize reducing false negatives
                }
            }
        },
        "visualization": {
            "color_palette": "colorblind",  # Accessible colors for healthcare dashboards
            "decimal_places": 4  # Higher precision for healthcare metrics
        }
    }
    
    # Save the configuration to a file
    os.makedirs('examples/configs', exist_ok=True)
    with open('examples/configs/healthcare_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Healthcare configuration saved to examples/configs/healthcare_config.json")
    return config

def run_with_config(config, dataset, name):
    """Run bias detection with a specific configuration."""
    print(f"\n=== Running with {name} Configuration ===")
    
    # Set the configuration
    fairness_config = FairnessConfig.get_instance()
    for key, value in config.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                fairness_config.set(subvalue, key, subkey)
        else:
            fairness_config.set(value, key)
    
    # Initialize components with the configuration
    bias_detector = BiasDetector()
    
    # Run bias detection
    results = bias_detector.detect_outcome_bias(
        dataset,
        attributes=['gender', 'age_group'],
        outcome_columns=['loan_approved']
    )
    
    # Print key results
    print(f"Configuration: {name}")
    print(f"Significance level: {bias_detector.significance_level}")
    print(f"Fairness threshold: {bias_detector.fairness_threshold}")
    
    # Print summary results
    if 'summary' in results and 'bias_detected' in results['summary']:
        print(f"Bias detected: {results['summary']['bias_detected']}")
        
        if 'attribute_summary' in results['summary']:
            for attr, attr_summary in results['summary']['attribute_summary'].items():
                print(f"  {attr}: {attr_summary['bias_detected']}")
                
                if 'outcome_summary' in attr_summary:
                    for outcome, outcome_summary in attr_summary['outcome_summary'].items():
                        print(f"    {outcome}: {outcome_summary['bias_detected']}")
                        
                        if 'metrics_summary' in outcome_summary:
                            for metric, metric_summary in outcome_summary['metrics_summary'].items():
                                print(f"      {metric}: {metric_summary['value']:.4f} (threshold: {metric_summary['threshold']:.4f})")
    
    return results

def main():
    """Run the example."""
    print("CustomerAI Fairness Framework - Configuration Examples")
    print("=====================================================")
    
    # Create a sample dataset
    print("\nCreating sample dataset with bias...")
    dataset = create_sample_dataset(size=1000, bias_level=0.2)
    print(f"Dataset shape: {dataset.shape}")
    print(dataset.head())
    
    # Generate example configurations
    financial_config = example_financial_services_config()
    healthcare_config = example_healthcare_config()
    
    # Default configuration
    print("\n=== Running with Default Configuration ===")
    default_detector = BiasDetector()
    default_results = default_detector.detect_outcome_bias(
        dataset,
        attributes=['gender', 'age_group'],
        outcome_columns=['loan_approved']
    )
    print(f"Default significance level: {default_detector.significance_level}")
    print(f"Default fairness threshold: {default_detector.fairness_threshold}")
    
    # Run with financial services configuration
    financial_results = run_with_config(financial_config, dataset, "Financial Services")
    
    # Run with healthcare configuration
    healthcare_results = run_with_config(healthcare_config, dataset, "Healthcare")
    
    # Compare results
    print("\n=== Comparison of Results ===")
    print("Default configuration detected bias:", default_results['summary'].get('bias_detected', 'N/A'))
    print("Financial services configuration detected bias:", financial_results['summary'].get('bias_detected', 'N/A'))
    print("Healthcare configuration detected bias:", healthcare_results['summary'].get('bias_detected', 'N/A'))
    
    print("\nThis example demonstrates how different organizations can customize")
    print("the fairness framework to match their specific requirements and policies.")
    print("See docs/CUSTOMIZATION_GUIDE.md for more details on customization options.")

if __name__ == "__main__":
    main()
