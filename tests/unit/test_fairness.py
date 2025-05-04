"""
Unit tests for the fairness framework components.

This module contains comprehensive tests for the BiasDetector, FairnessMitigation,
and FairnessDashboard classes to ensure they function correctly and handle
edge cases appropriately.

Copyright (c) 2025 Vikas Sahani
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime

from src.fairness.bias_detector import BiasDetector
from src.fairness.mitigation import FairnessMitigation
from src.utils.date_provider import DateProvider


class TestBiasDetector(unittest.TestCase):
    """Test cases for the BiasDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample dataset for testing
        self.data = pd.DataFrame({
            'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
            'age_group': ['18-25', '18-25', '26-35', '26-35', '36-45', '36-45', '46+', '46+'],
            'loan_approved': [1, 0, 1, 1, 1, 0, 1, 0],
            'loan_amount': [10000, 8000, 15000, 12000, 20000, 18000, 25000, 22000],
            'risk_score': [0.7, 0.8, 0.6, 0.5, 0.4, 0.9, 0.3, 0.6]
        })
        
        # Mock the DateProvider
        self.date_mock = datetime(2025, 5, 4, 12, 0, 0)
        DateProvider.set_mock_date(self.date_mock)
        
        # Initialize the BiasDetector
        self.bias_detector = BiasDetector()
    
    def tearDown(self):
        """Clean up after tests."""
        DateProvider.set_mock_date(None)
    
    def test_detect_outcome_bias_basic(self):
        """Test basic bias detection functionality."""
        results = self.bias_detector.detect_outcome_bias(
            self.data,
            attributes=['gender'],
            outcome_columns=['loan_approved']
        )
        
        # Verify the results structure
        self.assertIn('timestamp', results)
        self.assertIn('dataset_size', results)
        self.assertIn('attributes_analyzed', results)
        self.assertIn('outcomes_analyzed', results)
        self.assertIn('attribute_results', results)
        
        # Verify gender attribute results
        self.assertIn('gender', results['attribute_results'])
        gender_results = results['attribute_results']['gender']
        self.assertIn('values', gender_results)
        self.assertIn('distribution', gender_results)
        self.assertIn('outcome_metrics', gender_results)
        
        # Verify loan_approved outcome metrics
        self.assertIn('loan_approved', gender_results['outcome_metrics'])
        loan_metrics = gender_results['outcome_metrics']['loan_approved']
        self.assertIn('metrics', loan_metrics)
        self.assertIn('statistical_significance', loan_metrics)
    
    def test_empty_dataset(self):
        """Test behavior with an empty dataset."""
        empty_data = pd.DataFrame()
        results = self.bias_detector.detect_outcome_bias(
            empty_data,
            attributes=['gender'],
            outcome_columns=['loan_approved']
        )
        
        self.assertIn('error', results)
        self.assertEqual(results['error'], "Empty dataset provided")
    
    def test_missing_attributes(self):
        """Test behavior with non-existent attributes."""
        results = self.bias_detector.detect_outcome_bias(
            self.data,
            attributes=['nonexistent_attribute'],
            outcome_columns=['loan_approved']
        )
        
        self.assertIn('error', results)
        self.assertTrue("None of the provided attributes" in results['error'])
    
    def test_multiple_attributes_and_outcomes(self):
        """Test with multiple attributes and outcomes."""
        results = self.bias_detector.detect_outcome_bias(
            self.data,
            attributes=['gender', 'age_group'],
            outcome_columns=['loan_approved', 'risk_score']
        )
        
        # Verify both attributes are analyzed
        self.assertIn('gender', results['attribute_results'])
        self.assertIn('age_group', results['attribute_results'])
        
        # Verify both outcomes are analyzed for gender
        gender_metrics = results['attribute_results']['gender']['outcome_metrics']
        self.assertIn('loan_approved', gender_metrics)
        self.assertIn('risk_score', gender_metrics)


class TestFairnessMitigation(unittest.TestCase):
    """Test cases for the FairnessMitigation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample dataset for testing
        self.data = pd.DataFrame({
            'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
            'age_group': ['18-25', '18-25', '26-35', '26-35', '36-45', '36-45', '46+', '46+'],
            'education': ['high_school', 'college', 'graduate', 'high_school', 'college', 'graduate', 'high_school', 'college'],
            'loan_approved': [1, 0, 1, 1, 1, 0, 1, 0],
            'risk_score': [0.7, 0.8, 0.6, 0.5, 0.4, 0.9, 0.3, 0.6]
        })
        
        # Initialize the FairnessMitigation
        self.mitigation = FairnessMitigation()
    
    def test_reweighing(self):
        """Test the reweighing mitigation strategy."""
        result = self.mitigation.reweigh(
            self.data,
            protected_attribute='gender',
            outcome_column='loan_approved'
        )
        
        # Verify the result structure
        self.assertIn('mitigated_data', result)
        self.assertIn('weights', result)
        self.assertIn('mitigation_info', result)
        
        # Verify the mitigated data has the same number of rows
        self.assertEqual(len(result['mitigated_data']), len(self.data))
        
        # Verify weights are assigned to each row
        self.assertEqual(len(result['weights']), len(self.data))
        
        # Verify all weights are positive
        self.assertTrue(all(w > 0 for w in result['weights']))
    
    def test_balanced_sampling(self):
        """Test the balanced sampling mitigation strategy."""
        result = self.mitigation.balanced_sampling(
            self.data,
            protected_attribute='gender',
            outcome_column='loan_approved'
        )
        
        # Verify the result structure
        self.assertIn('mitigated_data', result)
        self.assertIn('sampling_info', result)
        
        # Verify the mitigated data has rows
        self.assertGreater(len(result['mitigated_data']), 0)
        
        # Verify the sampling info contains original and new counts
        self.assertIn('original_counts', result['sampling_info'])
        self.assertIn('new_counts', result['sampling_info'])
    
    def test_post_processing(self):
        """Test post-processing mitigation strategies."""
        # Create predictions and true values
        predictions = np.array([1, 0, 1, 1, 1, 0, 1, 0])
        true_values = np.array([1, 0, 1, 0, 1, 1, 1, 0])
        
        result = self.mitigation.equalized_odds(
            self.data,
            predictions=predictions,
            true_values=true_values,
            protected_attribute='gender',
            threshold=0.5
        )
        
        # Verify the result structure
        self.assertIn('adjusted_predictions', result)
        self.assertIn('adjustment_info', result)
        
        # Verify adjusted predictions have the same length
        self.assertEqual(len(result['adjusted_predictions']), len(predictions))


if __name__ == '__main__':
    unittest.main()
