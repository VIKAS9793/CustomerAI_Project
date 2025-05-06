import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fairness.bias_detector import BiasDetector


class TestFairness(unittest.TestCase):
    """Unit tests for the bias detection and fairness modules"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()

        # Create test dataset with demographic variations
        np.random.seed(42)  # For reproducible tests

        # Create balanced dataset
        self.balanced_data = pd.DataFrame(
            {
                "age_group": ["18-30", "31-45", "46-60", "60+"] * 25,
                "gender": ["male", "female", "non-binary"]
                * 34,  # Include non-binary as third gender option
                "satisfaction_score": [4, 4, 4, 4, 5, 5, 5, 5] * 12 + [3, 3, 3, 3] * 1,
                "resolved": [True] * 90 + [False] * 10,
            }
        )

        # Create biased dataset (biased against older age groups and gender)
        self.biased_data = pd.DataFrame(
            {
                "age_group": ["18-30", "31-45", "46-60", "60+"] * 25,
                "gender": ["male", "female", "non-binary"]
                * 34,  # Include non-binary as third gender option
                "satisfaction_score": [
                    *([5] * 20),  # 18-30 high satisfaction
                    *([4] * 20),  # 31-45 good satisfaction
                    *([3] * 20),  # 46-60 medium satisfaction
                    *([2] * 20),  # 60+ low satisfaction
                    *([5] * 15),  # Males: high satisfaction
                    *([3] * 5),  # Females: medium satisfaction
                    *([2] * 2),  # Non-binary: low satisfaction
                ],
                "resolved": [
                    *([True] * 20),  # 18-30 all resolved
                    *([True] * 18 + [False] * 2),  # 31-45 mostly resolved
                    *([True] * 15 + [False] * 5),  # 46-60 some unresolved
                    *([True] * 10 + [False] * 10),  # 60+ many unresolved
                    *([True] * 15),  # Males: all resolved
                    *([True] * 2 + [False] * 3),  # Females: mostly unresolved
                    *([False] * 2),  # Non-binary: all unresolved
                ],
            }
        )

        # Dataset with no protected attributes
        self.no_attributes_data = pd.DataFrame(
            {
                "satisfaction_score": [4, 5, 3, 2, 5, 4, 3, 5],
                "resolved": [True, True, False, True, True, False, True, True],
            }
        )

    def test_detect_outcome_bias_balanced(self):
        """Test bias detection on balanced dataset"""
        # Run bias detection
        results = self.detector.detect_outcome_bias(
            self.balanced_data,
            attributes=["age_group", "gender"],
            outcome_columns=["satisfaction_score", "resolved"],
        )

        # Generate fairness report
        fairness_report = self.detector.generate_fairness_report(results, threshold=0.8)

        # Balanced data should have high fairness score
        self.assertGreaterEqual(
            fairness_report["summary"]["fairness_score"],
            0.9,  # Allow for minor statistical variations
            "Balanced dataset should have high fairness score",
        )

        # Balanced data should have few or no bias findings
        self.assertLessEqual(
            len(fairness_report.get("detailed_findings", [])),
            1,  # Allow for minor statistical variations
            "Balanced dataset should have minimal bias findings",
        )

    def test_detect_outcome_bias_biased(self):
        """Test bias detection on biased dataset"""
        # Run bias detection
        results = self.detector.detect_outcome_bias(
            self.biased_data,
            attributes=["age_group", "gender"],
            outcome_columns=["satisfaction_score", "resolved"],
        )

        # Generate fairness report
        fairness_report = self.detector.generate_fairness_report(results, threshold=0.8)

        # Biased data should have lower fairness score
        self.assertLess(
            fairness_report["summary"]["fairness_score"],
            0.9,
            "Biased dataset should have reduced fairness score",
        )

        # Biased data should have multiple bias findings
        self.assertGreaterEqual(
            len(fairness_report.get("detailed_findings", [])),
            2,
            "Biased dataset should have multiple bias findings",
        )

        # Check if both biased attributes (age and gender) were flagged
        has_age_bias = any(
            finding["attribute"] == "age_group"
            for finding in fairness_report.get("detailed_findings", [])
        )
        has_gender_bias = any(
            finding["attribute"] == "gender"
            for finding in fairness_report.get("detailed_findings", [])
        )

        self.assertTrue(has_age_bias, "Should detect bias in age_group attribute")
        self.assertTrue(has_gender_bias, "Should detect bias in gender attribute")

    def test_no_attributes(self):
        """Test behavior when no protected attributes are found"""
        # Run bias detection with no valid attributes
        results = self.detector.detect_outcome_bias(
            self.no_attributes_data,
            attributes=["race", "religion"],  # Not in the dataset
            outcome_columns=["satisfaction_score", "resolved"],
        )

        # Should return error message
        self.assertIn("error", results)

    def test_invalid_inputs(self):
        """Test behavior with invalid inputs"""
        # Test with missing outcome columns
        results = self.detector.detect_outcome_bias(
            self.balanced_data, attributes=["age_group"], outcome_columns=[]
        )

        self.assertIn("error", results)

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        results = self.detector.detect_outcome_bias(
            empty_df, attributes=["age_group"], outcome_columns=["satisfaction_score"]
        )

        self.assertIn("error", results)

    def test_disparate_impact_calculation(self):
        """Test the disparate impact calculation function"""
        # Test standard case
        impact = self.detector._calculate_disparate_impact(0.8, 0.4)
        self.assertEqual(impact, 0.5)

        # Test edge case with zero
        impact = self.detector._calculate_disparate_impact(0, 0.5)
        self.assertEqual(impact, float("inf"))

    def test_generate_recommendations(self):
        """Test that recommendations are generated for biased results"""
        # Run bias detection on biased dataset
        results = self.detector.detect_outcome_bias(
            self.biased_data,
            attributes=["age_group", "gender"],
            outcome_columns=["satisfaction_score"],
        )

        # Generate fairness report
        fairness_report = self.detector.generate_fairness_report(results)

        # Should contain recommendations
        self.assertIn("recommendations", fairness_report["summary"])
        self.assertGreater(len(fairness_report["summary"]["recommendations"]), 0)


if __name__ == "__main__":
    unittest.main()
