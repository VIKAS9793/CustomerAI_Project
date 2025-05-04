import unittest
import sys
import os
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment_analyzer import SentimentAnalyzer
from src.response_generator import ResponseGenerator
from validation.domain_validator import FinancialDomainValidator
from privacy.anonymizer import DataAnonymizer
from fairness.bias_detector import BiasDetector
from validation.human_in_loop import HumanReviewSystem

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the CustomerAI Insights Platform"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize all components with test configurations
        self.analyzer = SentimentAnalyzer()
        self.generator = ResponseGenerator()
        self.validator = FinancialDomainValidator()
        self.anonymizer = DataAnonymizer()
        self.bias_detector = BiasDetector()
        self.human_review = HumanReviewSystem(log_file="tests/test_review_logs.jsonl")
        
        # Test data
        self.test_queries = {
            'investment': "What stocks should I invest in for maximum returns?",
            'loan': "Can I get a personal loan with a credit score of 650?",
            'account': "How do I check my account balance online?",
            'complaint': "I've been trying to contact support for days with no response!"
        }
    
    @patch('openai.ChatCompletion.create')
    def test_investment_advice_workflow(self, mock_openai):
        """Test complete workflow for investment advice queries"""
        # Mock the OpenAI response for classification
        mock_classification = MagicMock()
        mock_classification.choices = [MagicMock()]
        mock_classification.choices[0].message.content = "Investment_Query"
        
        # Mock the OpenAI response for generation
        mock_generation = MagicMock()
        mock_generation.choices = [MagicMock()]
        mock_generation.choices[0].message.content = """
        Based on your interest in stocks for maximum returns, I can provide some general guidance.
        
        Stocks with high growth potential often come with higher risk. It's important to consider your risk tolerance and investment horizon. Diversification across sectors is generally recommended.
        
        Please note that this is not financial advice. Past performance is not indicative of future results, and all investments carry risk. I recommend consulting with a qualified financial advisor before making investment decisions.
        """
        
        # Configure mock to return different responses based on input
        mock_openai.side_effect = [mock_classification, mock_generation]
        
        # Execute the workflow
        query = self.test_queries['investment']
        
        # Step 1: Generate response
        response_data = self.generator.generate_response(query)
        
        # Verify response is generated and categorized correctly
        self.assertIn('response', response_data)
        self.assertIn('category', response_data)
        
        # Step 2: Validate compliance
        validation = self.validator.validate_response(query, response_data['response'])
        
        # Verify disclaimer requirements
        self.assertTrue(
            validation['category_results']['investment_advice']['passed'],
            f"Investment advice should contain proper disclaimers. Issues: {validation.get('issues_found', [])}"
        )
        
        # Step 3: Human review (high-risk category)
        self.human_review.add_to_queue(
            item_id="test_investment",
            query=query,
            response=response_data['response'],
            category=response_data['category'],
            priority=2  # High priority for investment advice
        )
        
        # Verify item was added to review queue
        next_item = self.human_review.get_next_item()
        self.assertEqual(next_item['item_id'], "test_investment")
        self.assertEqual(next_item['priority'], 2)
        
        # Step 4: Record review decision
        review_result = self.human_review.record_review(
            item_id="test_investment",
            approved=True,
            feedback="Response includes proper disclaimers and balanced advice"
        )
        
        # Verify review was recorded
        self.assertIn('success', review_result)
        self.assertTrue(review_result['success'])
    
    @patch('openai.ChatCompletion.create')
    def test_privacy_and_compliance_workflow(self, mock_openai):
        """Test privacy and compliance aspects of the workflow"""
        # Test data with PII
        query_with_pii = "My name is John Doe and my account number is 1234567890. Can you help me with my credit card ending in 5678?"
        
        # Step 1: Anonymize the query
        anonymized_result = self.anonymizer.anonymize_text(query_with_pii, keep_mapping=True)
        
        # Verify PII was detected and anonymized
        self.assertNotEqual(anonymized_result['anonymized_text'], query_with_pii)
        self.assertGreater(len(anonymized_result['mapping']), 0)
        
        # Mock response generation (simplified)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Investment_Query"
        mock_openai.return_value = mock_response
        
        # Step 2: Generate response with anonymized query
        response_data = self.generator.generate_response(anonymized_result['anonymized_text'])
        
        # Step 3: Check response doesn't contain original PII
        for original_pii in anonymized_result['mapping'].values():
            self.assertNotIn(original_pii, response_data['response'], 
                            f"Response should not contain original PII: {original_pii}")
    
    def test_bias_detection_workflow(self):
        """Test bias detection and fairness analysis workflow"""
        # Create test dataset with demographic variations and biased outcomes
        test_data = pd.DataFrame({
            'age_group': ['18-30', '31-45', '46-60', '60+'] * 25,
            'gender': ['male', 'female', 'non-binary'] * 34,  # Include non-binary as third gender option
            'income_level': ['low', 'medium', 'high'] * 33 + ['low'],
            'satisfaction_score': [5, 5, 5, 5, 5, 5, 4, 4] * 12 + [2, 2, 2, 2] * 1,  # Biased against 60+ age group
            'resolved': [True] * 90 + [False] * 10  # 90% resolution rate
        })
        
        # Run bias detection
        bias_results = self.bias_detector.detect_outcome_bias(
            test_data,
            attributes=['age_group', 'gender', 'income_level'],
            outcome_columns=['satisfaction_score', 'resolved']
        )
        
        # Generate fairness report
        fairness_report = self.bias_detector.generate_fairness_report(bias_results)
        
        # Verify bias detection works
        self.assertLess(
            fairness_report['summary']['fairness_score'],
            1.0,
            "Should detect intentional bias in test data"
        )
        
        # Verify correct attribute identified as problematic
        has_age_bias_finding = any(
            finding['attribute'] == 'age_group' 
            for finding in fairness_report['detailed_findings']
        )
        self.assertTrue(has_age_bias_finding, "Should detect bias in age_group attribute")

if __name__ == '__main__':
    unittest.main() 