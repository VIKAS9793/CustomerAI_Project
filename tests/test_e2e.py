# Note: For local testing, use your own credentials in untracked files. Do not commit secret-like patterns.
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.cloud.dummy import DummyClient
from src.cloud.storage import CloudStorageClient
from src.exceptions import ServiceUnavailableError
from src.fairness.bias_detector import BiasDetector
from src.human_review.review_manager import ReviewManager
from src.privacy.anonymizer import DataAnonymizer
from src.response_generator import ResponseGenerator
from src.sentiment_analyzer import SentimentAnalyzer


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the CustomerAI Insights Platform"""

    def setUp(self):
        """Set up test fixtures"""
        # Initialize all components with test configurations
        self.analyzer = SentimentAnalyzer()
        self.generator = ResponseGenerator()
        self.review_manager = ReviewManager()
        self.bias_detector = BiasDetector()
        self.anonymizer = DataAnonymizer()

        # Test data
        self.test_queries = {
            "investment": "What stocks should I invest in for maximum returns?",
            "loan": "Can I get a personal loan with a credit score of 650?",
            "account": "How do I check my account balance online?",
            "complaint": "I've been trying to contact support for days with no response!",
        }

    @patch("openai.ChatCompletion.create")
    def test_investment_advice_workflow(self, mock_openai):
        """Test complete workflow for investment advice queries"""
        # Mock the OpenAI response for classification
        mock_classification = MagicMock()
        mock_classification.choices = [MagicMock()]
        mock_classification.choices[0].message.content = "Investment_Query"

        # Mock the OpenAI response for generation
        mock_generation = MagicMock()
        mock_generation.choices = [MagicMock()]
        mock_generation.choices[
            0
        ].message.content = """
        Based on your interest in stocks for maximum returns, I can provide some general guidance.

        Stocks with high growth potential often come with higher risk. It's important to consider your risk tolerance and investment horizon. Diversification across sectors is generally recommended.

        Please note that this is not financial advice. Past performance is not indicative of future results, and all investments carry risk. I recommend consulting with a qualified financial advisor before making investment decisions.
        """

        # Configure mock to return different responses based on input
        mock_openai.side_effect = [mock_classification, mock_generation]

        # Execute the workflow
        query = self.test_queries["investment"]

        # Step 1: Generate response
        response_data = self.generator.generate_response(query)

        # Verify response is generated and categorized correctly
        self.assertIn("response", response_data)
        self.assertIn("category", response_data)

        # Step 2: Validate compliance
        validation = self.validator.validate_response(query, response_data["response"])

        # Verify disclaimer requirements
        self.assertTrue(
            validation["category_results"]["investment_advice"]["passed"],
            f"Investment advice should contain proper disclaimers. Issues: {validation.get('issues_found', [])}",
        )

        # Step 3: Human review (high-risk category)
        self.human_review.add_to_queue(
            item_id="test_investment",
            query=query,
            response=response_data["response"],
            category=response_data["category"],
            priority=2,  # High priority for investment advice
        )

        # Verify item was added to review queue
        next_item = self.human_review.get_next_item()
        self.assertEqual(next_item["item_id"], "test_investment")
        self.assertEqual(next_item["priority"], 2)

        # Step 4: Record review decision
        review_result = self.human_review.record_review(
            item_id="test_investment",
            approved=True,
            feedback="Response includes proper disclaimers and balanced advice",
        )

        # Verify review was recorded
        self.assertIn("success", review_result)
        self.assertTrue(review_result["success"])

    @patch("openai.ChatCompletion.create")
    def test_privacy_and_compliance_workflow(self, mock_openai):
        """Test privacy and compliance aspects of the workflow"""
        # Test data with PII
        query_with_pii = "My name is John Doe and my account number is 1234567890. Can you help me with my credit card ending in 5678?"

        # Step 1: Anonymize the query
        anonymized_result = self.anonymizer.anonymize_text(query_with_pii, keep_mapping=True)

        # Verify PII was detected and anonymized
        self.assertNotEqual(anonymized_result["anonymized_text"], query_with_pii)
        self.assertGreater(len(anonymized_result["mapping"]), 0)

        # Mock response generation (simplified)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Investment_Query"
        mock_openai.return_value = mock_response

        # Step 2: Generate response with anonymized query
        response_data = self.generator.generate_response(anonymized_result["anonymized_text"])

        # Step 3: Check response doesn't contain original PII
        for original_pii in anonymized_result["mapping"].values():
            self.assertNotIn(
                original_pii,
                response_data["response"],
                f"Response should not contain original PII: {original_pii}",
            )

    def test_bias_detection_workflow(self):
        """Test bias detection and fairness analysis workflow"""
        # Create test dataset with demographic variations and biased outcomes
        test_data = pd.DataFrame(
            {
                "age_group": ["18-30", "31-45", "46-60", "60+"] * 25,
                "gender": ["male", "female", "non-binary"]
                * 34,  # Include non-binary as third gender option
                "income_level": ["low", "medium", "high"] * 33 + ["low"],
                "satisfaction_score": [5, 5, 5, 5, 5, 5, 4, 4] * 12
                + [2, 2, 2, 2] * 1,  # Biased against 60+ age group
                "resolved": [True] * 90 + [False] * 10,  # 90% resolution rate
            }
        )

        # Run bias detection
        bias_results = self.bias_detector.detect_outcome_bias(
            test_data,
            attributes=["age_group", "gender", "income_level"],
            outcome_columns=["satisfaction_score", "resolved"],
        )

        # Generate fairness report
        fairness_report = self.bias_detector.generate_fairness_report(bias_results)

        # Verify bias detection works
        self.assertLess(
            fairness_report["summary"]["fairness_score"],
            1.0,
            "Should detect intentional bias in test data",
        )

        # Verify correct attribute identified as problematic
        has_age_bias_finding = any(
            finding["attribute"] == "age_group" for finding in fairness_report["detailed_findings"]
        )
        self.assertTrue(has_age_bias_finding, "Should detect bias in age_group attribute")

    def test_cloud_service_integration(self):
        """Test cloud service integrations"""
        # Test storage operations
        storage_client = CloudStorageClient()

        # Test upload
        # upload_result = storage_client.upload("test.json", test_data)
        # self.assertTrue(upload_result["success"])

        # Test download
        # download_result = storage_client.download("test.json")
        # self.assertEqual(download_result["data"], test_data)

        # Test delete
        delete_result = storage_client.delete("test.json")
        self.assertTrue(delete_result["success"])

    def test_error_handling(self):
        """Test error handling scenarios"""
        # Test invalid API key
        with self.assertRaises(Exception):
            DummyClient(dummy_value="invalid")
        # Test rate limiting
        with self.assertRaises(Exception):
            for _ in range(100):  # Exceed rate limit
                self.generator.generate_response("test")

        # Test service unavailability
        with patch("requests.post") as mock_post:
            mock_post.side_effect = ConnectionError()
            with self.assertRaises(Exception):
                pass
            with self.assertRaises(ServiceUnavailableError):
                self.analyzer.analyze_sentiment("test")

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import asyncio

        async def make_request(query):
            return await self.generator.generate_response_async(query)

        # Create multiple concurrent requests
        queries = [f"test query {i}" for i in range(10)]
        tasks = [make_request(q) for q in queries]

        # Run concurrently and verify results
        results = asyncio.run(asyncio.gather(*tasks))
        self.assertEqual(len(results), len(queries))
        for result in results:
            self.assertIn("response", result)
            self.assertIn("category", result)


if __name__ == "__main__":
    unittest.main()
