import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Unit tests for the SentimentAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()

    def test_basic_sentiment_positive(self):
        """Test basic sentiment analysis for positive texts"""
        positive_texts = [
            "I'm very happy with your service",
            "Thank you for resolving my issue so quickly",
            "The new app feature is excellent, I love it",
            "My investment has grown beyond expectations",
        ]

        for text in positive_texts:
            result = self.analyzer.analyze_basic(text)
            self.assertEqual(result["sentiment"], "positive", f"Failed on: {text}")
            self.assertGreater(result["positive"], result["negative"], f"Failed on: {text}")

    def test_basic_sentiment_negative(self):
        """Test basic sentiment analysis for negative texts"""
        negative_texts = [
            "This is terrible service",
            "I've been waiting for days with no response",
            "My account was incorrectly charged a fee",
            "I lost money on the investment you recommended",
        ]

        for text in negative_texts:
            result = self.analyzer.analyze_basic(text)
            self.assertEqual(result["sentiment"], "negative", f"Failed on: {text}")
            self.assertGreater(result["negative"], result["positive"], f"Failed on: {text}")

    def test_basic_sentiment_neutral(self):
        """Test basic sentiment analysis for neutral texts"""
        neutral_texts = [
            "I'd like to check my account balance",
            "What is the current interest rate?",
            "When does your branch open tomorrow?",
            "Please send me the account statement",
        ]

        for text in neutral_texts:
            result = self.analyzer.analyze_basic(text)
            self.assertEqual(result["sentiment"], "neutral", f"Failed on: {text}")

    def test_financial_context(self):
        """Test that financial terms are properly interpreted"""
        financial_texts = [
            ("My SIP investments are not performing well", "negative"),
            ("The interest rate on my FD is excellent", "positive"),
            ("My loan application was rejected", "negative"),
            ("The new zero-fee debit card is great", "positive"),
        ]

        for text, expected in financial_texts:
            result = self.analyzer.analyze_basic(text)
            self.assertEqual(result["sentiment"], expected, f"Failed on: {text}")

    @patch("openai.ChatCompletion.create")
    def test_analyze_with_ai(self, mock_openai):
        """Test AI-powered sentiment analysis with mocked OpenAI API"""
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """
        Primary sentiment: negative
        Satisfaction score: 3/10
        Urgency level: high
        Key concerns: delay in service, lack of communication
        Improvement suggestions: provide regular updates, set clear expectations
        """
        mock_openai.return_value = mock_response

        result = self.analyzer.analyze_with_ai("I've been waiting for my loan approval for weeks")

        # Verify OpenAI was called with correct parameters
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args[1]
        self.assertEqual(call_args["temperature"], 0.3)

        # Verify result contains expected structure
        self.assertIn("analysis", result)
        self.assertIn("raw_response", result)

    def test_batch_analyze(self):
        """Test batch analysis of conversations"""
        test_df = pd.DataFrame(
            {
                "conversation_id": [1, 2, 3],
                "text": [
                    "I'm very happy with your service",
                    "This is terrible service",
                    "I'd like to check my account balance",
                ],
            }
        )

        results_df = self.analyzer.batch_analyze(test_df, "text")

        # Verify results were added to dataframe
        self.assertIn("sentiment_analysis", results_df.columns)
        self.assertEqual(len(results_df), 3)

        # Check sentiment matches individual analysis
        sentiments = [item["sentiment"] for item in results_df["sentiment_analysis"]]
        self.assertEqual(sentiments, ["positive", "negative", "neutral"])


if __name__ == "__main__":
    unittest.main()
