import os
import re
import json
import logging
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis for financial customer service interactions.
    Provides detailed sentiment insights specifically calibrated for financial queries and responses.
    """
    
    def __init__(self, 
                 model: str = "gpt-4-turbo", 
                 threshold_positive: float = 0.6,
                 threshold_negative: float = 0.4,
                 use_cached: bool = True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model: The OpenAI model to use
            threshold_positive: Score threshold for positive sentiment
            threshold_negative: Score threshold for negative sentiment
            use_cached: Whether to use cached results
        """
        self.model = model
        self.threshold_positive = threshold_positive
        self.threshold_negative = threshold_negative
        self.use_cached = use_cached
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Load finance-specific sentiment keywords
        self._load_keywords()
    
    def _load_keywords(self) -> None:
        """Load finance-specific sentiment keywords."""
        try:
            keywords_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "data", 
                "finance_keywords.json"
            )
            
            if os.path.exists(keywords_path):
                with open(keywords_path, 'r') as f:
                    self.keywords = json.load(f)
            else:
                logger.warning(f"Keywords file not found at {keywords_path}")
                self.keywords = {
                    "positive": ["gain", "profit", "growth", "approved", "increase", "benefit"],
                    "negative": ["decline", "loss", "debt", "charge", "penalty", "risk"],
                    "neutral": ["account", "statement", "transfer", "balance", "transaction"]
                }
        except Exception as e:
            logger.error(f"Error loading keywords: {str(e)}")
            self.keywords = {
                "positive": ["gain", "profit", "growth", "approved", "increase", "benefit"],
                "negative": ["decline", "loss", "debt", "charge", "penalty", "risk"],
                "neutral": ["account", "statement", "transfer", "balance", "transaction"]
            }
    
    def _calculate_keyword_score(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment score based on finance-specific keywords.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text = text.lower()
        scores = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        # Count keyword occurrences
        for sentiment, keywords in self.keywords.items():
            count = 0
            for keyword in keywords:
                # Use word boundary to match whole words
                pattern = r'\b' + re.escape(keyword) + r'\b'
                count += len(re.findall(pattern, text))
            
            # Normalize by total keywords
            if count > 0:
                scores[sentiment] = count / len(text.split())
        
        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            for sentiment in scores:
                scores[sentiment] /= total
        
        return scores
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        import hashlib
        # Use a hash of the text as the cache key
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result if available."""
        if not self.use_cached:
            return None
            
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
        
        return None
    
    def _save_to_cache(self, text: str, result: Dict[str, Any]) -> None:
        """Save analysis result to cache."""
        if not self.use_cached:
            return
            
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
    
    async def analyze_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using LLM.
        
        Args:
            text: The text to analyze
            
        Returns:
            Detailed sentiment analysis
        """
        try:
            from cloud.ai.llm_manager import get_llm_manager
            
            prompt = f"""
            You are a financial customer service sentiment analyzer. Analyze the following customer text and provide:
            1. Overall sentiment (positive, negative, or neutral)
            2. Sentiment scores (positive, negative, neutral) that sum to 1
            3. Satisfaction score (1-10)
            4. Key positive aspects mentioned (if any)
            5. Key areas of concern or improvement (if any)
            
            Return your analysis as a JSON object with the following structure:
            {{
              "sentiment": "positive|negative|neutral",
              "scores": {{
                "positive": float,
                "negative": float,
                "neutral": float
              }},
              "satisfaction_score": int,
              "key_positives": ["aspect1", "aspect2"],
              "key_negatives": ["concern1", "concern2"]
            }}
            
            Customer text: {text}
            """
            
            # Get the LLM manager and generate a response using the configured LLMs
            llm_manager = get_llm_manager()
            
            # Try to use a financial-specific client for sentiment analysis if available
            try:
                response = await llm_manager.generate_text(
                    prompt=prompt,
                    client_id="financial_sentiment",  # Try to use a dedicated financial sentiment client if configured
                    system_prompt="You are a specialized financial sentiment analysis system that focuses on customer feedback in financial services.",
                    temperature=0.1,
                    max_tokens=800
                )
            except ValueError:
                # Fall back to the default client if the specific one isn't available
                response = await llm_manager.generate_text(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=800
                )
            
            # Parse the JSON response
            import re
            import json
            
            content = response["text"]
            
            # Find and extract the JSON part
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            else:
                # Try to extract JSON without markdown formatting
                json_match = re.search(r'{.*}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            
            result = json.loads(content)
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            # Return a basic structure in case of error
            return {
                "sentiment": "neutral",
                "scores": {
                    "positive": 0.33,
                    "negative": 0.33,
                    "neutral": 0.34
                },
                "satisfaction_score": 5,
                "key_positives": [],
                "key_negatives": []
            }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        # Check cache first
        cached = self._get_cached_result(text)
        if cached:
            return cached
            
        # Calculate keyword-based score
        keyword_scores = self._calculate_keyword_score(text)
        
        # Determine sentiment
        sentiment = "neutral"
        if keyword_scores["positive"] > self.threshold_positive:
            sentiment = "positive"
        elif keyword_scores["negative"] > self.threshold_negative:
            sentiment = "negative"
        
        # Extract key aspects based on proximity to sentiment keywords
        words = text.lower().split()
        key_positives = []
        key_negatives = []
        
        # Basic extraction of key phrases
        for i, word in enumerate(words):
            if word in self.keywords["positive"]:
                # Get context (up to 5 words before and after)
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                phrase = " ".join(words[start:end])
                if phrase not in key_positives:
                    key_positives.append(phrase)
            
            elif word in self.keywords["negative"]:
                # Get context (up to 5 words before and after)
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                phrase = " ".join(words[start:end])
                if phrase not in key_negatives:
                    key_negatives.append(phrase)
        
        # Calculate satisfaction score (1-10)
        satisfaction_score = int(1 + keyword_scores["positive"] * 9)
        
        # Build result
        result = {
            "sentiment": sentiment,
            "positive": keyword_scores["positive"],
            "negative": keyword_scores["negative"],
            "neutral": keyword_scores["neutral"],
            "analysis": {
                "satisfaction_score": satisfaction_score,
                "key_positives": key_positives[:3],  # Limit to top 3
                "key_negatives": key_negatives[:3]   # Limit to top 3
            }
        }
        
        # Cache the result
        self._save_to_cache(text, result)
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of multiple sentiment analyses.
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {
                "count": 0,
                "sentiment_distribution": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                },
                "average_satisfaction": 0
            }
        
        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        satisfaction_scores = []
        
        for result in results:
            sentiment_counts[result["sentiment"]] += 1
            satisfaction_scores.append(result["analysis"]["satisfaction_score"])
        
        # Calculate distributions
        total = len(results)
        sentiment_distribution = {
            "positive": sentiment_counts["positive"] / total,
            "negative": sentiment_counts["negative"] / total,
            "neutral": sentiment_counts["neutral"] / total
        }
        
        # Calculate average satisfaction
        avg_satisfaction = sum(satisfaction_scores) / total if satisfaction_scores else 0
        
        return {
            "count": total,
            "sentiment_distribution": sentiment_distribution,
            "average_satisfaction": avg_satisfaction
        } 