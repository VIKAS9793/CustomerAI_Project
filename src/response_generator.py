import json
import logging
import os
from typing import Any, Dict, Optional

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    AI-powered response generator for financial customer service queries.
    Generates compliant responses tailored to financial regulations and best practices.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.2,
        use_cached: bool = True,
    ):
        """
        Initialize the response generator.

        Args:
            model: The OpenAI model to use
            temperature: Creativity level (lower is more deterministic)
            use_cached: Whether to use cached results
        """
        self.model = model
        self.temperature = temperature
        self.use_cached = use_cached
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "cache", "responses"
        )

        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Set OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Load response templates
        self._load_templates()

        # Load disclaimers
        self._load_disclaimers()

    def _load_templates(self) -> None:
        """Load response templates for different query categories."""
        try:
            templates_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "response_templates.json",
            )

            if os.path.exists(templates_path):
                with open(templates_path, "r") as f:
                    self.templates = json.load(f)
            else:
                logger.warning(f"Templates file not found at {templates_path}")
                self.templates = {
                    "investment_advice": [
                        "Based on general financial principles, {advice}. However, please note that this is not personalized investment advice. Past performance is not indicative of future results.",
                        "While I can provide general information, {advice}. Remember that investments carry risks. It's advisable to consult with a qualified financial advisor.",
                    ],
                    "account_inquiry": [
                        "Regarding your account inquiry, {response}. Is there anything else I can help with?",
                        "For your account question, {response}. Please let me know if you need any clarification.",
                    ],
                    "loan_inquiry": [
                        "Regarding your loan inquiry, {response}. Our team is happy to provide more details if needed.",
                        "About your loan question, {response}. Feel free to ask if you have further questions.",
                    ],
                    "complaint": [
                        "I'm sorry to hear about your experience. {response}. We appreciate your feedback and will work to improve our service.",
                        "We apologize for the inconvenience. {response}. Your satisfaction is important to us, and we're committed to resolving this issue.",
                    ],
                    "general": [
                        "{response}. Is there anything else you would like to know?",
                        "{response}. Please let me know if you have additional questions.",
                    ],
                }
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            self.templates = {"general": ["{response}"]}

    def _load_disclaimers(self) -> None:
        """Load regulatory disclaimers and compliance text."""
        try:
            disclaimers_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "disclaimers.json"
            )

            if os.path.exists(disclaimers_path):
                with open(disclaimers_path, "r") as f:
                    self.disclaimers = json.load(f)
            else:
                logger.warning(f"Disclaimers file not found at {disclaimers_path}")
                self.disclaimers = {
                    "investment_advice": "This information is provided for educational purposes only and is not intended as investment advice. Past performance is not indicative of future results. All investments involve risk, including loss of principal.",
                    "loan_information": "Loan approval is subject to credit approval and program guidelines. Interest rates and program terms are subject to change without notice. Not all applicants will qualify.",
                    "tax_advice": "This information is not intended as tax advice. Consult a qualified tax professional regarding your individual circumstances.",
                    "general": "The information provided is general in nature and may not apply to your specific situation. Please consult with appropriate professionals for advice tailored to your needs.",
                }
        except Exception as e:
            logger.error(f"Error loading disclaimers: {str(e)}")
            self.disclaimers = {
                "general": "This information is general in nature. Please consult professionals for specific advice."
            }

    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        import hashlib

        # Use a hash of the text as the cache key
        # Not used for security, just for caching
        return hashlib.sha256(query.encode()).hexdigest()

    def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        if not self.use_cached:
            return None

        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")

        return None

    def _save_to_cache(self, query: str, result: Dict[str, Any]) -> None:
        """Save response to cache."""
        if not self.use_cached:
            return

        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")

    def _classify_query(self, query: str) -> str:
        """
        Classify the customer query into categories.

        Args:
            query: The customer query

        Returns:
            Category string
        """
        try:
            prompt = f"""
            Classify the following financial customer service query into exactly one of these categories:
            - investment_advice: Questions about investments, stocks, funds, returns, etc.
            - account_inquiry: Questions about account status, balance, features, etc.
            - loan_inquiry: Questions about loans, mortgages, interest rates, etc.
            - complaint: Expressions of dissatisfaction or reports of issues
            - general: Any other general inquiries

            Return ONLY the category name, nothing else.

            Query: {query}
            """

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=20,
            )

            category = response.choices[0].message.content.strip().lower()

            # Normalize and validate the category
            valid_categories = [
                "investment_advice",
                "account_inquiry",
                "loan_inquiry",
                "complaint",
                "general",
            ]

            # Extract just the category if it's embedded in a sentence
            for valid_cat in valid_categories:
                if valid_cat in category:
                    return valid_cat

            # Default to general if category is not recognized
            return "general"

        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            return "general"

    def _generate_response_text(
        self, query: str, category: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response text using AI.

        Args:
            query: The customer query
            category: The query category
            context: Additional context information

        Returns:
            Generated response information
        """
        try:
            # Prepare context information
            context_str = ""
            if context:
                context_str = "Consider this additional context: "
                for key, value in context.items():
                    context_str += f"{key}: {value}, "

            # Get appropriate disclaimer
            disclaimer = self.disclaimers.get(category, self.disclaimers["general"])

            prompt = f"""
            As a financial customer service AI assistant, generate a helpful, accurate, and compliant response to the following query.

            Query category: {category}

            {context_str}

            For {category} queries, make sure to:
            """

            # Add category-specific instructions
            if category == "investment_advice":
                prompt += """
                - Never provide specific investment recommendations
                - Always include appropriate disclaimers
                - Emphasize consulting with a qualified financial advisor
                - Explain general principles rather than specific actions
                - Never make promises about returns or performance
                """
            elif category == "loan_inquiry":
                prompt += """
                - Provide general information about loan products
                - Do not pre-approve or guarantee acceptance
                - Explain that terms depend on individual circumstances and credit
                - Include appropriate disclaimers about loan approval
                - Be transparent about potential fees and requirements
                """
            elif category == "complaint":
                prompt += """
                - Show empathy and acknowledge the issue
                - Avoid defensive language
                - Provide clear next steps for resolution
                - Don't make promises that can't be kept
                - Offer to escalate appropriately
                """

            prompt += f"""
            Your response should be professional, concise, and helpful. Ensure it's compliant with financial regulations.

            IMPORTANT: Include this disclaimer for {category} queries: "{disclaimer}"

            Customer query: {query}

            Response:
            """

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=self.temperature,
                max_tokens=800,
            )

            response_text = response.choices[0].message.content.strip()

            # Generate alternative responses for high-risk categories
            alternative_responses = []
            if category in ["investment_advice", "loan_inquiry"]:
                # Generate one alternative with slightly different wording
                alt_response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt
                            + "\nProvide an alternative phrasing with the same meaning but different wording.",
                        }
                    ],
                    temperature=self.temperature + 0.3,  # Slightly more creative
                    max_tokens=800,
                )

                alternative_responses.append(alt_response.choices[0].message.content.strip())

            # Check if human review is likely needed
            requires_human_review = False
            if category == "investment_advice" or category == "loan_inquiry":
                requires_human_review = True
            elif "lawsuit" in query.lower() or "legal" in query.lower() or "sue" in query.lower():
                requires_human_review = True

            # Calculate confidence score (simplified)
            confidence = 0.95 if len(response_text) > 100 else 0.7

            return {
                "response": response_text,
                "category": category,
                "requires_human_review": requires_human_review,
                "confidence": confidence,
                "alternative_responses": alternative_responses,
            }

        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            # Fallback response
            return {
                "response": "I apologize, but I'm unable to provide a complete response at this time. Please contact our customer service team for assistance.",
                "category": category,
                "requires_human_review": True,
                "confidence": 0.1,
                "alternative_responses": [],
            }

    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a response for the given customer query.

        Args:
            query: The customer query
            context: Additional context information (optional)

        Returns:
            Dictionary with generated response and metadata
        """
        # Check cache first
        cached = self._get_cached_result(query)
        if cached:
            return cached

        # Classify the query
        category = self._classify_query(query)

        # Generate response text
        result = self._generate_response_text(query, category, context)

        # Cache the result
        self._save_to_cache(query, result)

        return result
