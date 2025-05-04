"""
LLM Provider Interface and Implementations

This module provides a unified interface for interacting with different
Large Language Model providers including OpenAI (GPT-4o), Anthropic (Claude), 
Google (Gemini), and others with specific optimizations for financial services.
"""

import os
import logging
import json
import time
import httpx
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    CUSTOM = "custom"

class LLMComplianceLevel(str, Enum):
    """Compliance levels for LLM usage"""
    STANDARD = "standard"
    FINANCIAL = "financial" 
    HEALTHCARE = "healthcare"
    GOVERNMENT = "government"

class LLMCapability(str, Enum):
    """LLM model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    EMBEDDINGS = "embeddings"
    CLASSIFICATION = "classification"
    IMAGE_UNDERSTANDING = "image_understanding"
    DOCUMENT_PROCESSING = "document_processing"
    MULTIMODAL = "multimodal"
    FINE_TUNING = "fine_tuning"
    RAG = "retrieval_augmented_generation"

class LLMConfig:
    """Configuration for LLM provider"""
    def __init__(
        self,
        provider: LLMProvider,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        compliance_level: LLMComplianceLevel = LLMComplianceLevel.STANDARD,
        capabilities: List[LLMCapability] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        organization_id: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.compliance_level = compliance_level
        self.capabilities = capabilities or [LLMCapability.TEXT_GENERATION]
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.organization_id = organization_id
        self.additional_headers = additional_headers or {}
        self.additional_params = additional_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excludes sensitive data)"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "compliance_level": self.compliance_level,
            "capabilities": [cap.value for cap in self.capabilities],
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "organization_id": self.organization_id
        }

class BaseLLMClient(ABC):
    """Base abstract class for all LLM provider clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        
    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the underlying client for the LLM provider"""
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    async def get_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Get embeddings for texts"""
        pass
    
    @abstractmethod
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Classify text into categories with confidence scores"""
        pass
    
    def with_retry(
        self, 
        operation: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute operation with retry logic"""
        max_retries = self.config.max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retry {attempt+1}/{max_retries} after error: {str(e)}. Sleeping {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed after {max_retries} retries: {str(e)}")
                    raise last_error

class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client implementation"""
    
    DEFAULT_MODELS = {
        LLMComplianceLevel.STANDARD: "gpt-4o",
        LLMComplianceLevel.FINANCIAL: "gpt-4o",  # Use 4o for financial compliance
        LLMComplianceLevel.HEALTHCARE: "gpt-4o",
        LLMComplianceLevel.GOVERNMENT: "gpt-4o"
    }
    
    def _initialize_client(self) -> Any:
        """Initialize OpenAI client"""
        try:
            import openai
            client = openai.AsyncClient(
                api_key=self.config.api_key,
                timeout=httpx.Timeout(self.config.timeout_seconds),
                max_retries=self.config.max_retries,
                default_headers=self.config.additional_headers
            )
            return client
        except ImportError:
            raise ImportError("OpenAI package not installed. Run 'pip install openai>=1.0.0'")
    
    async def generate_text(
        self, 
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using OpenAI models"""
        model = self.config.model_name or self.DEFAULT_MODELS[self.config.compliance_level]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
            **{**self.config.additional_params, **kwargs}
        )
        
        return {
            "text": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason,
            "model": response.model
        }
    
    async def get_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Get embeddings for texts from OpenAI"""
        model = kwargs.get("model", "text-embedding-3-large")
        
        response = await self.client.embeddings.create(
            model=model,
            input=texts,
            **self.config.additional_params
        )
        
        return [data.embedding for data in response.data]
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Classify text into categories using OpenAI"""
        # Use GPT to perform classification
        system_prompt = "You are a text classification system. Classify the text into one of the provided categories. Return ONLY a JSON object with categories as keys and confidence scores (0-1) as values. The scores should sum to 1."
        prompt = f"Classify the following text into these categories: {', '.join(categories)}\n\nText: {text}"
        
        result = await self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=150,
            **kwargs
        )
        
        # Parse the result - handle both JSON and text formats
        try:
            import re
            import json
            
            # Try to extract JSON
            content = result["text"]
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            scores = json.loads(content)
            return scores
        except Exception as e:
            logger.error(f"Error parsing classification result: {str(e)}")
            # Return equal distribution if parsing fails
            return {category: 1.0/len(categories) for category in categories}

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude LLM client implementation"""
    
    DEFAULT_MODELS = {
        LLMComplianceLevel.STANDARD: "claude-3-5-sonnet-20240620",
        LLMComplianceLevel.FINANCIAL: "claude-3-5-sonnet-20240620",
        LLMComplianceLevel.HEALTHCARE: "claude-3-5-sonnet-20240620",
        LLMComplianceLevel.GOVERNMENT: "claude-3-5-sonnet-20240620"
    }
    
    def _initialize_client(self) -> Any:
        """Initialize Anthropic client"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds
            )
            return client
        except ImportError:
            raise ImportError("Anthropic package not installed. Run 'pip install anthropic>=0.5.0'")
    
    async def generate_text(
        self, 
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Anthropic Claude models"""
        model = self.config.model_name or self.DEFAULT_MODELS[self.config.compliance_level]
        
        messages = [{"role": "user", "content": prompt}]
        
        # Claude uses system parameter instead of message
        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            stop_sequences=stop_sequences,
            **{**self.config.additional_params, **kwargs}
        )
        
        return {
            "text": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "model": response.model
        }
    
    async def get_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Get embeddings for texts from Anthropic"""
        # Claude doesn't have a dedicated embeddings endpoint
        # We'll use a 3rd party embeddings service or call OpenAI for this
        raise NotImplementedError("Embeddings not supported by Anthropic API directly")
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Classify text into categories using Anthropic Claude"""
        system_prompt = "You are a text classification system. Analyze the text and classify it into the provided categories. Return ONLY a JSON object with categories as keys and confidence scores (0-1) as values. The scores must sum to 1."
        prompt = f"Classify the following text into these categories: {', '.join(categories)}\n\nText: {text}"
        
        result = await self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=150,
            **kwargs
        )
        
        # Parse the JSON result
        try:
            import re
            import json
            
            content = result["text"]
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            scores = json.loads(content)
            return scores
        except Exception as e:
            logger.error(f"Error parsing classification result: {str(e)}")
            return {category: 1.0/len(categories) for category in categories}

class GoogleAIClient(BaseLLMClient):
    """Google AI (Gemini) LLM client implementation"""
    
    DEFAULT_MODELS = {
        LLMComplianceLevel.STANDARD: "gemini-1.5-pro",
        LLMComplianceLevel.FINANCIAL: "gemini-1.5-pro",
        LLMComplianceLevel.HEALTHCARE: "gemini-1.5-pro",
        LLMComplianceLevel.GOVERNMENT: "gemini-1.5-pro"
    }
    
    def _initialize_client(self) -> Any:
        """Initialize Google AI client"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.config.api_key)
            return genai
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Run 'pip install google-generativeai>=0.3.0'")
    
    async def generate_text(
        self, 
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Google Gemini models"""
        model = self.config.model_name or self.DEFAULT_MODELS[self.config.compliance_level]
        
        # Create the generation config
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
        }
        
        # Google's API is not fully async, run in thread pool
        def generate():
            model_instance = self.client.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "parts": [system_prompt]})
            messages.append({"role": "user", "parts": [prompt]})
            
            response = model_instance.generate_content(messages)
            return response
        
        # Run in thread pool
        with ThreadPoolExecutor() as executor:
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(executor, generate)
        
        # Extract and return the result
        return {
            "text": response.text,
            "usage": {},  # Google doesn't provide token usage info
            "model": model
        }
    
    async def get_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Get embeddings for texts from Google AI"""
        model = kwargs.get("model", "embedding-001")
        
        # Google's API is not fully async, run in thread pool
        def get_embeddings():
            result = []
            for text in texts:
                embedding = self.client.embed_content(
                    model=model,
                    content=text,
                    task_type=kwargs.get("task_type", "RETRIEVAL_QUERY")
                )
                result.append(embedding["embedding"])
            return result
        
        # Run in thread pool
        with ThreadPoolExecutor() as executor:
            import asyncio
            embeddings = await asyncio.get_event_loop().run_in_executor(executor, get_embeddings)
        
        return embeddings
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Classify text into categories using Google Gemini"""
        system_prompt = "You are a classification system. Analyze the text and return ONLY a JSON object with categories as keys and confidence scores (0-1) as values. Scores must sum to 1."
        prompt = f"Classify the following text into these categories: {', '.join(categories)}\n\nText: {text}\n\nReturn ONLY a JSON object with categories and scores."
        
        result = await self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=150,
            **kwargs
        )
        
        # Parse the JSON result
        try:
            import re
            import json
            
            content = result["text"]
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            scores = json.loads(content)
            return scores
        except Exception as e:
            logger.error(f"Error parsing classification result: {str(e)}")
            return {category: 1.0/len(categories) for category in categories}

class LLMClientFactory:
    """Factory for creating LLM clients based on provider"""
    
    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        """Create appropriate LLM client based on provider"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIClient(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(config)
        elif config.provider == LLMProvider.GOOGLE:
            return GoogleAIClient(config)
        elif config.provider == LLMProvider.AZURE_OPENAI:
            # For Azure, we'd implement a specialized OpenAI client
            # but for simplicity, we're not implementing it fully here
            raise NotImplementedError("Azure OpenAI client not implemented")
        elif config.provider == LLMProvider.COHERE:
            raise NotImplementedError("Cohere client not implemented")
        elif config.provider == LLMProvider.CUSTOM:
            raise ValueError("Custom provider requires implementation class to be provided")
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    @staticmethod
    def create_financial_client(
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseLLMClient:
        """Create LLM client optimized for financial services"""
        config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            compliance_level=LLMComplianceLevel.FINANCIAL,
            capabilities=[
                LLMCapability.TEXT_GENERATION,
                LLMCapability.CLASSIFICATION,
                LLMCapability.DOCUMENT_PROCESSING
            ],
            additional_params={
                "user": "financial-services",
                "safe_mode": True
            }
        )
        
        return LLMClientFactory.create_client(config) 