import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Set, Tuple, Union

import spacy

# Import security utilities for encryption
from src.utils.security import decrypt_data, encrypt_data, sanitize_input

# Setup logging
logger = logging.getLogger(__name__)


class DataAnonymizer:
    """
    Handles detection and anonymization of personally identifiable information (PII)
    with a focus on financial data protection regulations.
    """

    def __init__(
        self,
        load_spacy_model: bool = True,
        use_encryption: bool = True,
        use_consistent_tokens: bool = True,
    ):
        """
        Initialize the data anonymizer.

        Args:
            load_spacy_model: Whether to load the NLP model for advanced PII detection
            use_encryption: Whether to encrypt mapping data
            use_consistent_tokens: Whether to use consistent replacement tokens
        """
        self.use_encryption = use_encryption
        self.use_consistent_tokens = use_consistent_tokens

        # Set up NLP if enabled
        self.nlp = None
        if load_spacy_model:
            try:
                # Try to load the model - will require download with spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully loaded spaCy NLP model")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {str(e)}")

        # Load PII patterns
        self._load_patterns()

        # Initialize the token mapping for consistent replacements
        self.token_mapping = {}

    def _load_patterns(self) -> None:
        """Load regex patterns for PII detection."""
        # Define common PII regex patterns
        self.patterns = {
            # Contact Information
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            # Financial Information
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "account_number": r"\b\d{8,17}\b",  # 8-17 digit account numbers
            "routing_number": r"\b[0-9]{9}\b",  # 9-digit routing numbers
            # Personal Identifiers
            "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",  # Social Security Number
            "ein": r"\b\d{2}[-\s]?\d{7}\b",  # Employer Identification Number
            # Addresses
            "zip_code": r"\b\d{5}(?:[-\s]\d{4})?\b",
            "street_address": r"\b\d+\s+[A-Za-z0-9\s,\.]+(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Parkway|Pkwy)\b",
            # Dates
            "date": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
            "dob": r"\bDOB:?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            # URLs and IP addresses
            "url": r"\bhttps?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        }

        # Try to load additional patterns from file if available
        try:
            patterns_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "pii_patterns.json"
            )

            if os.path.exists(patterns_path):
                with open(patterns_path, "r") as f:
                    additional_patterns = json.load(f)
                    self.patterns.update(additional_patterns)
        except Exception as e:
            logger.error(f"Error loading additional PII patterns: {str(e)}")

    def _generate_replacement_token(self, pii_type: str, original_value: str) -> str:
        """
        Generate a replacement token for a specific PII type.

        Args:
            pii_type: Type of PII (e.g., "email", "phone")
            original_value: The original PII value

        Returns:
            A replacement token
        """
        # Check if we should use a consistent token for this value
        if self.use_consistent_tokens:
            # Create a unique but deterministic key for the value
            # Hash the value to avoid storing the actual PII in memory
            value_hash = hashlib.sha256(original_value.encode()).hexdigest()[:16]
            key = f"{pii_type}:{value_hash}"

            # Return existing token if we've seen this value before
            if key in self.token_mapping:
                return self.token_mapping[key]

        # Generate a new token
        token = f"[{pii_type.upper()}_{uuid.uuid4().hex[:8]}]"

        # Store in mapping if using consistent tokens
        if self.use_consistent_tokens:
            self.token_mapping[key] = token

        return token

    def _detect_pii_with_regex(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Detect PII in text using regex patterns.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping PII types to list of (value, replacement) tuples
        """
        pii_found = {}

        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                value = match.group()

                # Generate replacement token
                replacement = self._generate_replacement_token(pii_type, value)

                # Add to results
                if pii_type not in pii_found:
                    pii_found[pii_type] = []

                pii_found[pii_type].append((value, replacement))

        return pii_found

    def _detect_pii_with_nlp(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Detect PII in text using NLP model.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping PII types to list of (value, replacement) tuples
        """
        if not self.nlp:
            return {}

        # Process text with spaCy
        doc = self.nlp(text)

        # Map spaCy entity types to our PII types
        entity_map = {
            "PERSON": "name",
            "ORG": "organization",
            "GPE": "location",
            "LOC": "location",
            "DATE": "date",
            "MONEY": "financial",
            "CARDINAL": "number",
            "ORDINAL": "number",
        }

        pii_found = {}

        # Extract entities
        for ent in doc.ents:
            if ent.label_ in entity_map:
                pii_type = entity_map[ent.label_]
                value = ent.text

                # Generate replacement token
                replacement = self._generate_replacement_token(pii_type, value)

                # Add to results
                if pii_type not in pii_found:
                    pii_found[pii_type] = []

                pii_found[pii_type].append((value, replacement))

        return pii_found

    def anonymize_text(self, text: str, keep_mapping: bool = False) -> Dict[str, Any]:
        """
        Detect and anonymize PII in text.

        Args:
            text: Text to anonymize
            keep_mapping: Whether to return the mapping for de-anonymization

        Returns:
            Dictionary with anonymized text and optionally the mapping
        """
        # Sanitize input to prevent injection attacks
        text = sanitize_input(text)

        # Detect PII using regex
        regex_pii = self._detect_pii_with_regex(text)

        # Detect PII using NLP if available
        nlp_pii = self._detect_pii_with_nlp(text) if self.nlp else {}

        # Merge results
        all_pii = {}
        for pii_dict in [regex_pii, nlp_pii]:
            for pii_type, values in pii_dict.items():
                if pii_type not in all_pii:
                    all_pii[pii_type] = []
                all_pii[pii_type].extend(values)

        # Remove duplicates
        for pii_type in all_pii:
            all_pii[pii_type] = list(set(all_pii[pii_type]))

        # Create mapping for replacements
        mapping = {}
        for pii_type, values in all_pii.items():
            for original, replacement in values:
                mapping[replacement] = original

        # Anonymize text by replacing PII with tokens
        anonymized = text
        for pii_type, values in all_pii.items():
            for original, replacement in values:
                # Replace all occurrences
                anonymized = anonymized.replace(original, replacement)

        # Prepare result
        result = {
            "anonymized_text": anonymized,
            "pii_detected": list(all_pii.keys()),
            "pii_count": sum(len(values) for values in all_pii.values()),
        }

        # Add mapping if requested
        if keep_mapping:
            # Encrypt the mapping if enabled
            if self.use_encryption:
                try:
                    # Convert to JSON string and encrypt
                    mapping_str = json.dumps(mapping)
                    encrypted_mapping = encrypt_data(mapping_str)
                    result["mapping"] = encrypted_mapping
                    result["mapping_encrypted"] = True
                except Exception as e:
                    logger.error(f"Error encrypting mapping: {str(e)}")
                    result["mapping"] = mapping
                    result["mapping_encrypted"] = False
            else:
                result["mapping"] = mapping
                result["mapping_encrypted"] = False

        return result

    def deanonymize_text(self, anonymized_text: str, mapping: Union[Dict, str]) -> str:
        """
        Restore anonymized text using the mapping.

        Args:
            anonymized_text: Anonymized text
            mapping: Mapping dict or encrypted mapping string

        Returns:
            Original text with PII restored
        """
        # Handle encrypted mapping
        if isinstance(mapping, str):
            try:
                # Decrypt and parse
                mapping_str = decrypt_data(mapping).decode()
                mapping = json.loads(mapping_str)
            except Exception as e:
                logger.error(f"Error decrypting mapping: {str(e)}")
                return anonymized_text

        # Restore text by replacing tokens with original values
        restored = anonymized_text
        for token, original in mapping.items():
            restored = restored.replace(token, original)

        return restored

    def anonymize_json(self, data: Dict, sensitive_fields: Set[str] = None) -> Dict[str, Any]:
        """
        Anonymize sensitive fields in a JSON object.

        Args:
            data: JSON data to anonymize
            sensitive_fields: Set of field names to anonymize (if None, auto-detect)

        Returns:
            Dictionary with anonymized data and mapping
        """
        if sensitive_fields is None:
            # Default sensitive fields
            sensitive_fields = {
                "name",
                "email",
                "phone",
                "address",
                "ssn",
                "dob",
                "credit_card",
                "account_number",
                "password",
                "secret",
                "token",
            }

        # Recursively process the JSON
        anonymized_data = {}
        mapping = {}

        for key, value in data.items():
            # Check if this is a sensitive field
            is_sensitive = key.lower() in sensitive_fields or any(
                s in key.lower() for s in ["id", "key", "token", "secret", "password"]
            )

            if is_sensitive and isinstance(value, str):
                # Anonymize the value
                replacement = self._generate_replacement_token(key, value)
                anonymized_data[key] = replacement
                mapping[replacement] = value
            elif isinstance(value, dict):
                # Recurse into nested dictionary
                nested_result = self.anonymize_json(value, sensitive_fields)
                anonymized_data[key] = nested_result["anonymized_data"]
                mapping.update(nested_result["mapping"])
            elif isinstance(value, list):
                # Process list items
                anonymized_list = []
                for item in value:
                    if isinstance(item, dict):
                        nested_result = self.anonymize_json(item, sensitive_fields)
                        anonymized_list.append(nested_result["anonymized_data"])
                        mapping.update(nested_result["mapping"])
                    else:
                        anonymized_list.append(item)
                anonymized_data[key] = anonymized_list
            else:
                # Keep as is
                anonymized_data[key] = value

        return {"anonymized_data": anonymized_data, "mapping": mapping}

    def redact_text(self, text: str) -> str:
        """
        Completely redact PII without keeping a mapping.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        # Anonymize first
        result = self.anonymize_text(text, keep_mapping=False)
        return result["anonymized_text"]
