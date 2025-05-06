"""
Type definitions for the privacy module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, TypedDict


class PIIType(str, Enum):
    """Types of PII data"""

    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"


class AnonymizationMethod(str, Enum):
    """Available anonymization methods"""

    HASH = "hash"
    MASK = "mask"
    REDACT = "redact"
    ENCRYPT = "encrypt"
    TOKENIZE = "tokenize"


@dataclass
class AnonymizationConfig:
    """Configuration for data anonymization"""

    pii_types: Set[PIIType]
    method: AnonymizationMethod
    preserve_format: bool = False
    encryption_key: Optional[str] = None
    custom_patterns: Optional[List[str]] = None


class AnonymizationResult(TypedDict):
    """Result of anonymization process"""

    original_text: str
    anonymized_text: str
    pii_found: List[PIIType]
    method_used: AnonymizationMethod
    success: bool
    error: Optional[str]


# Type aliases
AnonymizationRules = Dict[PIIType, AnonymizationMethod]
PIILocations = Dict[PIIType, List[tuple[int, int]]]  # (start, end) positions
