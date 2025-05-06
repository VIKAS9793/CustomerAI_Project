# CustomerAI Type System Documentation

## Overview

The CustomerAI project uses a robust type system to ensure type safety, maintainability, and security. This document outlines the type system architecture, conventions, and best practices.

## Core Type System Components

### 1. Response Types
Located in `src/response/types.py`:

```python
from typing import Any, Dict, List, Optional, TypedDict

class ResponseContext(TypedDict):
    """Context information for response generation."""
    customer_history: Optional[List[Dict[str, str]]]
    previous_interactions: Optional[List[Dict[str, str]]]
    sentiment: Optional[SentimentLabel]
    priority: Optional[int]
    custom_data: Optional[Dict[str, Any]]

class ResponseMetadata(TypedDict):
    """Metadata about the generated response."""
    timestamp: str
    confidence: float
    conditions: Optional[Dict[str, str]]
    alternatives: Optional[List[str]]
```

### 2. Fairness Types
Located in `src/fairness/types.py`:

```python
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic, Type, NoReturn
from typing_extensions import Literal
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum
from datetime import datetime
import json

class MetricType(str, Enum):
    """Enum for different types of fairness metrics."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_ODDS = "equal_odds"
    EQUAL_OPPO = "equal_oppo"
    PREDICTIVE_PARITY = "predictive_parity"

class MitigationStrategy(str, Enum):
    """Enum for different mitigation strategies."""
    REWEIGHING = "reweighing"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    ADVERSARIAL = "adversarial"

class FairnessMetric(BaseModel):
    """Base class for fairness metrics."""
    name: str
    description: str
    type: MetricType
    threshold: float = Field(gt=0, le=1)

class FairnessConfig(BaseModel):
    """Configuration for fairness analysis."""
    metrics: List[FairnessMetric]
    sensitive_attributes: List[str]
    mitigation_strategy: MitigationStrategy
    confidence_level: float = Field(gt=0.9, le=1)
```

### 3. Security Types
Located in `src/utils/security_utils.py`:

```python
from typing import Any, Optional, List, Dict

class SecurityError(Exception):
    """Base class for security-related errors."""
    pass

class ValidationError(SecurityError):
    """Raised when input validation fails."""
    pass

class InjectionError(SecurityError):
    """Raised when potential injection is detected."""
    pass
```

## Type System Best Practices

1. **Explicit Optional Types**
   - Always use `Optional[T]` instead of `T = None`
   - Document default values in docstrings

2. **Type Validation**
   - Use Pydantic for data validation
   - Implement custom validators for complex types
   - Use type guards for runtime type checking

3. **Error Handling**
   - Use specific exception types
   - Document error conditions
   - Provide clear error messages

4. **Documentation**
   - Add docstrings to all public classes and methods
   - Document type constraints and invariants
   - Include examples where applicable

## Type System Maintenance

1. **Regular Updates**
   - Keep type definitions in sync with code changes
   - Update type hints during refactoring
   - Review type system periodically

2. **Testing**
   - Write type tests
   - Test edge cases
   - Verify type safety

3. **Code Review**
   - Check for type consistency
   - Verify type safety
   - Ensure proper documentation

## Common Type Patterns

### 1. Optional Parameters
```python
def process_data(
    data: List[str],
    context: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Process data with optional context and timeout.

    Args:
        data: Input data to process
        context: Optional context information
        timeout: Optional timeout in seconds

    Returns:
        Processed data with metadata
    """
```

### 2. Type Validation
```python
def validate_input(data: Any, max_length: int = 1024) -> None:
    """Validate input data for security.

    Args:
        data: Input data to validate
        max_length: Maximum allowed length

    Raises:
        ValidationError: If validation fails
    """
```

### 3. Error Handling
```python
class CustomError(Exception):
    """Custom error for specific conditions."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details
```

## Type System Roadmap

1. **Short-term Goals**
   - Fix remaining type system issues
   - Improve type coverage
   - Add more type tests

2. **Medium-term Goals**
   - Implement type-based security checks
   - Add runtime type validation
   - Improve error handling

3. **Long-term Goals**
   - Full type system documentation
   - Automated type checking
   - Type-based security monitoring
