# CustomerAI Fairness Framework Implementation Guide

This guide provides detailed instructions for implementing and integrating the CustomerAI fairness framework with real APIs, ML frameworks, and organization-specific values.

## Table of Contents

1. [Introduction](#introduction)
2. [Configuration System](#configuration-system)
3. [Bias Detection Implementation](#bias-detection-implementation)
4. [Mitigation Strategies Implementation](#mitigation-strategies-implementation)
5. [ML Framework Integration](#ml-framework-integration)
6. [API Integration](#api-integration)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Testing and Validation](#testing-and-validation)

## Introduction

The CustomerAI fairness framework is designed to be flexible and adaptable to different organizations' requirements. This guide explains how to replace placeholder implementations with real APIs and values specific to your use case.

## Configuration System

### Real Configuration Values

The framework uses a centralized configuration system that supports multiple sources:

1. **Configuration File**: `config/fairness_config.json`
2. **Environment Variables**: Prefixed with `FAIRNESS_`
3. **Direct Code Configuration**: Passed to component constructors

To replace placeholder values with real ones:

```json
// config/fairness_config.json
{
  "thresholds": {
    "disparate_impact": 0.85,  // Your organization's specific threshold
    "statistical_parity_difference": 0.05
  },
  "significance": {
    "pvalue_threshold": 0.01  // Your statistical significance level
  },
  "api": {
    "rate_limit": 500,  // Your API rate limit
    "max_payload_size_mb": 50  // Your payload size limit
  }
}
```

### Environment-Specific Configuration

For different environments (development, testing, production), use environment variables:

```bash
# Development
export FAIRNESS_THRESHOLDS_DISPARATE_IMPACT=0.8
export FAIRNESS_API_RATE_LIMIT=100

# Production
export FAIRNESS_THRESHOLDS_DISPARATE_IMPACT=0.85
export FAIRNESS_API_RATE_LIMIT=500
```

## Bias Detection Implementation

### Statistical Tests

Replace placeholder statistical tests with appropriate ones for your data:

```python
from scipy import stats

def custom_statistical_test(group1_data, group2_data):
    # Example: Use Mann-Whitney U test for non-parametric data
    u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data)
    return p_value

# Update BiasDetector._calculate_significance method
def _calculate_significance(self, rate1, rate2, n1, n2):
    # Use your custom statistical test
    return custom_statistical_test(rate1, rate2)
```

### Domain-Specific Metrics

Add domain-specific fairness metrics relevant to your industry:

```python
# Add to BiasDetector class
def calculate_custom_metric(self, data, protected_attribute, outcome):
    # Implement your domain-specific metric
    # Example: Risk ratio for healthcare
    # Example: Approval rate ratio for financial services
    pass
```

## Mitigation Strategies Implementation

### TensorFlow Integration for Adversarial Debiasing

To implement adversarial debiasing with TensorFlow:

```python
import tensorflow as tf

def create_adversarial_model(input_dim, output_dim, protected_dim):
    # Create predictor model
    predictor_input = tf.keras.Input(shape=(input_dim,))
    predictor_hidden = tf.keras.layers.Dense(64, activation='relu')(predictor_input)
    predictor_output = tf.keras.layers.Dense(output_dim, activation='sigmoid')(predictor_hidden)

    # Create adversary model
    adversary_input = tf.keras.layers.concatenate([predictor_hidden, predictor_output])
    adversary_hidden = tf.keras.layers.Dense(32, activation='relu')(adversary_input)
    adversary_output = tf.keras.layers.Dense(protected_dim, activation='softmax')(adversary_hidden)

    # Create combined model
    predictor_model = tf.keras.Model(inputs=predictor_input, outputs=predictor_output)
    adversary_model = tf.keras.Model(inputs=[predictor_hidden, predictor_output], outputs=adversary_output)

    return predictor_model, adversary_model

# Update FairnessMitigation.adversarial_debiasing method
def adversarial_debiasing(self, data, protected_attribute, features, outcome_column, ...):
    # Preprocess data
    X = data[features].values
    y = data[outcome_column].values
    protected = data[protected_attribute].values

    # Create models
    predictor_model, adversary_model = create_adversarial_model(
        input_dim=len(features),
        output_dim=1,
        protected_dim=len(np.unique(protected))
    )

    # Implement training loop with adversarial loss
    # ...

    return {
        'model': predictor_model,
        'metrics': {
            'accuracy': accuracy,
            'fairness_metrics': fairness_metrics
        }
    }
```

### Scikit-learn Integration for Reweighing

To implement reweighing with scikit-learn:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_with_weights(X, y, sample_weights):
    # Create and train a model with sample weights
    model = LogisticRegression()
    model.fit(X, y, sample_weight=sample_weights)
    return model

# Update FairnessMitigation.reweigh_samples method
def reweigh_samples(self, data, protected_attribute, outcome_column, ...):
    # Calculate weights
    weights = self._calculate_weights(data, protected_attribute, outcome_column)

    # Return weighted data
    data_with_weights = data.copy()
    data_with_weights['sample_weight'] = weights

    return data_with_weights, weights
```

## ML Framework Integration

### Integration with Existing ML Pipelines

To integrate with your existing ML pipelines:

```python
from your_ml_pipeline import ModelTrainer, ModelEvaluator

# Create a fairness-aware model trainer
class FairnessAwareTrainer:
    def __init__(self, fairness_mitigation):
        self.fairness_mitigation = fairness_mitigation
        self.model_trainer = ModelTrainer()

    def train(self, data, protected_attributes, features, outcome):
        # Apply fairness mitigation
        reweighted_data, weights = self.fairness_mitigation.reweigh_samples(
            data, protected_attributes[0], outcome
        )

        # Train model with weights
        model = self.model_trainer.train(
            reweighted_data[features],
            reweighted_data[outcome],
            sample_weights=weights
        )

        return model
```

### Custom Model Factory

Create a custom model factory for your specific ML framework:

```python
def custom_model_factory(input_dim, output_dim, framework='tensorflow'):
    if framework == 'tensorflow':
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_dim, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    elif framework == 'pytorch':
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}")
```

## API Integration

### RESTful API Implementation

To implement the fairness API with real authentication and rate limiting:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from your_auth_system import authenticate_user, get_current_user

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Authentication
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Fairness API endpoint
@app.post("/api/v1/fairness/detect_bias")
async def detect_bias(
    data: Dict,
    current_user: User = Depends(get_current_user)
):
    # Check rate limits based on user
    if not check_rate_limit(current_user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

    # Process request
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])

        # Run bias detection
        bias_detector = BiasDetector()
        results = bias_detector.detect_outcome_bias(
            df,
            attributes=data['protected_attributes'],
            outcome_columns=data['outcome_columns']
        )

        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
```

### GraphQL API Implementation

For GraphQL API integration:

```python
import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional

@strawberry.type
class BiasDetectionResult:
    bias_detected: bool
    attribute: str
    outcome: str
    metric: str
    value: float
    threshold: float
    p_value: float

@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Hello World"

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def detect_bias(
        self,
        data: List[Dict],
        protected_attributes: List[str],
        outcome_columns: List[str]
    ) -> List[BiasDetectionResult]:
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Run bias detection
        bias_detector = BiasDetector()
        results = bias_detector.detect_outcome_bias(
            df,
            attributes=protected_attributes,
            outcome_columns=outcome_columns
        )

        # Convert results to GraphQL type
        return [
            BiasDetectionResult(
                bias_detected=finding['bias_detected'],
                attribute=finding['attribute'],
                outcome=finding['outcome'],
                metric=finding['metric'],
                value=finding['value'],
                threshold=finding['threshold'],
                p_value=finding['p_value']
            )
            for finding in results.get('summary', {}).get('significant_findings', [])
        ]

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)
```

## Monitoring and Logging

### Prometheus Integration

To implement real-time monitoring with Prometheus:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
BIAS_DETECTION_REQUESTS = Counter(
    'bias_detection_requests_total',
    'Total number of bias detection requests',
    ['status', 'attribute']
)

BIAS_DETECTION_DURATION = Histogram(
    'bias_detection_duration_seconds',
    'Duration of bias detection requests',
    ['attribute']
)

BIAS_DETECTED = Counter(
    'bias_detected_total',
    'Number of times bias was detected',
    ['attribute', 'outcome', 'metric']
)

# Update BiasDetector.detect_outcome_bias method
def detect_outcome_bias(self, data, attributes, outcome_columns, ...):
    for attribute in attributes:
        with BIAS_DETECTION_DURATION.labels(attribute=attribute).time():
            # Existing bias detection code
            # ...

            # Record metrics
            BIAS_DETECTION_REQUESTS.labels(
                status='success',
                attribute=attribute
            ).inc()

            if bias_detected:
                BIAS_DETECTED.labels(
                    attribute=attribute,
                    outcome=outcome,
                    metric=metric
                ).inc()
```

### Structured Logging

Implement structured logging for better observability:

```python
import structlog

# Configure structured logger
logger = structlog.get_logger()

# Update BiasDetector.detect_outcome_bias method
def detect_outcome_bias(self, data, attributes, outcome_columns, ...):
    logger.info(
        "starting_bias_detection",
        dataset_size=len(data),
        attributes=attributes,
        outcomes=outcome_columns
    )

    # Existing bias detection code
    # ...

    logger.info(
        "bias_detection_complete",
        bias_detected=results['summary']['bias_detected'],
        significant_findings_count=len(results['summary'].get('significant_findings', [])),
        execution_time_ms=execution_time
    )
```

## Testing and Validation

### Unit Testing with Real Data

Create unit tests with real-world data samples:

```python
import pytest
import pandas as pd
from src.fairness.bias_detector import BiasDetector

# Load real test data
@pytest.fixture
def real_loan_data():
    # Load from your data source
    return pd.read_csv('tests/data/real_loan_data_sample.csv')

def test_bias_detection_with_real_data(real_loan_data):
    # Initialize with real thresholds
    bias_detector = BiasDetector({
        'significance_level': 0.01,
        'fairness_threshold': 0.85
    })

    # Run bias detection
    results = bias_detector.detect_outcome_bias(
        real_loan_data,
        attributes=['gender', 'race', 'age_group'],
        outcome_columns=['loan_approved']
    )

    # Validate results against expected outcomes for this dataset
    assert 'summary' in results
    assert 'bias_detected' in results['summary']

    # Check specific metrics based on known characteristics of the dataset
    gender_findings = [f for f in results['summary'].get('significant_findings', [])
                      if f['attribute'] == 'gender']
    assert len(gender_findings) > 0
```

### Integration Testing with Real APIs

Test integration with real ML frameworks and APIs:

```python
import pytest
import tensorflow as tf
from src.fairness.mitigation import FairnessMitigation

@pytest.fixture
def real_model_factory():
    def create_model(input_dim, output_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='sigmoid')
        ])
    return create_model

def test_adversarial_debiasing_with_real_tf(real_loan_data, real_model_factory):
    # Initialize with real parameters
    fairness_mitigation = FairnessMitigation({
        'default_strategy': 'adversarial_debiasing',
        'strategy_params': {
            'adversarial_debiasing': {
                'learning_rate': 0.001,
                'batch_size': 64
            }
        }
    })

    # Run adversarial debiasing
    result = fairness_mitigation.adversarial_debiasing(
        data=real_loan_data,
        protected_attribute='gender',
        features=['income', 'credit_score', 'loan_amount', 'loan_term'],
        outcome_column='loan_approved',
        model_factory=real_model_factory,
        epochs=5  # Small number for testing
    )

    # Validate result
    assert 'model' in result
    assert 'metrics' in result
    assert isinstance(result['model'], tf.keras.Model)
```

---

By following this implementation guide, you can replace placeholder APIs and values with real implementations specific to your organization's requirements and technical stack.

For any questions or assistance with implementation, please contact the CustomerAI team at support@customerai.example.com.
