# ML Framework Integration Guide

This guide provides detailed instructions for integrating the CustomerAI fairness framework with popular machine learning frameworks such as TensorFlow, PyTorch, and scikit-learn.

## Table of Contents

1. [Overview](#overview)
2. [TensorFlow Integration](#tensorflow-integration)
3. [PyTorch Integration](#pytorch-integration)
4. [Scikit-learn Integration](#scikit-learn-integration)
5. [Custom Model Integration](#custom-model-integration)
6. [Configuration Options](#configuration-options)
7. [Testing and Validation](#testing-and-validation)

## Overview

The fairness framework is designed to work with various ML frameworks through a flexible integration system. This allows organizations to use their preferred ML framework while still benefiting from the fairness capabilities.

Key integration points include:

1. **Model Training**: Integrating fairness constraints during model training
2. **Prediction Post-processing**: Applying fairness adjustments to model predictions
3. **Model Evaluation**: Evaluating models for fairness metrics

## TensorFlow Integration

### Prerequisites

- TensorFlow 2.4.0 or later
- CustomerAI fairness framework

### Basic Integration

```python
import tensorflow as tf
from src.fairness.mitigation import FairnessMitigation

# Initialize fairness mitigation
fairness_mitigation = FairnessMitigation()

# Create a TensorFlow model factory
def tf_model_factory(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Use adversarial debiasing with TensorFlow
result = fairness_mitigation.adversarial_debiasing(
    data=training_data,
    protected_attribute='gender',
    features=['income', 'education', 'occupation'],
    outcome_column='loan_approved',
    model_factory=tf_model_factory,
    framework='tensorflow'
)

# Get the debiased model
debiased_model = result['model']
```

### Advanced TensorFlow Integration

For more advanced integration, you can create a custom TensorFlow model with fairness constraints:

```python
import tensorflow as tf
from src.fairness.mitigation import FairnessMitigation

class FairModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, fairness_mitigation):
        super(FairModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')
        self.fairness_mitigation = fairness_mitigation

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

    def train_step(self, data):
        # Unpack the data
        x, y = data

        # Get protected attributes (assuming they're part of the input)
        protected = x[:, 0]  # Adjust index based on your data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

            # Add fairness constraint
            fairness_loss = self._calculate_fairness_loss(y_pred, y, protected)
            total_loss = loss + fairness_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['fairness_loss'] = fairness_loss
        return results

    def _calculate_fairness_loss(self, y_pred, y_true, protected):
        # Calculate demographic parity loss
        # This is a simplified example - replace with your specific fairness metric
        protected_mask = tf.cast(protected > 0, tf.float32)
        unprotected_mask = 1.0 - protected_mask

        protected_acceptance = tf.reduce_mean(y_pred * protected_mask)
        unprotected_acceptance = tf.reduce_mean(y_pred * unprotected_mask)

        demographic_parity_loss = tf.abs(protected_acceptance - unprotected_acceptance)

        return demographic_parity_loss * 0.1  # Weight for fairness constraint
```

## PyTorch Integration

### Prerequisites

- PyTorch 1.8.0 or later
- CustomerAI fairness framework

### Basic Integration

```python
import torch
import torch.nn as nn
import torch.optim as optim
from src.fairness.mitigation import FairnessMitigation

# Initialize fairness mitigation
fairness_mitigation = FairnessMitigation()

# Create a PyTorch model factory
def pytorch_model_factory(input_dim, output_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.2),
        nn.Linear(32, output_dim),
        nn.Sigmoid()
    )
    return model

# Use adversarial debiasing with PyTorch
result = fairness_mitigation.adversarial_debiasing(
    data=training_data,
    protected_attribute='gender',
    features=['income', 'education', 'occupation'],
    outcome_column='loan_approved',
    model_factory=pytorch_model_factory,
    framework='pytorch'
)

# Get the debiased model
debiased_model = result['model']
```

### Advanced PyTorch Integration

For more advanced integration, you can create a custom PyTorch model with fairness constraints:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from src.fairness.mitigation import FairnessMitigation

class FairModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FairModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.sigmoid(self.output_layer(x))
        return x

# Training with fairness constraints
def train_fair_model(model, train_loader, protected_idx, fairness_weight=0.1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Standard loss
            loss = criterion(output, target)

            # Fairness loss
            protected = data[:, protected_idx]
            protected_mask = (protected > 0).float()
            unprotected_mask = 1.0 - protected_mask

            protected_acceptance = torch.mean(output * protected_mask)
            unprotected_acceptance = torch.mean(output * unprotected_mask)

            fairness_loss = torch.abs(protected_acceptance - unprotected_acceptance)

            # Total loss
            total_loss = loss + fairness_weight * fairness_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, Fairness Loss: {fairness_loss.item()}')

    return model
```

## Scikit-learn Integration

### Prerequisites

- scikit-learn 0.24.0 or later
- CustomerAI fairness framework

### Basic Integration

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.fairness.mitigation import FairnessMitigation

# Initialize fairness mitigation
fairness_mitigation = FairnessMitigation()

# Reweigh samples for fairness
reweighted_data, weights = fairness_mitigation.reweigh_samples(
    data=training_data,
    protected_attribute='gender',
    outcome_column='loan_approved'
)

# Train a model with the weights
model = LogisticRegression()
model.fit(
    reweighted_data[features],
    reweighted_data[outcome_column],
    sample_weight=weights
)

# Make predictions
predictions = model.predict(test_data[features])

# Apply post-processing for equalized odds
adjusted_predictions = fairness_mitigation.equalized_odds_postprocessing(
    y_pred=predictions,
    y_true=test_data[outcome_column],
    protected_attributes=test_data[protected_attribute]
)
```

### Custom Scikit-learn Estimator

You can create a custom scikit-learn estimator that incorporates fairness:

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from src.fairness.mitigation import FairnessMitigation

class FairClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, protected_attribute_idx=0, base_estimator=None, fairness_threshold=0.8):
        self.protected_attribute_idx = protected_attribute_idx
        self.base_estimator = base_estimator or LogisticRegression()
        self.fairness_threshold = fairness_threshold
        self.fairness_mitigation = FairnessMitigation()

    def fit(self, X, y):
        # Check inputs
        X, y = check_X_y(X, y)

        # Extract protected attribute
        protected = X[:, self.protected_attribute_idx]

        # Convert to DataFrame for fairness mitigation
        import pandas as pd
        data = pd.DataFrame(X)
        data['outcome'] = y
        data['protected'] = protected

        # Apply reweighing
        reweighted_data, weights = self.fairness_mitigation.reweigh_samples(
            data=data,
            protected_attribute='protected',
            outcome_column='outcome'
        )

        # Train base estimator with weights
        self.base_estimator.fit(X, y, sample_weight=weights)

        # Save training data info
        self.classes_ = self.base_estimator.classes_
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Check input
        X = check_array(X)

        # Get base predictions
        base_predictions = self.base_estimator.predict(X)

        # Extract protected attribute
        protected = X[:, self.protected_attribute_idx]

        # Apply post-processing for fairness
        # This is a simplified example - in practice, you would need test data with ground truth
        # to apply equalized odds post-processing

        return base_predictions

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Check input
        X = check_array(X)

        # Get base probabilities
        return self.base_estimator.predict_proba(X)
```

## Custom Model Integration

If you're using a custom ML framework or model, you can still integrate with the fairness framework:

```python
from src.fairness.mitigation import FairnessMitigation
from src.fairness.bias_detector import BiasDetector

# Initialize components
fairness_mitigation = FairnessMitigation()
bias_detector = BiasDetector()

# Custom model training function
def train_custom_model(data, features, outcome, protected_attribute):
    # Step 1: Apply pre-processing fairness mitigation
    reweighted_data, weights = fairness_mitigation.reweigh_samples(
        data=data,
        protected_attribute=protected_attribute,
        outcome_column=outcome
    )

    # Step 2: Train your custom model with the weights
    # This is where you would integrate with your custom framework
    model = YourCustomModel()
    model.train(
        reweighted_data[features],
        reweighted_data[outcome],
        sample_weights=weights
    )

    # Step 3: Evaluate fairness of the model
    predictions = model.predict(data[features])
    fairness_results = bias_detector.detect_outcome_bias(
        data=data.assign(predictions=predictions),
        attributes=[protected_attribute],
        outcome_columns=['predictions']
    )

    # Step 4: Apply post-processing if needed
    if fairness_results['summary']['bias_detected']:
        adjusted_predictions = fairness_mitigation.equalized_odds_postprocessing(
            y_pred=predictions,
            y_true=data[outcome],
            protected_attributes=data[protected_attribute]
        )
        model.set_post_processor(lambda x: adjust_predictions(x, model, fairness_mitigation))

    return model, fairness_results

# Helper function for post-processing
def adjust_predictions(X, model, fairness_mitigation):
    base_predictions = model.predict(X)
    # Apply fairness adjustments
    # This is a placeholder - you would need to implement this based on your specific needs
    return base_predictions
```

## Configuration Options

The fairness framework provides several configuration options for ML framework integration:

```json
{
  "ml_integration": {
    "framework": "tensorflow",         // Options: tensorflow, pytorch, scikit-learn
    "version": "2.4.0",               // Minimum version required
    "gpu_enabled": false,             // Set to true to enable GPU acceleration
    "model_cache_size": 5,            // Number of models to keep in memory
    "batch_processing": true,         // Enable batch processing for large datasets
    "max_batch_size": 1000            // Maximum batch size for processing
  },
  "mitigation": {
    "adversarial_debiasing": {
      "learning_rate": 0.001,        // Learning rate for adversarial training
      "predictor_hidden_units": [64, 32], // Neural network architecture for predictor
      "adversary_hidden_units": [32],    // Neural network architecture for adversary
      "batch_norm": true,           // Whether to use batch normalization
      "dropout_rate": 0.2           // Dropout rate for regularization
    }
  }
}
```

You can customize these settings in your `fairness_config.json` file or through environment variables.

## Testing and Validation

After integrating with your ML framework, it's important to test and validate the fairness of your models:

```python
from src.fairness.bias_detector import BiasDetector

# Initialize bias detector
bias_detector = BiasDetector()

# Test fairness of model predictions
test_results = bias_detector.detect_outcome_bias(
    data=test_data.assign(predictions=model.predict(test_data[features])),
    attributes=[protected_attribute],
    outcome_columns=['predictions']
)

# Generate fairness report
fairness_report = bias_detector.generate_fairness_report(test_results)

# Visualize fairness metrics
bias_detector.visualize_fairness_metrics(test_results, save_path='fairness_metrics.png')

# Print summary
print(f"Bias detected: {fairness_report['summary']['bias_detected']}")
for finding in fairness_report['summary'].get('significant_findings', []):
    print(f"  - {finding['attribute']}: {finding['metric']} = {finding['value']:.4f} (threshold: {finding['threshold']:.4f})")

# Get mitigation recommendations if bias is detected
if fairness_report['summary']['bias_detected']:
    recommendations = fairness_mitigation.get_mitigation_recommendations(fairness_report)
    print("\nRecommended mitigation strategies:")
    for category, strategies in recommendations.items():
        print(f"\n{category.upper()}:")
        for strategy in strategies:
            print(f"  - {strategy['strategy']}: {strategy['description']} (Severity: {strategy['severity']})")
```

For more detailed testing and validation procedures, see the [Testing and Validation Guide](./testing_validation.md).

---

This guide provides a starting point for integrating the CustomerAI fairness framework with various ML frameworks. For specific implementation details or assistance with custom integrations, please contact the CustomerAI team at support@customerai.example.com.
