# CustomerAI Insights Platform - Cloud Infrastructure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive cloud infrastructure framework that provides a unified interface for building and deploying customer analysis solutions across major cloud providers (AWS, Azure, GCP).

## Overview

The CustomerAI Insights Platform enables businesses to harness the power of AI for customer analytics while maintaining the highest standards of security, compliance, and governance. It provides a flexible, multi-cloud architecture that supports rapid development and deployment of customer analysis applications.

## Key Features

### Cloud Integration
- **Multi-cloud Support**: Deploy on AWS, Azure, or GCP without vendor lock-in
- **Infrastructure as Code**: Terraform modules for all supported cloud platforms
- **Containerized Deployment**: Docker and Kubernetes integration

### AI and Machine Learning
- **LLM Integration**: Flexible integration with leading Large Language Models
  - OpenAI GPT-4o: State-of-the-art general purpose model
  - Anthropic Claude 3.5 Sonnet: Advanced model with financial compliance capabilities
  - Google Gemini 1.5 Pro: Multimodal AI with reasoning capabilities
- **Custom ML Models**: Train and deploy specialized models for customer analytics
- **Batch and Real-time Inference**: Support for both batch processing and real-time API

### Human-in-the-Loop Framework
- **AI Oversight System**: Tiered human review system for AI-generated content
- **Review Dashboard**: Web interface for human reviewers to evaluate AI outputs
- **Feedback Loops**: Human feedback incorporated into model improvements
- **SLA Management**: Configurable service level agreements for review turnaround time

### AI Safety and Governance
- **Model Cards**: Comprehensive documentation of all models following industry standards
- **LLM Guardrails**: Robust protection against harmful content, prompt injection, and other risks
- **Responsible AI Framework**: Comprehensive governance structure aligned with NIST AI RMF and EU AI Act
- **Enterprise-Ready Fairness Framework**: 
  - Advanced bias detection with multiple fairness metrics (disparate impact, statistical parity, equal opportunity, predictive parity)
  - Statistical significance testing with configurable thresholds for reliable bias detection
  - Interactive fairness visualization dashboard with memory-efficient data handling for large datasets
  - Comprehensive bias mitigation strategies:
    - Pre-processing techniques (reweighing, balanced sampling)
    - Post-processing adjustments (equalized odds, calibration)
    - ML framework integration for adversarial debiasing (TensorFlow, PyTorch)
  - Detailed fairness reporting with severity classifications and actionable insights
  - RESTful API endpoints for fairness analysis, dataset upload, and mitigation strategy application
  - Flexible configuration system for organization-specific customization
  - Industry-specific presets for financial services, healthcare, and other regulated sectors
  - Comprehensive implementation guides for real-world deployment

### Security and Compliance
- **Data Encryption**: End-to-end encryption for sensitive data
- **Role-based Access Control**: Granular permissions for different user types
- **Audit Logging**: Comprehensive logging of all system activities
- **Compliance Certifications**: Designed for GDPR, CCPA, SOC 2, and industry-specific regulations

### Observability
- **Performance Monitoring**: Real-time dashboards for system performance
- **Model Metrics**: Tracking of model accuracy, drift, and other quality indicators
- **Alerting System**: Proactive notifications of potential issues
- **Distributed Tracing**: End-to-end tracing of requests through the system

## Use Cases

The CustomerAI Insights Platform is designed for multiple industries:

### Financial Services
- Regulatory compliance monitoring with human oversight
- Customer sentiment analysis with explainable AI
- Risk assessment with model cards for regulatory transparency
- Fraud detection with protected group fairness monitoring

### Retail
- Customer journey analysis with multimodal processing
- Product recommendation systems with intervention capabilities
- Customer support automation with human escalation paths
- Voice of customer analytics with guard rails against misrepresentation

### Healthcare
- Patient experience analysis with PHI protection
- Clinical document analysis with medical expert oversight
- Care quality monitoring with governance documentation
- Patient engagement systems with safety verification

### Telecommunications
- Network experience analysis with privacy safeguards
- Churn prediction with demographic fairness monitoring
- Customer support optimization with oversight protections
- Service personalization with explainability requirements

### SaaS Companies
- User behavior analysis with strong user privacy controls
- Feature usage tracking with implementation explainability
- Feedback processing with bias mitigation
- Customer success prediction with human verification

## Getting Started

### Prerequisites
- Python 3.12
- Docker and Docker Compose
- AWS, Azure, or GCP account (depending on deployment target)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VIKAS9793/CustomerAI_Project.git
cd CustomerAI_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

4. Run the development server:
```bash
python -m uvicorn main:app --reload
```

### Docker Deployment

For production deployment:

```bash
docker-compose up -d
```

## Human-in-the-Loop Configuration

The human review system provides tiered oversight for AI-generated content:

1. Configure review thresholds in `config/review_config.json`
2. Set up reviewer roles and permissions in the admin dashboard
3. Access the review interface at `http://your-deployment/review`
4. Connect notification systems (Slack, Email) in the `.env` file

Learn more in the [Human Review Documentation](docs/HUMAN_REVIEW.md).

## LLM Integration

The CustomerAI Insights Platform includes a flexible LLM integration system that supports multiple providers:

- **OpenAI GPT-4o**: State-of-the-art general purpose model for text generation and analysis
- **Anthropic Claude 3.5 Sonnet**: Advanced model with strong performance in financial compliance
- **Google Gemini 1.5 Pro**: Powerful multimodal AI model with reasoning capabilities

The system allows developers to:

1. **Configure multiple LLM providers** through a simple JSON configuration file
2. **Select models based on specific use cases** (sentiment analysis, document processing, etc.)
3. **Implement compliance requirements** for different regulatory environments
4. **Fallback mechanisms** when primary models are unavailable
5. **Cost management** through intelligent routing and caching

> **Disclaimer for Developers**: The LLM integration system is designed to be highly customizable according to your specific business requirements and use cases. Developers have complete flexibility to configure token limits, model selection, temperature settings, system prompts, and other parameters for each provider. This allows for optimizing different aspects such as cost management, inference speed, output quality, and regulatory compliance based on your organization's priorities.

For implementation details, see [LLM Integration Guide](docs/LLM_INTEGRATION.md).

## AI Governance

This platform implements a comprehensive AI governance framework aligned with industry standards:

- **NIST AI Risk Management Framework**: Following the structured approach to AI risk
- **EU AI Act Compliance**: Meeting requirements for high-risk AI systems
- **Model Documentation**: Standardized model cards for all deployed models
- **Responsible AI Principles**: Comprehensive implementation of fairness, transparency, and safety

For details, see [AI Governance Documentation](docs/AI_GOVERNANCE.md).

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License applies only to the original code in this project, not to any third-party dependencies. See [DEPENDENCIES.md](DEPENDENCIES.md) for details on third-party components and their licenses.

## Author

**Vikas Sahani**
- GitHub: [https://github.com/VIKAS9793](https://github.com/VIKAS9793)
- Email: vikassahani17@gmail.com

## Acknowledgments

- This project implements industry best practices for cloud service integration and resilience patterns
- Inspired by enterprise-grade connection pooling and retry mechanisms

### Key Technologies

- **Python 3.12**: Latest stable Python version with improved performance and features
- **Modern AI & ML Stack**:
  - **JAX**: High-performance numerical computing with automatic differentiation
  - **Ray**: Distributed computing framework for scaling AI/ML workloads
  - **MLflow**: Platform for managing ML lifecycle including tracking, deployment
  - **Hugging Face Transformers**: State-of-the-art NLP models and tools
  - **DeepSpeed**: Optimization library for large-scale model training
- **Kubernetes Integration**: 
  - Native deployment on Kubernetes clusters
  - Kubeflow Pipelines for ML workflows
  - Seldon Core for model serving
- **Observability**:
  - OpenTelemetry for distributed tracing
  - Prometheus & Grafana for metrics visualization
  - Jaeger for end-to-end tracing
- **Cloud Provider SDKs**: Latest versions of AWS, Azure, and GCP Python SDKs

## Table of Contents
- [Quick Start Guide](#quick-start-guide)
- [Cloud and Deployment Guide](#cloud-and-deployment-guide)
- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Error Handling](#error-handling)
  - [Retry Mechanisms](#retry-mechanisms)
  - [Connection Pooling](#connection-pooling)
  - [Performance Metrics](#performance-metrics)
  - [Security and Encryption](#security-and-encryption)
  - [IAM Integration](#iam-integration)
  - [Observability](#observability)
  - [Chaos Testing](#chaos-testing)
  - [Load Testing](#load-testing)
  - [Deployment Automation](#deployment-automation)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
- [LLM Integration](#llm-integration)

## Quick Start Guide

Get up and running with CustomerAI Insights Platform in minutes:

### 1. Prerequisites
- Python 3.12 
- pip (Python package manager)
- Git

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/VIKAS9793/CustomerAI_Project.git
cd CustomerAI_Project

# Create and activate a virtual environment (recommended)
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Optional: Install GPU support (for CUDA-enabled systems)
pip install -r requirements-gpu.txt
```

### 3. Configuration

```bash
# Create environment configuration
cp env.example .env

# Edit .env file with your settings
# For Windows
notepad .env
# For macOS/Linux
nano .env
```

### 4. Run the Basic Example

```python
# examples/basic_usage.py
from cloud.config import CloudProvider
from cloud.factory import CloudFactory

# Create a cloud factory
factory = CloudFactory()

# Get a storage service for AWS
storage = factory.get_storage_service(CloudProvider.AWS)

# Use the storage service
bucket_name = "my-test-bucket"
storage.create_bucket(bucket_name)
print(f"Bucket {bucket_name} created successfully!")
```

Run the example:
```bash
python examples/basic_usage.py
```

### 5. Start the Demo Dashboard

```bash
# Start the dashboard
python app.py
```

Visit http://localhost:8000 in your browser.

### 6. Next Steps

- See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for more detailed examples
- Configure specific cloud providers in [CONFIGURATION.md](CONFIGURATION.md)
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues

## Cloud and Deployment Guide

This section provides instructions for deploying the CustomerAI Insights Platform to various environments for testing, development, and production.

### Docker Deployment

Deploy the full platform stack using Docker Compose:

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

The Docker Compose setup includes:
- API server (Port 8000)
- Dashboard UI (Port 8501)
- PostgreSQL database
- Redis for caching
- Prometheus for metrics
- Nginx for reverse proxy

### Cloud Provider Deployment

#### AWS Deployment

1. **Set up AWS credentials**
   ```bash
   # Configure AWS CLI
   aws configure
   
   # Or set environment variables
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_DEFAULT_REGION=us-east-1
   ```

2. **Deploy using Terraform**
   ```bash
   cd templates/aws
   terraform init
   terraform apply
   ```

#### Azure Deployment

1. **Set up Azure credentials**
   ```bash
   # Login with Azure CLI
   az login
   
   # Set subscription
   az account set --subscription <subscription-id>
   ```

2. **Deploy using Terraform**
   ```bash
   cd templates/azure
   terraform init
   terraform apply
   ```

#### GCP Deployment

1. **Set up GCP credentials**
   ```bash
   # Login with gcloud CLI
   gcloud auth login
   
   # Set project
   gcloud config set project <project-id>
   ```

2. **Deploy using Terraform**
   ```bash
   cd templates/gcp
   terraform init
   terraform apply
   ```

### CI/CD Pipeline

The project includes GitHub Actions workflows for continuous integration and deployment:

```yaml
# .github/workflows/ci.yml
- test: Runs unit tests, linting, and type checking
- security-scan: Performs security analysis
- build: Builds Python package
- docker: Builds and pushes Docker image
- deploy-dev: Deploys to development environment
- deploy-prod: Deploys to production environment
```

To use the CI/CD pipeline:
1. Fork the repository
2. Enable GitHub Actions
3. Set up required secrets in repository settings
4. Push to `develop` branch for dev deployment or `main` for production

### Testing in Virtual Environments

For isolated testing:

```bash
# Create testing environment
python -m venv test-env
source test-env/bin/activate  # Linux/macOS
test-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Local Kubernetes Deployment

For local Kubernetes testing with minikube:

```bash
# Start minikube
minikube start

# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods

# Forward ports
kubectl port-forward svc/customerai-api 8000:8000
```

Visit [CONFIGURATION.md](CONFIGURATION.md) for detailed settings for each deployment option.

## Features

- **Advanced LLM Integration**: Flexible system for working with multiple LLM providers (OpenAI, Anthropic, Google) with model specialization and failover capabilities
- **Financial Services Optimization**: LLMs configured for financial compliance and specialized financial use cases
- **Multi-cloud Support**: Unified interface for AWS, Azure, and GCP through factory pattern
- **Standardized Error Handling**: Consistent error types and proper logging
- **Resilient Retry Mechanism**: Configurable retry policies with various backoff strategies
- **Performance Metrics**: Real-time tracking of cloud operation performance
- **Connection Pooling**: Efficient management of cloud service connections
- **Enterprise Security**: Data encryption, key rotation, and IAM integration
- **Observability**: Prometheus integration for metrics collection and monitoring
- **Chaos Testing**: Formalized methodology for resilience testing
- **Load Testing**: Tools for scalability validation and performance benchmarking
- **Deployment Automation**: Terraform integration for infrastructure as code

## Installation

```bash
# Clone the repository
git clone https://github.com/VIKAS9793/CustomerAI_Project.git

# Install dependencies
pip install -r requirements.txt

# Install cloud-specific dependencies as needed
pip install boto3  # For AWS
pip install azure-storage-blob azure-cosmos  # For Azure
pip install google-cloud-storage google-cloud-firestore  # For GCP
```

## Architecture

The cloud infrastructure is designed with a layered architecture for maximum flexibility and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                   Your Application                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   Cloud Factory Layer                       │
│  (Provides unified interface for different cloud providers) │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                 Resilience Layer                            │
│ ┌─────────────┐ ┌─────────────┐ ┌────────────────┐          │
│ │ Retry       │ │ Connection  │ │ Error          │          │
│ │ Mechanisms  │ │ Pooling     │ │ Handling       │          │
│ └─────────────┘ └─────────────┘ └────────────────┘          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│               Provider-specific Implementations              │
│ ┌─────────────┐ ┌─────────────┐ ┌────────────────┐          │
│ │ AWS         │ │ Azure       │ │ GCP            │          │
│ │ Services    │ │ Services    │ │ Services       │          │
│ └─────────────┘ └─────────────┘ └────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Error Handling

The framework provides a standardized error handling system with consistent error types across all cloud providers.

```python
from cloud.errors import CloudError, CloudResourceNotFoundError, handle_cloud_error
from cloud.config import CloudProvider

# Example 1: Catching specific cloud errors
try:
    # Cloud operation that might fail
    storage_client.get_object(bucket="my-bucket", key="non-existent.txt")
except CloudResourceNotFoundError as e:
    # Handle specifically when a resource is not found
    print(f"Resource not found: {e.message}")
    e.log()  # Log with appropriate level and format
except CloudError as e:
    # Handle any cloud error
    print(f"Cloud operation failed: {e.message}")
    print(f"Error category: {e.category.value}")
    print(f"Provider: {e.provider.value if e.provider else 'unknown'}")
    print(f"Error details: {e.details}")
    
    # Log the error with custom level
    import logging
    e.log(level=logging.WARNING)

# Example 2: Using the error handler to convert provider-specific exceptions
import boto3
from botocore.exceptions import ClientError

try:
    # Direct AWS SDK call
    s3_client = boto3.client('s3')
    s3_client.get_object(Bucket="my-bucket", Key="non-existent.txt")
except ClientError as aws_error:
    # Convert to standardized error
    cloud_error = handle_cloud_error(aws_error, CloudProvider.AWS, "s3", "get_object")
    
    # Now you can handle the standardized error
    print(f"Standardized error: {cloud_error.message}")
    print(f"Error category: {cloud_error.category.value}")
    
    # Error details are preserved
    print(f"AWS error code: {cloud_error.error_code}")
```

### Retry Mechanisms

The framework includes configurable retry utilities to handle transient errors in cloud operations.

```python
from cloud.retry import retry_with_backoff, BackoffStrategy, CircuitBreaker
from cloud.config import CloudProvider
from cloud.errors import CloudError

# Example 1: Simple retry with exponential backoff
@retry_with_backoff(
    max_attempts=3,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    base_delay=1.0,
    max_delay=10.0,
    retryable_errors=[CloudError]
)
def upload_document(bucket, key, data):
    # Cloud operation that might fail with transient errors
    storage_client.put_object(bucket=bucket, key=key, data=data)
    return True

# Example 2: Retry with jitter and specific error types
@retry_with_backoff(
    max_attempts=5,
    backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
    base_delay=0.5,
    max_delay=30.0,
    retryable_errors=[CloudServiceUnavailableError, CloudNetworkError]
)
def query_database(table, query):
    # Database operation that might experience transient failures
    return database_client.query(table=table, query=query)

# Example 3: Circuit breaker pattern to prevent cascading failures
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    failure_status_codes=[500, 503]
)

@retry_with_backoff(
    max_attempts=3,
    backoff_strategy=BackoffStrategy.LINEAR,
    base_delay=2.0,
    circuit_breaker=circuit_breaker
)
def process_payment(payment_id, amount):
    # Payment processing that should fail fast if service is down
    return payment_service.process(payment_id=payment_id, amount=amount)
```

### Connection Pooling

Efficiently manage connections to cloud services to improve performance and resource utilization.

```python
from cloud.utils.pool import connection_pooled, get_pool_manager
from cloud.config import CloudProvider

# Example 1: Using the connection pooling decorator
@connection_pooled(provider=CloudProvider.AWS, service="s3")
def upload_file(bucket, key, data, connection_factory=None, connection=None):
    # The connection is automatically acquired from the pool
    # and released back when the function completes
    connection.put_object(Bucket=bucket, Key=key, Body=data)
    return True

# Create a connection factory
def s3_connection_factory():
    import boto3
    return boto3.client('s3')

# Use the pooled function
result = upload_file(
    bucket="my-bucket", 
    key="my-key", 
    data="my-data", 
    connection_factory=s3_connection_factory
)

# Example 2: Custom pool configuration
def dynamodb_connection_factory():
    import boto3
    return boto3.client('dynamodb')

# Get the global pool manager
pool_manager = get_pool_manager()

# Configure a specific pool
pool = pool_manager.get_pool(
    provider=CloudProvider.AWS,
    service="dynamodb",
    factory=dynamodb_connection_factory,
    max_size=20,     # Maximum number of connections
    min_size=5,      # Minimum connections to keep ready
    max_age=3600.0,  # Maximum connection lifetime in seconds
    max_idle=300.0,  # Maximum idle time before recycling
    max_uses=1000,   # Maximum times a connection can be used
    block_timeout=5.0  # Timeout for waiting for a connection
)

# Manually acquire and release a connection
connection = pool.acquire()
try:
    # Use the connection
    result = connection.query(
        TableName="my-table",
        KeyConditionExpression="id = :id",
        ExpressionAttributeValues={":id": {"S": "item-1"}}
    )
finally:
    # Always release the connection back to the pool
    pool.release(connection)

# Close pools when shutting down application
pool_manager.close_all_pools()
```

### Performance Metrics

Track and analyze the performance of cloud operations to identify bottlenecks and optimize performance.

```python
from cloud.utils.metrics import (
    track_performance, get_metrics_manager, 
    record_metric, MetricType
)
from cloud.config import CloudProvider

# Example 1: Using the performance tracking decorator
@track_performance(provider=CloudProvider.AWS, service="s3", operation="upload_file")
def upload_file(bucket, key, data):
    # This operation will be tracked automatically
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    return True

# The decorator records:
# - Latency (execution time)
# - Success rate
# - Error rate
# - Retry counts (if used with retry decorator)

# Example 2: Manual metrics recording
def process_batch(items):
    start_time = time.time()
    success_count = 0
    
    for item in items:
        try:
            # Process item
            process_item(item)
            success_count += 1
        except Exception as e:
            # Record failure metric
            record_metric(
                provider=CloudProvider.AZURE,
                service="cosmos_db",
                operation="process_item",
                metric_type=MetricType.ERROR_RATE,
                value=1.0
            )
    
    # Record batch throughput
    record_metric(
        provider=CloudProvider.AZURE,
        service="cosmos_db",
        operation="process_batch",
        metric_type=MetricType.THROUGHPUT,
        value=len(items) / (time.time() - start_time)
    )
    
    # Record success rate
    record_metric(
        provider=CloudProvider.AZURE,
        service="cosmos_db",
        operation="process_batch",
        metric_type=MetricType.SUCCESS_RATE,
        value=success_count / len(items)
    )

# Example 3: Generating performance reports
metrics_manager = get_metrics_manager()

# Generate text report
report = metrics_manager.generate_report(
    provider=CloudProvider.AWS,
    service="dynamodb",
    period=3600,  # Last hour
    format_type="text"
)
print(report)

# Get raw metrics data for custom analysis
metrics_data = metrics_manager.get_metrics(
    provider=CloudProvider.AWS,
    service="dynamodb",
    aggregated=True
)

# Export metrics as JSON
import json
with open("dynamodb_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)
```

### Security and Encryption

Implement data encryption for cloud operations:

```python
from cloud.security.encryption import encrypt_data, decrypt_data, EncryptionAlgorithm, with_encryption

# Simple encryption/decryption
encrypted_data = encrypt_data("sensitive data", "my-key", EncryptionAlgorithm.AES_256_GCM)
decrypted_data, metadata = decrypt_data(encrypted_data)

# Use encryption decorator
@with_encryption(key_name="api-key", algorithm=EncryptionAlgorithm.AES_256_GCM)
def store_api_key(storage_client, key_name, api_key):
    storage_client.put_item(key_name, api_key)
```

### IAM Integration

Integrate with enterprise IAM solutions:

```python
from cloud.security.iam import with_auth, get_iam_manager

# Check permissions with decorator
@with_auth(permission="s3:PutObject", resource="my-bucket")
def upload_to_s3(s3_client, file_path, bucket, key, token_id):
    s3_client.upload_file(file_path, bucket, key)

# Load credentials from environment
from cloud.security.iam import load_credentials_from_environment
load_credentials_from_environment()
```

### Observability

Integrate with Prometheus for monitoring:

```python
from cloud.observability.prometheus import start_prometheus_server, with_prometheus_metrics

# Start Prometheus server
start_prometheus_server(port=9090)

# Add Prometheus metrics to functions
@with_prometheus_metrics(name="s3_upload", labels={"bucket": "my-bucket"})
def upload_to_s3(s3_client, file_path, bucket, key):
    s3_client.upload_file(file_path, bucket, key)
```

### Chaos Testing

Test resilience with controlled failures:

```python
from cloud.testing.chaos import get_chaos_engine, inject_failure, FailureMode, InjectionScope

# Enable chaos testing
chaos_engine = get_chaos_engine()
chaos_engine.enable()

# Inject specific failure
inject_failure(
    failure_mode=FailureMode.LATENCY,
    scope=InjectionScope.SERVICE,
    target="aws",
    target_secondary="s3",
    probability=0.2,
    parameters={"latency": 2.0, "jitter": 0.5}
)

# Run a standard chaos test
from cloud.testing.chaos import run_standard_chaos_test
experiment_id = run_standard_chaos_test(
    provider="aws",
    service="s3",
    duration=300,
    stagger=True
)
```

### Load Testing

Validate scalability and performance:

```python
from cloud.testing.load import run_load_test, LoadPattern

# Run a load test
result = run_load_test(
    target_function=upload_to_s3,
    args=(s3_client, file_path, bucket, key),
    target_rps=50,
    duration=300,
    pattern=LoadPattern.RAMP,
    pattern_params={"initial_rps": 1.0},
    num_threads=20
)

# Get results
summary = result.get_summary()
print(f"Throughput: {summary['throughput']} RPS")
print(f"P95 Latency: {summary['latency']['p95']} seconds")
print(f"Success Rate: {summary['success_rate']}%")

# Save results to file
result.save_to_file("load_test_results.json")
```

### Deployment Automation

Automate infrastructure deployment:

```python
from cloud.deployment.terraform import deploy_infrastructure, destroy_infrastructure

# Deploy infrastructure
outputs = deploy_infrastructure(
    provider="aws",
    template_name="s3_bucket",
    variables={
        "bucket_name": "my-data-bucket",
        "region": "us-east-1"
    },
    backend_config={
        "bucket": "terraform-state",
        "key": "s3/terraform.tfstate",
        "region": "us-east-1"
    }
)

# Destroy infrastructure
destroy_infrastructure("deployments/aws-s3_bucket-1234567890")
```

## Configuration

The framework can be configured through environment variables or programmatically:

### Environment Variables

```
# General Configuration
CLOUD_DEFAULT_PROVIDER=aws  # Default cloud provider (aws, azure, gcp)
CLOUD_LOGGING_LEVEL=INFO    # Logging level (DEBUG, INFO, WARNING, ERROR)

# Retry Configuration
CLOUD_RETRY_MAX_ATTEMPTS=3        # Maximum retry attempts
CLOUD_RETRY_BACKOFF_STRATEGY=exponential  # Backoff strategy
CLOUD_RETRY_BASE_DELAY=1.0        # Base delay in seconds
CLOUD_RETRY_MAX_DELAY=30.0        # Maximum delay in seconds
CLOUD_RETRY_JITTER_FACTOR=0.1     # Jitter factor (0.0-1.0)

# Connection Pool Configuration
CLOUD_POOL_DEFAULT_MAX_SIZE=10    # Maximum connections per pool
CLOUD_POOL_DEFAULT_MIN_SIZE=1     # Minimum connections per pool
CLOUD_POOL_MAX_AGE=3600           # Maximum connection age in seconds
CLOUD_POOL_MAX_IDLE=300           # Maximum idle time in seconds
CLOUD_POOL_MAX_USES=1000          # Maximum uses per connection
CLOUD_POOL_BLOCK_TIMEOUT=30.0     # Timeout for acquiring connections

# Metrics Configuration
CLOUD_METRICS_ENABLED=true              # Enable metrics collection
CLOUD_METRICS_RETENTION_PERIOD=3600     # Metrics retention in seconds
CLOUD_METRICS_AGGREGATION_INTERVAL=60   # Aggregation interval in seconds
CLOUD_METRICS_MAX_SAMPLES=1000          # Maximum samples per metric
```

### Programmatic Configuration

```python
from cloud.config import CloudConfig
from cloud.utils.metrics import get_metrics_manager
from cloud.utils.pool import get_pool_manager

# Configure retry defaults
config = CloudConfig()
config.set_retry_defaults(
    max_attempts=5,
    backoff_strategy="exponential_jitter",
    base_delay=0.5,
    max_delay=20.0
)

# Configure metrics system
metrics_manager = get_metrics_manager()
metrics_manager.set_retention_period(7200)  # 2 hours
metrics_manager.set_aggregation_interval(30)  # 30 seconds
metrics_manager.set_max_samples(2000)

# Configure pool defaults
pool_manager = get_pool_manager()
pool_manager.set_default_config(
    max_size=20,
    min_size=2,
    max_age=1800.0,
    max_idle=300.0
)
```

## Best Practices

### Error Handling

1. **Use Specific Error Types**: Catch specific error types rather than generic CloudError
   ```python
   try:
       # Operation
   except CloudResourceNotFoundError:
       # Handle missing resources
   except CloudAuthorizationError:
       # Handle permission issues
   except CloudError:
       # Handle any other cloud error
   ```

2. **Log with Context**: Always log errors with contextual information
   ```python
   except CloudError as e:
       e.log()  # Uses built-in contextual logging
   ```

### Retry Strategies

1. **Match Retry to Error Type**: Use different retry strategies based on the error
   ```python
   # Network errors: More retries with exponential backoff
   @retry_with_backoff(max_attempts=5, backoff_strategy=BackoffStrategy.EXPONENTIAL)
   
   # Rate limiting: Longer delays with exponential backoff
   @retry_with_backoff(max_attempts=3, base_delay=2.0, max_delay=60.0)
   
   # Service errors: Circuit breaker to prevent cascading failures
   @retry_with_backoff(circuit_breaker=CircuitBreaker(failure_threshold=3))
   ```

2. **Don't Retry Non-Idempotent Operations**: Only retry operations that can be repeated safely
   ```python
   # Safe to retry (idempotent)
   @retry_with_backoff(max_attempts=3)
   def get_object(bucket, key):
       return storage_client.get_object(bucket=bucket, key=key)
   
   # Careful with non-idempotent operations
   @retry_with_backoff(max_attempts=1)  # No retries by default
   def create_unique_resource(name, data):
       return service_client.create_resource(name=name, data=data)
   ```

### Connection Pooling

1. **Size Pools Appropriately**: Match pool size to workload and resource constraints
   ```python
   # High throughput service
   pool_manager.get_pool(
       provider=CloudProvider.AWS,
       service="dynamodb",
       factory=dynamodb_factory,
       max_size=50,
       min_size=10
   )
   
   # Low volume service
   pool_manager.get_pool(
       provider=CloudProvider.AWS,
       service="sns",
       factory=sns_factory,
       max_size=5,
       min_size=1
   )
   ```

2. **Set Appropriate Timeouts**: Configure timeouts based on operation importance
   ```python
   # Critical path operations - fail fast
   pool_manager.get_pool(
       provider=CloudProvider.AWS,
       service="payment_service",
       factory=payment_factory,
       block_timeout=2.0  # Wait max 2 seconds for a connection
   )
   
   # Background operations - can wait longer
   pool_manager.get_pool(
       provider=CloudProvider.AWS,
       service="analytics",
       factory=analytics_factory,
       block_timeout=30.0  # Can wait up to 30 seconds
   )
   ```

## Troubleshooting

### Common Issues and Solutions

#### Connection Pool Exhaustion

**Symptoms:**
- `CloudTimeoutError: Timeout waiting for available connection`
- Performance degradation under load

**Solutions:**
1. Increase the maximum pool size:
   ```python
   pool_manager.get_pool(
       provider=CloudProvider.AWS,
       service="dynamodb",
       factory=dynamodb_factory,
       max_size=50  # Increase from default
   )
   ```

2. Decrease connection hold time:
   ```python
   # Avoid:
   connection = pool.acquire()
   # Long processing...
   pool.release(connection)
   
   # Better:
   connection = pool.acquire()
   data = connection.get_data()  # Get what you need quickly
   pool.release(connection)
   # Long processing on the obtained data...
   ```

3. Check for connection leaks:
   ```python
   # Monitor current pool size vs in-use connections
   pool_status = pool_manager.get_pool_status(
       provider=CloudProvider.AWS,
       service="dynamodb"
   )
   print(f"Total connections: {pool_status['size']}")
   print(f"In-use connections: {len(pool_status['in_use'])}")
   ```

#### Excessive Retries

**Symptoms:**
- Operations taking longer than expected
- Log full of retry messages

**Solutions:**
1. Adjust retry parameters:
   ```python
   @retry_with_backoff(
       max_attempts=3,  # Reduce max attempts
       base_delay=0.5,  # Start with shorter delays
       backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER  # Add jitter
   )
   ```

2. Use circuit breaker:
   ```python
   circuit_breaker = CircuitBreaker(
       failure_threshold=5,
       recovery_timeout=30.0,
   )
   
   @retry_with_backoff(circuit_breaker=circuit_breaker)
   def call_service():
       # Implementation
   ```

3. Implement fallback mechanisms:
   ```python
   @retry_with_backoff(max_attempts=2)
   def get_data_with_fallback():
       try:
           return primary_service.get_data()
       except CloudError:
           # If primary fails after retries, use backup
           return backup_service.get_data()
   ```

#### Metric System Memory Growth

**Symptoms:**
- Increasing memory usage
- Slow performance of metrics reporting

**Solutions:**
1. Reduce metrics retention period:
   ```python
   metrics_manager = get_metrics_manager()
   metrics_manager.set_retention_period(1800)  # 30 minutes
   ```

2. Reduce max samples:
   ```python
   metrics_manager.set_max_samples(500)
   ```

3. Increase aggregation frequency:
   ```python
   metrics_manager.set_aggregation_interval(30)  # Aggregate every 30 seconds
   ```

## Security Considerations

### Authentication and Authorization

- **Credential Management**: Never hardcode credentials
  ```python
  # BAD
  client = boto3.client('s3', 
      aws_access_key_id='AKIAXXXXXXXXXXXXXXXX',
      aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  )
  
  # GOOD
  # Use environment variables, IAM roles, or secure credential stores
  client = boto3.client('s3')  # Uses credentials from environment or instance profile
  ```

- **Least Privilege**: Use minimal permissions
  ```python
  # Configure service accounts with only required permissions
  # Example: S3 bucket policy granting only read access
  {
      "Version": "2012-10-17",
      "Statement": [
          {
              "Effect": "Allow",
              "Action": [
                  "s3:GetObject",
                  "s3:ListBucket"
              ],
              "Resource": [
                  "arn:aws:s3:::my-bucket",
                  "arn:aws:s3:::my-bucket/*"
              ]
          }
      ]
  }
  ```

### Data Protection

- **Encryption**: Encrypt sensitive data
  ```python
  # Example: Enabling encryption for S3 uploads
  @connection_pooled(provider=CloudProvider.AWS, service="s3")
  def upload_sensitive_file(bucket, key, data, connection_factory=None, connection=None):
      connection.put_object(
          Bucket=bucket,
          Key=key,
          Body=data,
          ServerSideEncryption='AES256'  # Enable encryption
      )
  ```

- **Data Validation**: Validate all inputs
  ```python
  def store_user_data(user_id, data):
      # Validate input
      if not isinstance(user_id, str) or not user_id.isalnum():
          raise ValueError("Invalid user ID format")
      
      # Sanitize data
      sanitized_data = sanitize_user_data(data)
      
      # Store data
      return db_client.put_item(
          TableName="users",
          Item={"id": user_id, "data": sanitized_data}
      )
  ```

## Performance Optimization

### Connection Pooling Optimization

- **Pool Sizing**: Optimize based on workload
  ```python
  # Formula: max_size = peak_concurrent_operations * (1 + buffer_factor)
  # Example for a service handling 20 concurrent operations with 20% buffer:
  pool = pool_manager.get_pool(
      provider=CloudProvider.AWS,
      service="dynamodb",
      factory=dynamodb_factory,
      max_size=20 * (1 + 0.2),  # 24 connections
      min_size=5  # Keep 5 warm connections
  )
  ```

- **Connection Lifecycle**: Tune connection recycling parameters
  ```python
  pool = pool_manager.get_pool(
      provider=CloudProvider.AWS,
      service="dynamodb",
      factory=dynamodb_factory,
      max_age=1800.0,  # Recycle connections after 30 minutes
      max_idle=300.0,  # Recycle connections idle for 5 minutes
      max_uses=1000    # Recycle after 1000 operations
  )
  ```

### Batch Operations

- **Use Batch APIs**: Combine multiple operations
  ```python
  # Instead of multiple single operations:
  for item in items:
      db_client.put_item(TableName="table", Item=item)  # Inefficient
  
  # Use batch operations:
  @connection_pooled(provider=CloudProvider.AWS, service="dynamodb")
  def batch_write_items(items, connection_factory=None, connection=None):
      # Prepare batch request
      batch_items = {"table": [{"PutRequest": {"Item": item}} for item in items]}
      
      # Execute batch operation
      connection.batch_write_item(RequestItems=batch_items)
  ```

### Metrics to Monitor

- **Track Key Performance Indicators**:
  ```python
  # Custom metric for operation rate
  record_metric(
      provider=CloudProvider.AWS,
      service="s3",
      operation="upload_file",
      metric_type=MetricType.THROUGHPUT,
      value=files_processed_per_second
  )
  
  # Custom metric for data volume
  record_metric(
      provider=CloudProvider.AWS,
      service="s3",
      operation="upload_file",
      metric_type="data_volume",  # Custom metric type
      value=total_bytes_uploaded
  )
  ```

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License applies only to the original code in this project, not to any third-party dependencies. See [DEPENDENCIES.md](DEPENDENCIES.md) for details on third-party components and their licenses.

## Author

**Vikas Sahani**
- GitHub: [https://github.com/VIKAS9793](https://github.com/VIKAS9793)
- Email: vikassahani17@gmail.com

## Acknowledgments

- This project implements industry best practices for cloud service integration and resilience patterns
- Inspired by enterprise-grade connection pooling and retry mechanisms

### Key Technologies

- **Python 3.12**: Latest stable Python version with improved performance and features
- **Modern AI & ML Stack**:
  - **JAX**: High-performance numerical computing with automatic differentiation
  - **Ray**: Distributed computing framework for scaling AI/ML workloads
  - **MLflow**: Platform for managing ML lifecycle including tracking, deployment
  - **Hugging Face Transformers**: State-of-the-art NLP models and tools
  - **DeepSpeed**: Optimization library for large-scale model training
- **Kubernetes Integration**: 
  - Native deployment on Kubernetes clusters
  - Kubeflow Pipelines for ML workflows
  - Seldon Core for model serving
- **Observability**:
  - OpenTelemetry for distributed tracing
  - Prometheus & Grafana for metrics visualization
  - Jaeger for end-to-end tracing
- **Cloud Provider SDKs**: Latest versions of AWS, Azure, and GCP Python SDKs

## LLM Integration

The CustomerAI Insights Platform includes a flexible LLM integration system that supports multiple providers:

- **OpenAI GPT-4o**: State-of-the-art general purpose model for text generation and analysis
- **Anthropic Claude 3.5 Sonnet**: Advanced model with strong performance in financial compliance
- **Google Gemini 1.5 Pro**: Powerful multimodal AI model with reasoning capabilities

The system allows developers to:

1. **Configure multiple LLM providers** through a simple JSON configuration file
2. **Select models based on specific use cases** (sentiment analysis, document processing, etc.)
3. **Implement compliance requirements** for different regulatory environments
4. **Fallback mechanisms** when primary models are unavailable

### Example LLM Usage

```python
from cloud.ai.llm_manager import get_llm_manager

# Get the global LLM manager
llm_manager = get_llm_manager()

# Generate text with default model
response = await llm_manager.generate_text(
    prompt="Summarize the benefits of our premium credit card",
    temperature=0.7,
    max_tokens=500
)

# Use a specific LLM for compliance tasks
compliance_response = await llm_manager.generate_text(
    prompt="Check if this document meets regulatory requirements: ...",
    client_id="claude_sonnet",  # Use Claude for compliance checking
    system_prompt="You are a financial compliance expert",
    temperature=0.1
)

# Get embeddings for semantic search
embeddings = await llm_manager.get_embeddings(
    texts=["Document 1", "Document 2", "Document 3"],
    client_id="embeddings"  # Use dedicated embeddings model
)
```

### Configuring LLMs

LLMs can be configured through the `config/llm_config.json` file or programmatically:

```python
from cloud.ai.llm_provider import LLMProvider, LLMComplianceLevel, LLMConfig
from cloud.ai.llm_manager import get_llm_manager

# Get manager and register a new LLM client
llm_manager = get_llm_manager()

# Register a financial-specific client
llm_manager.register_financial_client(
    client_id="gemini_financial",
    provider=LLMProvider.GOOGLE,
    model_name="gemini-1.5-pro"
)

# Set as default
llm_manager.set_default_client("gemini_financial")
```

> **Disclaimer for Developers**: The LLM integration system is highly configurable according to your specific use cases. Developers can customize token limits, model selection, system prompts, temperature settings, and other parameters for each provider. Refer to the [CONFIGURATION.md](CONFIGURATION.md) document for detailed instructions on how to optimize LLM settings for different scenarios, including cost optimization, performance tuning, and compliance requirements.

See the [LLM Configuration](docs/llm_configuration.md) document for more details.