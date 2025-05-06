# Dependencies Overview

Last updated: May 6, 2025

## Core Dependencies

### Python Environment
- Python 3.10 (standardized across all components)
- pip 23.0 or later
- virtualenv or conda for environment management

### Key Libraries
- argon2-cffi-bindings==23.1.0
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- torch==2.1.0
- transformers==4.35.0
- fastapi==0.104.0
- pydantic==2.4.2

### Development Tools
- mypy: Type checking
- ruff: Linting
- black: Code formatting
- pytest: Testing framework
- pre-commit: Git hooks

## Cloud Dependencies

### AWS
- boto3==1.28.0
- aws-cdk-lib==2.100.0

### Azure
- azure-storage-blob==12.18.0
- azure-cosmos==4.5.1

### GCP
- google-cloud-storage==2.10.0
- google-cloud-bigquery==3.11.0

## Monitoring & Observability
- prometheus-client==0.17.0
- opentelemetry-api==1.20.0
- grafana-api==2.0.0

## Web Dashboard
- Node.js 18.x LTS
- React 18
- TypeScript 5.0

## Version Management
All dependencies are pinned to specific versions to ensure consistency across environments. Updates are managed through our automated dependency update workflow.
