# Dependencies and Third-Party Licenses

This document lists the dependencies and third-party components used in the CustomerAI Insights Platform, along with their respective licenses. The MIT License applied to this project covers only the original code created by Vikas Sahani, not these dependencies.

## Cloud Provider SDKs

### AWS SDK for Python (Boto3)
- **License**: Apache License 2.0
- **Website**: https://github.com/boto/boto3
- **Copyright**: Copyright Amazon.com, Inc. or its affiliates

### Azure SDK for Python
- **License**: MIT License
- **Website**: https://github.com/Azure/azure-sdk-for-python
- **Copyright**: Copyright (c) Microsoft Corporation

### Google Cloud SDK for Python
- **License**: Apache License 2.0
- **Website**: https://github.com/googleapis/google-cloud-python
- **Copyright**: Copyright Google LLC

## AI/ML Frameworks and Libraries

### JAX
- **License**: Apache License 2.0
- **Website**: https://github.com/google/jax
- **Copyright**: Copyright 2018 Google LLC

### Ray
- **License**: Apache License 2.0
- **Website**: https://github.com/ray-project/ray
- **Copyright**: Copyright (c) 2019 The Ray Team

### MLflow
- **License**: Apache License 2.0
- **Website**: https://github.com/mlflow/mlflow
- **Copyright**: Copyright (c) MLflow Project

### Hugging Face Transformers
- **License**: Apache License 2.0
- **Website**: https://github.com/huggingface/transformers
- **Copyright**: Copyright 2018- The Hugging Face Team

### DeepSpeed
- **License**: MIT License
- **Website**: https://github.com/microsoft/DeepSpeed
- **Copyright**: Copyright (c) Microsoft Corporation

### PyTorch
- **License**: BSD License
- **Website**: https://github.com/pytorch/pytorch
- **Copyright**: Copyright (c) 2016-     Facebook, Inc

## Kubernetes & Orchestration

### Kubernetes Python Client
- **License**: Apache License 2.0
- **Website**: https://github.com/kubernetes-client/python
- **Copyright**: Copyright The Kubernetes Authors

### Kubeflow Pipelines
- **License**: Apache License 2.0
- **Website**: https://github.com/kubeflow/pipelines
- **Copyright**: Copyright 2018 The Kubeflow Authors

### Seldon Core
- **License**: Apache License 2.0
- **Website**: https://github.com/SeldonIO/seldon-core
- **Copyright**: Copyright 2017-2020 Seldon Technologies Ltd

## Observability

### OpenTelemetry
- **License**: Apache License 2.0
- **Website**: https://github.com/open-telemetry/opentelemetry-python
- **Copyright**: Copyright The OpenTelemetry Authors

### Jaeger Client
- **License**: Apache License 2.0
- **Website**: https://github.com/jaegertracing/jaeger-client-python
- **Copyright**: Copyright (c) 2016 Uber Technologies, Inc.

### Prometheus Client
- **License**: Apache License 2.0
- **Website**: https://github.com/prometheus/client_python
- **Copyright**: Copyright (c) 2015 The Prometheus Authors

## Python Standard Library Dependencies

The project uses various modules from the Python Standard Library, which is distributed under the Python Software Foundation License.

- threading
- logging
- time
- typing
- queue
- functools
- statistics
- datetime
- enum
- json
- random

## Note on Dependencies

When using this project with cloud provider SDKs:

1. You must comply with the license terms of each cloud provider SDK you use.
2. This project does not distribute cloud provider SDKs; you need to install them separately.
3. The MIT License of this project does not override or modify the licenses of these dependencies.

## License Disclaimer

The copyright and license notices included in the CustomerAI Insights Platform source files apply ONLY to the original code written by the authors of this project, and NOT to any third-party components or dependencies.

If you believe any copyright or license information is incorrect or missing, please contact the project owner.
