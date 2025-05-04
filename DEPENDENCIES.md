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