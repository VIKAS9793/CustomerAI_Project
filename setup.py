"""
CustomerAI project setup configuration.
"""

from setuptools import find_packages, setup

setup(
    name="customerai",
    version="1.0.0",
    description="A comprehensive AI-powered customer service platform",
    author="Vikas Sahani",
    author_email="vikas@example.com",
    packages=find_packages(),
    package_data={
        "customerai": ["py.typed", "**/*.pyi"],
    },
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.68.0",
        "pydantic>=1.8.2",
        "spacy>=3.1.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.2",
        "transformers>=4.9.0",
        "torch>=1.9.0",
        "streamlit>=0.85.0",
        "plotly>=5.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "isort>=5.9.3",
            "mypy>=0.910",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
