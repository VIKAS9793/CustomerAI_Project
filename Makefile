# Makefile for CustomerAI Project

.PHONY: all clean test lint run coverage security docker

# Default target
all: lint test coverage

# Clean build files
.PHONY: clean
clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Install dependencies
.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Run tests
.PHONY: test
test:
	pytest tests/ --cov=src

# Run linting
.PHONY: lint
lint:
	flake8 src/ tests/
	black --check src/ tests/
	mypy src/

# Run coverage
.PHONY: coverage
coverage:
	pytest --cov=src --cov-report=term-missing

# Run security checks
.PHONY: security
security:
	safety check
	pip-audit

# Run the application
.PHONY: run
run:
	python src/app.py

# Build Docker image
.PHONY: docker
docker:
	docker build -t customerai:latest .

# Run Docker container
docker-run:
	docker run -p 8000:8000 customerai:latest

# Run Docker with compose
docker-compose:
	docker-compose up --build

# Stop Docker containers
docker-down:
	docker-compose down

# Show help
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make clean       - Clean build files"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make coverage    - Run coverage report"
	@echo "  make security    - Run security checks"
	@echo "  make run         - Run the application"
	@echo "  make docker      - Build Docker image"
	@echo "  make docker-run  - Run Docker container"
	@echo "  make docker-down - Stop Docker containers"
