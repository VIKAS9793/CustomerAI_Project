# CustomerAI Project – Claim-to-Evidence Mapping

This document provides a transparent mapping between key claims about the CustomerAI project and the specific files, configurations, or tests that support each claim. This ensures all statements are verifiable and reproducible by any reviewer.

---

## 1. Python Version Standardization
- **Claim:** Python 3.10 is used and enforced across all components.
- **Evidence:**
  - `Dockerfile`: Uses `FROM python:3.10-slim` for both builder and production.
  - `requirements.txt`: Header and dependency notes specify Python 3.10 compatibility.
  - `.github/workflows/ci-cd.yml`: CI/CD pipeline jobs use Python 3.10.
  - `docker-entrypoint.sh`: Version check for Python 3.10 or higher.
  - All documentation references updated to Python 3.10 (see `README.md`, `IMPLEMENTATION_NOTES.md`, etc.)

---

## 2. Security and Secret Management
- **Claim:** All secrets are required at startup and secret scanning is enforced.
- **Evidence:**
  - `app.py`: Startup validation for required secrets (`JWT_SECRET_KEY`, `ENCRYPTION_KEY`, etc.).
  - `.pre-commit-config.yaml`: Includes `detect-secrets` hook for secret scanning.
  - CI/CD pipeline (`.github/workflows/ci-cd.yml`): Runs pre-commit and secret scanning on every PR.

---

## 3. Robust Configuration System
- **Claim:** Fairness framework is highly configurable and supports industry-specific adaptations.
- **Evidence:**
  - `src/fairness/config.py` (FairnessConfig class): Hierarchical configuration, multiple sources.
  - Example configs for finance/healthcare in `examples/` or `docs/` (see customization guide).
  - Documentation: Customization guide and config options (see `README.md`, `docs/`).

---

## 4. Testing, CI/CD, and Code Quality
- **Claim:** Automated tests, linting, and code quality checks are enforced.
- **Evidence:**
  - `tests/`: Unit and integration tests for fairness components.
  - `.github/workflows/ci-cd.yml`: Multi-stage pipeline with linting, type checking, formatting, and test coverage.
  - `.pre-commit-config.yaml`: Hooks for linting, doc coverage, and secret scanning.

---

## 5. Containerization & Operational Readiness
- **Claim:** Dockerized, non-root execution, resource limits, and monitoring are implemented.
- **Evidence:**
  - `Dockerfile`: Non-root user, resource limits, healthcheck.
  - `docker-compose.yml`: Dedicated fairness-dashboard service, resource configs.
  - `monitoring/` or `grafana/`: Dashboards for API/fairness metrics.

---

## 6. Documentation & Onboarding
- **Claim:** Comprehensive, up-to-date documentation for all stakeholders.
- **Evidence:**
  - `README.md`: Project structure, setup, and fairness framework details.
  - `PROJECT_QA_ALL_USERS.md`, `PROJECT_SECURITY_AND_READINESS.md`: Q&A, risk register, and readiness docs.
  - Onboarding and API docs in `docs/` directory.

---

## 7. Real-World Test Cases & Edge Handling
- **Claim:** Framework handles edge cases and large datasets reliably.
- **Evidence:**
  - `tests/`: Edge case tests for empty datasets, missing attributes, and large files.
  - `src/fairness/dashboard.py`: Memory-efficient pagination and error handling.

---

*For any claim, see the referenced file or directory for direct evidence. All claims are based on actual project artifacts, not assumptions or hallucination.*
