from pathlib import Path

import pytest


# Set up test environment
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture(scope="function")
def mock_env(monkeypatch):
    """Mock environment variables for tests."""
    monkeypatch.setenv("CUSTOMERAI_ENV", "test")
    monkeypatch.setenv("PYTHON_VERSION", "3.10")
    monkeypatch.setenv("FAIRNESS_CONFIG_PATH", "/tmp/fairness_config.yaml")
    return monkeypatch
