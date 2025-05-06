from pathlib import Path

import pytest

from scripts.secure_subprocess import (
    CommandArgs,
    CommandValidationError,
    PathValidationError,
    SecureSubprocess,
    TimeoutError,
)


class TestSecureSubprocess:
    def test_validate_path_valid(self, project_root: Path):
        """Test path validation with valid path."""
        test_path = str(project_root / "tests")
        result = SecureSubprocess.validate_path(test_path, project_root)
        assert isinstance(result, Path)
        assert result.exists()

    def test_validate_path_invalid(self, project_root: Path):
        """Test path validation with invalid path."""
        with pytest.raises(PathValidationError):
            SecureSubprocess.validate_path("/nonexistent/path", project_root)

    def test_sanitize_args_valid(self):
        """Test argument sanitization with valid arguments."""
        args = ["python", "-c", "print('hello')"]
        result = SecureSubprocess.sanitize_args(args)
        assert isinstance(result, CommandArgs)
        assert len(result.args) == 3

    def test_sanitize_args_invalid(self):
        """Test argument sanitization with invalid arguments."""
        args = ["python", "-c", "print('hello'); rm -rf /"]
        with pytest.raises(CommandValidationError):
            SecureSubprocess.sanitize_args(args)

    def test_run_success(self, project_root: Path):
        """Test successful subprocess run."""
        result = SecureSubprocess.run(
            ["python", "-c", "print('test')"], cwd=str(project_root), capture_output=True
        )
        assert result.returncode == 0
        assert result.stdout.decode().strip() == "test"

    def test_run_timeout(self, project_root: Path):
        """Test subprocess with timeout."""
        with pytest.raises(TimeoutError):
            SecureSubprocess.run(
                ["python", "-c", "import time; time.sleep(2)"], cwd=str(project_root), timeout=1
            )
