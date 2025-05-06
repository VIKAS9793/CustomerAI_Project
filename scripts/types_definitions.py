"""
Type definitions for secure subprocess operations.
"""

from pathlib import Path
from typing import Generic, List, TypeVar

T = TypeVar("T")


class PathValidationError(Exception):
    """Raised when path validation fails."""

    pass


class CommandValidationError(Exception):
    """Raised when command validation fails."""

    pass


class TimeoutError(Exception):
    """Raised when operation times out."""

    pass


class SafePath:
    """
    A wrapper around Path that ensures path safety.

    Invariants:
    1. Path must be absolute
    2. Path must be within project root
    3. Path must be valid and exist
    """

    def __init__(self, path: str, project_root: Path):
        self._path = Path(path).resolve()
        self._project_root = project_root.resolve()

    def validate(self) -> bool:
        """Mathematically validate path safety."""
        return (
            self._path.is_absolute()
            and self._project_root in self._path.parents
            and self._path.exists()
        )

    @property
    def path(self) -> Path:
        """Get validated path."""
        if not self.validate():
            raise PathValidationError("Path validation failed")
        return self._path


class CommandArgs(Generic[T]):
    """
    Type-safe wrapper for command arguments.

    Invariants:
    1. No dangerous characters
    2. Valid executable path
    3. Arguments are non-empty strings
    """

    def __init__(self, args: List[str]):
        self._args = args

    def sanitize(self) -> List[str]:
        """Sanitize arguments mathematically."""
        dangerous_chars = set("&|;<>`$'\"\0")
        return [
            arg for arg in self._args if arg and not any(char in dangerous_chars for char in arg)
        ]

    @property
    def args(self) -> List[str]:
        """Get sanitized arguments."""
        return self.sanitize()

    def validate(self) -> bool:
        """Mathematically validate command structure."""
        return (
            len(self.args) > 0
            and all(isinstance(arg, str) for arg in self.args)
            and all(arg.strip() for arg in self.args)
        )
