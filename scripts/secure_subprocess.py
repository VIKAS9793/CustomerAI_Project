"""
Secure subprocess wrapper with proper path validation and security measures.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from scripts.types_definitions import (
    CommandArgs,
    CommandValidationError,
    PathValidationError,
    SafePath,
    TimeoutError,
)
from src.utils.security_utils import (
    InjectionError,
    SecurityError,
    ValidationError,
    sanitize_input,
    validate_command_args,
    validate_input,
    validate_path,
)


class SecureSubprocess:
    """
    Secure wrapper for subprocess operations.

    This class provides a safe interface for executing subprocess commands
    with built-in security measures to prevent injection attacks and
    ensure proper validation of all inputs.

    Security Features:
    - Path validation and sanitization
    - Command argument validation
    - Environment variable sanitization
    - Output size limiting
    - Timeout protection
    - Shell injection prevention
    """

    def __init__(self, project_root: Path, max_args: int = 100, max_path_length: int = 4096):
        """
        Initialize the secure subprocess wrapper.

        Args:
            project_root: Root directory for path validation
            max_args: Maximum number of command arguments
            max_path_length: Maximum length of paths
        """
        self.project_root = project_root
        self.max_args = max_args
        self.max_path_length = max_path_length

    def validate_path(self, path: str) -> SafePath:
        """
        Validate that a path is safe and within project root.

        Args:
            path: Path to validate

        Returns:
            SafePath object

        Raises:
            PathValidationError: If path validation fails
        """
        try:
            # Validate path length and basic safety
            validate_input(path, max_length=self.max_path_length)

            # Validate path against project root
            safe_path = validate_path(path, str(self.project_root))

            return SafePath(safe_path, self.project_root)

        except (ValidationError, SecurityError) as e:
            raise PathValidationError(str(e))

    def sanitize_args(self, args: List[str]) -> CommandArgs:
        """
        Sanitize command arguments to prevent injection attacks.

        Args:
            args: List of command arguments

        Returns:
            CommandArgs object with sanitized arguments

        Raises:
            CommandValidationError: If argument validation fails
        """
        try:
            # Validate input length and basic safety
            for arg in args:
                validate_input(arg, max_length=256)  # Limit argument length

            # Validate and sanitize command arguments
            sanitized = validate_command_args(args)

            # Additional sanitization
            sanitized = [sanitize_input(arg) for arg in sanitized]

            return CommandArgs(sanitized)

        except (ValidationError, InjectionError, SecurityError) as e:
            raise CommandValidationError(str(e))

    def run(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        max_output_size: int = 1024 * 1024,  # 1MB
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """
        Run a command with enhanced security checks.

        Args:
            args: List of command arguments
            cwd: Current working directory
            timeout: Maximum execution time in seconds
            max_output_size: Maximum allowed output size in bytes
            **kwargs: Additional subprocess arguments

        Returns:
            CompletedProcess object

        Raises:
            PathValidationError: If working directory validation fails
            CommandValidationError: If command validation fails
            TimeoutError: If command times out
        """
        if not cwd:
            raise PathValidationError("Current working directory is required")

        try:
            # Validate and sanitize working directory
            safe_cwd = self.validate_path(cwd)

            # Sanitize command arguments
            safe_args = self.sanitize_args(args)

            # Prepare secure environment
            secure_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "LD_LIBRARY_PATH": "",
                "PAGER": "cat",
            }

            # Update with any additional environment variables
            if "env" in kwargs:
                secure_env.update(kwargs["env"])

            # Set up process options
            options = {
                "args": safe_args.args(),
                "cwd": str(safe_cwd.path),
                "timeout": timeout,
                "check": True,
                "capture_output": True,
                "env": secure_env,
                "encoding": "utf-8",
                "shell": False,  # Never allow shell=True
                "preexec_fn": os.setsid,  # Create new process group
            }

            # Execute command
            result = subprocess.run(**options)

            # Validate output size
            if len(result.stdout) > max_output_size:
                raise CommandValidationError(f"Output size too large ({len(result.stdout)} bytes)")

            return result

        except subprocess.TimeoutExpired:
            raise TimeoutError("Command timed out")
        except subprocess.CalledProcessError as e:
            # Log error details securely
            error_details = {
                "returncode": e.returncode,
                "cmd": " ".join(e.cmd),
                "output": e.output[:1000] if e.output else None,  # Limit error output
            }
            raise CommandValidationError(f"Command failed: {error_details}")
        except Exception as e:
            attempt = 1
            max_retries = 3
            retry_delay = 1
            last_error = e
            while attempt < max_retries:
                try:
                    return subprocess.run(
                        safe_args.args(),
                        cwd=str(safe_cwd.path),
                        capture_output=True,
                        check=True,
                        timeout=timeout,
                        shell=False,
                        executable=sys.executable if args[0] == "python" else None,
                    )
                except subprocess.TimeoutExpired as e:
                    last_error = e
                    attempt += 1
                    if attempt < max_retries:
                        import time

                        time.sleep(retry_delay)
                    else:
                        raise TimeoutError("Command timed out after retries") from e
                except Exception as e:
                    last_error = e
                    attempt += 1
                    if attempt < max_retries:
                        import time

                        time.sleep(retry_delay)
                    else:
                        raise CommandValidationError(f"Command failed: {str(e)}") from last_error
