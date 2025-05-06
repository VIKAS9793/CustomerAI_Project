"""
Date Provider Utility

This module provides a centralized date/time service for the application.
It enables consistent date formatting and facilitates testing by allowing
date mocking in test environments.

Copyright (c) 2025 Vikas Sahani
"""

from datetime import date, datetime
from typing import ClassVar, Optional


class DateProvider:
    """
    Centralized date provider for the application.

    This class provides a single source of truth for date/time operations,
    allowing for consistent formatting and enabling test mocking.
    """

    _instance: ClassVar[Optional["DateProvider"]] = None
    _mock_date: ClassVar[Optional[datetime]] = None

    @classmethod
    def get_instance(cls) -> "DateProvider":
        """Get the singleton instance of DateProvider."""
        if cls._instance is None:
            cls._instance = DateProvider()
        return cls._instance

    @classmethod
    def set_mock_date(cls, mock_date: Optional[datetime]) -> None:
        """
        Set a mock date for testing purposes.

        Args:
            mock_date: The datetime to use for all date operations, or None to use real time
        """
        cls._mock_date = mock_date

    def now(self) -> datetime:
        """
        Get the current datetime, or the mock date if set.

        Returns:
            Current datetime or mock date
        """
        if self._mock_date is not None:
            return self._mock_date
        return datetime.now()

    def today(self) -> date:
        """
        Get the current date, or the mock date if set.

        Returns:
            Current date or mock date
        """
        return self.now().date()

    def iso_format(self) -> str:
        """
        Get the current datetime in ISO format.

        Returns:
            ISO formatted datetime string
        """
        return self.now().isoformat()

    def format(self, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format the current datetime with the specified format string.

        Args:
            format_string: The format string to use (default: "%Y-%m-%d %H:%M:%S")

        Returns:
            Formatted datetime string
        """
        return self.now().strftime(format_string)

    def timestamp(self) -> float:
        """
        Get the current timestamp.

        Returns:
            Current timestamp as a float
        """
        return self.now().timestamp()
