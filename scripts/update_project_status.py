#!/usr/bin/env python3
"""
Script to automatically update project status documentation.
This script analyzes the codebase and git history to generate an updated PROJECT_STATUS.md
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from .secure_subprocess import SecureSubprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProjectAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.status_file = self.project_root / "PROJECT_STATUS.md"

    def get_git_info(self) -> Dict:
        """Get recent git activity and changes."""
        try:
            # Get last commit date
            try:
                result = SecureSubprocess.run(
                    ["git", "log", "-1", "--format=%cd", "--date=iso"],
                    cwd=str(self.project_root),
                    capture_output=True,
                )
                last_commit = result.stdout.decode().strip()
            except Exception as e:
                logger.error(f"Error getting last commit: {e}")
                last_commit = "Unknown"

            # Get recent changes
            try:
                result = SecureSubprocess.run(
                    ["git", "log", "-10", "--pretty=format:%s"],
                    cwd=str(self.project_root),
                    capture_output=True,
                )
                recent_changes = result.stdout.decode().split("\n")
            except Exception as e:
                logger.error(f"Error getting recent changes: {e}")
                recent_changes = []

            # Get modified files
            try:
                result = SecureSubprocess.run(
                    ["git", "diff", "--name-only"], cwd=str(self.project_root), capture_output=True
                )
                modified_files = result.stdout.decode().split("\n")
            except Exception as e:
                logger.error(f"Error getting modified files: {e}")
                modified_files = []

            return {
                "last_commit": last_commit,
                "recent_changes": recent_changes,
                "modified_files": [f for f in modified_files if f],
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting git info: {e}")
            return {}

    def analyze_dependencies(self) -> Dict:
        """Analyze project dependencies from requirements.txt."""
        deps = {}
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file) as f:
                for line in f:
                    if "==" in line:
                        name, version = line.strip().split("==")
                        deps[name] = version
        return deps

    def get_test_coverage(self) -> Tuple[float, Dict]:
        """Get test coverage information."""
        try:
            # Run pytest with coverage
            try:
                SecureSubprocess.run(
                    ["pytest", "--cov=src", "--cov-report=json"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    check=True,
                    timeout=300,  # 5 minute timeout for tests
                )
            except Exception as e:
                logger.error(f"Error running tests: {e}")

            # Read coverage data
            coverage_file = self.project_root / ".coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    total = coverage_data.get("total", {}).get("coverage", 0)
                    return total, coverage_data
        except Exception as e:
            logger.error(f"Error getting test coverage: {e}")
        return 0.0, {}

    def analyze_code_quality(self) -> Dict:
        """Analyze code quality metrics."""
        try:
            # Run flake8 for code quality and capture output
            try:
                SecureSubprocess.run(
                    [
                        "python",
                        "-m",
                        "flake8",
                        "--statistics",
                        "--tee",
                        "--output-file=flake8.txt",
                    ],
                    cwd=str(self.project_root),
                    capture_output=True,
                    check=True,
                    timeout=300,  # 5 minute timeout for flake8
                )
            except Exception as e:
                logger.error(f"Error running flake8: {e}")

            # Parse flake8 output
            issues = {"errors": 0, "warnings": 0, "style": 0}
            if (self.project_root / "flake8.txt").exists():
                with open(self.project_root / "flake8.txt") as f:
                    for line in f:
                        if ":" in line:
                            if "E" in line:
                                issues["errors"] += 1
                            elif "W" in line:
                                issues["warnings"] += 1
                            else:
                                issues["style"] += 1

            return issues
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {}

    def generate_status_markdown(self) -> str:
        """Generate the PROJECT_STATUS.md content."""
        git_info = self.get_git_info()
        deps = self.analyze_dependencies()
        coverage, coverage_data = self.get_test_coverage()
        code_quality = self.analyze_code_quality()

        # Format the markdown content
        content = [
            "# CustomerAI Project Status Report",
            f"*Last Updated: {datetime.now().strftime('%B %d, %Y')}*\n",
            "## Recent Changes",
            "\n".join(f"- {change}" for change in git_info.get("recent_changes", [])),
            "\n## Core Components Status\n",
        ]

        # Add test coverage information
        content.extend(
            [
                "### Test Coverage",
                f"- Overall coverage: {coverage:.1f}%",
                "- Status by component:",
            ]
        )

        # Add code quality metrics
        content.extend(
            [
                "\n### Code Quality Metrics",
                f"- Errors: {code_quality.get('errors', 0)}",
                f"- Warnings: {code_quality.get('warnings', 0)}",
                f"- Style issues: {code_quality.get('style', 0)}",
            ]
        )

        # Add dependency information
        content.extend(
            [
                "\n### Dependencies",
                "Key dependencies and versions:",
            ]
        )
        for dep, version in deps.items():
            content.append(f"- {dep}: {version}")

        # Add modified files
        if git_info.get("modified_files"):
            content.extend(
                ["\n### Recently Modified Files", *[f"- {f}" for f in git_info["modified_files"]]]
            )

        return "\n".join(content)

    def update_status_file(self):
        """Update the PROJECT_STATUS.md file."""
        try:
            content = self.generate_status_markdown()

            # Backup existing file
            if self.status_file.exists():
                backup_path = self.status_file.with_suffix(".md.bak")
                self.status_file.rename(backup_path)

            # Write new content
            with open(self.status_file, "w") as f:
                f.write(content)

            logger.info(f"Successfully updated {self.status_file}")
            return True
        except Exception as e:
            logger.error(f"Error updating status file: {e}")
            return False


def main():
    """Main entry point for the script."""
    try:
        # Get project root (assumes script is in scripts/ directory)
        project_root = Path(__file__).parent.parent

        # Create analyzer and update status
        analyzer = ProjectAnalyzer(project_root)
        if analyzer.update_status_file():
            logger.info("Project status updated successfully")
            return 0
        else:
            logger.error("Failed to update project status")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
