#!/usr/bin/env python
"""
Timestamp Standardization Script

This script updates all timestamp generation in the codebase to use the centralized
DateProvider utility. This ensures consistent date formatting and enables easier
testing by allowing date mocking.

Usage:
    python scripts/update_timestamps.py

Copyright (c) 2025 Vikas Sahani
"""

import os
import re
from pathlib import Path


def update_file(file_path):
    """Update DateProvider.get_instance().now() calls in a file to use DateProvider."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if file already imports DateProvider
    has_import = re.search(r"from\s+src\.utils\.date_provider\s+import\s+DateProvider", content)

    # Replace DateProvider.get_instance().iso_format() with DateProvider
    updated_content = re.sub(
        r"datetime\.now\(\)\.isoformat\(\)",
        r"DateProvider.get_instance().iso_format()",
        content,
    )

    # Replace other DateProvider.get_instance().now() calls
    updated_content = re.sub(
        r"datetime\.now\(\)", r"DateProvider.get_instance().now()", updated_content
    )

    # Add import if needed
    if updated_content != content and not has_import:
        # Find the imports section
        import_match = re.search(r"(import\s+[^\n]+\n|from\s+[^\n]+\n)+", updated_content)
        if import_match:
            import_section = import_match.group(0)
            new_import = import_section + "from src.utils.date_provider import DateProvider\n"
            updated_content = updated_content.replace(import_section, new_import)

    # Write updated content back to file
    if updated_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        return True

    return False


def main():
    """Main function to update timestamp generation in the codebase."""
    # Get project root directory
    project_root = Path(__file__).parent.parent.absolute()

    # Python files to check
    python_files = []
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    # Update files
    updated_files = []
    for file_path in python_files:
        if update_file(file_path):
            updated_files.append(file_path)

    # Print summary
    print(f"Updated {len(updated_files)} files to use DateProvider:")
    for file in updated_files:
        print(f"  - {os.path.relpath(file, project_root)}")


if __name__ == "__main__":
    main()
