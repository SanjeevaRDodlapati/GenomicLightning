#!/usr/bin/env python3
"""
Debug script to find syntax errors in the codebase.
"""

import os
import py_compile
import sys


def check_python_files(directory):
    """Check all Python files for syntax errors."""
    errors = []

    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git directories
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".pytest_cache"]]

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    py_compile.compile(filepath, doraise=True)
                    print(f"✅ {filepath}")
                except py_compile.PyCompileError as e:
                    error_msg = f"❌ {filepath}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
                except Exception as e:
                    error_msg = f"❌ {filepath}: Unexpected error: {e}"
                    print(error_msg)
                    errors.append(error_msg)

    return errors


if __name__ == "__main__":
    print("🔍 Checking Python syntax in the codebase...")

    # Check main directories
    directories_to_check = ["genomic_lightning", "tests", "examples", "scripts"]

    all_errors = []

    for directory in directories_to_check:
        if os.path.exists(directory):
            print(f"\n📁 Checking {directory}/")
            errors = check_python_files(directory)
            all_errors.extend(errors)

    # Check root level Python files
    print(f"\n📁 Checking root level files")
    for file in os.listdir("."):
        if file.endswith(".py"):
            try:
                py_compile.compile(file, doraise=True)
                print(f"✅ {file}")
            except py_compile.PyCompileError as e:
                error_msg = f"❌ {file}: {e}"
                print(error_msg)
                all_errors.append(error_msg)

    print(f"\n📊 Summary:")
    if all_errors:
        print(f"❌ Found {len(all_errors)} syntax errors:")
        for error in all_errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("✅ No syntax errors found!")
        sys.exit(0)
