#!/usr/bin/env python3
"""
Simple script to check flake8 errors that are causing CI failures.
"""

import subprocess
import sys


def run_flake8_check():
    """Run the same flake8 checks as the CI workflow."""

    print("🔍 Running flake8 checks (same as CI workflow)...")

    # Check 1: Critical errors only (E9, F63, F7, F82)
    print("\n1️⃣ Critical errors check:")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flake8",
                ".",
                "--count",
                "--select=E9,F63,F7,F82",
                "--show-source",
                "--statistics",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Return code: {result.returncode}")
        if result.stdout.strip():
            print("STDOUT:")
            print(result.stdout)
        if result.stderr.strip():
            print("STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            print("❌ Critical errors found!")
            return False
        else:
            print("✅ No critical errors found")

    except subprocess.TimeoutExpired:
        print("❌ Flake8 timed out")
        return False
    except FileNotFoundError:
        print("❌ Flake8 not found")
        return False

    # Check 2: Full flake8 check with warnings
    print("\n2️⃣ Full flake8 check:")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flake8",
                ".",
                "--count",
                "--exit-zero",
                "--max-complexity=12",
                "--max-line-length=88",
                "--statistics",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Return code: {result.returncode}")
        if result.stdout.strip():
            print("STDOUT (first 50 lines):")
            lines = result.stdout.split("\n")[:50]
            print("\n".join(lines))
        if result.stderr.strip():
            print("STDERR:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("❌ Full flake8 check timed out")
    except FileNotFoundError:
        print("❌ Flake8 not found")

    return True


if __name__ == "__main__":
    print("🚀 GenomicLightning Flake8 Checker")
    print("=" * 50)

    # Check if flake8 is installed
    try:
        subprocess.run(
            [sys.executable, "-m", "flake8", "--version"],
            capture_output=True,
            check=True,
        )
        print("✅ Flake8 is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Flake8 not installed. Installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "flake8"], check=True
            )
            print("✅ Flake8 installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install flake8")
            sys.exit(1)

    success = run_flake8_check()

    if success:
        print("\n✅ Flake8 checks completed successfully!")
    else:
        print("\n❌ Flake8 checks found critical errors!")
        sys.exit(1)
