#!/usr/bin/env python3
"""Simple flake8 test to see what errors remain."""

import subprocess
import sys


def run_simple_flake8():
    """Run a simple flake8 check."""
    print("🔍 Running flake8 check...")

    try:
        # Run the critical errors check first
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flake8",
                "genomic_lightning/",
                "--count",
                "--select=E9,F63,F7,F82",
                "--show-source",
                "--statistics",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        print(f"Return code: {result.returncode}")
        if result.stdout.strip():
            print("STDOUT:")
            print(result.stdout)
        if result.stderr.strip():
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ No critical flake8 errors found in genomic_lightning/")
        else:
            print("❌ Critical flake8 errors found!")

    except subprocess.TimeoutExpired:
        print("❌ Flake8 timed out")
    except Exception as e:
        print(f"❌ Error running flake8: {e}")


if __name__ == "__main__":
    run_simple_flake8()
