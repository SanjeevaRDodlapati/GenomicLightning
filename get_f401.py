#!/usr/bin/env python3
"""Get remaining F401 violations."""

import subprocess
import sys

try:
    result = subprocess.run(
        [sys.executable, '-m', 'flake8', '--select=F401', '--format=%(path)s:%(row)d: %(code)s %(text)s'],
        capture_output=True, text=True, timeout=30
    )
    
    lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
    
    print(f"Found {len(lines)} F401 violations:")
    for i, line in enumerate(lines[:30]):  # Show first 30
        if line.strip():
            print(f"{i+1:2d}. {line}")
    
    if len(lines) > 30:
        print(f"... and {len(lines) - 30} more")
        
except Exception as e:
    print(f"Error: {e}")
