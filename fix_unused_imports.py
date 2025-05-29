#!/usr/bin/env python3
"""Script to systematically fix unused imports and other flake8 violations."""

import os
import re
import subprocess
import sys


def run_flake8_for_specific_violations(violation_code):
    """Get specific flake8 violations."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', f'--select={violation_code}'],
            capture_output=True, text=True, cwd='.'
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception as e:
        print(f"Error running flake8: {e}")
        return []


def fix_unused_imports():
    """Fix F401 unused import violations."""
    violations = run_flake8_for_specific_violations('F401')
    print(f"Found {len(violations)} F401 violations")
    
    # Group violations by file
    file_violations = {}
    for violation in violations:
        if violation.strip() and ':' in violation:
            parts = violation.split(':')
            if len(parts) >= 4:
                file_path = parts[0]
                line_no = int(parts[1])
                error_msg = parts[3].strip()
                
                if file_path not in file_violations:
                    file_violations[file_path] = []
                file_violations[file_path].append((line_no, error_msg))
    
    # Process each file
    for file_path, violations in file_violations.items():
        if os.path.exists(file_path):
            print(f"\nProcessing {file_path} with {len(violations)} unused imports")
            fix_file_unused_imports(file_path, violations)


def fix_file_unused_imports(file_path, violations):
    """Fix unused imports in a specific file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Sort violations by line number in reverse order to avoid index issues
        violations_sorted = sorted(violations, key=lambda x: x[0], reverse=True)
        
        modified = False
        for line_no, error_msg in violations_sorted:
            if line_no <= len(lines):
                line_content = lines[line_no - 1].strip()
                
                # Check if it's really an import line we can safely remove
                if (line_content.startswith('import ') or 
                    line_content.startswith('from ') or
                    'import' in line_content):
                    
                    # Extract the unused import name from error message
                    match = re.search(r"'([^']+)' imported but unused", error_msg)
                    if match:
                        unused_name = match.group(1)
                        
                        # Remove the line if it's a standalone import
                        if (f"import {unused_name}" in line_content or
                            f"from {unused_name}" in line_content or
                            line_content.strip() == f"import {unused_name}" or
                            line_content.strip().endswith(f"import {unused_name}")):
                            
                            print(f"  Removing line {line_no}: {line_content}")
                            lines.pop(line_no - 1)
                            modified = True
        
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(lines)
            print(f"  Fixed unused imports in {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    print("Starting systematic flake8 fixes...")
    fix_unused_imports()
    print("\nDone!")
