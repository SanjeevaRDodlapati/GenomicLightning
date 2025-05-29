#!/usr/bin/env python3
"""
Comprehensive script to fix F401 unused import violations.
"""

import subprocess
import sys
import re
import ast
import os
from pathlib import Path


def get_all_f401_violations():
    """Get all F401 violations from flake8."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', '--select=F401', '--format=%(path)s:%(row)d: %(code)s %(text)s'],
            capture_output=True, text=True, timeout=60
        )
        
        violations = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip() and 'F401' in line:
                    violations.append(line.strip())
        
        return violations
    except Exception as e:
        print(f"Error getting F401 violations: {e}")
        return []


def parse_violation(violation_line):
    """Parse a flake8 violation line."""
    # Format: ./path/file.py:line: F401 'module' imported but unused
    match = re.match(r'([^:]+):(\d+):\s+F401\s+(.+)', violation_line)
    if match:
        filepath = match.group(1)
        line_num = int(match.group(2))
        message = match.group(3)
        
        # Extract the import name from the message
        import_match = re.search(r"'([^']+)' imported but unused", message)
        import_name = import_match.group(1) if import_match else None
        
        return {
            'file': filepath,
            'line': line_num,
            'message': message,
            'import_name': import_name
        }
    return None


def fix_simple_unused_imports():
    """Fix simple cases of unused imports."""
    violations = get_all_f401_violations()
    print(f"Found {len(violations)} F401 violations")
    
    # Group by file
    files_to_fix = {}
    for violation_line in violations:
        violation = parse_violation(violation_line)
        if violation:
            filepath = violation['file']
            if filepath.startswith('./'):
                filepath = filepath[2:]  # Remove leading ./
            
            if filepath not in files_to_fix:
                files_to_fix[filepath] = []
            files_to_fix[filepath].append(violation)
    
    print(f"Files to fix: {len(files_to_fix)}")
    
    for filepath, file_violations in files_to_fix.items():
        if os.path.exists(filepath):
            print(f"\nFixing {filepath} ({len(file_violations)} violations)")
            fix_file_imports(filepath, file_violations)


def fix_file_imports(filepath, violations):
    """Fix unused imports in a specific file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        original_lines = lines.copy()
        
        # Sort violations by line number in reverse order
        violations.sort(key=lambda x: x['line'], reverse=True)
        
        removed_count = 0
        for violation in violations:
            line_idx = violation['line'] - 1
            import_name = violation['import_name']
            
            if line_idx < len(lines):
                line_content = lines[line_idx].strip()
                
                # Only remove if it's clearly an unused import line
                if (line_content.startswith('import ') or 
                    line_content.startswith('from ') or
                    'import' in line_content):
                    
                    # Check if this is a simple single-line import we can safely remove
                    if (line_content == f"import {import_name}" or
                        line_content.endswith(f"import {import_name}") or
                        (import_name and import_name in line_content and len(line_content.split(',')) == 1)):
                        
                        print(f"  Removing line {violation['line']}: {line_content}")
                        lines.pop(line_idx)
                        removed_count += 1
                    else:
                        print(f"  Skipping complex import at line {violation['line']}: {line_content}")
        
        if removed_count > 0:
            with open(filepath, 'w') as f:
                f.writelines(lines)
            print(f"  Removed {removed_count} unused imports from {filepath}")
        else:
            print(f"  No simple imports to remove from {filepath}")
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    print("🔧 Fixing F401 unused import violations...")
    fix_simple_unused_imports()
    print("\n✅ Done!")
