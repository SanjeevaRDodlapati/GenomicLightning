#!/usr/bin/env python3
"""
Test version management functionality without requiring full package imports.
"""

import sys
import os
import re
from pathlib import Path

def test_version_manager_existence():
    """Test that version manager scripts exist in all repositories."""
    repos = ['FuGEP', 'UAVarPrior', 'GenomicLightning']
    base_path = Path('/home/sdodl001')

    for repo in repos:
        version_manager = base_path / repo / 'scripts' / 'version_manager.py'
        get_version = base_path / repo / 'scripts' / 'get_version.py'
        bump_version = base_path / repo / 'scripts' / 'bump_version.py'

        assert version_manager.exists(), f"version_manager.py missing in {repo}"
        assert get_version.exists(), f"get_version.py missing in {repo}"
        assert bump_version.exists(), f"bump_version.py missing in {repo}"

        print(f"✅ {repo}: All version management scripts present")

def test_workflow_files_existence():
    """Test that GitHub workflow files exist."""
    repos = ['FuGEP', 'UAVarPrior', 'GenomicLightning']
    base_path = Path('/home/sdodl001')

    for repo in repos:
        workflows_dir = base_path / repo / '.github' / 'workflows'
        testing_workflow = workflows_dir / 'testing.yml'
        release_workflow = workflows_dir / 'release.yml'

        assert workflows_dir.exists(), f"Workflows directory missing in {repo}"
        assert testing_workflow.exists(), f"testing.yml missing in {repo}"
        assert release_workflow.exists(), f"release.yml missing in {repo}"

        print(f"✅ {repo}: GitHub workflows present")

def test_version_extraction():
    """Test version extraction from project files."""
    repos = {
        'FuGEP': Path('/home/sdodl001/FuGEP'),
        'UAVarPrior': Path('/home/sdodl001/UAVarPrior'),
        'GenomicLightning': Path('/home/sdodl001/GenomicLightning')
    }

    for repo_name, repo_path in repos.items():
        # Check pyproject.toml
        pyproject = repo_path / 'pyproject.toml'
        if pyproject.exists():
            content = pyproject.read_text()
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                version = version_match.group(1)
                print(f"✅ {repo_name}: Found version {version} in pyproject.toml")

                # Validate version format (semantic versioning)
                if re.match(r'\d+\.\d+\.\d+', version):
                    print(f"✅ {repo_name}: Version format is valid")
                else:
                    print(f"⚠️ {repo_name}: Version format may not be semantic")

def test_test_directories():
    """Test that test directories exist."""
    repos = ['FuGEP', 'UAVarPrior', 'GenomicLightning']
    base_path = Path('/home/sdodl001')

    for repo in repos:
        tests_dir = base_path / repo / 'tests'
        smoke_dir = tests_dir / 'smoke'
        unit_dir = tests_dir / 'unit'

        assert tests_dir.exists(), f"Tests directory missing in {repo}"
        assert smoke_dir.exists(), f"Smoke tests directory missing in {repo}"
        assert unit_dir.exists(), f"Unit tests directory missing in {repo}"

        print(f"✅ {repo}: Test directories present")

def test_basic_workflow_syntax():
    """Test basic YAML syntax of workflow files."""
    repos = ['FuGEP', 'UAVarPrior', 'GenomicLightning']
    base_path = Path('/home/sdodl001')

    for repo in repos:
        workflows_dir = base_path / repo / '.github' / 'workflows'

        for workflow_file in ['testing.yml', 'release.yml']:
            workflow_path = workflows_dir / workflow_file
            if workflow_path.exists():
                content = workflow_path.read_text()

                # Basic YAML checks
                assert 'name:' in content, f"Missing name in {repo}/{workflow_file}"
                assert 'on:' in content, f"Missing triggers in {repo}/{workflow_file}"
                assert 'jobs:' in content, f"Missing jobs in {repo}/{workflow_file}"

                print(f"✅ {repo}/{workflow_file}: Basic YAML structure valid")

def main():
    """Run all tests."""
    print("🧪 Testing CI/CD Infrastructure Implementation")
    print("=" * 50)

    try:
        test_version_manager_existence()
        print()

        test_workflow_files_existence()
        print()

        test_version_extraction()
        print()

        test_test_directories()
        print()

        test_basic_workflow_syntax()
        print()

        print("🎉 All infrastructure tests passed!")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
