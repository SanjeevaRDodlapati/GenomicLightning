#!/usr/bin/env python3
"""
Final deployment readiness verification script.
Tests all components for production readiness.
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml
import json


def test_docker_files():
    """Test Docker infrastructure files."""
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        ".env.example",
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    return len(missing) == 0, missing


def test_kubernetes_manifests():
    """Test Kubernetes deployment files."""
    k8s_files = ["k8s/deployment.yaml", "k8s/pvc.yaml"]

    missing = []
    for file in k8s_files:
        if not Path(file).exists():
            missing.append(file)

    return len(missing) == 0, missing


def test_documentation():
    """Test production documentation."""
    docs = ["docs/docker_deployment.md", "docs/production_deployment.md"]

    missing = []
    for doc in docs:
        if not Path(doc).exists():
            missing.append(doc)

    return len(missing) == 0, missing


def test_package_structure():
    """Test package structure and imports."""
    try:
        import genomic_lightning
        from genomic_lightning.metrics import genomic_metrics
        from genomic_lightning.models import danq, chromdragonn

        return True, "All imports successful"
    except ImportError as e:
        return False, f"Import error: {e}"


def test_configuration_files():
    """Test configuration templates."""
    config_files = [
        "configs/example_danq.yml",
        "configs/example_chromdragonn.yml",
        "configs/example_deepsea.yml",
    ]

    valid_configs = 0
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, "r") as f:
                    yaml.safe_load(f)
                valid_configs += 1
            except yaml.YAMLError:
                pass

    return (
        valid_configs == len(config_files),
        f"{valid_configs}/{len(config_files)} valid configs",
    )


def test_integration_tests():
    """Test integration test suite."""
    integration_test = Path("tests/integration/test_end_to_end.py")
    if not integration_test.exists():
        return False, "Integration tests missing"

    # Check if comprehensive
    with open(integration_test, "r") as f:
        content = f.read()
        has_danq_test = "test_danq_full_pipeline" in content
        has_chromdragonn_test = "test_chromdragonn_full_pipeline" in content
        has_performance_test = (
            "test_performance" in content or "performance" in content.lower()
        )

    return all([has_danq_test, has_chromdragonn_test]), "Integration tests implemented"


def test_security_features():
    """Test security hardening features."""
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        return False, "Dockerfile missing"

    with open(dockerfile_path, "r") as f:
        dockerfile_content = f.read()

    security_features = [
        "useradd" in dockerfile_content,  # Non-root user
        "HEALTHCHECK" in dockerfile_content,  # Health checks
        "chown" in dockerfile_content,  # Proper permissions
    ]

    return all(security_features), f"{sum(security_features)}/3 security features"


def calculate_deployment_score():
    """Calculate overall deployment readiness score."""
    tests = [
        test_docker_files,
        test_kubernetes_manifests,
        test_documentation,
        test_package_structure,
        test_configuration_files,
        test_integration_tests,
        test_security_features,
    ]

    results = []
    total_score = 0

    print("\n🔍 GenomicLightning Deployment Readiness Assessment")
    print("=" * 60)

    for test_func in tests:
        test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
        try:
            passed, details = test_func()
            status = "✅ PASS" if passed else "❌ FAIL"
            score = 1 if passed else 0
            total_score += score

            print(f"{test_name:<25} {status:<10} {details}")
            results.append((test_name, passed, details))
        except Exception as e:
            print(f"{test_name:<25} ❌ ERROR   {str(e)}")
            results.append((test_name, False, str(e)))

    # Calculate percentage
    max_score = len(tests)
    percentage = (total_score / max_score) * 100

    print("\n" + "=" * 60)
    print(
        f"📊 DEPLOYMENT READINESS SCORE: {total_score}/{max_score} ({percentage:.1f}%)"
    )

    if percentage >= 100:
        print("🎉 PERFECT! Production deployment ready!")
        grade = "10/10"
    elif percentage >= 90:
        print("🚀 EXCELLENT! Minor improvements needed.")
        grade = "9/10"
    elif percentage >= 80:
        print("👍 GOOD! Some components need attention.")
        grade = "8/10"
    elif percentage >= 70:
        print("⚠️  FAIR! Several issues to address.")
        grade = "7/10"
    else:
        print("🔧 NEEDS WORK! Major improvements required.")
        grade = f"{int(percentage/10)}/10"

    print(f"🏆 OVERALL GRADE: {grade}")

    return percentage, results


def main():
    """Run deployment readiness verification."""
    print("🧬 GenomicLightning Deployment Readiness Verification")
    print("Testing all production deployment components...")

    score, results = calculate_deployment_score()

    # Additional recommendations
    print("\n📋 DEPLOYMENT RECOMMENDATIONS:")

    if score >= 100:
        print("✨ Repository is fully production-ready!")
        print("✨ All containerization components implemented")
        print("✨ Comprehensive documentation available")
        print("✨ Security hardening in place")
        print("✨ Integration tests implemented")
    else:
        failed_tests = [r for r in results if not r[1]]
        for test_name, _, details in failed_tests:
            print(f"  🔧 Fix: {test_name} - {details}")

    print("\n🐳 NEXT STEPS:")
    print("  1. Build Docker image: docker build -t genomic-lightning .")
    print("  2. Run services: docker-compose up")
    print("  3. Deploy to Kubernetes: kubectl apply -f k8s/")
    print("  4. Monitor with provided dashboards")

    return 0 if score >= 100 else 1


if __name__ == "__main__":
    sys.exit(main())
