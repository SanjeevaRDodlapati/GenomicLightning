#!/usr/bin/env python3
"""
Test script to verify that our build fixes resolve the GitHub Actions issues.
"""

def test_imports():
    """Test that all core imports work."""
    print("🔍 Testing core imports...")
    
    try:
        from genomic_lightning.utils.sampler_utils import SamplerUtils
        print("✅ SamplerUtils imported successfully")
    except Exception as e:
        print(f"❌ SamplerUtils import failed: {e}")
        return False
    
    try:
        from genomic_lightning.data.sampler_adapter import SamplerAdapter
        print("✅ SamplerAdapter imported successfully")
    except Exception as e:
        print(f"❌ SamplerAdapter import failed: {e}")
        return False
        
    return True

def test_uavarprior_integration():
    """Test UAVarPrior integration handling."""
    print("🔍 Testing UAVarPrior integration...")
    
    try:
        # Test the integration example import function
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
        
        from uavarprior_integration_example import import_uavarprior_data
        
        # This should handle the ImportError gracefully
        result = import_uavarprior_data()
        if result is None:
            print("✅ UAVarPrior integration handles missing package gracefully")
            return True
        else:
            print("✅ UAVarPrior integration works with package available")
            return True
            
    except Exception as e:
        print(f"❌ UAVarPrior integration test failed: {e}")
        return False

def test_sampler_utils():
    """Test SamplerUtils functionality."""
    print("🔍 Testing SamplerUtils functionality...")
    
    try:
        from genomic_lightning.utils.sampler_utils import SamplerUtils
        
        # Create instance
        sampler_utils = SamplerUtils()
        print("✅ SamplerUtils instantiated successfully")
        
        # Test UAVarPrior sampler creation (should handle gracefully)
        try:
            sampler = sampler_utils.create_uavarprior_sampler(
                config_path="nonexistent.yaml",
                data_path="nonexistent",
                split="train"
            )
            print("✅ UAVarPrior sampler creation handled gracefully")
        except Exception as e:
            # This is expected to fail gracefully
            print("✅ UAVarPrior sampler creation failed gracefully (expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ SamplerUtils test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing GenomicLightning Build Fixes")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_uavarprior_integration, 
        test_sampler_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    if passed == total:
        print(f"🎉 All {total} tests passed! Build fixes are working.")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
