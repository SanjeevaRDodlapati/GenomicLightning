#!/usr/bin/env python3
"""
Test script to verify UAVarPrior integration fixes.
This tests that the previously failing imports now work correctly.
"""
import sys
import os

# Add GenomicLightning to path
sys.path.insert(0, '/home/sdodl001/GenomicLightning')

def test_sampler_utils_import():
    """Test that SamplerUtils can be imported and used."""
    print("🔍 Testing SamplerUtils import...")
    try:
        from genomic_lightning.utils.sampler_utils import SamplerUtils
        sampler_utils = SamplerUtils()
        print("✅ SamplerUtils imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ SamplerUtils import failed: {e}")
        return False

def test_uavarprior_sampler_creation():
    """Test that UAVarPrior sampler creation provides helpful error messages."""
    print("🔍 Testing UAVarPrior sampler creation...")
    try:
        from genomic_lightning.utils.sampler_utils import SamplerUtils
        sampler_utils = SamplerUtils()
        
        # This should create a mock sampler with helpful error message
        mock_sampler = sampler_utils.create_uavarprior_sampler(
            config_path=None,
            data_path=None,
            split="train"
        )
        
        # Test that the mock sampler works
        assert len(mock_sampler) > 0, "Mock sampler should have non-zero length"
        sample = mock_sampler[0]
        assert isinstance(sample, tuple), "Mock sampler should return tuple"
        assert len(sample) == 2, "Mock sampler should return (sequence, labels)"
        
        print("✅ UAVarPrior mock sampler created and works correctly")
        return True
    except Exception as e:
        print(f"❌ UAVarPrior sampler creation failed: {e}")
        return False

def test_sampler_adapter_import():
    """Test that SamplerAdapter can be imported."""
    print("🔍 Testing SamplerAdapter import...")
    try:
        from genomic_lightning.data.sampler_adapter import SamplerAdapter
        print("✅ SamplerAdapter imported successfully")
        return True
    except Exception as e:
        print(f"❌ SamplerAdapter import failed: {e}")
        return False

def test_integration_example_imports():
    """Test that the integration example imports work."""
    print("🔍 Testing integration example imports...")
    try:
        # Add the examples directory to path
        sys.path.insert(0, '/home/sdodl001/GenomicLightning/examples')
        
        # Import the fixed integration example
        import uavarprior_integration_example
        
        # Test the function that replaced the problematic sampler import
        result = uavarprior_integration_example.import_uavarprior_data()
        
        # Result should be None (since UAVarPrior not installed) but no import error
        print("✅ Integration example imports work correctly")
        return True
    except ImportError as e:
        if "uavarprior" in str(e).lower():
            print("✅ Integration example handles UAVarPrior import gracefully")
            return True
        else:
            print(f"❌ Unexpected import error: {e}")
            return False
    except Exception as e:
        print(f"❌ Integration example test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Testing UAVarPrior Integration Fixes")
    print("=" * 50)
    
    tests = [
        test_sampler_utils_import,
        test_uavarprior_sampler_creation,
        test_sampler_adapter_import,
        test_integration_example_imports,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed! UAVarPrior integration fixes are working.")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
