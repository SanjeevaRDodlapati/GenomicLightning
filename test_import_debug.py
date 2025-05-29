#!/usr/bin/env python3
import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, '.')

print("🔍 Testing GenomicLightning import...")

try:
    import genomic_lightning
    print('✅ Basic import successful')
    print(f'Version: {genomic_lightning.__version__}')
    print(f'Package location: {genomic_lightning.__file__}')
except Exception as e:
    print(f'❌ Import failed: {e}')
    print("Full traceback:")
    traceback.print_exc()

print("\n🔍 Testing specific modules...")

# Test core modules
modules_to_test = [
    'genomic_lightning.models',
    'genomic_lightning.data',
    'genomic_lightning.metrics',
    'genomic_lightning.lightning_modules'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f'✅ {module} imported successfully')
    except Exception as e:
        print(f'❌ {module} failed: {e}')
