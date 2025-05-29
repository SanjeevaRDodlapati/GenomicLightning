#!/bin/tcsh
# Pre-push verification of fixes

echo "🧪 Pre-Push Verification for GenomicLightning"
echo "============================================="

cd /home/sdodl001/GenomicLightning

echo "\n1. 📦 Testing Package Imports..."
crun -p ~/envs/GenomicLightningTF2170 python -c "
try:
    import genomic_lightning
    print('✅ Main package import successful')
    
    from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC, TopKAccuracy, PositionalAUROC
    print('✅ Metric imports successful')
    
    from genomic_lightning.models.danq import DanQ
    from genomic_lightning.models.chromdragonn import ChromDragoNNModel
    print('✅ Model imports successful')
    
    print('🎉 All critical imports working!')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

if ($status != 0) then
    echo "❌ Import test failed - aborting push"
    exit 1
endif

echo "\n2. 🧮 Testing Metric Compatibility..."
crun -p ~/envs/GenomicLightningTF2170 python -c "
import torch
from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC, PositionalAUROC, TopKAccuracy

try:
    # Test GenomicAUPRC
    auprc = GenomicAUPRC(num_classes=10)
    preds = torch.randn(8, 10)
    targets = torch.randint(0, 2, (8, 10)).float()
    auprc.update(preds, targets)
    result = auprc.compute()
    print(f'✅ GenomicAUPRC working: {result.mean():.4f}')
    
    # Test PositionalAUROC  
    pos_auroc = PositionalAUROC(num_classes=10, sequence_length=100)
    preds = torch.randn(4, 10, 100)
    targets = torch.randint(0, 2, (4, 10, 100)).float()
    pos_auroc.update(preds, targets)
    result = pos_auroc.compute()
    print(f'✅ PositionalAUROC working: {result.mean():.4f}')
    
    # Test TopKAccuracy
    topk = TopKAccuracy(k=3)
    preds = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    topk.update(preds, targets)
    result = topk.compute()
    print(f'✅ TopKAccuracy working: {result:.4f}')
    
    print('🎉 All metrics working correctly!')
    
except Exception as e:
    print(f'❌ Metric test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if ($status != 0) then
    echo "❌ Metric test failed - aborting push"
    exit 1
endif

echo "\n3. 🏗️ Testing Model Creation..."
crun -p ~/envs/GenomicLightningTF2170 python -c "
try:
    from genomic_lightning.models.danq import DanQ
    from genomic_lightning.models.chromdragonn import ChromDragoNNModel
    
    # Test DanQ
    danq = DanQ(sequence_length=1000, num_classes=919)
    print('✅ DanQ model created successfully')
    
    # Test ChromDragoNN
    chromdragonn = ChromDragoNNModel(sequence_length=1000, num_classes=919)
    print('✅ ChromDragoNN model created successfully')
    
    print('🎉 All models working correctly!')
    
except Exception as e:
    print(f'❌ Model test failed: {e}')
    exit(1)
"

if ($status != 0) then
    echo "❌ Model test failed - aborting push"
    exit 1
endif

echo "\n✅ PRE-PUSH VERIFICATION COMPLETE!"
echo "================================="
echo "🎉 All critical components working correctly"
echo "📤 Ready to push and trigger GitHub Actions"
echo ""
echo "Run the following to push and test GitHub Actions:"
echo "tcsh test_github_actions.csh"
