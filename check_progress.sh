#!/bin/bash

cd /nfs/usrhome2/africanstu/kangwa/m3d/M3D

echo "=========================================="
echo "DOWNLOAD PROGRESS CHECK"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Training set
if [ -d "ctrate_volumes/train" ]; then
    TRAIN_IMGS=$(ls -1 ctrate_volumes/train/*.nii.gz 2>/dev/null | wc -l)
    TRAIN_REPORTS=$(ls -1 ctrate_reports/train/*.txt 2>/dev/null | wc -l)
    TRAIN_SIZE=$(du -sh ctrate_volumes/train/ 2>/dev/null | cut -f1)
    
    echo "✓ TRAINING SET:"
    echo "  Images: $TRAIN_IMGS / 150 (target: 150)"
    if [ "$TRAIN_IMGS" -gt 0 ]; then
        PERCENT=$(echo "scale=1; $TRAIN_IMGS*100/150" | bc)
        echo "  Progress: ${PERCENT}%"
    fi
    echo "  Reports: $TRAIN_REPORTS"
    echo "  Size: $TRAIN_SIZE"
else
    echo "✗ Training directory not found"
fi

echo ""

# Test set
if [ -d "ctrate_volumes/test" ]; then
    TEST_IMGS=$(ls -1 ctrate_volumes/test/*.nii.gz 2>/dev/null | wc -l)
    TEST_REPORTS=$(ls -1 ctrate_reports/test/*.txt 2>/dev/null | wc -l)
    TEST_SIZE=$(du -sh ctrate_volumes/test/ 2>/dev/null | cut -f1)
    
    echo "✓ TEST SET:"
    echo "  Images: $TEST_IMGS files"
    echo "  Reports: $TEST_REPORTS files"
    echo "  Size: $TEST_SIZE"
else
    echo "✗ Test directory not found"
fi

echo ""
echo "=========================================="