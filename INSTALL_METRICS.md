# Quick Start: Installing LPIPS and FID Metrics

This guide will help you install the required dependencies for LPIPS and FID metrics.

## Option 1: Install Everything (Recommended)

```bash
pip install -r requirements.txt
```

This installs all dependencies including the advanced metrics.

---

## Option 2: Add Only Advanced Metrics

If you already have the basic application working, add just the new metrics:

```bash
pip install torch torchvision lpips
```

**Note**: `scipy` is required for FID and should already be in your environment.

---

## Option 3: CPU-Only PyTorch (Smaller Download)

If you don't have a GPU or want a smaller install:

```bash
# For CPU-only PyTorch (much smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install lpips
```

---

## Verify Installation

Run this command to check if metrics are available:

```bash
python test_new_metrics.py
```

Or in Python:

```python
from models.metrics import print_metric_availability
print_metric_availability()
```

Expected output:
```
=== Image Quality Metrics Availability ===
PSNR:  ✓ Available (scikit-image)
SSIM:  ✓ Available (scikit-image)
MSE:   ✓ Available (numpy)
LPIPS: ✓ Available
FID:   ✓ Available
============================================
```

---

## Troubleshooting

### LPIPS Not Available

```bash
pip install lpips torch torchvision
```

### FID Not Available

```bash
pip install torch torchvision scipy
```

### CUDA Out of Memory

If you get CUDA memory errors:
- Use CPU mode: `use_gpu=False` in method calls
- Close other GPU-using applications
- Reduce batch size for FID

### Import Errors

Make sure you're in the correct Python environment:
```bash
# Check Python version (needs 3.7+)
python --version

# Check installed packages
pip list | grep -E "torch|lpips|scipy"
```

---

## Download Sizes

Approximate download sizes:

| Package | Size | Notes |
|---------|------|-------|
| torch (CUDA) | ~2 GB | Full PyTorch with GPU support |
| torch (CPU) | ~200 MB | CPU-only version |
| torchvision | ~20 MB | Image transforms and models |
| lpips | ~5 MB | + pretrained weights (~50 MB on first use) |

**First run**: LPIPS and FID will download pretrained model weights:
- LPIPS (AlexNet): ~50 MB
- FID (Inception v3): ~100 MB

These are downloaded once and cached.

---

## GPU Acceleration

To use GPU acceleration:

1. **Check CUDA availability**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

2. **Install CUDA-enabled PyTorch**:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check https://pytorch.org/get-started/locally/ for your specific CUDA version.

---

## Performance Comparison

### LPIPS Speed
- **CPU**: ~1-2 seconds per image pair
- **GPU**: ~0.1-0.3 seconds per image pair
- **Speedup**: ~10x faster on GPU

### FID Speed (for 50 images)
- **CPU**: ~5-10 minutes
- **GPU**: ~1-2 minutes
- **Speedup**: ~5x faster on GPU

---

## Minimal Installation (No Advanced Metrics)

If you don't want LPIPS/FID:

```bash
pip install PySide6 opencv-python numpy scipy scikit-image numba matplotlib seaborn pandas
```

The application will work with traditional metrics (PSNR, SSIM, MSE) only.

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run `test_new_metrics.py` to verify
3. ✅ Read `METRICS_GUIDE.md` for usage details
4. ✅ Try comparison mode with LPIPS
5. ✅ Run batch processing with 50+ images to test FID

Enjoy the new metrics! 🎉

