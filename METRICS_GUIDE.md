# Image Quality Metrics Guide

This guide explains the image quality metrics available in the inpainting application.

## Available Metrics

### 1. PSNR (Peak Signal-to-Noise Ratio)
- **Type**: Pixel-based metric
- **Range**: 0-100+ dB (higher is better)
- **Description**: Measures the ratio between maximum signal power and corrupting noise
- **Always Available**: Yes (scikit-image)

**Quality Thresholds:**
- Excellent: ≥ 40.0 dB
- Good: ≥ 30.0 dB
- Fair: ≥ 20.0 dB
- Poor: < 20.0 dB

**Use Case**: Quick pixel-level similarity assessment

---

### 2. SSIM (Structural Similarity Index)
- **Type**: Structural similarity metric
- **Range**: 0.0-1.0 (higher is better)
- **Description**: Measures structural similarity considering luminance, contrast, and structure
- **Always Available**: Yes (scikit-image)
- **Extra Feature**: Provides difference map visualization

**Quality Thresholds:**
- Excellent: ≥ 0.95
- Good: ≥ 0.85
- Fair: ≥ 0.70
- Poor: < 0.70

**Use Case**: Assesses perceptual quality better than PSNR

---

### 3. MSE (Mean Squared Error)
- **Type**: Pixel-based error metric
- **Range**: 0-∞ (lower is better)
- **Description**: Average squared difference between pixel values
- **Always Available**: Yes (numpy)

**Use Case**: Simple error measurement, related to PSNR

---

### 4. LPIPS (Learned Perceptual Image Patch Similarity) ✨ NEW
- **Type**: Deep learning-based perceptual metric
- **Range**: 0.0-1.0+ (lower is better)
- **Description**: Uses deep neural networks (AlexNet/VGG) to measure perceptual similarity
- **Requires**: `torch`, `torchvision`, `lpips`
- **GPU Accelerated**: Yes (if CUDA available)

**Quality Thresholds:**
- Excellent: ≤ 0.1
- Good: ≤ 0.3
- Fair: ≤ 0.5
- Poor: > 0.5

**Use Case**: Best for assessing perceptual quality as humans would perceive it

**Installation:**
```bash
pip install torch torchvision lpips
```

**Key Features:**
- More aligned with human perception than PSNR/SSIM
- Captures semantic differences
- GPU acceleration supported
- Can use different backbones: AlexNet (default), VGG, SqueezeNet

**Note**: LPIPS is a distance metric, so **LOWER values mean MORE similar images** (opposite of PSNR/SSIM)

---

### 5. FID (Fréchet Inception Distance) ✨ NEW
- **Type**: Distribution-based metric for image sets
- **Range**: 0-∞ (lower is better)
- **Description**: Measures the distance between feature distributions of real and generated images
- **Requires**: `torch`, `torchvision`, `scipy`
- **GPU Accelerated**: Yes (if CUDA available)

**⚠️ IMPORTANT LIMITATIONS:**
- **Minimum 50 images required** per set for reliable results
- Only available for batch processing
- Computing FID on small batches leads to unreliable and high-variance results
- Not suitable for single image comparisons

**Installation:**
```bash
pip install torch torchvision scipy
```

**Key Features:**
- Uses Inception v3 network for feature extraction
- Captures overall quality of a set of images
- Standard metric for evaluating generative models
- Requires sufficient sample size for statistical reliability

**Use Case**: 
- Evaluating overall quality of batch inpainting results
- Comparing different parameter configurations on large datasets
- Research and quantitative analysis

**Why 50+ images?**
FID estimates the distance between two multivariate Gaussian distributions. With fewer samples:
- Covariance matrix estimation becomes unstable
- Results have high variance and are unreliable
- Statistical significance is questionable

---

## Usage Examples

### Single Image Comparison (with LPIPS)

```python
from models.metrics import ImageMetrics

# Calculate all metrics including LPIPS
metrics = ImageMetrics.calculate_all_metrics(
    original_image, 
    processed_image,
    include_lpips=True
)

print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"LPIPS: {metrics['lpips']:.4f}")
```

### Batch Processing with FID

```python
from models.metrics import ImageMetrics

# Prepare lists of images (must have 50+ images each)
original_images = [img1, img2, ..., img50, ...]  # 50+ images
processed_images = [proc1, proc2, ..., proc50, ...]  # 50+ images

# Calculate FID
fid_score = ImageMetrics.calculate_fid(
    original_images,
    processed_images,
    use_gpu=True
)

print(f"FID Score: {fid_score:.2f}")
```

### Check Metric Availability

```python
from models.metrics import get_available_metrics, print_metric_availability

# Print status
print_metric_availability()

# Get availability dict
available = get_available_metrics()
if available['lpips']:
    print("LPIPS is available!")
if available['fid']:
    print("FID is available!")
```

---

## Comparison Panel Integration

The comparison panel now supports LPIPS calculation:

1. Load original and inpainted images
2. Check "Include LPIPS" option (if available)
3. Click "Calculate Metrics"
4. View results including LPIPS score and interpretation

---

## Batch Exhaustive Research

For batch processing with 50+ image pairs:
- LPIPS will be calculated per image pair
- FID will be calculated across all images in the batch
- Results include both per-image and distribution-level metrics

---

## Performance Considerations

### LPIPS
- **CPU**: ~1-2 seconds per image pair
- **GPU**: ~0.1-0.3 seconds per image pair
- **Memory**: ~500MB (model) + image batch
- **Recommendation**: Use GPU for faster processing

### FID
- **CPU**: ~5-10 minutes for 50-100 images
- **GPU**: ~1-2 minutes for 50-100 images
- **Memory**: ~1GB (Inception model) + features
- **Recommendation**: 
  - Use GPU for faster processing
  - Only enable for large batches (50+ images)
  - Pre-download Inception model weights

---

## Metric Selection Guidelines

| Scenario | Recommended Metrics |
|----------|-------------------|
| Quick quality check | PSNR, SSIM |
| Perceptual quality assessment | LPIPS, SSIM |
| Single image comparison | PSNR, SSIM, LPIPS |
| Batch evaluation (< 50 images) | PSNR, SSIM, LPIPS |
| Batch evaluation (50+ images) | PSNR, SSIM, LPIPS, FID |
| Research/publication | All metrics |
| Real-time processing | PSNR, SSIM (faster) |

---

## Troubleshooting

### LPIPS not available
```bash
# Install required packages
pip install lpips torch torchvision
```

### FID not available
```bash
# Install required packages
pip install torch torchvision scipy
```

### CUDA out of memory
- Reduce batch size for FID calculation
- Use CPU mode: `use_gpu=False`
- Process images in smaller groups

### FID ValueError: Requires 50+ images
- Ensure you have at least 50 images in each set
- Consider using LPIPS instead for smaller batches
- Wait until more images are processed

---

## References

1. **PSNR/SSIM**: Available through scikit-image
2. **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)
3. **FID**: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS 2017)

---

## Version History

- **v1.0**: Added LPIPS and FID metrics
- Initial metrics: PSNR, SSIM, MSE

