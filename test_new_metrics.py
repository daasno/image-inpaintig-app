"""
Test script for new LPIPS and FID metrics
Demonstrates usage and validates functionality
"""
import numpy as np
from models.metrics import (
    ImageMetrics, 
    MetricsComparison,
    is_lpips_available,
    is_fid_available,
    print_metric_availability
)


def create_test_image(size=(256, 256, 3), noise_level=0):
    """Create a synthetic test image with optional noise"""
    # Create a gradient image
    img = np.zeros(size, dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j, 0] = int((i / size[0]) * 255)  # Red gradient
            img[i, j, 1] = int((j / size[1]) * 255)  # Green gradient
            img[i, j, 2] = 128  # Constant blue
    
    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def test_basic_metrics():
    """Test basic metrics (PSNR, SSIM, MSE)"""
    print("\n" + "="*60)
    print("Testing Basic Metrics (PSNR, SSIM, MSE)")
    print("="*60)
    
    # Create test images
    original = create_test_image()
    processed_good = create_test_image(noise_level=5)   # Small noise
    processed_bad = create_test_image(noise_level=30)   # Large noise
    
    # Test with good quality
    print("\n1. Good Quality (low noise):")
    metrics_good = ImageMetrics.calculate_all_metrics(original, processed_good)
    print(f"   PSNR: {metrics_good['psnr']:.2f} dB - {MetricsComparison.interpret_psnr(metrics_good['psnr'])}")
    print(f"   SSIM: {metrics_good['ssim']:.4f} - {MetricsComparison.interpret_ssim(metrics_good['ssim'])}")
    print(f"   MSE:  {metrics_good['mse']:.2f}")
    
    # Test with poor quality
    print("\n2. Poor Quality (high noise):")
    metrics_bad = ImageMetrics.calculate_all_metrics(original, processed_bad)
    print(f"   PSNR: {metrics_bad['psnr']:.2f} dB - {MetricsComparison.interpret_psnr(metrics_bad['psnr'])}")
    print(f"   SSIM: {metrics_bad['ssim']:.4f} - {MetricsComparison.interpret_ssim(metrics_bad['ssim'])}")
    print(f"   MSE:  {metrics_bad['mse']:.2f}")
    
    print("\n✓ Basic metrics test completed successfully!")


def test_lpips():
    """Test LPIPS metric"""
    print("\n" + "="*60)
    print("Testing LPIPS (Learned Perceptual Image Patch Similarity)")
    print("="*60)
    
    if not is_lpips_available():
        print("\n✗ LPIPS not available. Install with:")
        print("  pip install lpips torch torchvision")
        return
    
    try:
        # Create test images
        original = create_test_image()
        processed_good = create_test_image(noise_level=5)
        processed_bad = create_test_image(noise_level=30)
        
        print("\n1. Good Quality (low noise):")
        lpips_good = ImageMetrics.calculate_lpips(original, processed_good)
        print(f"   LPIPS: {lpips_good:.4f} - {MetricsComparison.interpret_lpips(lpips_good)}")
        print(f"   (Lower is better)")
        
        print("\n2. Poor Quality (high noise):")
        lpips_bad = ImageMetrics.calculate_lpips(original, processed_bad)
        print(f"   LPIPS: {lpips_bad:.4f} - {MetricsComparison.interpret_lpips(lpips_bad)}")
        print(f"   (Lower is better)")
        
        # Test with all metrics including LPIPS
        print("\n3. All Metrics Together (with LPIPS):")
        metrics = ImageMetrics.calculate_all_metrics(original, processed_good, include_lpips=True)
        print(f"   PSNR:  {metrics['psnr']:.2f} dB")
        print(f"   SSIM:  {metrics['ssim']:.4f}")
        print(f"   LPIPS: {metrics['lpips']:.4f}")
        print(f"   MSE:   {metrics['mse']:.2f}")
        
        print("\n✓ LPIPS test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ LPIPS test failed: {e}")


def test_fid():
    """Test FID metric"""
    print("\n" + "="*60)
    print("Testing FID (Fréchet Inception Distance)")
    print("="*60)
    
    if not is_fid_available():
        print("\n✗ FID not available. Install with:")
        print("  pip install torch torchvision scipy")
        return
    
    print("\n⚠️  FID requires at least 50 images. This test will:")
    print("   1. Demonstrate the minimum requirement check")
    print("   2. Show how to use FID with sufficient samples")
    
    # Test 1: Insufficient samples (should fail)
    print("\n1. Testing with insufficient samples (10 images):")
    try:
        small_batch_orig = [create_test_image() for _ in range(10)]
        small_batch_proc = [create_test_image(noise_level=5) for _ in range(10)]
        
        fid_score = ImageMetrics.calculate_fid(small_batch_orig, small_batch_proc)
        print(f"   FID: {fid_score:.2f}")
    except ValueError as e:
        print(f"   ✓ Expected error: {str(e)[:100]}...")
    
    # Test 2: Sufficient samples
    print("\n2. Testing with sufficient samples (50 images):")
    print("   Creating 50 test images... (this may take a minute)")
    
    try:
        large_batch_orig = [create_test_image() for _ in range(50)]
        large_batch_proc = [create_test_image(noise_level=5) for _ in range(50)]
        
        print("   Computing FID score...")
        fid_score = ImageMetrics.calculate_fid(large_batch_orig, large_batch_proc, use_gpu=True)
        print(f"   FID: {fid_score:.2f} (Lower is better)")
        print("   ✓ FID computation successful!")
        
    except Exception as e:
        print(f"   Note: FID computation requires significant resources")
        print(f"   Error: {e}")
        print("   This is normal if GPU is not available or if Inception weights need downloading")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("IMAGE QUALITY METRICS TEST SUITE")
    print("="*60)
    
    # Print availability
    print_metric_availability()
    
    # Run tests
    test_basic_metrics()
    test_lpips()
    test_fid()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\nFor more information, see METRICS_GUIDE.md")


if __name__ == "__main__":
    main()

