"""
Image quality metrics calculation module
Provides PSNR, SSIM, LPIPS, and FID calculation functions for image comparison
"""
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple, Optional, List
import warnings

# Optional imports for advanced metrics
try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS not available. Install with: pip install lpips torch torchvision")

try:
    from scipy import linalg
    import torch
    import torchvision.models as models
    from torchvision import transforms
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    warnings.warn("FID dependencies not available. Install with: pip install scipy torch torchvision")


class ImageMetrics:
    """Class for calculating image quality metrics"""
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, processed: np.ndarray, 
                      data_range: Optional[float] = None) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
        
        Args:
            original: Original reference image
            processed: Processed/inpainted image to compare
            data_range: Data range of the image (default: auto-detect)
            
        Returns:
            PSNR value in decibels (dB)
        """
        try:
            # Ensure images have the same shape
            if original.shape != processed.shape:
                raise ValueError(f"Image shapes don't match: {original.shape} vs {processed.shape}")
            
            # Auto-detect data range if not provided
            if data_range is None:
                if original.dtype == np.uint8:
                    data_range = 255
                elif original.dtype == np.float32 or original.dtype == np.float64:
                    data_range = 1.0
                else:
                    data_range = np.max(original) - np.min(original)
            
            # Calculate PSNR
            psnr_value = psnr(original, processed, data_range=data_range)
            
            return float(psnr_value)
            
        except Exception as e:
            raise RuntimeError(f"Error calculating PSNR: {str(e)}")
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, processed: np.ndarray,
                      data_range: Optional[float] = None,
                      multichannel: bool = True) -> Tuple[float, np.ndarray]:
        """
        Calculate Structural Similarity Index (SSIM) between two images
        
        Args:
            original: Original reference image
            processed: Processed/inpainted image to compare
            data_range: Data range of the image (default: auto-detect)
            multichannel: Whether to treat the image as multichannel (color)
            
        Returns:
            Tuple of (SSIM value, SSIM difference image)
        """
        try:
            # Ensure images have the same shape
            if original.shape != processed.shape:
                raise ValueError(f"Image shapes don't match: {original.shape} vs {processed.shape}")
            
            # Auto-detect data range if not provided
            if data_range is None:
                if original.dtype == np.uint8:
                    data_range = 255
                elif original.dtype == np.float32 or original.dtype == np.float64:
                    data_range = 1.0
                else:
                    data_range = np.max(original) - np.min(original)
            
            # For color images, use channel_axis parameter
            if len(original.shape) == 3 and multichannel:
                ssim_value, ssim_diff = ssim(original, processed, 
                                           data_range=data_range,
                                           channel_axis=2,
                                           full=True)
            else:
                ssim_value, ssim_diff = ssim(original, processed,
                                           data_range=data_range,
                                           full=True)
            
            return float(ssim_value), ssim_diff
            
        except Exception as e:
            raise RuntimeError(f"Error calculating SSIM: {str(e)}")
    
    @staticmethod
    def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE) between two images
        
        Args:
            original: Original reference image
            processed: Processed/inpainted image to compare
            
        Returns:
            MSE value
        """
        try:
            # Ensure images have the same shape
            if original.shape != processed.shape:
                raise ValueError(f"Image shapes don't match: {original.shape} vs {processed.shape}")
            
            # Calculate MSE
            mse_value = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
            
            return float(mse_value)
            
        except Exception as e:
            raise RuntimeError(f"Error calculating MSE: {str(e)}")
    
    @staticmethod
    def calculate_lpips(original: np.ndarray, processed: np.ndarray, 
                       net: str = 'alex', use_gpu: bool = True) -> float:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two images
        
        LPIPS measures perceptual similarity using deep neural networks.
        Lower values indicate more similar images (more perceptually similar).
        
        Args:
            original: Original reference image (numpy array, RGB, 0-255 or 0-1)
            processed: Processed/inpainted image to compare
            net: Network to use for LPIPS ('alex', 'vgg', or 'squeeze')
            use_gpu: Whether to use GPU if available
            
        Returns:
            LPIPS value (lower is better, typically 0.0-1.0)
            
        Raises:
            RuntimeError: If LPIPS is not available or calculation fails
        """
        if not LPIPS_AVAILABLE:
            raise RuntimeError("LPIPS not available. Install with: pip install lpips torch torchvision")
        
        try:
            # Ensure images have the same shape
            if original.shape != processed.shape:
                raise ValueError(f"Image shapes don't match: {original.shape} vs {processed.shape}")
            
            # Initialize LPIPS model (cached after first call)
            if not hasattr(ImageMetrics, '_lpips_model'):
                device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
                ImageMetrics._lpips_model = lpips.LPIPS(net=net).to(device)
                ImageMetrics._lpips_device = device
            
            # Convert numpy arrays to torch tensors
            # LPIPS expects input in range [-1, 1] with shape (N, C, H, W)
            def preprocess_for_lpips(img):
                # Ensure float and normalize to [0, 1]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                # Convert from HWC to CHW
                if len(img.shape) == 3:
                    img = np.transpose(img, (2, 0, 1))
                
                # Convert to tensor and normalize to [-1, 1]
                tensor = torch.from_numpy(img).float()
                tensor = tensor * 2.0 - 1.0
                
                # Add batch dimension
                tensor = tensor.unsqueeze(0)
                
                return tensor.to(ImageMetrics._lpips_device)
            
            img1_tensor = preprocess_for_lpips(original)
            img2_tensor = preprocess_for_lpips(processed)
            
            # Calculate LPIPS
            with torch.no_grad():
                lpips_value = ImageMetrics._lpips_model(img1_tensor, img2_tensor)
            
            return float(lpips_value.item())
            
        except Exception as e:
            raise RuntimeError(f"Error calculating LPIPS: {str(e)}")
    
    @staticmethod
    def calculate_fid(original_images: List[np.ndarray], 
                     processed_images: List[np.ndarray],
                     use_gpu: bool = True) -> float:
        """
        Calculate FID (Fréchet Inception Distance) between two sets of images
        
        FID measures the distance between feature distributions of real and generated images.
        Lower values indicate more similar distributions (better quality).
        
        **IMPORTANT**: FID requires at least 50 images per set for reliable results.
        Computing FID on smaller batches can lead to unreliable and high-variance results.
        
        Args:
            original_images: List of original images (numpy arrays, RGB)
            processed_images: List of processed/inpainted images
            use_gpu: Whether to use GPU if available
            
        Returns:
            FID value (lower is better)
            
        Raises:
            RuntimeError: If FID dependencies not available or calculation fails
            ValueError: If fewer than 50 images provided in either set
        """
        if not FID_AVAILABLE:
            raise RuntimeError("FID dependencies not available. Install with: pip install scipy torch torchvision")
        
        # Validate minimum sample size (FID requirement)
        MIN_SAMPLES = 50
        if len(original_images) < MIN_SAMPLES or len(processed_images) < MIN_SAMPLES:
            raise ValueError(
                f"FID requires at least {MIN_SAMPLES} images per set for reliable results. "
                f"Provided: {len(original_images)} original, {len(processed_images)} processed. "
                f"Computing FID on small batches leads to unreliable and high-variance results."
            )
        
        try:
            # Initialize Inception model for feature extraction
            if not hasattr(ImageMetrics, '_inception_model'):
                device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
                inception = models.inception_v3(pretrained=True, transform_input=False)
                inception.fc = torch.nn.Identity()  # Remove final classification layer
                inception.eval()
                ImageMetrics._inception_model = inception.to(device)
                ImageMetrics._fid_device = device
            
            # Preprocessing transform for Inception
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            def extract_features(images):
                """Extract Inception features from a list of images"""
                features = []
                
                with torch.no_grad():
                    for img in images:
                        # Ensure uint8 format
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        
                        # Preprocess and extract features
                        img_tensor = preprocess(img).unsqueeze(0).to(ImageMetrics._fid_device)
                        feat = ImageMetrics._inception_model(img_tensor)
                        features.append(feat.cpu().numpy().flatten())
                
                return np.array(features)
            
            # Extract features from both sets
            features_original = extract_features(original_images)
            features_processed = extract_features(processed_images)
            
            # Calculate statistics (mean and covariance)
            mu1, sigma1 = features_original.mean(axis=0), np.cov(features_original, rowvar=False)
            mu2, sigma2 = features_processed.mean(axis=0), np.cov(features_processed, rowvar=False)
            
            # Calculate FID
            diff = mu1 - mu2
            
            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError(f"Imaginary component {m} too large")
                covmean = covmean.real
            
            fid_value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
            
            return float(fid_value)
            
        except Exception as e:
            raise RuntimeError(f"Error calculating FID: {str(e)}")
    
    @staticmethod
    def calculate_all_metrics(original: np.ndarray, processed: np.ndarray, 
                             include_lpips: bool = False) -> dict:
        """
        Calculate all available metrics between two images
        
        Args:
            original: Original reference image
            processed: Processed/inpainted image to compare
            include_lpips: Whether to include LPIPS calculation (slower, requires torch)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        try:
            metrics = {}
            
            # Calculate PSNR
            metrics['psnr'] = ImageMetrics.calculate_psnr(original, processed)
            
            # Calculate SSIM
            ssim_value, ssim_diff = ImageMetrics.calculate_ssim(original, processed)
            metrics['ssim'] = ssim_value
            metrics['ssim_diff'] = ssim_diff
            
            # Calculate MSE
            metrics['mse'] = ImageMetrics.calculate_mse(original, processed)
            
            # Calculate LPIPS if requested and available
            if include_lpips:
                if LPIPS_AVAILABLE:
                    try:
                        metrics['lpips'] = ImageMetrics.calculate_lpips(original, processed)
                    except Exception as e:
                        print(f"Warning: LPIPS calculation failed: {e}")
                        metrics['lpips'] = None
                else:
                    metrics['lpips'] = None
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Error calculating metrics: {str(e)}")
    
    @staticmethod
    def format_metrics(metrics: dict) -> str:
        """
        Format metrics dictionary into a readable string
        
        Args:
            metrics: Dictionary containing calculated metrics
            
        Returns:
            Formatted string with metrics
        """
        try:
            formatted = []
            
            if 'psnr' in metrics:
                formatted.append(f"PSNR: {metrics['psnr']:.2f} dB")
            
            if 'ssim' in metrics:
                formatted.append(f"SSIM: {metrics['ssim']:.4f}")
            
            if 'lpips' in metrics and metrics['lpips'] is not None:
                formatted.append(f"LPIPS: {metrics['lpips']:.4f}")
            
            if 'mse' in metrics:
                formatted.append(f"MSE: {metrics['mse']:.2f}")
            
            return " | ".join(formatted)
            
        except Exception as e:
            return f"Error formatting metrics: {str(e)}"


class MetricsComparison:
    """Class for comparing metrics and providing interpretation"""
    
    # Quality thresholds
    PSNR_EXCELLENT = 40.0
    PSNR_GOOD = 30.0
    PSNR_FAIR = 20.0
    
    SSIM_EXCELLENT = 0.95
    SSIM_GOOD = 0.85
    SSIM_FAIR = 0.7
    
    # LPIPS thresholds (lower is better for LPIPS, opposite of PSNR/SSIM)
    LPIPS_EXCELLENT = 0.1  # Very low distance = excellent
    LPIPS_GOOD = 0.3
    LPIPS_FAIR = 0.5
    
    @staticmethod
    def interpret_psnr(psnr_value: float) -> str:
        """Interpret PSNR value quality"""
        if psnr_value >= MetricsComparison.PSNR_EXCELLENT:
            return "Excellent"
        elif psnr_value >= MetricsComparison.PSNR_GOOD:
            return "Good"
        elif psnr_value >= MetricsComparison.PSNR_FAIR:
            return "Fair"
        else:
            return "Poor"
    
    @staticmethod
    def interpret_ssim(ssim_value: float) -> str:
        """Interpret SSIM value quality"""
        if ssim_value >= MetricsComparison.SSIM_EXCELLENT:
            return "Excellent"
        elif ssim_value >= MetricsComparison.SSIM_GOOD:
            return "Good"
        elif ssim_value >= MetricsComparison.SSIM_FAIR:
            return "Fair"
        else:
            return "Poor"
    
    @staticmethod
    def interpret_lpips(lpips_value: float) -> str:
        """
        Interpret LPIPS value quality
        
        Note: LPIPS is a distance metric, so LOWER is BETTER
        """
        if lpips_value <= MetricsComparison.LPIPS_EXCELLENT:
            return "Excellent"
        elif lpips_value <= MetricsComparison.LPIPS_GOOD:
            return "Good"
        elif lpips_value <= MetricsComparison.LPIPS_FAIR:
            return "Fair"
        else:
            return "Poor"
    
    @staticmethod
    def get_quality_summary(metrics: dict) -> str:
        """Get overall quality summary from metrics"""
        try:
            summaries = []
            
            if 'psnr' in metrics:
                psnr_quality = MetricsComparison.interpret_psnr(metrics['psnr'])
                summaries.append(f"PSNR: {psnr_quality}")
            
            if 'ssim' in metrics:
                ssim_quality = MetricsComparison.interpret_ssim(metrics['ssim'])
                summaries.append(f"SSIM: {ssim_quality}")
            
            if 'lpips' in metrics and metrics['lpips'] is not None:
                lpips_quality = MetricsComparison.interpret_lpips(metrics['lpips'])
                summaries.append(f"LPIPS: {lpips_quality}")
            
            return " | ".join(summaries)
            
        except Exception as e:
            return f"Error interpreting metrics: {str(e)}" 


# Helper functions for checking metric availability
def is_lpips_available() -> bool:
    """Check if LPIPS metric is available"""
    return LPIPS_AVAILABLE


def is_fid_available() -> bool:
    """Check if FID metric is available"""
    return FID_AVAILABLE


def get_available_metrics() -> dict:
    """
    Get dictionary of available metrics and their status
    
    Returns:
        Dictionary with metric names and availability status
    """
    return {
        'psnr': True,  # Always available (scikit-image)
        'ssim': True,  # Always available (scikit-image)
        'mse': True,   # Always available (numpy)
        'lpips': LPIPS_AVAILABLE,
        'fid': FID_AVAILABLE
    }


def print_metric_availability():
    """Print status of all metrics"""
    print("\n=== Image Quality Metrics Availability ===")
    print(f"PSNR:  ✓ Available (scikit-image)")
    print(f"SSIM:  ✓ Available (scikit-image)")
    print(f"MSE:   ✓ Available (numpy)")
    print(f"LPIPS: {'✓ Available' if LPIPS_AVAILABLE else '✗ Not Available (pip install lpips torch torchvision)'}")
    print(f"FID:   {'✓ Available' if FID_AVAILABLE else '✗ Not Available (pip install scipy torch torchvision)'}")
    print("=" * 44) 