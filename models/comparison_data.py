"""
Comparison data model for managing image comparison state
"""
import numpy as np
from typing import Optional, Dict, Any
from .metrics import ImageMetrics, MetricsComparison


class ComparisonData:
    """Data model for image comparison functionality"""
    
    def __init__(self):
        self.original_image: Optional[np.ndarray] = None
        self.inpainted_image: Optional[np.ndarray] = None
        self.original_path: Optional[str] = None
        self.inpainted_path: Optional[str] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self._metrics_calculated = False
    
    @property
    def has_original_image(self) -> bool:
        """Check if original image is loaded"""
        return self.original_image is not None
    
    @property
    def has_inpainted_image(self) -> bool:
        """Check if inpainted image is loaded"""
        return self.inpainted_image is not None
    
    @property
    def has_both_images(self) -> bool:
        """Check if both images are loaded"""
        return self.has_original_image and self.has_inpainted_image
    
    @property
    def metrics_available(self) -> bool:
        """Check if metrics have been calculated"""
        return self._metrics_calculated and self.metrics is not None
    
    @property
    def is_ready_for_comparison(self) -> bool:
        """Check if ready for comparison"""
        return self.has_both_images
    
    def load_original_image(self, image_path: str) -> bool:
        """
        Load original image from file path
        
        Args:
            image_path: Path to the original image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.original_image = image
            self.original_path = image_path
            
            # Reset metrics when new image is loaded
            self._reset_metrics()
            
            return True
            
        except Exception as e:
            print(f"Error loading original image: {e}")
            return False
    
    def load_inpainted_image(self, image_path: str) -> bool:
        """
        Load inpainted image from file path
        
        Args:
            image_path: Path to the inpainted image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.inpainted_image = image
            self.inpainted_path = image_path
            
            # Reset metrics when new image is loaded
            self._reset_metrics()
            
            return True
            
        except Exception as e:
            print(f"Error loading inpainted image: {e}")
            return False
    
    def set_original_image_array(self, image_array: np.ndarray) -> bool:
        """
        Set original image from numpy array
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.original_image = image_array.copy()
            self.original_path = None  # No file path for array data
            
            # Reset metrics when new image is set
            self._reset_metrics()
            
            return True
            
        except Exception as e:
            print(f"Error setting original image array: {e}")
            return False
    
    def set_inpainted_image_array(self, image_array: np.ndarray) -> bool:
        """
        Set inpainted image from numpy array
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.inpainted_image = image_array.copy()
            self.inpainted_path = None  # No file path for array data
            
            # Reset metrics when new image is set
            self._reset_metrics()
            
            return True
            
        except Exception as e:
            print(f"Error setting inpainted image array: {e}")
            return False
    
    def calculate_metrics(self) -> bool:
        """
        Calculate comparison metrics between the two images
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_ready_for_comparison:
                return False
            
            # Ensure images have the same dimensions
            if self.original_image.shape != self.inpainted_image.shape:
                # Try to resize inpainted to match original
                import cv2
                target_shape = self.original_image.shape[:2]  # (height, width)
                self.inpainted_image = cv2.resize(
                    self.inpainted_image, 
                    (target_shape[1], target_shape[0]),  # cv2.resize expects (width, height)
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Calculate all metrics including LPIPS
            self.metrics = ImageMetrics.calculate_all_metrics(
                self.original_image, 
                self.inpainted_image,
                include_lpips=True
            )
            
            self._metrics_calculated = True
            return True
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self.metrics = None
            self._metrics_calculated = False
            return False
    
    def get_metrics_summary(self) -> str:
        """
        Get formatted summary of calculated metrics
        
        Returns:
            Formatted string with metrics information
        """
        if not self.metrics_available:
            return "No metrics calculated"
        
        return ImageMetrics.format_metrics(self.metrics)
    
    def get_quality_summary(self) -> str:
        """
        Get quality interpretation summary
        
        Returns:
            Formatted string with quality interpretation
        """
        if not self.metrics_available:
            return "No quality assessment available"
        
        return MetricsComparison.get_quality_summary(self.metrics)
    
    def get_detailed_metrics(self) -> Dict[str, str]:
        """
        Get detailed metrics with interpretations
        
        Returns:
            Dictionary with metric names, values, and interpretations
        """
        if not self.metrics_available:
            return {}
        
        detailed = {}
        
        if 'psnr' in self.metrics:
            psnr_val = self.metrics['psnr']
            psnr_quality = MetricsComparison.interpret_psnr(psnr_val)
            detailed['PSNR'] = f"{psnr_val:.2f} dB ({psnr_quality})"
        
        if 'ssim' in self.metrics:
            ssim_val = self.metrics['ssim']
            ssim_quality = MetricsComparison.interpret_ssim(ssim_val)
            detailed['SSIM'] = f"{ssim_val:.4f} ({ssim_quality})"
        
        if 'lpips' in self.metrics and self.metrics['lpips'] is not None:
            lpips_val = self.metrics['lpips']
            lpips_quality = MetricsComparison.interpret_lpips(lpips_val)
            detailed['LPIPS'] = f"{lpips_val:.4f} ({lpips_quality})"
        
        if 'mse' in self.metrics:
            mse_val = self.metrics['mse']
            detailed['MSE'] = f"{mse_val:.2f}"
        
        return detailed
    
    def get_ssim_difference_image(self) -> Optional[np.ndarray]:
        """
        Get SSIM difference image if available
        
        Returns:
            SSIM difference image or None
        """
        if self.metrics_available and 'ssim_diff' in self.metrics:
            return self.metrics['ssim_diff']
        return None
    
    def validate_images(self) -> tuple[bool, str]:
        """
        Validate that both images are suitable for comparison
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.has_original_image:
            return False, "Original image not loaded"
        
        if not self.has_inpainted_image:
            return False, "Inpainted image not loaded"
        
        # Check if images can be compared (after potential resizing)
        orig_shape = self.original_image.shape
        inp_shape = self.inpainted_image.shape
        
        # Must have same number of channels
        if len(orig_shape) != len(inp_shape):
            return False, "Images have different number of dimensions"
        
        if len(orig_shape) == 3 and orig_shape[2] != inp_shape[2]:
            return False, "Images have different number of channels"
        
        return True, "Images are valid for comparison"
    
    def reset(self):
        """Reset all comparison data"""
        self.original_image = None
        self.inpainted_image = None
        self.original_path = None
        self.inpainted_path = None
        self.metrics = None
        self._metrics_calculated = False
    
    def _reset_metrics(self):
        """Reset calculated metrics"""
        self.metrics = None
        self._metrics_calculated = False 