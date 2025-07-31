"""
Image quality metrics calculation module
Provides PSNR and SSIM calculation functions for image comparison
"""
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple, Optional


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
    def calculate_all_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
        """
        Calculate all available metrics between two images
        
        Args:
            original: Original reference image
            processed: Processed/inpainted image to compare
            
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
            
            return " | ".join(summaries)
            
        except Exception as e:
            return f"Error interpreting metrics: {str(e)}" 