"""
Advanced Memory Management System
"""
import gc
import cv2
import numpy as np
import psutil
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import weakref

from .settings import AppSettings, MemoryMode
from .logging_config import AppLogger, LogCategory, MemoryMonitor


class ResizeStrategy(Enum):
    """Image resize strategies"""
    NONE = "none"
    MAINTAIN_ASPECT = "maintain_aspect" 
    SMART_CROP = "smart_crop"
    ADAPTIVE = "adaptive"


@dataclass
class MemoryInfo:
    """Memory usage information"""
    used_mb: float
    available_mb: float
    total_mb: float
    percent_used: float
    gpu_memory_mb: float = 0.0
    gpu_available_mb: float = 0.0


class ImageProcessor:
    """Intelligent image processing with memory management"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.logger = AppLogger.get_logger(LogCategory.MEMORY)
        
    def validate_and_resize_image(self, image: np.ndarray, image_type: str = "input") -> Tuple[np.ndarray, bool]:
        """
        Validate image size and resize if necessary
        
        Args:
            image: Input image array
            image_type: Type of image ("input", "mask", "result")
            
        Returns:
            Tuple of (processed_image, was_resized)
        """
        original_shape = image.shape
        memory_limits = self.settings.get_memory_limits()
        
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Check if resize is needed
        needs_resize = False
        target_width, target_height = width, height
        
        if total_pixels > memory_limits['max_image_pixels']:
            needs_resize = True
            # Calculate target size maintaining aspect ratio
            aspect_ratio = width / height
            max_pixels = memory_limits['max_image_pixels']
            
            if aspect_ratio > 1:  # Wider than tall
                target_width = int(np.sqrt(max_pixels * aspect_ratio))
                target_height = int(target_width / aspect_ratio)
            else:  # Taller than wide
                target_height = int(np.sqrt(max_pixels / aspect_ratio))
                target_width = int(target_height * aspect_ratio)
            
            # Ensure dimensions don't exceed max_dimension
            max_dim = memory_limits['max_dimension']
            if target_width > max_dim:
                target_width = max_dim
                target_height = int(max_dim / aspect_ratio)
            if target_height > max_dim:
                target_height = max_dim
                target_width = int(max_dim * aspect_ratio)
        
        elif max(width, height) > memory_limits['max_dimension']:
            needs_resize = True
            # Scale down to fit max dimension
            scale = memory_limits['max_dimension'] / max(width, height)
            target_width = int(width * scale)
            target_height = int(height * scale)
        
        if needs_resize:
            self.logger.warning(
                f"Resizing {image_type} image from {width}x{height} to {target_width}x{target_height}"
            )
            
            # Choose interpolation method based on image type
            if image_type == "mask":
                interpolation = cv2.INTER_NEAREST  # Preserve binary mask
            else:
                interpolation = cv2.INTER_LANCZOS4  # High quality for images
            
            resized_image = cv2.resize(
                image, 
                (target_width, target_height),
                interpolation=interpolation
            )
            
            # Log memory savings
            original_size_mb = self._calculate_image_memory(original_shape)
            new_size_mb = self._calculate_image_memory(resized_image.shape)
            
            self.logger.info(
                f"Memory reduction: {original_size_mb:.1f}MB -> {new_size_mb:.1f}MB "
                f"({((original_size_mb - new_size_mb) / original_size_mb * 100):.1f}% savings)"
            )
            
            return resized_image, True
        
        return image, False
    
    def _calculate_image_memory(self, shape: tuple) -> float:
        """Calculate memory usage of an image in MB"""
        if len(shape) == 3:
            height, width, channels = shape
        else:
            height, width = shape
            channels = 1
        
        # Estimate memory for different data types
        bytes_per_pixel = 4 if channels == 3 else 1  # Assume float32 for RGB, uint8 for grayscale
        total_bytes = height * width * channels * bytes_per_pixel
        
        return total_bytes / (1024 * 1024)
    
    def optimize_image_for_processing(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for processing (data type, memory layout)"""
        # Ensure contiguous memory layout
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
            self.logger.debug("Made image memory contiguous")
        
        # Optimize data type
        if image.dtype != np.uint8 and np.max(image) <= 255:
            image = image.astype(np.uint8)
            self.logger.debug("Converted image to uint8 for memory efficiency")
        
        return image


class GPUMemoryManager:
    """GPU memory management for CUDA operations"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.logger = AppLogger.get_logger(LogCategory.MEMORY)
        self._gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            from numba import cuda
            return cuda.is_available()
        except ImportError:
            return False
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information"""
        if not self._gpu_available:
            return {'total': 0.0, 'used': 0.0, 'free': 0.0}
        
        try:
            from numba import cuda
            
            # Get memory info from CUDA context
            meminfo = cuda.current_context().get_memory_info()
            free_bytes, total_bytes = meminfo
            used_bytes = total_bytes - free_bytes
            
            return {
                'total': total_bytes / (1024 * 1024),  # MB
                'used': used_bytes / (1024 * 1024),    # MB
                'free': free_bytes / (1024 * 1024)     # MB
            }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory info: {e}")
            return {'total': 0.0, 'used': 0.0, 'free': 0.0}
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if not self._gpu_available:
            return
        
        try:
            from numba import cuda
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            cuda.current_context().memory_manager.clear()
            
            self.logger.info("GPU memory cleanup completed")
        except Exception as e:
            self.logger.warning(f"GPU memory cleanup failed: {e}")
    
    def estimate_gpu_memory_needed(self, image_shape: tuple, patch_size: int) -> float:
        """Estimate GPU memory needed for inpainting operation"""
        height, width = image_shape[:2]
        channels = image_shape[2] if len(image_shape) > 2 else 1
        
        # Base image memory (original + working + lab if color)
        base_memory = height * width * channels * 4  # float32
        if channels == 3:
            base_memory += height * width * 3 * 4  # LAB image
        
        # Mask and confidence maps
        base_memory += height * width * 4 * 2  # mask + confidence (float32)
        
        # Additional buffers for processing
        patch_memory = patch_size * patch_size * channels * 4 * 100  # Estimate for patch operations
        
        total_bytes = base_memory + patch_memory
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def check_gpu_memory_sufficient(self, image_shape: tuple, patch_size: int) -> Tuple[bool, str]:
        """Check if GPU has sufficient memory for the operation"""
        if not self._gpu_available:
            return False, "GPU not available"
        
        needed_mb = self.estimate_gpu_memory_needed(image_shape, patch_size)
        gpu_info = self.get_gpu_memory_info()
        
        available_mb = gpu_info['free']
        usable_mb = available_mb * self.settings.gpu_memory_fraction
        
        if needed_mb > usable_mb:
            return False, f"Insufficient GPU memory: need {needed_mb:.0f}MB, available {usable_mb:.0f}MB"
        
        if needed_mb > available_mb * 0.8:  # Warning threshold
            return True, f"GPU memory will be heavily utilized: {needed_mb:.0f}MB of {available_mb:.0f}MB"
        
        return True, ""


class SystemMemoryManager:
    """System RAM memory management"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.logger = AppLogger.get_logger(LogCategory.MEMORY)
        self._image_cache = weakref.WeakValueDictionary()
        
    def get_system_memory_info(self) -> MemoryInfo:
        """Get system memory information"""
        try:
            virtual_mem = psutil.virtual_memory()
            process = psutil.Process()
            process_mem = process.memory_info()
            
            return MemoryInfo(
                used_mb=process_mem.rss / (1024 * 1024),
                available_mb=virtual_mem.available / (1024 * 1024),
                total_mb=virtual_mem.total / (1024 * 1024),
                percent_used=virtual_mem.percent
            )
        except ImportError:
            # Fallback if psutil not available
            return MemoryInfo(
                used_mb=0.0,
                available_mb=1024.0,  # Assume 1GB available
                total_mb=4096.0,      # Assume 4GB total
                percent_used=50.0
            )
    
    def check_system_memory_sufficient(self, operation_memory_mb: float) -> Tuple[bool, str]:
        """Check if system has sufficient memory for operation"""
        memory_info = self.get_system_memory_info()
        
        # Conservative check - ensure we have 150% of needed memory available
        safety_factor = 1.5
        needed_memory = operation_memory_mb * safety_factor
        
        if needed_memory > memory_info.available_mb:
            return False, f"Insufficient system memory: need {needed_memory:.0f}MB, available {memory_info.available_mb:.0f}MB"
        
        # Warning if using more than 70% of available memory
        if needed_memory > memory_info.available_mb * 0.7:
            return True, f"High memory usage warning: will use {needed_memory:.0f}MB of {memory_info.available_mb:.0f}MB available"
        
        return True, ""
    
    def cleanup_system_memory(self):
        """Clean up system memory"""
        # Clear image cache
        self._image_cache.clear()
        
        # Force garbage collection
        collected = gc.collect()
        
        self.logger.info(f"System memory cleanup: collected {collected} objects")
        
        # Log memory status after cleanup
        MemoryMonitor.log_memory_status("After cleanup")
    
    def cache_image(self, key: str, image: np.ndarray):
        """Cache an image with weak references"""
        if self.settings.cache_enabled:
            self._image_cache[key] = image
    
    def get_cached_image(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached image"""
        return self._image_cache.get(key)


class MemoryManager:
    """Central memory management coordinator"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.logger = AppLogger.get_logger(LogCategory.MEMORY)
        
        self.image_processor = ImageProcessor(settings)
        self.gpu_manager = GPUMemoryManager(settings)
        self.system_manager = SystemMemoryManager(settings)
        
        # Track memory usage
        self._memory_warnings_shown = set()
        
    def prepare_images_for_processing(self, input_image: np.ndarray, 
                                    mask_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare images for processing with memory optimization
        
        Returns:
            Tuple of (processed_input, processed_mask, info_dict)
        """
        info = {
            'input_resized': False,
            'mask_resized': False,
            'memory_warnings': [],
            'optimizations_applied': []
        }
        
        # Log initial memory status
        MemoryMonitor.log_memory_status("Before image preparation")
        
        # Process input image
        processed_input, input_resized = self.image_processor.validate_and_resize_image(
            input_image, "input"
        )
        info['input_resized'] = input_resized
        
        if input_resized:
            info['optimizations_applied'].append("Input image resized for memory efficiency")
        
        # Process mask image (resize to match input if needed)
        if input_resized:
            h, w = processed_input.shape[:2]
            processed_mask = cv2.resize(mask_image, (w, h), interpolation=cv2.INTER_NEAREST)
            info['mask_resized'] = True
            info['optimizations_applied'].append("Mask resized to match input image")
        else:
            processed_mask, mask_resized = self.image_processor.validate_and_resize_image(
                mask_image, "mask"
            )
            info['mask_resized'] = mask_resized
            if mask_resized:
                info['optimizations_applied'].append("Mask image resized for memory efficiency")
        
        # Optimize memory layout
        processed_input = self.image_processor.optimize_image_for_processing(processed_input)
        processed_mask = self.image_processor.optimize_image_for_processing(processed_mask)
        
        # Check memory requirements
        self._check_memory_requirements(processed_input.shape, info)
        
        # Log final memory status
        MemoryMonitor.log_memory_status("After image preparation")
        
        return processed_input, processed_mask, info
    
    def _check_memory_requirements(self, image_shape: tuple, info: Dict[str, Any]):
        """Check memory requirements for processing"""
        # Check system memory
        estimated_memory = self.image_processor._calculate_image_memory(image_shape) * 3  # Conservative estimate
        
        sufficient, message = self.system_manager.check_system_memory_sufficient(estimated_memory)
        if not sufficient:
            info['memory_warnings'].append(f"System memory: {message}")
        elif message:  # Warning
            info['memory_warnings'].append(f"System memory warning: {message}")
        
        # Check GPU memory if using GPU
        if self.settings.preferred_implementation == "GPU":
            patch_size = self.settings.default_patch_size
            sufficient, message = self.gpu_manager.check_gpu_memory_sufficient(image_shape, patch_size)
            if not sufficient:
                info['memory_warnings'].append(f"GPU memory: {message}")
            elif message:  # Warning
                info['memory_warnings'].append(f"GPU memory warning: {message}")
    
    def cleanup_after_processing(self):
        """Clean up memory after processing operations"""
        self.logger.info("Starting post-processing memory cleanup")
        
        # Clean up GPU memory
        self.gpu_manager.cleanup_gpu_memory()
        
        # Clean up system memory
        self.system_manager.cleanup_system_memory()
        
        # Clear memory warnings
        self._memory_warnings_shown.clear()
        
        self.logger.info("Memory cleanup completed")
    
    def get_memory_status_report(self) -> Dict[str, Any]:
        """Get comprehensive memory status report"""
        system_info = self.system_manager.get_system_memory_info()
        gpu_info = self.gpu_manager.get_gpu_memory_info()
        
        return {
            'system': {
                'used_mb': system_info.used_mb,
                'available_mb': system_info.available_mb,
                'total_mb': system_info.total_mb,
                'percent_used': system_info.percent_used
            },
            'gpu': gpu_info,
            'memory_mode': self.settings.memory_mode,
            'limits': self.settings.get_memory_limits(),
            'cache_enabled': self.settings.cache_enabled
        }
    
    def suggest_memory_optimizations(self, image_shape: tuple) -> list:
        """Suggest memory optimizations based on current state"""
        suggestions = []
        
        memory_info = self.system_manager.get_system_memory_info()
        memory_limits = self.settings.get_memory_limits()
        
        height, width = image_shape[:2]
        total_pixels = height * width
        
        # Check if image is large
        if total_pixels > memory_limits['warning_threshold']:
            suggestions.append("Consider switching to Conservative memory mode for large images")
        
        # Check system memory
        if memory_info.percent_used > 80:
            suggestions.append("System memory is high - close other applications")
            suggestions.append("Consider reducing image size or using Conservative mode")
        
        # Check memory mode recommendations
        if self.settings.memory_mode == MemoryMode.PERFORMANCE.value and memory_info.available_mb < 2048:
            suggestions.append("Switch to Balanced memory mode with less than 2GB available RAM")
        
        # GPU-specific suggestions
        if self.settings.preferred_implementation == "GPU":
            gpu_info = self.gpu_manager.get_gpu_memory_info()
            if gpu_info['free'] < 1024:  # Less than 1GB GPU memory
                suggestions.append("Limited GPU memory - consider using CPU implementation for large images")
        
        return suggestions 