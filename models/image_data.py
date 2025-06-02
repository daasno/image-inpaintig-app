"""
Image data model for handling image operations
"""
import os
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from config.settings import AppConstants


class ImageData:
    """Model for handling image data and operations"""
    
    def __init__(self):
        self.input_image: Optional[np.ndarray] = None
        self.mask_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        
        self.input_image_path: Optional[str] = None
        self.mask_image_path: Optional[str] = None
        
    @property
    def has_input_image(self) -> bool:
        """Check if input image is loaded"""
        return self.input_image is not None
    
    @property
    def has_mask_image(self) -> bool:
        """Check if mask image is loaded"""
        return self.mask_image is not None
    
    @property
    def has_result_image(self) -> bool:
        """Check if result image is available"""
        return self.result_image is not None
    
    @property
    def is_ready_for_processing(self) -> bool:
        """Check if both input and mask are loaded"""
        return self.has_input_image and self.has_mask_image
    
    @property
    def image_dimensions(self) -> Optional[Tuple[int, int]]:
        """Get image dimensions (height, width)"""
        if self.input_image is not None:
            return self.input_image.shape[:2]
        return None
    
    def load_input_image(self, file_path: str) -> bool:
        """
        Load input image from file path
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File does not exist: {file_path}")
            
            # Load with OpenCV
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read the image file")
            
            # Check image size for memory safety
            height, width = image.shape[:2]
            total_pixels = height * width
            
            if total_pixels > AppConstants.MAX_SAFE_IMAGE_PIXELS:
                raise ValueError(
                    f"Image too large ({width}x{height} = {total_pixels:,} pixels). "
                    f"Maximum safe size is {AppConstants.MAX_SAFE_IMAGE_PIXELS:,} pixels."
                )
            
            self.input_image = image
            self.input_image_path = file_path
            
            # If mask is already loaded, resize it to match
            if self.mask_image is not None:
                self._resize_mask_to_match_image()
            
            return True
            
        except Exception as e:
            print(f"Error loading input image: {e}")
            return False
    
    def load_mask_image(self, file_path: str) -> bool:
        """
        Load mask image from file path
        
        Args:
            file_path: Path to the mask image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File does not exist: {file_path}")
            
            # Load as grayscale
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Could not read the mask file")
            
            # Ensure binary mask (0 or 255)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            self.mask_image = mask
            self.mask_image_path = file_path
            
            # Resize mask to match input image if loaded
            if self.input_image is not None:
                self._resize_mask_to_match_image()
            
            return True
            
        except Exception as e:
            print(f"Error loading mask image: {e}")
            return False
    
    def set_mask_image_array(self, mask_array: np.ndarray) -> bool:
        """
        Set mask image from numpy array (e.g., from mask editor)
        
        Args:
            mask_array: Numpy array representing the mask
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if mask_array is None:
                raise ValueError("Mask array is None")
            
            # Ensure it's a 2D array
            if len(mask_array.shape) != 2:
                raise ValueError("Mask must be a 2D array")
            
            # Ensure it's uint8 and binary
            if mask_array.dtype != np.uint8:
                mask_array = mask_array.astype(np.uint8)
            
            # Ensure binary values (0 or 255)
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
            
            self.mask_image = mask_array
            self.mask_image_path = None  # No file path for created masks
            
            # Resize mask to match input image if loaded
            if self.input_image is not None:
                self._resize_mask_to_match_image()
            
            return True
            
        except Exception as e:
            print(f"Error setting mask image array: {e}")
            return False
    
    def _resize_mask_to_match_image(self):
        """Resize mask to match input image dimensions"""
        if self.input_image is not None and self.mask_image is not None:
            h, w = self.input_image.shape[:2]
            self.mask_image = cv2.resize(
                self.mask_image, (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
    
    def validate_images(self) -> Tuple[bool, str]:
        """
        Validate that images are ready for processing
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not self.has_input_image:
            return False, "No input image loaded"
        
        if not self.has_mask_image:
            return False, "No mask image loaded"
        
        # Check dimensions match
        if self.input_image.shape[:2] != self.mask_image.shape:
            return False, "Input image and mask dimensions don't match"
        
        # Check if mask has any pixels to inpaint
        if np.sum(self.mask_image > 127) == 0:
            return False, "Mask is empty (no pixels to inpaint)"
        
        return True, ""
    
    def save_result_image(self, file_path: str) -> bool:
        """
        Save result image to file
        
        Args:
            file_path: Path where to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.has_result_image:
                raise ValueError("No result image to save")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the image
            success = cv2.imwrite(file_path, self.result_image)
            if not success:
                raise ValueError("Failed to write image file")
            
            return True
            
        except Exception as e:
            print(f"Error saving result image: {e}")
            return False
    
    def set_result_image(self, result: np.ndarray):
        """Set the result image"""
        if result is not None:
            self.result_image = np.copy(result)
        else:
            self.result_image = None
    
    def reset(self):
        """Reset all image data"""
        self.input_image = None
        self.mask_image = None
        self.result_image = None
        self.input_image_path = None
        self.mask_image_path = None
    
    def get_image_info(self) -> dict:
        """Get information about loaded images"""
        info = {
            'has_input': self.has_input_image,
            'has_mask': self.has_mask_image,
            'has_result': self.has_result_image,
            'dimensions': self.image_dimensions,
            'input_path': self.input_image_path,
            'mask_path': self.mask_image_path
        }
        
        if self.input_image is not None:
            info['input_channels'] = len(self.input_image.shape) if len(self.input_image.shape) == 2 else self.input_image.shape[2]
            info['input_dtype'] = str(self.input_image.dtype)
        
        return info 