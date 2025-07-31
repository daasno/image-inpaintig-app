"""
Batch processing data model for handling multiple image/mask pairs
"""
import os
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class ImagePair:
    """Represents a matched image/mask pair"""
    image_path: str
    mask_path: str
    result_name: str
    number: int
    
    @property
    def image_filename(self) -> str:
        return os.path.basename(self.image_path)
    
    @property
    def mask_filename(self) -> str:
        return os.path.basename(self.mask_path)
    
    def load_input_image(self) -> Optional[np.ndarray]:
        """Load the input image as numpy array"""
        try:
            image = cv2.imread(self.image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return None
        except Exception:
            return None
    
    def load_mask_image(self) -> Optional[np.ndarray]:
        """Load the mask image as numpy array"""
        try:
            mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
        except Exception:
            return None


class BatchData:
    """Manages batch processing data and file operations"""
    
    def __init__(self):
        self.images_folder: Optional[str] = None
        self.masks_folder: Optional[str] = None
        self.results_folder: Optional[str] = None
        self.image_pairs: List[ImagePair] = []
        self.supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    def set_images_folder(self, folder_path: str) -> bool:
        """Set the images folder and scan for files"""
        if os.path.isdir(folder_path):
            self.images_folder = folder_path
            self._scan_and_match_files()
            return True
        return False
    
    def set_masks_folder(self, folder_path: str) -> bool:
        """Set the masks folder and scan for files"""
        if os.path.isdir(folder_path):
            self.masks_folder = folder_path
            self._scan_and_match_files()
            return True
        return False
    
    def set_results_folder(self, folder_path: str) -> bool:
        """Set the results folder"""
        if os.path.isdir(folder_path):
            self.results_folder = folder_path
            return True
        return False
    
    def _extract_number_from_filename(self, filename: str, prefix: str) -> Optional[int]:
        """Extract number from filename like 'img1.jpg' or 'mask25.png'"""
        pattern = rf'{prefix}(\d+)'
        match = re.search(pattern, filename, re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def _scan_folder(self, folder_path: str, prefix: str) -> dict:
        """Scan folder for files with given prefix and return number->path mapping"""
        files_dict = {}
        if not folder_path or not os.path.isdir(folder_path):
            return files_dict
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(self.supported_extensions):
                number = self._extract_number_from_filename(filename, prefix)
                if number is not None:
                    files_dict[number] = os.path.join(folder_path, filename)
        
        return files_dict
    
    def _scan_and_match_files(self):
        """Scan both folders and create matched pairs"""
        self.image_pairs.clear()
        
        if not self.images_folder or not self.masks_folder:
            return
        
        # Scan both folders
        image_files = self._scan_folder(self.images_folder, 'img')
        mask_files = self._scan_folder(self.masks_folder, 'mask')
        
        # Create pairs where both img{N} and mask{N} exist
        common_numbers = set(image_files.keys()) & set(mask_files.keys())
        
        for number in sorted(common_numbers):
            pair = ImagePair(
                image_path=image_files[number],
                mask_path=mask_files[number],
                result_name=f"result{number}",
                number=number
            )
            self.image_pairs.append(pair)
    
    def get_unmatched_images(self) -> List[str]:
        """Get list of image files that don't have corresponding masks"""
        if not self.images_folder:
            return []
        
        image_files = self._scan_folder(self.images_folder, 'img')
        mask_files = self._scan_folder(self.masks_folder, 'mask')
        
        unmatched = []
        for number, path in image_files.items():
            if number not in mask_files:
                unmatched.append(os.path.basename(path))
        
        return unmatched
    
    def get_unmatched_masks(self) -> List[str]:
        """Get list of mask files that don't have corresponding images"""
        if not self.masks_folder:
            return []
        
        image_files = self._scan_folder(self.images_folder, 'img')
        mask_files = self._scan_folder(self.masks_folder, 'mask')
        
        unmatched = []
        for number, path in mask_files.items():
            if number not in image_files:
                unmatched.append(os.path.basename(path))
        
        return unmatched
    
    @property
    def is_ready_for_processing(self) -> bool:
        """Check if batch is ready for processing"""
        return (
            self.images_folder is not None and
            self.masks_folder is not None and
            self.results_folder is not None and
            len(self.image_pairs) > 0
        )
    
    @property
    def total_pairs(self) -> int:
        """Get total number of matched pairs"""
        return len(self.image_pairs)
    
    def validate_pair(self, pair: ImagePair) -> Tuple[bool, str]:
        """Validate that an image pair can be processed"""
        try:
            # Check if files exist
            if not os.path.exists(pair.image_path):
                return False, f"Image file not found: {pair.image_filename}"
            
            if not os.path.exists(pair.mask_path):
                return False, f"Mask file not found: {pair.mask_filename}"
            
            # Try to load images to verify they're valid
            image = cv2.imread(pair.image_path)
            if image is None:
                return False, f"Cannot read image: {pair.image_filename}"
            
            mask = cv2.imread(pair.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return False, f"Cannot read mask: {pair.mask_filename}"
            
            # Check dimensions match
            if image.shape[:2] != mask.shape:
                return False, f"Image and mask dimensions don't match for pair {pair.number}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_result_path(self, pair: ImagePair, extension: str = '.jpg') -> str:
        """Get the full path for saving result image"""
        filename = f"{pair.result_name}{extension}"
        return os.path.join(self.results_folder, filename)
    
    def create_results_folder(self) -> bool:
        """Create results folder if it doesn't exist"""
        if self.results_folder:
            try:
                os.makedirs(self.results_folder, exist_ok=True)
                return True
            except Exception:
                return False
        return False 