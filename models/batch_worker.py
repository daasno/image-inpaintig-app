"""
Background worker for batch inpainting operations
"""
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from typing import List, Dict, Any, Tuple

from .batch_data import BatchData, ImagePair
from .image_data import ImageData
from .inpaint_worker import InpaintWorker


class BatchInpaintWorker(QThread):
    """Background worker for batch inpainting operations"""
    
    # Signals
    progress_updated = Signal(int, int)  # current_index, total_count
    pair_started = Signal(str, str)      # image_filename, mask_filename
    pair_completed = Signal(str, bool, str)  # result_filename, success, message
    batch_completed = Signal(dict)       # summary statistics
    error_occurred = Signal(str)         # error message
    
    def __init__(self, batch_data: BatchData, settings: Dict[str, Any]):
        super().__init__()
        self.batch_data = batch_data
        self.settings = settings
        self.should_stop = False
        
        # Statistics
        self.successful_pairs = []
        self.failed_pairs = []
        self.processing_times = []
    
    def stop_processing(self):
        """Stop the batch processing"""
        self.should_stop = True
    
    def run(self):
        """Main processing loop"""
        try:
            self._process_batch()
        except Exception as e:
            self.error_occurred.emit(f"Batch processing error: {str(e)}")
    
    def _process_batch(self):
        """Process all image pairs in the batch"""
        total_pairs = len(self.batch_data.image_pairs)
        
        if total_pairs == 0:
            self.error_occurred.emit("No image pairs found to process")
            return
        
        # Create results folder
        if not self.batch_data.create_results_folder():
            self.error_occurred.emit("Failed to create results folder")
            return
        
        # Process each pair
        for i, pair in enumerate(self.batch_data.image_pairs):
            if self.should_stop:
                break
            
            # Emit progress
            self.progress_updated.emit(i, total_pairs)
            self.pair_started.emit(pair.image_filename, pair.mask_filename)
            
            # Process this pair
            success, message = self._process_single_pair(pair)
            
            # Record result
            if success:
                self.successful_pairs.append(pair)
            else:
                self.failed_pairs.append((pair, message))
            
            # Emit completion for this pair
            result_filename = f"{pair.result_name}.jpg"
            self.pair_completed.emit(result_filename, success, message)
        
        # Emit final progress
        self.progress_updated.emit(total_pairs, total_pairs)
        
        # Emit batch completion summary
        summary = {
            'total_pairs': total_pairs,
            'successful': len(self.successful_pairs),
            'failed': len(self.failed_pairs),
            'failed_pairs': self.failed_pairs,
            'processing_times': self.processing_times
        }
        self.batch_completed.emit(summary)
    
    def _process_single_pair(self, pair: ImagePair) -> Tuple[bool, str]:
        """Process a single image/mask pair"""
        try:
            import time
            start_time = time.time()
            
            # Validate the pair first
            is_valid, validation_message = self.batch_data.validate_pair(pair)
            if not is_valid:
                return False, validation_message
            
            # Load images
            input_image = cv2.imread(pair.image_path)
            mask_image = cv2.imread(pair.mask_path, cv2.IMREAD_GRAYSCALE)
            
            if input_image is None:
                return False, f"Failed to load image: {pair.image_filename}"
            
            if mask_image is None:
                return False, f"Failed to load mask: {pair.mask_filename}"
            
            # Create ImageData instance for this pair
            image_data = ImageData()
            image_data.input_image = input_image
            image_data.mask_image = mask_image
            
            # Create inpaint worker
            inpaint_worker = InpaintWorker(
                implementation=self.settings.get('implementation', 'CPU'),
                image=input_image,
                mask=mask_image,
                patch_size=self.settings.get('patch_size', 9),
                p_value=self.settings.get('p_value', 1.0)
            )
            
            # Set up result capturing
            result_image = None
            error_message = None
            
            def on_process_complete(image):
                nonlocal result_image
                result_image = image
            
            def on_error_occurred(msg):
                nonlocal error_message
                error_message = msg
            
            # Connect signals
            inpaint_worker.process_complete.connect(on_process_complete)
            inpaint_worker.error_occurred.connect(on_error_occurred)
            
            # Run inpainting synchronously
            inpaint_worker.run()
            
            # Wait for completion
            inpaint_worker.wait()
            
            # Check if processing was successful
            if result_image is not None:
                # Save result
                result_path = self.batch_data.get_result_path(pair, '.jpg')
                success = cv2.imwrite(result_path, result_image)
                
                if success:
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    return True, f"Processed successfully in {processing_time:.2f}s"
                else:
                    return False, "Failed to save result image"
            else:
                error_msg = error_message or 'Unknown inpainting error'
                return False, f"Inpainting failed: {error_msg}"
                
        except Exception as e:
            return False, f"Processing error: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'total_processed': len(self.successful_pairs) + len(self.failed_pairs),
            'successful': len(self.successful_pairs),
            'failed': len(self.failed_pairs),
            'success_rate': len(self.successful_pairs) / max(1, len(self.successful_pairs) + len(self.failed_pairs)) * 100,
            'average_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_time': sum(self.processing_times),
            'processing_times': self.processing_times
        } 