"""
Batch Exhaustive Research Worker
Combines batch processing with exhaustive parameter research
"""
import time
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from PySide6.QtCore import QThread, Signal, Slot

from models.inpaint_worker import InpaintWorker
from models.batch_data import ImagePair


@dataclass
class ExhaustiveResult:
    """Container for a single exhaustive research result"""
    image_pair: ImagePair
    result_image: np.ndarray
    parameters: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: str = ""
    metrics: Dict[str, float] = None  # Optional: PSNR, SSIM metrics


@dataclass
class PairResults:
    """Container for all results from one image pair"""
    image_pair: ImagePair
    results: List[ExhaustiveResult]
    best_result: ExhaustiveResult = None
    best_criteria: str = ""  # "fastest", "best_psnr", "best_ssim", etc.


class BatchExhaustiveWorker(QThread):
    """Worker thread for batch exhaustive research"""
    
    # Signals
    progress_updated = Signal(int, int)  # current_pair, total_pairs
    pair_started = Signal(str, str)  # image_filename, mask_filename
    combination_started = Signal(str, str, dict)  # image_filename, mask_filename, parameters
    combination_completed = Signal(str, ExhaustiveResult)  # image_filename, result
    pair_completed = Signal(str, PairResults)  # image_filename, all_results_for_pair
    batch_completed = Signal(dict)  # summary statistics
    error_occurred = Signal(str)  # error message
    status_update = Signal(str)  # status message
    
    def __init__(self, image_pairs, parameter_combinations, best_criteria="fastest"):
        """
        Initialize batch exhaustive worker
        
        Args:
            image_pairs: List of ImagePair instances to process
            parameter_combinations: List of parameter dictionaries to test
            best_criteria: Criteria for selecting best result ("fastest", "best_psnr", "best_ssim")
        """
        super().__init__()
        self.image_pairs = image_pairs
        self.parameter_combinations = parameter_combinations
        self.best_criteria = best_criteria
        self.should_stop = False
        
        # Results storage
        self.all_pair_results: List[PairResults] = []
        
        # Current processing state
        self.current_result = None
        self.current_error = None
    
    def run(self):
        """Execute batch exhaustive research"""
        try:
            total_pairs = len(self.image_pairs)
            total_combinations = len(self.parameter_combinations)
            
            self.status_update.emit(
                f"Starting batch exhaustive research: {total_pairs} pairs × {total_combinations} combinations = "
                f"{total_pairs * total_combinations} total operations"
            )
            
            successful_pairs = 0
            failed_pairs = 0
            total_processing_time = 0
            
            # Process each image pair
            for pair_idx, image_pair in enumerate(self.image_pairs):
                if self.should_stop:
                    break
                
                # Emit pair started signal
                self.progress_updated.emit(pair_idx, total_pairs)
                self.pair_started.emit(image_pair.image_filename, image_pair.mask_filename)
                
                # Process this pair with all parameter combinations
                pair_results = self.process_image_pair(image_pair, pair_idx + 1, total_pairs)
                
                if pair_results and pair_results.results:
                    # Find best result for this pair
                    pair_results.best_result = self.find_best_result(pair_results.results)
                    pair_results.best_criteria = self.best_criteria
                    
                    self.all_pair_results.append(pair_results)
                    successful_pairs += 1
                    
                    # Add to total processing time
                    total_processing_time += sum(r.processing_time for r in pair_results.results if r.success)
                    
                    # Emit pair completed signal
                    self.pair_completed.emit(image_pair.image_filename, pair_results)
                else:
                    failed_pairs += 1
            
            # Emit completion signal with summary
            summary = {
                'total_pairs': total_pairs,
                'successful_pairs': successful_pairs,
                'failed_pairs': failed_pairs,
                'total_combinations_tested': successful_pairs * total_combinations,
                'total_processing_time': total_processing_time,
                'average_time_per_combination': total_processing_time / (successful_pairs * total_combinations) if successful_pairs > 0 else 0,
                'parameter_combinations': self.parameter_combinations,
                'best_criteria': self.best_criteria
            }
            
            self.batch_completed.emit(summary)
            
        except Exception as e:
            self.error_occurred.emit(f"Batch exhaustive research failed: {str(e)}")
    
    def process_image_pair(self, image_pair: ImagePair, pair_number: int, total_pairs: int) -> PairResults:
        """Process a single image pair with all parameter combinations"""
        try:
            # Load images
            input_image = image_pair.load_input_image()
            mask_image = image_pair.load_mask_image()
            
            if input_image is None or mask_image is None:
                self.error_occurred.emit(f"Failed to load images for pair: {image_pair.image_filename}")
                return None
            
            pair_results = PairResults(
                image_pair=image_pair,
                results=[]
            )
            
            # Test each parameter combination
            for combo_idx, params in enumerate(self.parameter_combinations):
                if self.should_stop:
                    break
                
                # Emit combination started signal
                self.combination_started.emit(
                    image_pair.image_filename, 
                    image_pair.mask_filename, 
                    params
                )
                
                # Update status
                self.status_update.emit(
                    f"Pair {pair_number}/{total_pairs}: {image_pair.image_filename} - "
                    f"Combination {combo_idx + 1}/{len(self.parameter_combinations)} "
                    f"(Patch={params['patch_size']}, P={params['p_value']}, {params['implementation']})"
                )
                
                # Process this combination
                result = self.process_single_combination(
                    input_image, mask_image, image_pair, params
                )
                
                pair_results.results.append(result)
                
                # Emit combination completed signal
                self.combination_completed.emit(image_pair.image_filename, result)
            
            return pair_results
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing pair {image_pair.image_filename}: {str(e)}")
            return None
    
    def process_single_combination(self, input_image: np.ndarray, mask_image: np.ndarray, 
                                 image_pair: ImagePair, parameters: Dict[str, Any]) -> ExhaustiveResult:
        """Process a single parameter combination"""
        start_time = time.time()
        
        try:
            # Create worker for this combination
            worker = InpaintWorker(
                implementation=parameters['implementation'],
                image=input_image.copy(),
                mask=mask_image.copy(),
                patch_size=parameters['patch_size'],
                p_value=parameters['p_value']
            )
            
            # Reset result variables
            self.current_result = None
            self.current_error = None
            
            # Connect worker signals
            worker.process_complete.connect(self._on_worker_complete)
            worker.error_occurred.connect(self._on_worker_error)
            
            # Run synchronously (we're already in a thread)
            worker.run()
            
            # Wait for worker to complete
            timeout_counter = 0
            max_timeout = 600  # 60 seconds timeout (600 * 0.1 second intervals)
            
            while self.current_result is None and self.current_error is None and timeout_counter < max_timeout:
                self.msleep(100)  # Wait 100ms
                timeout_counter += 1
            
            processing_time = time.time() - start_time
            
            # Create result object
            if self.current_result is not None:
                # Optionally calculate metrics here
                metrics = None
                if hasattr(self, 'calculate_metrics') and self.calculate_metrics:
                    try:
                        from models.metrics import ImageMetrics
                        metrics = ImageMetrics.calculate_all_metrics(input_image, self.current_result)
                    except Exception as e:
                        print(f"Failed to calculate metrics: {e}")
                
                result = ExhaustiveResult(
                    image_pair=image_pair,
                    result_image=self.current_result,
                    parameters=parameters.copy(),
                    processing_time=processing_time,
                    success=True,
                    metrics=metrics
                )
            elif self.current_error is not None:
                result = ExhaustiveResult(
                    image_pair=image_pair,
                    result_image=None,
                    parameters=parameters.copy(),
                    processing_time=processing_time,
                    success=False,
                    error_message=self.current_error
                )
            else:
                # Timeout occurred
                result = ExhaustiveResult(
                    image_pair=image_pair,
                    result_image=None,
                    parameters=parameters.copy(),
                    processing_time=processing_time,
                    success=False,
                    error_message="Processing timeout occurred"
                )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExhaustiveResult(
                image_pair=image_pair,
                result_image=None,
                parameters=parameters.copy(),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def find_best_result(self, results: List[ExhaustiveResult]) -> ExhaustiveResult:
        """Find the best result based on the specified criteria"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return None
        
        if self.best_criteria == "fastest":
            return min(successful_results, key=lambda r: r.processing_time)
        elif self.best_criteria == "best_psnr" and successful_results[0].metrics:
            return max(successful_results, key=lambda r: r.metrics.get('psnr', 0) if r.metrics else 0)
        elif self.best_criteria == "best_ssim" and successful_results[0].metrics:
            return max(successful_results, key=lambda r: r.metrics.get('ssim', 0) if r.metrics else 0)
        else:
            # Default to fastest if criteria not available
            return min(successful_results, key=lambda r: r.processing_time)
    
    @Slot(object)
    def _on_worker_complete(self, result_image):
        """Handle worker completion"""
        self.current_result = result_image
    
    @Slot(str)
    def _on_worker_error(self, error_message):
        """Handle worker error"""
        self.current_error = error_message
    
    def stop_processing(self):
        """Stop the batch processing"""
        self.should_stop = True
    
    def set_calculate_metrics(self, calculate: bool):
        """Enable/disable metrics calculation"""
        self.calculate_metrics = calculate
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all results"""
        if not self.all_pair_results:
            return {}
        
        total_combinations = sum(len(pair.results) for pair in self.all_pair_results)
        successful_combinations = sum(len([r for r in pair.results if r.success]) for pair in self.all_pair_results)
        
        # Get timing statistics
        successful_results = []
        for pair in self.all_pair_results:
            successful_results.extend([r for r in pair.results if r.success])
        
        if successful_results:
            processing_times = [r.processing_time for r in successful_results]
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
        else:
            avg_time = min_time = max_time = 0
        
        return {
            'total_pairs': len(self.all_pair_results),
            'total_combinations': total_combinations,
            'successful_combinations': successful_combinations,
            'success_rate': (successful_combinations / total_combinations * 100) if total_combinations > 0 else 0,
            'average_processing_time': avg_time,
            'min_processing_time': min_time,
            'max_processing_time': max_time,
            'best_criteria_used': self.best_criteria
        }
    
    def export_results_to_csv(self, filepath: str) -> bool:
        """Export all results to CSV file"""
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'image_filename', 'mask_filename', 'patch_size', 'p_value', 
                    'implementation', 'processing_time', 'success', 'error_message',
                    'is_best_for_pair', 'psnr', 'ssim', 'mse'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for pair_result in self.all_pair_results:
                    for result in pair_result.results:
                        row = {
                            'image_filename': result.image_pair.image_filename,
                            'mask_filename': result.image_pair.mask_filename,
                            'patch_size': result.parameters['patch_size'],
                            'p_value': result.parameters['p_value'],
                            'implementation': result.parameters['implementation'],
                            'processing_time': result.processing_time,
                            'success': result.success,
                            'error_message': result.error_message,
                            'is_best_for_pair': result == pair_result.best_result,
                            'psnr': result.metrics.get('psnr', '') if result.metrics else '',
                            'ssim': result.metrics.get('ssim', '') if result.metrics else '',
                            'mse': result.metrics.get('mse', '') if result.metrics else ''
                        }
                        writer.writerow(row)
            
            return True
            
        except Exception as e:
            print(f"Failed to export results to CSV: {e}")
            return False 