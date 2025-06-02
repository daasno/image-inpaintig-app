"""
Worker thread for inpainting operations
"""
import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from gpu_inpainter import GPUInpainter as CudaInpainter
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPU inpainter not available")

try:
    from papier_test import Inpainter_Dyalna as CPUInpainter
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    print("Warning: CPU inpainter not available")


class InpaintWorker(QThread):
    """Worker thread for running the inpainting process"""
    progress_update = Signal(int)  # Signal for progress updates (0-100)
    process_complete = Signal(np.ndarray)  # Signal emitting the result image
    error_occurred = Signal(str)  # Signal for error handling
    status_update = Signal(str)  # Signal for status messages

    def __init__(self, implementation: str, image: np.ndarray, mask: np.ndarray, 
                 patch_size: int, p_value: float = 1.0):
        super().__init__()
        self.implementation = implementation
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.p_value = p_value
        
        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters"""
        if self.image is None:
            raise ValueError("Input image is None")
        
        if self.mask is None:
            raise ValueError("Mask image is None")
        
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError("Image and mask dimensions don't match")
        
        if self.implementation == "GPU" and not GPU_AVAILABLE:
            raise ValueError("GPU implementation requested but not available")
        
        if self.implementation == "CPU" and not CPU_AVAILABLE:
            raise ValueError("CPU implementation requested but not available")

    def run(self):
        """Main worker thread execution"""
        try:
            if self.implementation == "CPU":
                self._run_cpu_inpainting()
            elif self.implementation == "GPU":
                self._run_gpu_inpainting()
            else:
                self.error_occurred.emit(f"Unknown implementation: {self.implementation}")

        except Exception as e:
            import traceback
            error_msg = f"Inpainting failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            print("--- TRACEBACK ---")
            traceback.print_exc()
            print("--- END TRACEBACK ---")
            self.error_occurred.emit(error_msg)

    def _run_cpu_inpainting(self):
        """Run CPU-based inpainting"""
        self.status_update.emit("Starting CPU inpainting...")
        
        # Prepare mask for CPU implementation (might need 0-1 instead of 0-255)
        # The CPU implementation expects mask with 1 for target region (where inpainting happens)
        mask_binary = (self.mask > 127).astype(np.uint8)

        # Get the initial mask count to track actual progress
        height, width = mask_binary.shape
        initial_mask_sum = mask_binary.sum()
        if initial_mask_sum == 0:
            self.error_occurred.emit("No pixels to inpaint (mask is empty)")
            return
        
        # Emit 0% at the beginning
        self.progress_update.emit(0)

        # Initialize inpainter with plot_progress=False to avoid GUI conflicts
        inpainter = CPUInpainter(
            self.image, mask_binary, self.patch_size, 
            plot_progress=False, p=self.p_value
        )

        # Custom progress update for CPU implementation
        # We'll patch the _finished method to emit progress signals
        original_finished = inpainter._finished

        def patched_finished():
            result = original_finished()
            current_mask_sum = inpainter.working_mask.sum()
            # Calculate progress based on how much of the initial mask has been filled
            completed_mask = initial_mask_sum - current_mask_sum
            percentage = int((completed_mask / initial_mask_sum) * 100)
            self.progress_update.emit(percentage)
            return result

        inpainter._finished = patched_finished

        # Run the inpainting process
        self.status_update.emit("Processing with CPU implementation...")
        result = inpainter.inpaint()
        
        # Ensure we emit 100% when done
        self.progress_update.emit(100)
        self.status_update.emit("CPU inpainting completed")
        
        # Signal completion with result
        self.process_complete.emit(result)

    def _run_gpu_inpainting(self):
        """Run GPU-based inpainting"""
        try:
            self.status_update.emit("Starting GPU inpainting...")
            
            # New GPUInpainter expects mask with 0=source, 1=target region
            # Convert the 0/255 mask from image loading to 0/1
            if np.max(self.mask) > 1:
                mask_binary = (self.mask > 127).astype(np.uint8)
            else:
                # Ensure it's uint8 if already 0/1
                mask_binary = self.mask.astype(np.uint8)

            initial_target_count = np.sum(mask_binary == 1)
            # No need to check initial_target_count == 0 here, as GPUInpainter handles it
            # and will yield (0,0) and return the original image.

            # Emit 0% at the beginning - still useful before inpainter starts
            self.progress_update.emit(0)

            # Initialize the new GPUInpainter
            self.status_update.emit(f"Initializing GPU with patch_size={self.patch_size}, p_value={self.p_value}")
            inpainter = CudaInpainter(self.image, mask_binary, self.patch_size, self.p_value)
            
            self.status_update.emit("Processing with GPU implementation...")
            
            inpainter_generator = inpainter.inpaint()
            result_image = None
            last_emitted_percentage = 0  # Track last emitted to avoid redundant signals

            try:
                while True:
                    filled_count, total_to_fill = next(inpainter_generator)
                    
                    current_percentage = 0
                    if total_to_fill > 0:
                        current_percentage = int((filled_count / total_to_fill) * 100)
                    elif filled_count == 0 and total_to_fill == 0:  # Case: nothing to fill
                        current_percentage = 100
                    
                    if current_percentage != last_emitted_percentage:
                        self.progress_update.emit(current_percentage)
                        last_emitted_percentage = current_percentage

            except StopIteration as e:
                result_image = e.value  # Capture the returned image from the generator
            
            self.status_update.emit("GPU inpainting completed")

            # Ensure 100% is emitted if it wasn't the last reported state
            if last_emitted_percentage < 100 and result_image is not None:
                self.progress_update.emit(100)
            
            # If result_image is None here, it means StopIteration didn't have a value, or an error occurred.
            if result_image is None:
                # This implies the generator finished without returning a value as expected.
                # The inpainter.inpaint() should always return an image.
                raise RuntimeError("GPU Inpainting process completed but did not return a final image.")

            # Signal completion with result
            self.process_complete.emit(result_image)

        except Exception as e:
            # Catch potential initialization or runtime errors from GPUInpainter
            import traceback
            print("--- GPU ERROR TRACEBACK ---")
            traceback.print_exc()
            print("--- END TRACEBACK ---")
            self.error_occurred.emit(f"GPU error: {str(e)}. Try using CPU implementation instead.")
            return

    @staticmethod
    def check_gpu_availability() -> bool:
        """Check if GPU implementation is available"""
        if not GPU_AVAILABLE:
            return False
        
        try:
            import numba
            return numba.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def check_cpu_availability() -> bool:
        """Check if CPU implementation is available"""
        return CPU_AVAILABLE 