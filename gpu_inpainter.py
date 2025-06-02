import numpy as np
from numba import cuda
import math
import time
# from skimage.color import rgb2lab # Import for LAB conversion - NO LONGER NEEDED
import cv2 # Import OpenCV
from skimage.filters import laplace
from numba.core.types import float32, int32 # Import type definitions for shared arrays
import cupy as cp

class GPUInpainter:
    def __init__(self, image, mask, patch_size=9, p_value=1.0):
        """
        Initializes the GPUInpainter.

        Args:
            image (np.ndarray): The input image (uint8, HxWxC or HxW).
            mask (np.ndarray): The mask image (uint8, HxW), where 1 indicates the region to inpaint.
            patch_size (int): The side length of the square patch (should be odd).
            p_value (float): The Minkowski order parameter for patch distance calculation.
            
        Note: This implementation uses GPU acceleration for computationally intensive operations
        (gradient calculation, patch distance computation) but uses CPU-side NumPy operations
        for priority selection and patch matching to ensure identical tie-breaking behavior
        with the CPU-only implementation.
        """
        print("Initializing GPUInpainter...")
        self.patch_size = patch_size
        self.p_value = p_value
        self.half_patch_size = (patch_size - 1) // 2

        # --- Validate Inputs (Basic) ---
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same height and width.")
        if patch_size % 2 == 0:
            raise ValueError("Patch size must be odd.")
        if image.dtype != np.uint8:
            print(f"Warning: Input image dtype is {image.dtype}, converting to uint8.")
            image = image.astype(np.uint8)
        if mask.dtype != np.uint8:
            print(f"Warning: Input mask dtype is {mask.dtype}, converting to uint8.")
            # Assuming mask uses 1 for target, 0 for source, like CPU version
            mask = mask.round().astype(np.uint8)

        self.height, self.width = image.shape[:2]
        self.is_color = len(image.shape) == 3 and image.shape[2] == 3

        # --- Convert to LAB if color ---
        lab_image_cpu = None
        if self.is_color:
            print("Converting image to LAB color space using cv2.COLOR_BGR2LAB...")
            # CPU path: BGR uint8 -> BGR float32 [0,1] -> cv2.COLOR_BGR2LAB -> LAB float32
            image_float32 = image.astype(np.float32) / 255.0
            lab_image_cpu = cv2.cvtColor(image_float32, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Ensure image is in RGB format for skimage (OpenCV reads BGR)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Need to import cv2
            # lab_image_cpu = rgb2lab(image_rgb).astype(np.float32)
            print("Conversion to LAB done.")

        # --- GPU Data Allocation ---
        print("Allocating GPU memory...")
        # Ensure image is contiguous for transfer
        img_gpu = np.ascontiguousarray(image)
        mask_gpu = np.ascontiguousarray(mask)

        self.d_image = cuda.to_device(img_gpu) # Original image (read-only during patch search)
        self.d_mask = cuda.to_device(mask_gpu) # 0=source, 1=target (updated during inpainting)
        self.d_working_image = cuda.to_device(img_gpu) # Image being modified (uint8 RGB/Gray)
        # Allocate LAB image on GPU only if it exists
        self.d_lab_image = None
        if lab_image_cpu is not None:
            self.d_lab_image = cuda.to_device(np.ascontiguousarray(lab_image_cpu))

        self.d_confidence = cuda.to_device((1.0 - mask_gpu).astype(np.float32)) # Confidence map
        self.d_data = cuda.device_array((self.height, self.width), dtype=np.float32) # Data term (gradient info)
        self.d_priority = cuda.device_array((self.height, self.width), dtype=np.float32) # Priority map
        self.d_front = cuda.device_array((self.height, self.width), dtype=np.uint8) # Fill front pixels

        # For gradient calculation, to match CPU's np.gradient on float gray image
        self.d_prepared_gray_for_gradient = cuda.device_array((self.height, self.width), dtype=np.float32)

        # Initialize data and priority to zero
        self.d_data.copy_to_device(np.zeros((self.height, self.width), dtype=np.float32))
        self.d_priority.copy_to_device(np.zeros((self.height, self.width), dtype=np.float32))
        self.d_front.copy_to_device(np.zeros((self.height, self.width), dtype=np.uint8))

        print("GPU Memory allocated.")

        # --- CUDA Kernel Configuration (Example) ---
        # Define typical block and grid sizes (can be adjusted)
        self.threadsperblock = (16, 16)
        grid_x = math.ceil(self.width / self.threadsperblock[0])
        grid_y = math.ceil(self.height / self.threadsperblock[1])
        self.blockspergrid = (grid_x, grid_y)

        print(f"CUDA Configuration: Threads={self.threadsperblock}, Grid={self.blockspergrid}")


    def inpaint(self):
        """
        Performs efficient GPU-accelerated image inpainting.
        The algorithm follows the Criminisi paper but uses GPU acceleration
        for significant performance improvements.
        Yields progress tuples (pixels_filled, total_pixels_to_fill) and
        returns the final inpainted image.
        """
        print("Starting GPU Inpainting Process...")
        start_time = time.time()
        
        # Get initial count of pixels to fill
        total_pixels_to_fill = np.sum(self.d_mask.copy_to_host() == 1)
        pixels_filled = 0
        iterations = 0

        if total_pixels_to_fill == 0:
            print("No pixels to fill. Returning original image.")
            yield 0, 0 # Yield initial state even if nothing to do
            return self.d_working_image.copy_to_host()
        
        # Yield initial progress
        yield pixels_filled, total_pixels_to_fill

        # Main inpainting loop
        while pixels_filled < total_pixels_to_fill:
            # Only print progress every 10 iterations to reduce console spam
            if iterations % 10 == 0:
                remaining = total_pixels_to_fill - pixels_filled
                print(f"Inpainting progress: {pixels_filled}/{total_pixels_to_fill} pixels filled. {remaining} remaining.")
            
            # 1. Find fill front
            self._find_front_gpu()
            cuda.synchronize()
            
            # 2. Update priority
            self._update_priority_gpu()
            cuda.synchronize()
            
            # 3. Find highest priority pixel
            target_y, target_x = self._find_highest_priority_pixel_gpu()
            cuda.synchronize()
            
            # Check if we found a valid target pixel
            if target_y < 0 or target_x < 0:
                print("No valid front pixels found. Inpainting finished.")
                break
            
            # 4. Find best source patch
            source_center_y, source_center_x = self._find_best_patch_gpu(target_y, target_x)
            cuda.synchronize()
            
            # Check if we found a valid source patch
            if source_center_y < 0 or source_center_x < 0:
                print(f"Could not find valid source patch for target at ({target_y}, {target_x}). Stopping.")
                break
            
            # 5. Update the image
            # For efficiency, we'll use the GPU patch update kernel
            d_counter = cuda.to_device(np.array([0], dtype=np.int32))
            _update_patch_kernel[self.blockspergrid, self.threadsperblock](
                target_y, target_x,
                source_center_y, source_center_x,  # Now using center coordinates
                self.d_working_image,
                self.d_mask, 
                self.d_confidence,
                d_counter,
                self.patch_size, self.half_patch_size,
                self.height, self.width, self.is_color
            )
            cuda.synchronize()
            
            # Get count of filled pixels in this iteration
            pixels_newly_filled = int(d_counter.copy_to_host()[0])
            pixels_filled += pixels_newly_filled

            # Yield current progress
            yield pixels_filled, total_pixels_to_fill
            
            # 6. Update the LAB image for next iteration
            if self.is_color and pixels_newly_filled > 0:
                # This is expensive but necessary for accurate results
                # Match the CPU's BGR uint8 -> BGR float32 [0,1] -> cv2.COLOR_BGR2LAB path
                h_working_image = self.d_working_image.copy_to_host() # This is BGR uint8
                h_working_image_float32 = h_working_image.astype(np.float32) / 255.0
                lab_image_cpu = cv2.cvtColor(h_working_image_float32, cv2.COLOR_BGR2LAB).astype(np.float32)
                # image_rgb = cv2.cvtColor(h_working_image, cv2.COLOR_BGR2RGB)
                # lab_image_cpu = rgb2lab(image_rgb).astype(np.float32)
                self.d_lab_image.copy_to_device(np.ascontiguousarray(lab_image_cpu))
            
            iterations += 1
            
            # Safety check to avoid infinite loops
            if pixels_newly_filled == 0:
                print("Warning: No pixels filled in this iteration. Stopping to avoid infinite loop.")
                break
        
        print(f"GPU Inpainting completed in {time.time() - start_time:.2f} seconds.")
        print(f"Filled {pixels_filled} pixels in {iterations} iterations.")
        
        # Ensure final progress is yielded if loop completes
        yield pixels_filled, total_pixels_to_fill
        
        # Return the completed image
        return self.d_working_image.copy_to_host()

    # --- Placeholder GPU Kernel Launchers ---

    def _find_front_gpu(self):
        """
        Identifies the fill front using a CUDA kernel that implements a Laplacian-like
        operation on the mask.
        
        This uses a GPU algorithm but ensures results match the CPU implementation.
        """
        # Launch custom Laplacian kernel to identify boundary pixels
        _find_front_laplacian_kernel[self.blockspergrid, self.threadsperblock](
            self.d_mask,
            self.d_front,
            self.height,
            self.width
        )
        
        # No need for synchronization as the main loop handles it

    def _update_priority_gpu(self):
        """
        Updates the priority map using GPU kernels that match the CPU implementation.
        Priority = Confidence * Data term, where Data term is based on the
        dot product of the normal and gradient vectors.
        """
        # 0. Prepare a float32 grayscale image (0-1 range) with masked areas as 0.0
        # This mimics the CPU's preprocessing before np.gradient
        _prepare_grayscale_for_gradient_kernel[self.blockspergrid, self.threadsperblock](
            self.d_working_image, self.d_mask,
            self.d_prepared_gray_for_gradient,
            self.height, self.width, self.is_color
        )

        # 1. Update Confidence Term (unchanged GPU kernel)
        d_new_confidence = cuda.device_array_like(self.d_confidence)
        _update_confidence_kernel[self.blockspergrid, self.threadsperblock](
            self.d_confidence, self.d_front, self.d_mask,
            d_new_confidence, self.height, self.width, self.half_patch_size
        )
        self.d_confidence = d_new_confidence
        
        # 2. Calculate Mask Normals (unchanged GPU kernel)
        d_normal_x = cuda.device_array_like(self.d_data)
        d_normal_y = cuda.device_array_like(self.d_data)
        _calculate_normal_kernel[self.blockspergrid, self.threadsperblock](
            self.d_mask, d_normal_x, d_normal_y, self.height, self.width
        )
        
        # 3. Calculate Image Gradients using the prepared grayscale image
        # These are the gradients at each pixel of the image
        d_gradient_y = cuda.device_array_like(self.d_data) # Re-use d_data's shape/type
        d_gradient_x = cuda.device_array_like(self.d_data)
        d_gradient_mag = cuda.device_array_like(self.d_data) 
        _calculate_image_gradient_kernel[self.blockspergrid, self.threadsperblock](
            self.d_prepared_gray_for_gradient, # INPUT: pre-processed float32 grayscale image
            d_gradient_y, d_gradient_x, d_gradient_mag,
            self.height, self.width
            # No longer needs self.d_mask or self.is_color, as these are handled by the prepare kernel
        )
        
        # 4. Calculate Data Term using gradients at the front pixel
        _calculate_data_term_kernel[self.blockspergrid, self.threadsperblock](
            self.d_front, d_normal_x, d_normal_y,
            d_gradient_x, d_gradient_y, # Pass the full image gradients
            self.d_data, self.height, self.width
        )
        
        # 5. Calculate Priority (Confidence * Data Term)
        _calculate_priority_kernel[self.blockspergrid, self.threadsperblock](
            self.d_confidence,
            self.d_data,
            self.d_front,
            self.d_priority,
            self.height, self.width
        )

    def _find_highest_priority_pixel_gpu(self):
        """
        Finds the (y, x) coordinates of the pixel with the highest priority
        on the current fill front using CPU-side argmax for consistent tie-breaking.
        Returns: tuple (y, x) or (-1, -1) if no front pixels exist.
        """
        # Copy priority and front maps to CPU for consistent tie-breaking
        h_priority = self.d_priority.copy_to_host()
        h_front = self.d_front.copy_to_host()
        
        # Create masked priority array (set non-front pixels to -inf)
        masked_priority = np.where(h_front == 1, h_priority, -np.inf)
        
        # Check if any front pixels exist
        if not np.any(h_front == 1):
            return -1, -1
        
        # Use numpy's argmax which gives consistent row-major tie-breaking (same as CPU)
        linear_idx = np.argmax(masked_priority)
        target_y = linear_idx // self.width
        target_x = linear_idx % self.width
        
        return target_y, target_x

    def _find_best_patch_gpu(self, target_y, target_x):
        """
        Finds the best matching source patch for target patch centered at (target_y, target_x).
        Uses GPU for parallel distance calculation but CPU-side argmin for consistent tie-breaking.
        Returns CENTER coordinates of the best source patch.
        """
        # 1. Calculate target patch bounds with proper boundary handling (like CPU)
        half_ps = self.half_patch_size
        target_top = max(0, target_y - half_ps)
        target_bottom = min(self.height - 1, target_y + half_ps)
        target_left = max(0, target_x - half_ps)
        target_right = min(self.width - 1, target_x + half_ps)
        
        # The actual patch dimensions might be smaller than patch_size if near image boundary
        patch_height = target_bottom - target_top + 1
        patch_width = target_right - target_left + 1
        
        # 2. Setup search area for valid source patches
        # Initialize distance array on GPU with infinity
        d_distances = cuda.to_device(np.full((self.height, self.width), np.inf, dtype=np.float32))
        
        # 3. Launch the patch distance calculation kernel
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(self.width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(self.height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        _calculate_patch_distances_kernel[blocks_per_grid, threads_per_block](
            target_top, target_left, patch_height, patch_width,
            self.d_working_image, self.d_lab_image, self.d_mask,
            d_distances, self.height, self.width, self.p_value,
            self.is_color
        )
        cuda.synchronize()
        
        # 4. Copy distances to CPU and use numpy argmin for consistent tie-breaking
        h_distances = d_distances.copy_to_host()
        
        # Check if any valid patches were found
        if np.all(np.isinf(h_distances)):
            return -1, -1
        
        # Use numpy's argmin which gives consistent row-major tie-breaking (same as CPU)
        linear_idx = np.argmin(h_distances)
        source_top = linear_idx // self.width
        source_left = linear_idx % self.width
        
        # Convert top-left to center coordinates to match what the kernel expects
        source_center_y = source_top + patch_height // 2
        source_center_x = source_left + patch_width // 2
        
        return source_center_y, source_center_x

    def _update_image_gpu(self, target_y, target_x, source_y, source_x):
        """
        Uses a CPU approach similar to the original implementation to update the image.
        This should provide the most consistent results.
        """
        # Get the necessary data from GPU to CPU
        working_image_cpu = self.d_working_image.copy_to_host()
        mask_cpu = self.d_mask.copy_to_host()
        confidence_cpu = self.d_confidence.copy_to_host()
        
        # Get the target patch bounds (similar to CPU _get_patch)
        half_ps = self.half_patch_size
        height, width = self.height, self.width
        
        target_top = max(0, target_y - half_ps)
        target_bottom = min(height - 1, target_y + half_ps)
        target_left = max(0, target_x - half_ps)
        target_right = min(width - 1, target_x + half_ps)
        
        target_patch = [
            [target_top, target_bottom],
            [target_left, target_right]
        ]
        
        # Calculate source patch bounds based on how CPU version does it
        source_top = max(0, source_y - half_ps)
        source_bottom = min(height - 1, source_y + half_ps)
        source_left = max(0, source_x - half_ps)
        source_right = min(width - 1, source_x + half_ps)
        
        source_patch = [
            [source_top, source_bottom],
            [source_left, source_right]
        ]
        
        # Get the pixels in the target patch that need to be filled (mask==1)
        target_mask_data = mask_cpu[target_top:target_bottom+1, target_left:target_right+1]
        pixels_to_fill = np.argwhere(target_mask_data == 1)
        
        # Convert local patch coordinates to global image coordinates
        pixels_global = pixels_to_fill + [target_top, target_left]
        
        # Get the confidence value of the target center pixel
        patch_confidence = confidence_cpu[target_y, target_x]
        
        # Update the confidence of all filled pixels
        for point in pixels_global:
            confidence_cpu[point[0], point[1]] = patch_confidence
        
        # Get the mask for the target patch and convert it to RGB mask if needed
        mask = mask_cpu[target_top:target_bottom+1, target_left:target_right+1]
        if self.is_color:
            rgb_mask = mask.reshape(mask.shape[0], mask.shape[1], 1).repeat(3, axis=2)
        else:
            rgb_mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        
        # Get the data from source and target patches
        source_data = working_image_cpu[source_top:source_bottom+1, source_left:source_right+1]
        target_data = working_image_cpu[target_top:target_bottom+1, target_left:target_right+1]
        
        # Create new data by combining source and target data based on mask
        # source data where mask==1, target data where mask==0
        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)
        
        # Update the working image
        working_image_cpu[target_top:target_bottom+1, target_left:target_right+1] = new_data
        
        # Update the mask (set to 0 for the entire target patch)
        mask_cpu[target_top:target_bottom+1, target_left:target_right+1] = 0
        
        # Copy updated data back to GPU
        self.d_working_image.copy_to_device(working_image_cpu)
        self.d_mask.copy_to_device(mask_cpu)
        self.d_confidence.copy_to_device(confidence_cpu)
        
        # Return the count of newly filled pixels
        return len(pixels_to_fill)


    # --- Helper Functions / Potential Kernels ---
    # (Need kernels for laplacian, gradient, patch summing, patch copying, etc.)

@cuda.jit
def _find_front_laplacian_kernel(mask, front, height, width):
    """
    CUDA kernel that implements a Laplacian-like operation to identify
    the fill front. Similar to the CPU implementation that uses scipy.laplace().
    
    A pixel is on the front if it's inside the hole (mask=1) but has at least
    one neighbor outside the hole (mask=0).
    """
    y, x = cuda.grid(2)
    
    if y >= height or x >= width:
        return
    
    # Default: not on front
    front[y, x] = 0
    
    # Front check: must be inside the target region (mask=1)
    if mask[y, x] == 1:
        # Check if any of the 4-neighbors is outside the mask (mask=0)
        has_outside_neighbor = False
        
        # Check 4-neighbors (with boundary checks)
        if y > 0 and mask[y-1, x] == 0:
            has_outside_neighbor = True
        elif y < height-1 and mask[y+1, x] == 0:
            has_outside_neighbor = True
        elif x > 0 and mask[y, x-1] == 0:
            has_outside_neighbor = True
        elif x < width-1 and mask[y, x+1] == 0:
            has_outside_neighbor = True
            
        if has_outside_neighbor:
            front[y, x] = 1

@cuda.jit
def _update_confidence_kernel(confidence, front, mask, new_confidence, height, width, half_patch_size):
    """
    CUDA kernel to update the confidence term for pixels on the fill front.
    Reads from 'confidence', writes updated values for front pixels into 'new_confidence'.
    """
    y, x = cuda.grid(2)

    if y >= height or x >= width:
        return

    # Initialize new confidence with the old value
    new_confidence[y, x] = confidence[y, x]

    # Only update confidence for pixels on the front
    if front[y, x] == 1:
        sum_confidence = 0.0
        count = 0

        # Iterate over the patch centered at (y, x)
        for j in range(-half_patch_size, half_patch_size + 1):
            for i in range(-half_patch_size, half_patch_size + 1):
                py = y + j
                px = x + i

                # Check patch pixel bounds
                if 0 <= py < height and 0 <= px < width:
                        # Check if the pixel in the patch is *not* part of the original hole (mask == 0)
                        # This matches the logic of averaging only known pixels' confidence
                        # Or should we average *all* pixels' confidence in the patch?
                        # The CPU version averages confidence values from the 'confidence' map
                        # within the patch area. Let's stick to that.
                    sum_confidence += confidence[py, px]
                    count += 1

        # Calculate average confidence for the patch
        if count > 0:
            avg_confidence = sum_confidence / count
            new_confidence[y, x] = avg_confidence
        else:
            # Should not happen if the center pixel (y,x) is on the front
            # and contributes to the count, but as a fallback:
            new_confidence[y, x] = 0.0


@cuda.jit
def _calculate_normal_kernel(mask, normal_x, normal_y, height, width):
    """Calculates normalized normals for the mask."""
    y, x = cuda.grid(2)
    if y >= height or x >= width: return

    # Sobel-like kernels from CPU version for normal calculation
    Kx = ((-0.25, 0.0, 0.25), (-0.5, 0.0, 0.5), (-0.25, 0.0, 0.25))
    Ky = ((-0.25, -0.5, -0.25), (0.0, 0.0, 0.0), (0.25, 0.5, 0.25))

    nx = 0.0
    ny = 0.0

    # Convolve using float mask values
    for j in range(-1, 2):
        for i in range(-1, 2):
            py = y + j
            px = x + i
            if 0 <= py < height and 0 <= px < width:
                mask_val = float(mask[py, px]) # Ensure float for calculation
                nx += mask_val * Kx[j+1][i+1]
                ny += mask_val * Ky[j+1][i+1]

    # Normalize
    norm = math.sqrt(nx**2 + ny**2)
    if norm > 1e-6: # Avoid division by zero
        normal_x[y, x] = nx / norm
        normal_y[y, x] = ny / norm
    else:
        normal_x[y, x] = 0.0
        normal_y[y, x] = 0.0


@cuda.jit
def _calculate_image_gradient_kernel(prepared_gray_image,  # Changed from working_image, mask, is_color
                                     gradient_y, gradient_x, gradient_mag, 
                                     height, width):
    """
    Calculates image gradients (gy, gx) and magnitude using central/first-order differences.
    Assumes input `prepared_gray_image` is a float32 grayscale image (range 0-1) 
    where masked regions have already been set to 0.0.
    Matches np.gradient behavior for boundary and central differences.
    """
    y, x = cuda.grid(2)
    if y >= height or x >= width:
        return

    # Initialize gradients to zero (though typically they will be overwritten)
    # gradient_y[y, x] = 0.0
    # gradient_x[y, x] = 0.0
    # gradient_mag[y, x] = 0.0

    # Vertical gradient (gy)
    # np.gradient(f) at f[0] is f[1] - f[0]
    # np.gradient(f) at f[N-1] is f[N-1] - f[N-2]
    # np.gradient(f) at f[i] is (f[i+1] - f[i-1]) / 2 for 0 < i < N-1
    if y == 0: # Top boundary: f[1] - f[0]
        # Ensure y+1 is within bounds for single-row images, though unlikely for inpainting
        val_plus = prepared_gray_image[y + 1, x] if y + 1 < height else prepared_gray_image[y,x]
        val_center = prepared_gray_image[y, x]
        gy = val_plus - val_center
    elif y == height - 1: # Bottom boundary: f[N-1] - f[N-2]
        # Ensure y-1 is within bounds for single-row images
        val_minus = prepared_gray_image[y - 1, x] if y - 1 >= 0 else prepared_gray_image[y,x]
        val_center = prepared_gray_image[y, x]
        gy = val_center - val_minus
    else: # Central difference
        val_plus = prepared_gray_image[y + 1, x]
        val_minus = prepared_gray_image[y - 1, x]
        gy = (val_plus - val_minus) / 2.0
    gradient_y[y, x] = gy

    # Horizontal gradient (gx)
    # np.gradient(f) at f[0] is f[1] - f[0]
    # np.gradient(f) at f[N-1] is f[N-1] - f[N-2]
    # np.gradient(f) at f[i] is (f[i+1] - f[i-1]) / 2 for 0 < i < N-1
    if x == 0: # Left boundary: f[1] - f[0]
        val_plus = prepared_gray_image[y, x + 1] if x + 1 < width else prepared_gray_image[y,x]
        val_center = prepared_gray_image[y, x]
        gx = val_plus - val_center
    elif x == width - 1: # Right boundary: f[N-1] - f[N-2]
        val_minus = prepared_gray_image[y, x - 1] if x - 1 >=0 else prepared_gray_image[y,x]
        val_center = prepared_gray_image[y, x]
        gx = val_center - val_minus
    else: # Central difference
        val_plus = prepared_gray_image[y, x + 1]
        val_minus = prepared_gray_image[y, x - 1]
        gx = (val_plus - val_minus) / 2.0
    gradient_x[y, x] = gx
        
    gradient_mag[y, x] = math.sqrt(gx*gx + gy*gy)


@cuda.jit
def _prepare_grayscale_for_gradient_kernel(working_image, mask, prepared_gray_image, height, width, is_color):
    """
    Prepares a float32 grayscale image (0-1 range) for gradient calculation.
    - Converts color images to grayscale.
    - Normalizes uint8 grayscale to float32 [0,1].
    - Sets masked regions (mask == 1) to 0.0.
    """
    y, x = cuda.grid(2)
    if y >= height or x >= width:
        return

    if mask[y, x] == 1:
        prepared_gray_image[y, x] = 0.0
    else:
        if is_color:
            # OpenCV BGR order: R=2, G=1, B=0
            r_val = float(working_image[y, x, 2])
            g_val = float(working_image[y, x, 1])
            b_val = float(working_image[y, x, 0])
            # Standard RGB to Grayscale conversion formula, then normalize to [0,1]
            gray_val = (0.2989 * r_val + 0.5870 * g_val + 0.1140 * b_val) / 255.0
            prepared_gray_image[y, x] = gray_val
        else: # Grayscale image
            # Normalize uint8 grayscale to float32 [0,1]
            prepared_gray_image[y, x] = float(working_image[y, x, 0]) / 255.0


@cuda.jit
def _calculate_data_term_kernel(front, normal_x, normal_y, 
                                gradient_x, gradient_y, # Changed from max_gradient_x, max_gradient_y
                                data_term, height, width):
    """
    Calculates the data term for front pixels using the dot product of the normal
    and the image gradient *at the front pixel*.
    Data = |normal_x * gradient_x_at_pixel + normal_y * gradient_y_at_pixel| + epsilon
    """
    y, x = cuda.grid(2)
    if y >= height or x >= width:
        return

    data_term[y, x] = 0.0  # Initialize data term

    if front[y, x] == 1:  # Process only front pixels
        nx = normal_x[y, x]
        ny = normal_y[y, x]
        
        # Use the gradient at the current pixel (y,x)
        gx_at_pixel = gradient_x[y, x] 
        gy_at_pixel = gradient_y[y, x]
        
        # Calculate dot product: (nx * gx_at_pixel) + (ny * gy_at_pixel)
        dot_product = nx * gx_at_pixel + ny * gy_at_pixel
        
        # Add epsilon (same as CPU and previous GPU versions)
        data_term[y, x] = abs(dot_product) + 0.001

@cuda.jit
def _calculate_priority_kernel(confidence, data_term, front, priority, height, width):
    """Calculates priority = confidence * data_term for front pixels."""
    y, x = cuda.grid(2)

    if y >= height or x >= width:
        return

    if front[y, x] == 1:
        priority[y, x] = confidence[y, x] * data_term[y, x]
    else:
        priority[y, x] = 0.0 # Priority is zero elsewhere

@cuda.jit
def _block_argmax_priority_kernel(priority, front, block_max_vals, block_max_indices, height, width):
    """
    Reduction kernel: Each block finds the max priority value (where front==1)
    and its linear index within the block's assigned region.
    Writes the block's max value and index to global arrays.
    """
    # Shared memory for reduction within the block
    # Store pairs of (value, index)
    s_vals = cuda.shared.array(shape=(16*16), dtype=np.float32) # Assuming 16x16 blocks
    s_indices = cuda.shared.array(shape=(16*16), dtype=np.int32)

    # Thread's global coordinates and linear index
    gy, gx = cuda.grid(2)
    block_dim_y = cuda.blockDim.y
    block_dim_x = cuda.blockDim.x
    thread_id_x = cuda.threadIdx.x
    thread_id_y = cuda.threadIdx.y
    # Linear index within the block
    tid_in_block = thread_id_y * block_dim_x + thread_id_x
    # Global linear index (potential candidate)
    linear_idx = gy * width + gx

    # Initialize shared memory (using the first thread is not efficient here, better parallel init)
    # Initialize directly based on global memory load
    if gy < height and gx < width:
        p_val = priority[gy, gx]
        f_val = front[gy, gx]
        # Use a very small value (or -1) if not on front, so it doesn't win max
        s_vals[tid_in_block] = p_val if f_val == 1 else -1.0
        s_indices[tid_in_block] = linear_idx
    else:
        # Out of bounds threads initialize with non-winning values
        s_vals[tid_in_block] = -1.0
        s_indices[tid_in_block] = -1 # Invalid index

    cuda.syncthreads() # Ensure shared memory is populated

    # --- In-block reduction ---
    s = block_dim_x * block_dim_y // 2 # Start with half the block size
    while s > 0:
        if tid_in_block < s:
            # Compare thread i with thread i + s
            if s_vals[tid_in_block] < s_vals[tid_in_block + s]:
                # If other thread has higher value, copy its value and index
                s_vals[tid_in_block] = s_vals[tid_in_block + s]
                s_indices[tid_in_block] = s_indices[tid_in_block + s]
            elif s_vals[tid_in_block] == s_vals[tid_in_block + s]:
                # Tie-breaking: prefer lower index if values are equal
                s_indices[tid_in_block] = min(s_indices[tid_in_block], s_indices[tid_in_block + s])


        cuda.syncthreads() # Synchronize after each comparison level
        s //= 2 # Reduce comparison stride

    # --- Write block's result ---
    if tid_in_block == 0: # Only the first thread in the block writes the result
        block_id = cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x
        block_max_vals[block_id] = s_vals[0]
        block_max_indices[block_id] = s_indices[0]

@cuda.jit(device=True) # Mark as device function for potential reuse/clarity
def _calculate_pixel_distance(p1_val, p2_val, p_value):
    """Calculates Minkowski^p and Chebyshev distance components for a single pixel value pair.
       Assumes input values are scalar numbers (e.g., uint8 or float).
    """
    # Cast inputs to float for safe arithmetic
    f_p1_val = float(p1_val)
    f_p2_val = float(p2_val)
    diff = abs(f_p1_val - f_p2_val)
    # Ensure p_value is float for power calculation
    f_p_value = float(p_value)
    # Handle p_value=0 case for power
    mink_comp = diff ** f_p_value if f_p_value != 0 else 0.0
    cheb_comp = diff # Chebyshev is just the max absolute difference
    return mink_comp, cheb_comp

@cuda.jit
def _calculate_patch_distances_kernel(target_top, target_left, patch_height, patch_width,
                                     working_image, lab_image, mask,
                                     distances, height, width, p_value,
                                     is_color):
    """
    Calculates the distance between the target patch and a potential source patch.
    Each thread calculates one source patch's distance.
    Only valid source patches get a finite distance value.
    """
    # Each thread represents a potential source patch (top-left corner)
    source_top, source_left = cuda.grid(2)
    
    # Check global boundaries
    if source_top >= height or source_left >= width:
        return
    
    # Check if this source patch would go out of bounds
    if source_top + patch_height > height or source_left + patch_width > width:
        distances[source_top, source_left] = float('inf')
        return
    
    # Check if source patch contains any unknown pixels (must be all known)
    for j in range(patch_height):
        for i in range(patch_width):
            sy = source_top + j
            sx = source_left + i
            if mask[sy, sx] == 1:  # Unknown pixel
                distances[source_top, source_left] = float('inf')
                return
    
    # Prepare for distance calculation - matches CPU algorithm
    total_cheb_dist = 0.0    # Maximum absolute difference (Chebyshev)
    total_mink_sum = 0.0     # Sum for Minkowski distance
    valid_pixel_count = 0    # Count pixels that can be compared
    
    # Compare corresponding pixels in target and source patches
    for j in range(patch_height):
        for i in range(patch_width):
            ty = target_top + j
            tx = target_left + i
            sy = source_top + j
            sx = source_left + i
            
            # Only compare if target pixel is known (mask==0)
            if mask[ty, tx] == 0:
                valid_pixel_count += 1
                
                # Color comparison uses LAB colorspace
                if is_color:
                    # Compare each channel
                    for c in range(3):
                        t_val = lab_image[ty, tx, c]
                        s_val = lab_image[sy, sx, c]
                        
                        diff = abs(float(t_val) - float(s_val))
                        
                        # Update Chebyshev distance (max of all channels and pixels)
                        total_cheb_dist = max(total_cheb_dist, diff)
                        
                        # Update Minkowski distance sum
                        if p_value != 0:
                            total_mink_sum += pow(diff, p_value)
                else:
                    # Grayscale comparison
                    t_val = working_image[ty, tx, 0]
                    s_val = working_image[sy, sx, 0]
                    
                    diff = abs(float(t_val) - float(s_val))
                    
                    # Update Chebyshev distance
                    total_cheb_dist = max(total_cheb_dist, diff)
                    
                    # Update Minkowski distance sum
                    if p_value != 0:
                        total_mink_sum += pow(diff, p_value)
    
    # If no valid pixels to compare, mark as invalid
    if valid_pixel_count == 0:
        distances[source_top, source_left] = float('inf')
        return
    
    # Calculate final Minkowski distance
    minkowski_dist = 0.0
    if p_value != 0:
        minkowski_dist = pow(total_mink_sum, 1.0 / p_value)
    
    # Final distance = Chebyshev + Minkowski (matching CPU algorithm)
    distances[source_top, source_left] = total_cheb_dist + minkowski_dist

@cuda.jit
def _global_argmin_kernel(distances, min_val, min_idx, height, width):
    """
    Global reduction kernel to find the minimum value and its index.
    Uses one thread block with multiple threads for parallel reduction.
    """
    # Shared memory for reduction within a block
    shared_vals = cuda.shared.array(shape=128, dtype=float32)
    shared_idxs = cuda.shared.array(shape=128, dtype=int32)
    
    tid = cuda.threadIdx.x
    
    # Initialize shared memory
    shared_vals[tid] = float('inf')
    shared_idxs[tid] = -1
    
    # Grid-stride loop to process all elements
    for i in range(tid, height * width, 128):
        y = i // width
        x = i % width
        
        if y < height and x < width:
            val = distances[y, x]
            
            # Keep track of minimum
            if not math.isinf(val) and (math.isinf(shared_vals[tid]) or val < shared_vals[tid]):
                shared_vals[tid] = val
                shared_idxs[tid] = i
    
    cuda.syncthreads()
    
    # Reduction within block
    s = 64
    while s > 0:
        if tid < s:
            if shared_vals[tid + s] < shared_vals[tid]:
                shared_vals[tid] = shared_vals[tid + s]
                shared_idxs[tid] = shared_idxs[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write result to global memory (thread 0 only)
    if tid == 0:
        if shared_vals[0] < min_val[0]:
            min_val[0] = shared_vals[0]
            min_idx[0] = shared_idxs[0]

@cuda.jit
def _update_patch_kernel(target_center_y, target_center_x,
                         source_center_y, source_center_x,
                         working_image, mask, confidence,
                         pixels_filled_counter, # Device array (size 1) for atomic counter
                         patch_size, half_patch_size,
                         height, width, is_color):
    """
    CUDA kernel to update the target patch based on the source patch.
    Launched over the entire image grid. Each thread checks if it's inside the target patch.
    Updates working_image, mask, and confidence for pixels where mask was 1.
    Atomically increments pixels_filled_counter.
    
    Closely matches the CPU implementation's _update_image method.
    """
    y, x = cuda.grid(2)

    # Check image bounds
    if y >= height or x >= width:
        return

    # Calculate target patch bounds (ensuring we don't go out of the image boundaries)
    target_top = max(0, target_center_y - half_patch_size)
    target_bottom = min(height - 1, target_center_y + half_patch_size)
    target_left = max(0, target_center_x - half_patch_size)
    target_right = min(width - 1, target_center_x + half_patch_size)

    # Check if the current pixel (y, x) is within the target patch bounds
    is_in_target_patch = (target_top <= y <= target_bottom and 
                           target_left <= x <= target_right)

    if is_in_target_patch:
        # Original confidence value at target center - will be propagated to filled pixels
        patch_confidence = confidence[target_center_y, target_center_x]
        
        # Check if this pixel needs to be updated (is part of the hole)
        if mask[y, x] == 1:
            # Calculate relative offset from target patch's top-left
            rel_y = y - target_top
            rel_x = x - target_left
            
            # Calculate corresponding source patch's top-left
            source_top = max(0, source_center_y - half_patch_size)
            source_left = max(0, source_center_x - half_patch_size)
            
            # Calculate corresponding source coordinates
            sy = source_top + rel_y
            sx = source_left + rel_x
            
            # Additional bounds check for source coordinates
            if 0 <= sy < height and 0 <= sx < width:
                # 1. Copy pixel data from source to working image
                if is_color:
                    for c in range(3):
                        working_image[y, x, c] = working_image[sy, sx, c]
                else:
                    working_image[y, x, 0] = working_image[sy, sx, 0]
                
                # 2. Update mask (pixel is now known)
                mask[y, x] = 0
                
                # 3. Update confidence (propagate confidence from target center)
                confidence[y, x] = patch_confidence
                
                # 4. Atomically increment the counter for filled pixels
                cuda.atomic.add(pixels_filled_counter, 0, 1)
