#!/usr/bin/python

import sys, os, time
import math, cv2
import numpy as np
import numba as nb
from numba import jit, cuda

print(cuda.select_device(0))


# Move these CUDA kernels outside the Inpainter class, at the top of the file
@cuda.jit
def compute_patch_errors_kernel(source_patches, target_patch, source_mask, target_mask, patch_errors, p_value):
    """
    CUDA kernel to compute errors between source patches and target patch
    using Chebyshev and Minkowski distance metrics
    """
    patch_idx = cuda.grid(1)  # Get current thread index
    if patch_idx < source_patches.shape[0]:  # Check if index is within bounds
        # Initialize metrics
        max_diff = 0.0  # For Chebyshev distance
        sum_abs_diff_p = 0.0  # For Minkowski distance with power p
        valid_pixels = 0

        # Iterate through patch pixels
        for i in range(target_patch.shape[0]):
            for j in range(target_patch.shape[1]):
                # We only compare pixels that are:
                # 1. Valid in the source patch (source_mask == 1)
                # 2. Valid in the target patch (target_mask == 1) - these are pixels we DON'T need to fill
                # This ensures we match based on the known context around the hole
                if source_mask[patch_idx, i, j] == 1 and target_mask[i, j] == 1:
                    valid_pixels += 1
                    
                    # Calculate absolute differences for each channel
                    pixel_abs_diff = 0.0
                    for c in range(3):
                        abs_diff = abs(float(source_patches[patch_idx, i, j, c]) - float(target_patch[i, j, c]))
                        pixel_abs_diff = max(pixel_abs_diff, abs_diff)

                    # Update Chebyshev distance (max absolute difference)
                    max_diff = max(max_diff, pixel_abs_diff)

                    # Add to Minkowski distance sum
                    sum_abs_diff_p += pixel_abs_diff ** p_value

        # Store the combined error metric
        if valid_pixels > 0:
            # Calculate Minkowski distance
            minkowski_dist = sum_abs_diff_p ** (1.0 / p_value) if p_value > 0.0 else sum_abs_diff_p

            # Combined metric: Chebyshev + Minkowski (matching CPU implementation)
            combined_error = max_diff + minkowski_dist
            
            # Store the error and valid pixel count
            patch_errors[patch_idx, 0] = combined_error
            patch_errors[patch_idx, 1] = float(valid_pixels)
        else:
            # If no valid comparison pixels, mark as invalid patch
            patch_errors[patch_idx, 0] = 1e10  # Very large value
            patch_errors[patch_idx, 1] = 0.0


@cuda.jit
def update_matrices_kernel(work_image, gradient_x, gradient_y, confidence,
                           source_region, target_region, updated_mask,
                           target_confidence, target_x, target_y,
                           patch_ul_x, patch_ul_y, best_ul_x, best_ul_y,
                           patch_coords_i, patch_coords_j):
    """
    CUDA kernel for parallel matrix updates
    """
    idx = cuda.grid(1)
    if idx < patch_coords_i.shape[0]:
        # Get pixel coordinates relative to patch
        i = patch_coords_i[idx]
        j = patch_coords_j[idx]

        # Calculate absolute coordinates
        target_px_y = patch_ul_y + i
        target_px_x = patch_ul_x + j
        source_px_y = best_ul_y + i
        source_px_x = best_ul_x + j

        # Update work image (copy colors)
        for c in range(3):
            work_image[target_px_y, target_px_x, c] = work_image[source_px_y, source_px_x, c]

        # Update other matrices
        gradient_x[target_px_y, target_px_x] = gradient_x[source_px_y, source_px_x]
        gradient_y[target_px_y, target_px_x] = gradient_y[source_px_y, source_px_x]
        confidence[target_px_y, target_px_x] = target_confidence
        source_region[target_px_y, target_px_x] = 1
        target_region[target_px_y, target_px_x] = 0
        updated_mask[target_px_y, target_px_x] = 0

@cuda.jit
def find_fill_front_kernel(source_region, output_mask, normal_x, normal_y):
    """
    CUDA kernel to find fill front pixels in parallel
    """
    # Get current pixel coordinates
    y, x = cuda.grid(2)
    height, width = source_region.shape

    # Check if within bounds
    if y < height and x < width:
        # Check if pixel needs to be filled
        if source_region[y, x] == 0:  # Unfilled pixel
            # Check if it has any filled neighbors (boundary pixel)
            is_boundary = False
            normal_dx, normal_dy = 0.0, 0.0
            filled_neighbors = 0

            # Check 8-connected neighbors
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if source_region[ny, nx] == 1:  # Filled neighbor
                            is_boundary = True
                            # Accumulate normal direction (pointing from filled to unfilled)
                            normal_dx -= dx
                            normal_dy -= dy
                            filled_neighbors += 1

            if is_boundary:
                output_mask[y, x] = 1

                # Normalize the normal vector if we have neighbors
                if filled_neighbors > 0:
                    norm = (normal_dx ** 2 + normal_dy ** 2) ** 0.5
                    if norm > 0:
                        normal_x[y, x] = normal_dx / norm
                        normal_y[y, x] = normal_dy / norm


@cuda.jit
def compute_confidence_kernel(confidence, source_region, target_region, fill_front_x, fill_front_y, half_patch_width):
    """
    CUDA kernel to compute confidence values for all fill front points in parallel
    """
    idx = cuda.grid(1)  # Get current thread index

    if idx < fill_front_x.shape[0]:  # Check if index is within bounds
        # Get point coordinates
        px = fill_front_x[idx]
        py = fill_front_y[idx]

        # Get patch bounds
        height, width = confidence.shape
        min_x = max(px - half_patch_width, 0)
        max_x = min(px + half_patch_width, width - 1)
        min_y = max(py - half_patch_width, 0)
        max_y = min(py + half_patch_width, height - 1)

        # Compute confidence
        total = 0.0
        count = 0

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if target_region[y, x] == 0:  # If pixel is in source region
                    total += confidence[y, x]
                    count += 1

        # Update confidence at this point
        if count > 0:
            confidence[py, px] = total / count


@cuda.jit
def compute_data_kernel(data, gradient_x, gradient_y, normal_x, normal_y, fill_front_x, fill_front_y):
    """
    CUDA kernel to compute data term values for all fill front points in parallel
    """
    idx = cuda.grid(1)  # Get current thread index

    if idx < fill_front_x.shape[0]:  # Check if index is within bounds
        # Get point coordinates
        px = fill_front_x[idx]
        py = fill_front_y[idx]

        # Get normal vector
        nx = normal_x[idx]
        ny = normal_y[idx]

        # Compute data term (dot product of gradient and normal)
        value = abs(gradient_x[py, px] * nx + gradient_y[py, px] * ny) + 0.001

        # Update data at this point
        data[py, px] = value


@cuda.jit
def process_gradients_kernel(gradient_x, gradient_y, source_region):
    """
    CUDA kernel to process gradients in parallel
    - Zero out gradients in unfilled regions
    - Normalize gradients to [0,1] range
    """
    # Get pixel coordinates
    y, x = cuda.grid(2)
    height, width = source_region.shape

    # Check if within bounds
    if y < height and x < width:
        # Zero out gradients in unfilled regions
        if source_region[y, x] == 0:
            gradient_x[y, x] = 0.0
            gradient_y[y, x] = 0.0
        else:
            # Normalize gradients to range [0,1]
            gradient_x[y, x] = gradient_x[y, x] / 255.0
            gradient_y[y, x] = gradient_y[y, x] / 255.0


class Inpainter():
    DEFAULT_HALF_PATCH_WIDTH = 3
    MODE_ADDITION = 0
    MODE_MULTIPLICATION = 1

    ERROR_INPUT_MAT_INVALID_TYPE = 0
    ERROR_INPUT_MASK_INVALID_TYPE = 1
    ERROR_MASK_INPUT_SIZE_MISMATCH = 2
    ERROR_HALF_PATCH_WIDTH_ZERO = 3
    CHECK_VALID = 4

    inputImage = None
    mask = updatedMask = None
    result = None
    workImage = None
    sourceRegion = None
    targetRegion = None
    originalSourceRegion = None
    gradientX = None
    gradientY = None
    confidence = None
    data = None
    LAPLACIAN_KERNEL = NORMAL_KERNELX = NORMAL_KERNELY = None
    # cv::Point2i
    bestMatchUpperLeft = bestMatchLowerRight = None
    patchHeight = patchWidth = 0
    # std::vector<cv::Point> -> list[(y,x)]
    fillFront = []
    # std::vector<cv::Point2f>
    normals = []
    sourcePatchULList = []
    targetPatchSList = []
    targetPatchTList = []
    mode = None
    halfPatchWidth = None
    targetIndex = None

    def __init__(self, inputImage, mask, halfPatchWidth=4, mode=1):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.updatedMask = np.copy(mask)
        self.workImage = np.copy(inputImage)
        self.result = np.ndarray(shape=inputImage.shape, dtype=inputImage.dtype)
        self.mode = mode
        self.halfPatchWidth = halfPatchWidth

    def checkValidInputs(self):
        if not self.inputImage.dtype == np.uint8:  # CV_8UC3
            return self.ERROR_INPUT_MAT_INVALID_TYPE
        if not self.mask.dtype == np.uint8:  # CV_8UC1
            return self.ERROR_INPUT_MASK_INVALID_TYPE
        if not self.mask.shape == self.inputImage.shape[:2]:  # CV_ARE_SIZES_EQ
            return self.ERROR_MASK_INPUT_SIZE_MISMATCH
        if self.halfPatchWidth == 0:
            return self.ERROR_HALF_PATCH_WIDTH_ZERO
        return self.CHECK_VALID

    def inpaint(self):
        """Main inpainting function"""
        print("Starting inpainting process with GPU approach")
        # Initialize matrices needed for inpainting
        self.initializeMats()
        
        # Initial gradient calculation
        self.calculateGradients()
        
        # Save visualization of initial state
        vis_image = self.workImage.copy()
        mask_vis = np.zeros_like(self.workImage)
        mask_vis[self.targetRegion == 1] = [0, 0, 255]  # Red for unfilled areas
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_vis, 0.3, 0)
        cv2.imwrite("initial_state.jpg", vis_image)
        
        # Main inpainting loop
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        last_remaining = float('inf')

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Find the fill front (boundary of source/target regions)
            self.computeFillFront()
            if not self.fillFront:
                print("No more fill front points!")
                break

            # Compute confidence and data terms
            self.computeConfidence()
            self.computeData()
            
            # Find the target pixel with highest priority
            self.computeTarget()
            
            # Find the best patch to copy from
            p_value = 1.0  # Default Minkowski parameter
            if hasattr(self, 'p_value'):
                p_value = self.p_value
                print(f"Using custom p-value: {p_value}")
                
            self.computeBestPatch(p=p_value)
            
            # Update the image with the best patch
            self.updateMats()

            # Check if we're making progress
            current_remaining = np.sum(self.targetRegion)
            if current_remaining >= last_remaining:
                print(f"WARNING: No progress made! {current_remaining} pixels still remain")
                # Increase patch size to try to make progress
                self.halfPatchWidth += 1
                print(f"Increasing patch size to {self.halfPatchWidth}")
                if self.halfPatchWidth > 10:  # Limit patch size growth
                    print("Maximum patch size reached, stopping")
                    break
            last_remaining = current_remaining

            # Track progress
            self.track_progress()

            # Check if we're done
            if np.sum(self.targetRegion) == 0:
                print("Inpainting completed!")
                break
                
            # Every 5 iterations, save a progress image
            if iteration % 5 == 0 or iteration == 1:
                vis_image = self.workImage.copy()
                mask_vis = np.zeros_like(self.workImage)
                mask_vis[self.targetRegion == 1] = [0, 0, 255]  # Red for unfilled areas
                vis_image = cv2.addWeighted(vis_image, 0.7, mask_vis, 0.3, 0)
                cv2.imwrite(f"progress_iter_{iteration}.jpg", vis_image)
        
        # Save the final result
        self.result = self.workImage.copy()
        print(f"Inpainting finished after {iteration} iterations")
        
        # Save final visualization
        cv2.imwrite("final_result.jpg", self.result)

    def initializeMats(self):
        """Initialize necessary matrices for inpainting"""
        print("DEBUG: Starting initialization...")
        print(f"DEBUG: Input mask shape: {self.mask.shape}, dtype: {self.mask.dtype}")
        print(f"DEBUG: Mask min: {np.min(self.mask)}, max: {np.max(self.mask)}")
        
        # CRITICAL: In GPU implementation, we need:
        # - sourceRegion = 1 for known/filled pixels, 0 for unknown/to-be-filled
        # - targetRegion = 1 for pixels that need filling, 0 for known pixels
        # The mask is assumed to be 0 for source region, 255 for target region
        
        # Create sourceRegion (1=known/source, 0=unknown/target)
        _, self.sourceRegion = cv2.threshold(self.mask, 127, 1, cv2.THRESH_BINARY_INV)
        self.originalSourceRegion = np.copy(self.sourceRegion)

        # Create confidence map (same as sourceRegion initially)
        self.confidence = self.sourceRegion.astype(np.float32)
        
        # Create targetRegion (1=target/to-fill, 0=source/known)
        self.targetRegion = 1 - self.sourceRegion
        
        # Initialize updatedMask for visualization
        self.updatedMask = np.copy(self.mask)
        
        # Debug outputs
        print(f"DEBUG: sourceRegion shape: {self.sourceRegion.shape}, sum: {np.sum(self.sourceRegion)}")
        print(f"DEBUG: targetRegion shape: {self.targetRegion.shape}, sum: {np.sum(self.targetRegion)}")
        source_percent = (np.sum(self.sourceRegion) / (self.sourceRegion.shape[0] * self.sourceRegion.shape[1])) * 100
        print(f"DEBUG: Source region is {source_percent:.2f}% of the image")
        
        # Initialize data term
        self.data = np.zeros(self.inputImage.shape[:2], dtype=np.float32)
        
        # Initialize kernels for gradient computation
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype=np.float32)
        self.LAPLACIAN_KERNEL[1, 1] = -8
        self.NORMAL_KERNELX = np.zeros((3, 3), dtype=np.float32)
        self.NORMAL_KERNELX[1, 0] = -1
        self.NORMAL_KERNELX[1, 2] = 1
        self.NORMAL_KERNELY = cv2.transpose(self.NORMAL_KERNELX)

    def calculateGradients(self):
        """GPU-accelerated gradient calculation"""
        # Use OpenCV for the initial Scharr gradient calculation (already optimized)
        srcGray = cv2.cvtColor(self.workImage, cv2.COLOR_BGR2GRAY)

        # Calculate X and Y gradients using Scharr operator
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0)
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)

        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)

        # Transfer data to GPU
        d_gradient_x = cuda.to_device(self.gradientX)
        d_gradient_y = cuda.to_device(self.gradientY)
        d_source_region = cuda.to_device(self.sourceRegion)

        # Configure CUDA grid for 2D processing
        height, width = self.sourceRegion.shape[:2]
        threads_per_block = (16, 16)  # 256 threads per block in a 2D grid
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch kernel
        process_gradients_kernel[blocks_per_grid, threads_per_block](
            d_gradient_x, d_gradient_y, d_source_region
        )

        # Copy results back to host
        d_gradient_x.copy_to_host(self.gradientX)
        d_gradient_y.copy_to_host(self.gradientY)

        print(f"Processed gradients for {height}x{width} image using GPU")

    def computeFillFront(self):
        """GPU-accelerated approach to find fill front pixels"""
        del self.fillFront[:]
        del self.normals[:]
        height, width = self.sourceRegion.shape[:2]

        # Create output arrays
        output_mask = np.zeros((height, width), dtype=np.uint8)
        normal_x = np.zeros((height, width), dtype=np.float32)
        normal_y = np.zeros((height, width), dtype=np.float32)

        # Transfer data to GPU
        d_source_region = cuda.to_device(self.sourceRegion)
        d_output_mask = cuda.to_device(output_mask)
        d_normal_x = cuda.to_device(normal_x)
        d_normal_y = cuda.to_device(normal_y)

        # Configure CUDA grid
        threads_per_block = (16, 16)  # 256 threads per block
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch kernel
        find_fill_front_kernel[blocks_per_grid, threads_per_block](
            d_source_region, d_output_mask, d_normal_x, d_normal_y
        )

        # Copy results back to host
        d_output_mask.copy_to_host(output_mask)
        d_normal_x.copy_to_host(normal_x)
        d_normal_y.copy_to_host(normal_y)

        # Extract fill front points and normals
        for y in range(height):
            for x in range(width):
                if output_mask[y, x] == 1:
                    self.fillFront.append((x, y))
                    self.normals.append((normal_x[y, x], normal_y[y, x]))

        print(f"Found {len(self.fillFront)} points in fill front")

        # Fallback if no points found (similar to original implementation)
        if len(self.fillFront) == 0:
            mask = (self.sourceRegion == 0).astype(np.uint8)
            ys, xs = np.where(mask == 1)
            if len(ys) > 0:
                self.fillFront.append((xs[0], ys[0]))
                self.normals.append((0, 1))
                print(f"Added single unfilled pixel at ({xs[0]}, {ys[0]})")

    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = max(centerX - self.halfPatchWidth, 0)
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        minY = max(centerY - self.halfPatchWidth, 0)
        maxY = min(centerY + self.halfPatchWidth, height - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight

    def computeConfidence(self):
        """GPU-accelerated confidence computation"""
        if not self.fillFront:
            return

        # Extract fill front coordinates
        fill_front_x = np.array([p[0] for p in self.fillFront], dtype=np.int32)
        fill_front_y = np.array([p[1] for p in self.fillFront], dtype=np.int32)

        # Transfer data to GPU
        d_confidence = cuda.to_device(self.confidence)
        d_source_region = cuda.to_device(self.sourceRegion)
        d_target_region = cuda.to_device(self.targetRegion)
        d_fill_front_x = cuda.to_device(fill_front_x)
        d_fill_front_y = cuda.to_device(fill_front_y)

        # Configure CUDA grid
        threads_per_block = 256
        blocks_per_grid = (len(self.fillFront) + threads_per_block - 1) // threads_per_block

        # Launch kernel
        compute_confidence_kernel[blocks_per_grid, threads_per_block](
            d_confidence, d_source_region, d_target_region,
            d_fill_front_x, d_fill_front_y, self.halfPatchWidth
        )

        # Copy results back to host
        d_confidence.copy_to_host(self.confidence)

        print(f"Computed confidence for {len(self.fillFront)} points using GPU")

    def computeData(self):
        """GPU-accelerated data term computation"""
        if not self.fillFront:
            return

        # Extract fill front coordinates and normals
        fill_front_x = np.array([p[0] for p in self.fillFront], dtype=np.int32)
        fill_front_y = np.array([p[1] for p in self.fillFront], dtype=np.int32)
        normal_x = np.array([n[0] for n in self.normals], dtype=np.float32)
        normal_y = np.array([n[1] for n in self.normals], dtype=np.float32)

        # Transfer data to GPU
        d_data = cuda.to_device(self.data)
        d_gradient_x = cuda.to_device(self.gradientX)
        d_gradient_y = cuda.to_device(self.gradientY)
        d_normal_x = cuda.to_device(normal_x)
        d_normal_y = cuda.to_device(normal_y)
        d_fill_front_x = cuda.to_device(fill_front_x)
        d_fill_front_y = cuda.to_device(fill_front_y)

        # Configure CUDA grid
        threads_per_block = 256
        blocks_per_grid = (len(self.fillFront) + threads_per_block - 1) // threads_per_block

        # Launch kernel
        compute_data_kernel[blocks_per_grid, threads_per_block](
            d_data, d_gradient_x, d_gradient_y,
            d_normal_x, d_normal_y, d_fill_front_x, d_fill_front_y
        )

        # Copy results back to host
        d_data.copy_to_host(self.data)

        print(f"Computed data term for {len(self.fillFront)} points using GPU")

    def computeTarget(self):
        self.targetIndex = 0
        maxPriority = 0
        found_valid_target = False

        # Create a front mask similar to the Python implementation
        front_mask = np.zeros_like(self.sourceRegion, dtype=np.uint8)
        for x, y in self.fillFront:
            front_mask[y, x] = 1

        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]

            # Check if this point actually needs filling
            if self.sourceRegion[y, x] == 0:
                # Direct multiplication as in the Python implementation
                priority = self.confidence[y, x] * self.data[y, x] * front_mask[y, x]

                if priority > maxPriority:
                    maxPriority = priority
                    self.targetIndex = i
                    found_valid_target = True

        if not found_valid_target:
            print("WARNING: No valid target found!")

    def computeBestPatch(self, p=1.0):
        """
        CUDA-accelerated version of computeBestPatch that matches the CPU version's behavior
        by using LAB color space and the same distance metrics.
        """
        print(f'Finding best patch with GPU acceleration, using LAB color space and p={p}')
        if not self.fillFront:
            print("ERROR: Fill front is empty!")
            return

        # Get the current target point and its patch
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1

        print(f"Target point: {currentPoint}, Patch: ({aX},{aY}) to ({bX},{bY}), Size: {pWidth}x{pHeight}")

        # Classify pixels in the target patch
        self.targetPatchSList = []  # Source pixels in target patch (known)
        self.targetPatchTList = []  # Target pixels in target patch (unknown)

        for i in range(pHeight):
            for j in range(pWidth):
                if self.sourceRegion[aY + i, aX + j] == 1:
                    self.targetPatchSList.append((i, j))
                else:
                    self.targetPatchTList.append((i, j))

        if not self.targetPatchTList:
            print("No pixels to fill in this patch!")
            return

        print(f"Target patch has {len(self.targetPatchSList)} known pixels and {len(self.targetPatchTList)} unknown pixels")
        
        # Extract target patch
        target_patch = self.workImage[aY:bY + 1, aX:bX + 1].copy()
        target_mask = self.sourceRegion[aY:bY + 1, aX:bX + 1].copy()
        
        # Convert entire work image to LAB color space to match CPU version
        try:
            from skimage.color import rgb2lab
            # Convert BGR to RGB, then to LAB
            work_image_rgb = cv2.cvtColor(self.workImage, cv2.COLOR_BGR2RGB)
            lab_image = rgb2lab(work_image_rgb)
            print("Successfully converted image to LAB color space")
        except ImportError:
            print("WARNING: skimage.color module not available, using BGR color space instead")
            lab_image = self.workImage  # Fallback to BGR if skimage not available
        
        # Lists to store candidate source patches
        source_patches = []
        source_masks = []
        self.sourcePatchULList = []

        # Prepare to search for source patches
        height, width = self.workImage.shape[:2]
        
        # Use a step size to speed up search while finding enough patches
        step = max(1, min(4, pWidth // 4))
        print(f"Using step size {step} for patch search")
        
        # Collect source patches - limit to a reasonable number for GPU processing
        max_patches = 5000
        patches_found = 0
        patches_examined = 0
        
        # Create LAB target patch - convert the target patch to LAB for matching
        if 'lab_image' in locals() and lab_image is not self.workImage:
            lab_target_patch = lab_image[aY:bY + 1, aX:bX + 1].copy()
        else:
            lab_target_patch = target_patch
        
        # Search for source patches in the image
        for y in range(0, height - pHeight + 1, step):
            for x in range(0, width - pWidth + 1, step):
                patches_examined += 1
                
                # Skip patches that overlap with the target
                if (x <= bX and x + pWidth > aX and y <= bY and y + pHeight > aY):
                    continue
                
                # Check if this patch contains any masked (unfilled) pixels
                # CPU version skips patches that have ANY masked pixels
                patch_mask = self.sourceRegion[y:y+pHeight, x:x+pWidth]
                if np.any(patch_mask == 0):  # If any pixel is masked/unfilled
                    continue
                
                # Extract patch in LAB color space
                if 'lab_image' in locals() and lab_image is not self.workImage:
                    lab_source_patch = lab_image[y:y+pHeight, x:x+pWidth].copy()
                else:
                    lab_source_patch = self.workImage[y:y+pHeight, x:x+pWidth].copy()
                
                # Get source patch and its mask
                source_mask = np.ones((pHeight, pWidth), dtype=np.uint8)  # All valid
                
                source_patches.append(lab_source_patch)
                source_masks.append(source_mask)
                self.sourcePatchULList.append((y, x))
                patches_found += 1
                
                if patches_found >= max_patches:
                    break
            
            if patches_found >= max_patches:
                break
        
        print(f"Found {patches_found} valid source patches out of {patches_examined} examined")
        
        if not source_patches:
            print("ERROR: No valid source patches found!")
            return

        # Convert to numpy arrays
        source_patches = np.array(source_patches)
        source_masks = np.array(source_masks)
        
        # Transfer data to GPU
        d_source_patches = cuda.to_device(source_patches)
        d_target_patch = cuda.to_device(lab_target_patch)
        d_source_masks = cuda.to_device(source_masks)
        d_target_mask = cuda.to_device(target_mask)
        d_patch_errors = cuda.to_device(np.zeros((len(source_patches), 2), dtype=np.float32))

        # Configure CUDA kernel
        threadsperblock = 256
        blockspergrid = (len(source_patches) + (threadsperblock - 1)) // threadsperblock

        # Launch kernel
        compute_patch_errors_kernel[blockspergrid, threadsperblock](
            d_source_patches, d_target_patch, d_source_masks, d_target_mask, d_patch_errors, p
        )

        # Get results back from GPU
        patch_errors = d_patch_errors.copy_to_host()

        # Find best patch
        min_error = float('inf')
        best_idx = 0

        for idx in range(len(source_patches)):
            error = patch_errors[idx, 0]
            valid_pixels = patch_errors[idx, 1]

            # If valid pixels found and better error, update best match
            if valid_pixels > 0 and error < min_error:
                min_error = error
                best_idx = idx

        # Set best match coordinates
        y, x = self.sourcePatchULList[best_idx]
        self.bestMatchUpperLeft = (x, y)
        self.bestMatchLowerRight = (x + pWidth - 1, y + pHeight - 1)

        print(f"Best matching patch found at ({x},{y}) with error {patch_errors[best_idx, 0]:.2f}")
        
        # For debugging: save a visualization of the match
        debug_img = self.workImage.copy()
        cv2.rectangle(debug_img, (aX, aY), (bX, bY), (0, 0, 255), 2)  # Target in red
        cv2.rectangle(debug_img, (x, y), (x + pWidth - 1, y + pHeight - 1), (0, 255, 0), 2)  # Source in green
        cv2.imwrite("debug_match.jpg", debug_img)

    def updateMats(self):
        """
        Update matrices with the best matching patch,
        following the CPU implementation's approach
        """
        print("Updating matrices with best patch...")
        if not self.targetPatchTList:
            print("No pixels to fill in this patch!")
            return

        # Get coordinates for target patch and best source patch
        targetPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(targetPoint)
        (bulX, bulY) = self.bestMatchUpperLeft
        
        # Get patch dimensions
        pHeight, pWidth = bY - aY + 1, bX - aX + 1

        print(f"Target patch: ({aX},{aY}) to ({bX},{bY})")
        print(f"Source patch: ({bulX},{bulY}) to ({bulX+pWidth-1},{bulY+pHeight-1})")
        print(f"Number of pixels to update: {len(self.targetPatchTList)}")

        # Following CPU implementation in papier_test.py: _update_image method
        # 1. Update confidence for all pixels in target patch
        patch_confidence = self.confidence[targetPoint[1], targetPoint[0]]
        for i, j in self.targetPatchTList:
            target_y, target_x = aY + i, aX + j
            if 0 <= target_y < self.workImage.shape[0] and 0 <= target_x < self.workImage.shape[1]:
                self.confidence[target_y, target_x] = patch_confidence
        
        # 2. Extract mask, source data and target data
        mask = np.zeros((pHeight, pWidth), dtype=np.uint8)
        for i, j in self.targetPatchTList:
            if 0 <= aY+i < mask.shape[0] and 0 <= aX+j < mask.shape[1]:
                mask[i, j] = 1
        
        # Convert mask to 3-channel for element-wise multiplication
        rgb_mask = np.stack([mask, mask, mask], axis=2)
        
        # Extract source and target data
        source_data = self.workImage[bulY:bulY+pHeight, bulX:bulX+pWidth].copy()
        target_data = self.workImage[aY:aY+pHeight, aX:aX+pWidth].copy()
        
        # 3. Combine source and target data using the mask
        # Source data where mask is 1, target data where mask is 0
        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)
        
        # 4. Copy new data to target patch
        self.workImage[aY:aY+pHeight, aX:aX+pWidth] = new_data
        
        # 5. Update working mask to mark these pixels as filled
        for i, j in self.targetPatchTList:
            target_y, target_x = aY + i, aX + j
            if 0 <= target_y < self.sourceRegion.shape[0] and 0 <= target_x < self.sourceRegion.shape[1]:
                self.sourceRegion[target_y, target_x] = 1
                self.targetRegion[target_y, target_x] = 0
                self.updatedMask[target_y, target_x] = 0
        
        # Count updated pixels
        updated_count = len(self.targetPatchTList)
        print(f"Updated {updated_count} pixels")
        
        # Save updated image for debugging
        cv2.imwrite("workImage_update.jpg", self.workImage)
        
        # Create a visualization showing what we've filled
        vis_image = self.workImage.copy()
        # Draw rectangles showing the patches
        cv2.rectangle(vis_image, (aX, aY), (bX, bY), (0, 0, 255), 2)  # Target in red
        cv2.rectangle(vis_image, (bulX, bulY), (bulX + pWidth - 1, bulY + pHeight - 1), (0, 255, 0), 2)  # Source in green
        # Show mask
        mask_vis = np.zeros_like(vis_image)
        mask_vis[self.targetRegion == 1] = [0, 0, 255]  # Red for unfilled areas
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_vis, 0.3, 0)
        cv2.imwrite("update_visualization.jpg", vis_image)

    def track_progress(self):
        """
        Tracks the progress of inpainting by calculating the ratio of filled pixels
        Returns True if inpainting is complete, False otherwise
        """
        height, width = self.workImage.shape[:2]
        # Count remaining pixels to be filled (where targetRegion is 1)
        remaining = np.sum(self.targetRegion)
        total = np.sum(self.originalSourceRegion == 0)  # Count of original pixels to fill
        
        # Avoid division by zero
        if total == 0:
            percentage = 100
        else:
            # Calculate percentage completed
            completed = total - remaining
            percentage = (completed / total) * 100

        print(f'Progress: {completed} of {total} pixels completed ({percentage:.2f}%)')
        return remaining == 0
