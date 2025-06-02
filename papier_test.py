import numpy as np
import matplotlib.pyplot as plt
import time
import cv2  # Add OpenCV import
from skimage import io, color
from skimage.color import rgb2gray  # Keep only rgb2gray
from skimage.filters import laplace
from scipy.ndimage.filters import convolve

class Inpainter_Dyalna:
    def __init__(self, image, mask, patch_size=9, plot_progress=False, p=1):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.plot_progress = plot_progress
        self.p = p

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.priority = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()

            target_pixel = self._find_highest_priority_pixel()
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            print('Time to find best: %f seconds'
                  % (time.time()-find_start_time))

            self._update_image(target_pixel, source_patch)

            keep_going = not self._finished()

        print('Took %f seconds to complete' % (time.time() - start_time))
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.5)  # Pauses for 0.001 seconds to update the figure.

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]

        # Use float32 explicitly for confidence
        self.confidence = (1 - self.mask).astype(np.float32)
        # Use float32 for data term
        self.data = np.zeros([height, width], dtype=np.float32)

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # TODO: check if scipy's laplace filter is faster than scikit's

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        # Ensure float32 precision for priority calculation
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            # Ensure float32 precision in confidence calculation
            new_confidence[point[0], point[1]] = np.float32(sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch))

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        # Calculate dot product between normal and gradient vectors
        # Original Criminisi formula: D(p) = |∇I_p⊥ · n_p| / α
        # This is the absolute value of the dot product
        dot_product = normal[:,:,0] * gradient[:,:,0] + normal[:,:,1] * gradient[:,:,1]
        # Ensure float32 precision
        self.data = np.abs(dot_product).astype(np.float32) + np.float32(0.001)  # To be sure to have a greater than 0 data

    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]], dtype=np.float32)
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]], dtype=np.float32)

        # Ensure float32 precision for convolution
        x_normal = convolve(self.working_mask.astype(np.float32), x_kernel)
        y_normal = convolve(self.working_mask.astype(np.float32), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        # Avoid division by zero
        norm[norm == 0] = 1

        # Ensure float32 precision for normal calculation
        unit_normal = (normal/norm).astype(np.float32)
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        # Ensure float32 precision for gradient calculations
        # Ensure LAB conversion is in float32 - match GPU implementation exactly
        if len(self.working_image.shape) == 3 and self.working_image.shape[2] == 3:
            # Color image: use same conversion as GPU (BGR uint8 -> BGR float32 [0,1] -> LAB)
            image_float32 = self.working_image.astype(np.float32) / 255.0
            lab_image = cv2.cvtColor(image_float32, cv2.COLOR_BGR2LAB).astype(np.float32)
        else:
            # Grayscale: convert to LAB-like representation
            gray_image = rgb2gray(self.working_image).astype(np.float32)
            # For grayscale, create a 3-channel version for consistent processing
            lab_image = np.stack([gray_image, gray_image, gray_image], axis=2)

        grey_image = lab_image[:,:,0]
        grey_image[self.working_mask == 1] = np.nan

        gradient = np.nan_to_num(np.array(np.gradient(grey_image))).astype(np.float32)
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2).astype(np.float32)
        max_gradient = np.zeros([height, width, 2], dtype=np.float32)

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        # Ensure LAB conversion is in float32 - match GPU implementation exactly
        if len(self.working_image.shape) == 3 and self.working_image.shape[2] == 3:
            # Color image: use same conversion as GPU (BGR uint8 -> BGR float32 [0,1] -> LAB)
            image_float32 = self.working_image.astype(np.float32) / 255.0
            lab_image = cv2.cvtColor(image_float32, cv2.COLOR_BGR2LAB).astype(np.float32)
        else:
            # Grayscale: convert to LAB-like representation
            gray_image = rgb2gray(self.working_image).astype(np.float32)
            # For grayscale, create a 3-channel version for consistent processing
            lab_image = np.stack([gray_image, gray_image, gray_image], axis=2)

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self._patch_data(self.working_mask, source_patch) \
                   .sum() != 0:
                    continue

                difference = self._calc_patch_difference_dyalna(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    def _calc_patch_difference_dyalna(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask

        # Ensure float32 precision for distance calculations
        cheb_dist = np.max(np.abs(target_data - source_data)).astype(np.float32)

        # p = 1 (or whatever user specified)
        # Ensure float32 precision for Minkowski distance
        p_float = np.float32(self.p)
        diff_abs = np.abs(target_data - source_data).astype(np.float32)
        diff_pow = np.power(diff_abs, p_float).astype(np.float32)
        minkowski_dist = np.power(np.sum(diff_pow), np.float32(1.0 / p_float)).astype(np.float32)

        return cheb_dist + minkowski_dist

    def _finished(self):
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        completed = total - remaining
        percentage_completed = (completed / total) * 100  # Calculate percentage
        print(f'{percentage_completed:.2f}% of the execution completed ({completed} of {total})')
        return remaining == 0

    @staticmethod
    def _patch_area(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
