import sys
import os
import time
import cv2
import numpy as np
from numba import cuda
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QSlider, QComboBox,
                               QFileDialog, QProgressBar, QMessageBox, QButtonGroup,
                               QGroupBox, QSplitter, QFrame, QStatusBar, QSpinBox, QSizePolicy,
                               QDoubleSpinBox, QStyle)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QIcon
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
#from crim_cuda import Inpainter as CudaInpainter
from gpu_inpainter import GPUInpainter as CudaInpainter # Use the new GPU implementation
from papier_test import Inpainter_Dyalna as CPUInpainter


class InpaintThread(QThread):
    """Worker thread for running the inpainting process"""
    progress_update = Signal(int)  # Signal for progress updates (0-100)
    process_complete = Signal(np.ndarray)  # Signal emitting the result image
    error_occurred = Signal(str)  # Signal for error handling

    def __init__(self, implementation, image, mask, patch_size, p_value=1.0):
        super().__init__()
        self.implementation = implementation
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.p_value = p_value

    def run(self):
        try:
            if self.implementation == "CPU":
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
                inpainter = CPUInpainter(self.image, mask_binary, self.patch_size, plot_progress=False, p=self.p_value)

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
                result = inpainter.inpaint()
                
                # Ensure we emit 100% when done
                self.progress_update.emit(100)
                # Signal completion with result
                self.process_complete.emit(result)

            elif self.implementation == "GPU":
                try:
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
                    print(f"Initializing GPUInpainter with patch_size={self.patch_size}, p_value={self.p_value}")
                    inpainter = CudaInpainter(self.image, mask_binary, self.patch_size, self.p_value)
                    print("GPUInpainter initialized successfully.")

                    print("Starting GPU inpaint process...")
                    
                    inpainter_generator = inpainter.inpaint()
                    result_image = None
                    last_emitted_percentage = 0 # Track last emitted to avoid redundant signals

                    try:
                        while True:
                            filled_count, total_to_fill = next(inpainter_generator)
                            
                            current_percentage = 0
                            if total_to_fill > 0:
                                current_percentage = int((filled_count / total_to_fill) * 100)
                            elif filled_count == 0 and total_to_fill == 0: # Case: nothing to fill
                                current_percentage = 100
                            
                            if current_percentage != last_emitted_percentage:
                                self.progress_update.emit(current_percentage)
                                last_emitted_percentage = current_percentage

                    except StopIteration as e:
                        result_image = e.value # Capture the returned image from the generator
                    
                    print("GPU inpaint process finished.")

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

        except Exception as e:
            self.error_occurred.emit(str(e))


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with proper scaling"""

    def __init__(self, title=""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setFrameShape(QFrame.Box)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)

        # Add a title label
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-weight: bold;")

        # Create a layout for the image and title
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.title, 0, Qt.AlignTop)
        self.layout.addStretch()

        # Initialize with placeholder
        self.setPlaceholder()

    def setPlaceholder(self):
        """Set a placeholder when no image is loaded"""
        self.clear()
        self.setText("No Image")
        self.setStyleSheet("QLabel { background-color : #f0f0f0; color : #888; }")

    def setImage(self, image):
        """Set an image to display (accepts cv2/numpy image)"""
        if image is None:
            self.setPlaceholder()
            return

        # Convert cv2 image (BGR) to QImage (RGB)
        if len(image.shape) == 3:  # Color image
            if image.shape[2] == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                # Handle other channel counts if needed
                return
        else:  # Grayscale image
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        # Create pixmap and set it
        pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(self.scaled_pixmap(pixmap))
        self.setStyleSheet("")

    def scaled_pixmap(self, pixmap):
        """Scale the pixmap to fit the label while preserving aspect ratio"""
        return pixmap.scaled(self.width(), self.height(),
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def resizeEvent(self, event):
        """Handle resize events to scale the image appropriately"""
        super().resizeEvent(event)
        if not self.pixmap():
            return

        self.setPixmap(self.scaled_pixmap(self.pixmap()))


class InpaintingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.input_image = None
        self.mask_image = None
        self.result_image = None
        self.inpaint_thread = None

        # Set up the UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Image Inpainting Application")
        self.setMinimumSize(1200, 600)

        # Icon provider
        style = self.style()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top bar with file controls
        top_bar = QHBoxLayout()
        load_image_btn = QPushButton("Load Image")
        load_image_btn.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        load_image_btn.clicked.connect(self.load_image)

        load_mask_btn = QPushButton("Load Mask")
        load_mask_btn.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon)) # A generic file icon, or be more specific
        load_mask_btn.clicked.connect(self.load_mask)

        save_result_btn = QPushButton("Save Result")
        save_result_btn.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        save_result_btn.clicked.connect(self.save_result)
        save_result_btn.setEnabled(False)
        self.save_result_btn = save_result_btn

        reset_btn = QPushButton("Reset")
        reset_btn.setIcon(style.standardIcon(QStyle.SP_BrowserReload)) # Or SP_TrashIcon or SP_DialogResetButton
        reset_btn.clicked.connect(self.reset)

        top_bar.addWidget(load_image_btn)
        top_bar.addWidget(load_mask_btn)
        top_bar.addStretch()
        top_bar.addWidget(reset_btn)
        top_bar.addWidget(save_result_btn)

        main_layout.addLayout(top_bar)

        # Image display area with splitter for resizing
        image_splitter = QSplitter(Qt.Horizontal)

        # Create image display labels
        self.input_image_label = ImageLabel("Original Image")
        self.mask_image_label = ImageLabel("Mask")
        self.result_image_label = ImageLabel("Result")

        image_splitter.addWidget(self.input_image_label)
        image_splitter.addWidget(self.mask_image_label)
        image_splitter.addWidget(self.result_image_label)

        # Set equal sizes for the splitter sections
        image_splitter.setSizes([400, 400, 400])

        main_layout.addWidget(image_splitter, 1)  # Give it a stretch factor

        # Controls group box
        controls_group = QGroupBox("Inpainting Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Implementation selection
        impl_layout = QHBoxLayout()
        impl_label = QLabel("Implementation:")
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")

        # Set default selection based on CUDA availability
        try:
            import numba
            has_cuda = numba.cuda.is_available()
            if has_cuda:
                self.gpu_radio.setChecked(True)
            else:
                self.cpu_radio.setChecked(True)
                self.gpu_radio.setEnabled(False)
                self.gpu_radio.setToolTip("CUDA is not available on this system")
        except ImportError:
            self.cpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip("Numba CUDA is not installed")

        # Group the radio buttons
        impl_group = QButtonGroup(self)
        impl_group.addButton(self.cpu_radio)
        impl_group.addButton(self.gpu_radio)

        impl_layout.addWidget(impl_label)
        impl_layout.addWidget(self.cpu_radio)
        impl_layout.addWidget(self.gpu_radio)
        impl_layout.addStretch()

        controls_layout.addLayout(impl_layout)

        # Patch size control
        patch_layout = QHBoxLayout()
        patch_label = QLabel("Patch Size:")
        self.patch_size_spinner = QSpinBox()
        self.patch_size_spinner.setRange(3, 21)
        self.patch_size_spinner.setValue(9)
        self.patch_size_spinner.setSingleStep(2)  # Only allow odd numbers

        # Force odd numbers only
        self.patch_size_spinner.valueChanged.connect(self.validate_patch_size)

        #patch_info = QLabel("()")

        patch_layout.addWidget(patch_label)
        patch_layout.addWidget(self.patch_size_spinner)
        #patch_layout.addWidget(patch_info)
        patch_layout.addStretch()

        controls_layout.addLayout(patch_layout)
        
        # Minkowski order parameter control
        p_layout = QHBoxLayout()
        p_label = QLabel("Minkowski Order (p):")
        self.p_value_spinner = QDoubleSpinBox()
        self.p_value_spinner.setRange(0.1, 10.0)
        self.p_value_spinner.setValue(1.0)
        self.p_value_spinner.setSingleStep(0.1)
        self.p_value_spinner.setDecimals(1)
        p_info = QLabel("(Affects patch matching - 1.0 for Manhattan, 2.0 for Euclidean)")
        
        p_layout.addWidget(p_label)
        p_layout.addWidget(self.p_value_spinner)
        p_layout.addWidget(p_info)
        p_layout.addStretch()
        
        controls_layout.addLayout(p_layout)

        main_layout.addWidget(controls_group)

        # Bottom area with progress bar and run button
        bottom_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.run_btn = QPushButton("Run Inpainting")
        self.run_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay)) # Or SP_ArrowRightIcon
        self.run_btn.clicked.connect(self.run_inpainting)
        self.run_btn.setEnabled(False)

        bottom_layout.addWidget(self.progress_bar, 1)
        bottom_layout.addWidget(self.run_btn)

        main_layout.addLayout(bottom_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Apply some styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                padding: 5px 15px;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5c85d6;
            }
        """)

    def validate_patch_size(self, value):
        """Ensure patch size is always odd"""
        if value % 2 == 0:
            self.patch_size_spinner.setValue(value + 1)

    def load_image(self):
        """Load an input image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path:
            try:
                # Load with OpenCV
                self.input_image = cv2.imread(file_path)
                if self.input_image is None:
                    raise ValueError("Could not read the image file")

                # Update UI
                self.input_image_label.setImage(self.input_image)
                self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_path)}")

                # If mask is already loaded, resize it to match the image
                if self.mask_image is not None:
                    h, w = self.input_image.shape[:2]
                    self.mask_image = cv2.resize(self.mask_image, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.mask_image_label.setImage(self.mask_image)

                # Check if we can enable the run button
                self.update_run_button_state()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")

    def load_mask(self):
        """Load a mask image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path:
            try:
                # Load with OpenCV
                self.mask_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.mask_image is None:
                    raise ValueError("Could not read the mask file")

                # Ensure binary mask
                _, self.mask_image = cv2.threshold(self.mask_image, 127, 255, cv2.THRESH_BINARY)

                # Resize mask to match input image if loaded
                if self.input_image is not None:
                    h, w = self.input_image.shape[:2]
                    self.mask_image = cv2.resize(self.mask_image, (w, h),
                                                 interpolation=cv2.INTER_NEAREST)

                # Update UI
                self.mask_image_label.setImage(self.mask_image)
                self.status_bar.showMessage(f"Loaded mask: {os.path.basename(file_path)}")

                # Check if we can enable the run button
                self.update_run_button_state()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading mask: {str(e)}")

    def update_run_button_state(self):
        """Update the state of the run button based on loaded images"""
        can_run = (self.input_image is not None and
                   self.mask_image is not None)

        self.run_btn.setEnabled(can_run)

    def run_inpainting(self):
        """Updated function for starting the inpainting process with proper error handling"""
        if self.input_image is None or self.mask_image is None:
            QMessageBox.warning(self, "Warning", "Please load both an image and a mask first")
            return

        # Get parameters
        if self.gpu_radio.isChecked():
            implementation = "GPU"
            # Check CUDA availability at runtime
            try:
                import numba
                if not numba.cuda.is_available():
                    QMessageBox.warning(self, "Warning",
                                        "CUDA is not available. Switching to CPU implementation.")
                    implementation = "CPU"
                    self.cpu_radio.setChecked(True)
            except ImportError:
                QMessageBox.warning(self, "Warning",
                                    "Numba CUDA is not installed. Switching to CPU implementation.")
                implementation = "CPU"
                self.cpu_radio.setChecked(True)
        else:
            implementation = "CPU"

        patch_size = self.patch_size_spinner.value()
        p_value = self.p_value_spinner.value()
        
        # Disable UI elements during processing
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"Processing using {implementation} implementation with p={p_value}...")

        # Start processing thread
        self.inpaint_thread = InpaintThread(
            implementation, self.input_image, self.mask_image, patch_size, p_value
        )

        # Connect signals
        self.inpaint_thread.progress_update.connect(self.update_progress)
        self.inpaint_thread.process_complete.connect(self.process_complete)
        self.inpaint_thread.error_occurred.connect(self.process_error)

        # Start the thread
        self.inpaint_thread.start()

    @Slot(int)
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    @Slot(np.ndarray)
    def process_complete(self, result_image):
        """Enhanced completion handling with additional verification"""
        # Verify the result image
        if result_image is None:
            self.process_error("Inpainting produced a null result")
            return

        try:
            # Check that the result is a valid image
            h, w = result_image.shape[:2]
            if h <= 0 or w <= 0:
                self.process_error("Inpainting produced an invalid image size")
                return

            # Make a safety copy of the result
            self.result_image = np.copy(result_image)

            # Update the UI
            self.result_image_label.setImage(self.result_image)
            self.status_bar.showMessage("Inpainting completed successfully")
            self.run_btn.setEnabled(True)
            self.save_result_btn.setEnabled(True)

            # Show success message
            QMessageBox.information(self, "Success", "Inpainting completed successfully!")

        except Exception as e:
            self.process_error(f"Error processing result: {str(e)}")

    @Slot(str)
    def process_error(self, error_message):
        """Enhanced error handling with more specific messages"""
        # Log the error for debugging
        print(f"ERROR: {error_message}")

        # Check if this is a CUDA-specific error
        if "CUDA" in error_message:
            QMessageBox.critical(self, "CUDA Error",
                                 f"{error_message}\n\nTry using the CPU implementation instead.")
        else:
            QMessageBox.critical(self, "Error", f"Inpainting failed: {error_message}")

        self.status_bar.showMessage("Inpainting failed")
        self.run_btn.setEnabled(True)

    def save_result(self):
        """Save the result image to file"""
        if self.result_image is None:
            QMessageBox.warning(self, "Warning", "No result image to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        if file_path:
            try:
                # Save the image
                cv2.imwrite(file_path, self.result_image)
                self.status_bar.showMessage(f"Result saved to: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save result: {str(e)}")

    def reset(self):
        """Reset the application state"""
        # Clear images
        self.input_image = None
        self.mask_image = None
        self.result_image = None

        # Reset UI
        self.input_image_label.setPlaceholder()
        self.mask_image_label.setPlaceholder()
        self.result_image_label.setPlaceholder()

        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)

        self.status_bar.showMessage("Reset complete")


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("logo.png")) # Set application icon
    app.setStyle("Fusion")  # Apply Fusion style
    window = InpaintingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()