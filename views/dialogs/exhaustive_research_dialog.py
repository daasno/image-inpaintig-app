"""
Exhaustive Research Dialog for running multiple parameter combinations
"""
import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QCheckBox, QLabel, QProgressBar,
                              QScrollArea, QWidget, QGridLayout, QFrame,
                              QSplitter, QListWidget, QListWidgetItem,
                              QMessageBox, QSpinBox, QLineEdit, QFileDialog)
from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QPixmap, QPainter, QFont

import time
import os
import webbrowser
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from models.inpaint_worker import InpaintWorker
from config.settings import AppConstants

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.style.use('default')  # Use default matplotlib style
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn pandas")


@dataclass
class BatchResult:
    """Container for a single batch processing result"""
    image: np.ndarray
    parameters: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: str = ""


class BatchProcessingThread(QThread):
    """Thread for running multiple inpainting operations"""
    
    # Signals
    progress_update = Signal(int)  # Overall progress (0-100)
    status_update = Signal(str)  # Status message
    combination_complete = Signal(BatchResult)  # Emitted when one combination is done
    all_complete = Signal()  # Emitted when all combinations are done
    
    def __init__(self, input_image, mask_image, combinations):
        super().__init__()
        self.input_image = input_image
        self.mask_image = mask_image
        self.combinations = combinations
        self.should_stop = False
        self.current_result = None
        self.current_error = None
        
    def run(self):
        """Run all parameter combinations"""
        total_combinations = len(self.combinations)
        
        for idx, params in enumerate(self.combinations):
            if self.should_stop:
                break
                
            # Update progress
            progress = int((idx / total_combinations) * 100)
            self.progress_update.emit(progress)
            
            # Update status
            self.status_update.emit(
                f"Processing combination {idx + 1}/{total_combinations}: "
                f"Patch={params['patch_size']}, P={params['p_value']}, {params['implementation']}"
            )
            
            # Process this combination
            start_time = time.time()
            
            try:
                # Create worker for this combination
                worker = InpaintWorker(
                    implementation=params['implementation'],
                    image=self.input_image.copy(),
                    mask=self.mask_image.copy(),
                    patch_size=params['patch_size'],
                    p_value=params['p_value']
                )
                
                # Reset result variables
                self.current_result = None
                self.current_error = None
                
                # Connect worker signals
                worker.process_complete.connect(self._on_worker_complete)
                worker.error_occurred.connect(self._on_worker_error)
                
                # Run synchronously (we're already in a thread)
                worker.run()
                
                # Wait for worker to complete (since run() might return before signals are emitted)
                # We'll use a simple loop to wait for either result or error
                timeout_counter = 0
                max_timeout = 300  # 30 seconds timeout (300 * 0.1 second intervals)
                
                while self.current_result is None and self.current_error is None and timeout_counter < max_timeout:
                    self.msleep(100)  # Wait 100ms
                    timeout_counter += 1
                
                processing_time = time.time() - start_time
                
                # Create result object
                if self.current_result is not None:
                    result = BatchResult(
                        image=self.current_result,
                        parameters=params.copy(),
                        processing_time=processing_time,
                        success=True
                    )
                elif self.current_error is not None:
                    result = BatchResult(
                        image=None,
                        parameters=params.copy(),
                        processing_time=processing_time,
                        success=False,
                        error_message=self.current_error
                    )
                else:
                    # Timeout occurred
                    result = BatchResult(
                        image=None,
                        parameters=params.copy(),
                        processing_time=processing_time,
                        success=False,
                        error_message="Processing timeout occurred"
                    )
                
            except Exception as e:
                # Handle errors during worker creation
                processing_time = time.time() - start_time
                result = BatchResult(
                    image=None,
                    parameters=params.copy(),
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e)
                )
            
            # Emit result
            self.combination_complete.emit(result)
        
        # Update final progress
        self.progress_update.emit(100)
        self.status_update.emit("All combinations completed!")
        self.all_complete.emit()
    
    def _on_worker_complete(self, result_image):
        """Handle worker completion"""
        self.current_result = result_image
    
    def _on_worker_error(self, error_message):
        """Handle worker error"""
        self.current_error = error_message
    
    def stop(self):
        """Stop the batch processing"""
        self.should_stop = True


class ResultThumbnail(QFrame):
    """Widget for displaying a single result thumbnail"""
    
    clicked = Signal(BatchResult)
    selection_changed = Signal(bool)  # Signal emitted when selection state changes
    
    def __init__(self, result: BatchResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the thumbnail UI"""
        self.setFrameStyle(QFrame.Box)
        self.setFixedSize(130, 180)  # Increased height to accommodate checkbox
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Selection checkbox at the top
        self.selection_checkbox = QCheckBox()
        self.selection_checkbox.setText("Select")
        self.selection_checkbox.setStyleSheet("font-size: 10px; font-weight: bold;")
        self.selection_checkbox.toggled.connect(self.on_selection_changed)
        layout.addWidget(self.selection_checkbox)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(120, 100)  # Reduced height to make room for checkbox
        
        if self.result.success and self.result.image is not None:
            # Convert numpy array to QPixmap
            pixmap = self._numpy_to_pixmap(self.result.image)
            self.image_label.setPixmap(pixmap.scaled(120, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText("Error")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("background-color: #ffcccc;")
        
        layout.addWidget(self.image_label)
        
        # Parameters label
        params_text = f"Patch: {self.result.parameters['patch_size']}, P: {self.result.parameters['p_value']}\n"
        params_text += f"{self.result.parameters['implementation']} - {self.result.processing_time:.2f}s"
        
        params_label = QLabel(params_text)
        params_label.setWordWrap(True)
        params_label.setAlignment(Qt.AlignCenter)
        params_label.setStyleSheet("font-size: 9px;")  # Slightly smaller font
        layout.addWidget(params_label)
        
        # Initialize styling
        self.update_selection_style()
    
    def _numpy_to_pixmap(self, image):
        """Convert numpy array to QPixmap"""
        from PySide6.QtGui import QImage
        import cv2
        
        if len(image.shape) == 3:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            if channel == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return QPixmap.fromImage(q_image)
    
    def on_selection_changed(self):
        """Handle selection state change"""
        self.selection_changed.emit(self.selection_checkbox.isChecked())
        self.update_selection_style()
    
    def is_selected(self):
        """Check if this thumbnail is selected"""
        return self.selection_checkbox.isChecked()
    
    def set_selected(self, selected):
        """Set the selection state"""
        self.selection_checkbox.setChecked(selected)
    
    def update_selection_style(self):
        """Update styling based on selection state"""
        if self.result.success:
            if self.is_selected():
                self.setStyleSheet("""
                    ResultThumbnail {
                        border: 3px solid #28a745;
                        background-color: #f8fff8;
                    }
                    ResultThumbnail:hover {
                        border: 3px solid #1e7e34;
                        background-color: #e8f8e8;
                    }
                """)
            else:
                self.setStyleSheet("""
                    ResultThumbnail {
                        border: 2px solid #ccc;
                        background-color: white;
                    }
                    ResultThumbnail:hover {
                        border: 2px solid #007acc;
                        background-color: #f0f8ff;
                    }
                """)
        else:
            self.setStyleSheet("""
                ResultThumbnail {
                    border: 2px solid #ff6666;
                    background-color: #fff0f0;
                }
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.result)


class ExhaustiveResearchDialog(QDialog):
    """Dialog for configuring and running exhaustive parameter research"""
    
    def __init__(self, parent, input_image, mask_image):
        super().__init__(parent)
        self.input_image = input_image
        self.mask_image = mask_image
        self.results = []
        self.batch_thread = None
        
        self.setWindowTitle("Exhaustive Research - Image Inpainting")
        self.setModal(True)
        self.resize(800, 500)  # Much smaller, more reasonable dialog size
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Configuration
        config_widget = self.create_configuration_panel()
        splitter.addWidget(config_widget)
        
        # Right panel - Results
        results_widget = self.create_results_panel()
        splitter.addWidget(results_widget)
        
        # Set splitter sizes
        splitter.setSizes([280, 520])  # Adjusted for 800px width: config panel 280, results panel 520
        
        # Bottom controls
        controls_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Configure parameters and click Start")
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.close_btn)
        
        layout.addLayout(controls_layout)
    
    def create_configuration_panel(self):
        """Create the configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Parameter Configuration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Patch sizes group
        patch_group = QGroupBox("Patch Sizes")
        patch_layout = QVBoxLayout(patch_group)
        
        patch_label = QLabel("Enter patch sizes (comma-separated):")
        patch_layout.addWidget(patch_label)
        
        self.patch_input = QLineEdit()
        self.patch_input.setPlaceholderText("e.g., 3,5,7,9,11,13,15")
        self.patch_input.setText("7,9,11")  # Default values
        self.patch_input.setStyleSheet("padding: 8px; font-size: 12px;")
        patch_layout.addWidget(self.patch_input)
        
        patch_help = QLabel("Valid range: 3-50 (odd numbers recommended)")
        patch_help.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        patch_layout.addWidget(patch_help)
        
        layout.addWidget(patch_group)
        
        # P-values group
        p_group = QGroupBox("Minkowski Orders (P-values)")
        p_layout = QVBoxLayout(p_group)
        
        p_label = QLabel("Enter p-values (comma-separated):")
        p_layout.addWidget(p_label)
        
        self.p_input = QLineEdit()
        self.p_input.setPlaceholderText("e.g., 0.5,1.0,2.0,3.0")
        self.p_input.setText("1.0,2.0")  # Default values
        self.p_input.setStyleSheet("padding: 8px; font-size: 12px;")
        p_layout.addWidget(self.p_input)
        
        p_help = QLabel("Valid range: 0.1-10.0 (common values: 0.5, 1.0, 2.0)")
        p_help.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        p_layout.addWidget(p_help)
        
        layout.addWidget(p_group)
        
        # Implementation group
        impl_group = QGroupBox("Implementations")
        impl_layout = QVBoxLayout(impl_group)
        
        self.cpu_checkbox = QCheckBox("CPU")
        self.cpu_checkbox.setChecked(True)
        impl_layout.addWidget(self.cpu_checkbox)
        
        self.gpu_checkbox = QCheckBox("GPU (CUDA)")
        # Check GPU availability
        if InpaintWorker.check_gpu_availability():
            self.gpu_checkbox.setChecked(True)
        else:
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setToolTip("GPU not available")
        impl_layout.addWidget(self.gpu_checkbox)
        
        layout.addWidget(impl_group)
        
        # Summary
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("background-color: #000000; color: #ffffff; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.summary_label)
        
        # Start button
        self.start_btn = QPushButton("🚀 Start Exhaustive Research")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self.start_research)
        layout.addWidget(self.start_btn)
        
        # Now update summary after start_btn is created
        self.update_summary()
        
        # Connect signals to update summary
        self.patch_input.textChanged.connect(self.update_summary)
        self.p_input.textChanged.connect(self.update_summary)
        self.cpu_checkbox.toggled.connect(self.update_summary)
        self.gpu_checkbox.toggled.connect(self.update_summary)
        
        layout.addStretch()
        
        return widget
    
    def create_results_panel(self):
        """Create the results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title and selection controls
        header_layout = QHBoxLayout()
        
        title = QLabel("Results Gallery")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Selection controls
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #005999;
            }
        """)
        self.select_all_btn.clicked.connect(self.select_all_results)
        header_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        self.deselect_all_btn.clicked.connect(self.deselect_all_results)
        header_layout.addWidget(self.deselect_all_btn)
        
        self.download_btn = QPushButton("💾 Download Selected")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.download_btn.clicked.connect(self.download_selected_results)
        self.download_btn.setEnabled(False)
        header_layout.addWidget(self.download_btn)
        
        # Plot results button
        self.plot_btn = QPushButton("📊 Plot Results")
        self.plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.plot_btn.clicked.connect(self.plot_results)
        self.plot_btn.setEnabled(False)
        if not PLOTTING_AVAILABLE:
            self.plot_btn.setToolTip("Matplotlib/Seaborn not installed. Install with: pip install matplotlib seaborn pandas")
        header_layout.addWidget(self.plot_btn)
        
        layout.addLayout(header_layout)
        
        # Selection status label
        self.selection_status_label = QLabel("No results yet")
        self.selection_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px 10px;")
        layout.addWidget(self.selection_status_label)
        
        # Scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 12px;
                border: none;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.results_widget = QWidget()
        self.results_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
            }
        """)
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(10)
        
        scroll_area.setWidget(self.results_widget)
        layout.addWidget(scroll_area)
        
        return widget
    
    def update_summary(self):
        """Update the summary label with current selections"""
        # Parse text inputs
        patch_sizes = self.parse_patch_sizes()
        p_values = self.parse_p_values()
        
        # Count selections
        patch_count = len(patch_sizes)
        p_count = len(p_values)
        impl_count = sum(1 for cb in [self.cpu_checkbox, self.gpu_checkbox] if cb.isChecked() and cb.isEnabled())
        
        total_combinations = patch_count * p_count * impl_count
        
        # Estimate time (rough estimate: 3 seconds per combination)
        estimated_time = total_combinations * 3
        time_str = f"{estimated_time // 60}m {estimated_time % 60}s" if estimated_time >= 60 else f"{estimated_time}s"
        
        # Check for parsing errors
        patch_error = len(patch_sizes) == 0 and self.patch_input.text().strip() != ""
        p_error = len(p_values) == 0 and self.p_input.text().strip() != ""
        
        summary = f"<b>Total Combinations:</b> {total_combinations}<br>"
        summary += f"<b>Estimated Time:</b> ~{time_str}<br>"
        summary += f"<b>Selected:</b> {patch_count} patch sizes × {p_count} p-values × {impl_count} implementations"
        
        if patch_error or p_error:
            summary += "<br><font color='red'><b>Warning:</b> Invalid input format detected</font>"
        
        self.summary_label.setText(summary)
        
        # Enable/disable start button
        self.start_btn.setEnabled(total_combinations > 0 and not patch_error and not p_error)
    
    def parse_patch_sizes(self):
        """Parse patch sizes from text input"""
        try:
            text = self.patch_input.text().strip()
            if not text:
                return []
            
            sizes = []
            for size_str in text.split(','):
                size_str = size_str.strip()
                if size_str:
                    size = int(size_str)
                    if 3 <= size <= 50:  # Reasonable range
                        sizes.append(size)
            return sizes
        except (ValueError, TypeError):
            return []
    
    def parse_p_values(self):
        """Parse p-values from text input"""
        try:
            text = self.p_input.text().strip()
            if not text:
                return []
            
            values = []
            for p_str in text.split(','):
                p_str = p_str.strip()
                if p_str:
                    p = float(p_str)
                    if 0.1 <= p <= 10.0:  # Reasonable range
                        values.append(p)
            return values
        except (ValueError, TypeError):
            return []
    
    def get_selected_combinations(self):
        """Get all selected parameter combinations"""
        combinations = []
        
        # Get parsed values
        selected_patches = self.parse_patch_sizes()
        selected_p_values = self.parse_p_values()
        selected_impls = []
        if self.cpu_checkbox.isChecked():
            selected_impls.append("CPU")
        if self.gpu_checkbox.isChecked() and self.gpu_checkbox.isEnabled():
            selected_impls.append("GPU")
        
        # Generate all combinations
        for patch_size in selected_patches:
            for p_value in selected_p_values:
                for implementation in selected_impls:
                    combinations.append({
                        'patch_size': patch_size,
                        'p_value': p_value,
                        'implementation': implementation
                    })
        
        return combinations
    
    def start_research(self):
        """Start the exhaustive research"""
        # Get combinations
        combinations = self.get_selected_combinations()
        
        if not combinations:
            QMessageBox.warning(self, "No Selection", "Please select at least one value for each parameter.")
            return
        
        # Confirm start
        reply = QMessageBox.question(
            self,
            "Confirm Start",
            f"This will run {len(combinations)} inpainting operations.\n"
            f"This may take several minutes. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Clear previous results
        self.clear_results()
        
        # Disable controls
        self.start_btn.setEnabled(False)
        self.set_configuration_enabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start batch processing
        self.batch_thread = BatchProcessingThread(
            self.input_image,
            self.mask_image,
            combinations
        )
        
        # Connect signals
        self.batch_thread.progress_update.connect(self.on_progress_update)
        self.batch_thread.status_update.connect(self.on_status_update)
        self.batch_thread.combination_complete.connect(self.on_combination_complete)
        self.batch_thread.all_complete.connect(self.on_all_complete)
        
        # Start thread
        self.batch_thread.start()
    
    def clear_results(self):
        """Clear all results from the gallery"""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.results = []
    
    def set_configuration_enabled(self, enabled):
        """Enable/disable configuration controls"""
        self.patch_input.setEnabled(enabled)
        self.p_input.setEnabled(enabled)
        self.cpu_checkbox.setEnabled(enabled)
        if InpaintWorker.check_gpu_availability():
            self.gpu_checkbox.setEnabled(enabled)
    
    def select_all_results(self):
        """Select all result thumbnails"""
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget():
                thumbnail = item.widget()
                if isinstance(thumbnail, ResultThumbnail):
                    thumbnail.set_selected(True)
        self.update_selection_status()
    
    def deselect_all_results(self):
        """Deselect all result thumbnails"""
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget():
                thumbnail = item.widget()
                if isinstance(thumbnail, ResultThumbnail):
                    thumbnail.set_selected(False)
        self.update_selection_status()
    
    def update_selection_status(self):
        """Update the selection status label and download button"""
        total_results = 0
        selected_results = 0
        
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget():
                thumbnail = item.widget()
                if isinstance(thumbnail, ResultThumbnail):
                    total_results += 1
                    if thumbnail.is_selected():
                        selected_results += 1
        
        if total_results == 0:
            self.selection_status_label.setText("No results yet")
        else:
            self.selection_status_label.setText(f"{selected_results} of {total_results} results selected")
        
        # Enable download button only if at least one result is selected
        self.download_btn.setEnabled(selected_results > 0)
        
        # Enable plot button if we have any results and plotting libraries are available
        self.plot_btn.setEnabled(total_results > 0 and PLOTTING_AVAILABLE)
    
    def download_selected_results(self):
        """Download all selected results"""
        selected_thumbnails = []
        
        # Collect selected thumbnails
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget():
                thumbnail = item.widget()
                if isinstance(thumbnail, ResultThumbnail) and thumbnail.is_selected():
                    selected_thumbnails.append(thumbnail)
        
        if not selected_thumbnails:
            QMessageBox.warning(self, "No Selection", "Please select at least one result to download.")
            return
        
        # Ask user to choose directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose Directory to Save Results",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not directory:
            return  # User cancelled
        
        # Save selected images
        import os
        import cv2
        saved_count = 0
        
        try:
            for idx, thumbnail in enumerate(selected_thumbnails):
                result = thumbnail.result
                if result.success and result.image is not None:
                    # Generate filename with parameters
                    filename = f"inpaint_patch{result.parameters['patch_size']}_p{result.parameters['p_value']}_{result.parameters['implementation'].lower()}_{idx + 1:03d}.png"
                    filepath = os.path.join(directory, filename)
                    
                    # Convert from RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(filepath, image_bgr)
                    
                    if success:
                        saved_count += 1
            
            # Show success message
            QMessageBox.information(
                self,
                "Download Complete",
                f"Successfully saved {saved_count} of {len(selected_thumbnails)} selected images to:\n{directory}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Download Error",
                f"An error occurred while saving images:\n{str(e)}"
            )
    
    def plot_results(self):
        """Create and display a bar chart of processing times using seaborn"""
        print("Plot results button clicked!")
        
        if not PLOTTING_AVAILABLE:
            QMessageBox.warning(
                self,
                "Plotting Libraries Not Available",
                "Matplotlib/Seaborn are not installed. Please install with:\npip install matplotlib seaborn pandas"
            )
            return
        
        # Collect data from all results
        data = []
        
        for result in self.results:
            if result.success:
                data.append({
                    'Configuration': f"Patch{result.parameters['patch_size']} P{result.parameters['p_value']} {result.parameters['implementation']}",
                    'Processing Time (seconds)': result.processing_time,
                    'Implementation': result.parameters['implementation'],
                    'Patch Size': result.parameters['patch_size'],
                    'P Value': result.parameters['p_value']
                })
        
        if not data:
            QMessageBox.warning(self, "No Data", "No successful results to plot.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Define colors for different implementations
        colors = {'GPU': '#00a8ff', 'CPU': '#ff6b6b'}
        
        # Create bar plot
        ax = sns.barplot(
            data=df, 
            x='Configuration', 
            y='Processing Time (seconds)',
            hue='Implementation',
            palette=colors,
            dodge=False  # No dodging since we might only have one implementation
        )
        
        # Customize the plot
        plt.title('Inpainting Processing Times by Configuration', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Configuration (Patch Size, P-Value, Implementation)', fontsize=12, fontweight='bold')
        plt.ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(df['Processing Time (seconds)']):
            ax.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        from tkinter import filedialog
        import tkinter as tk
        
        # Create a temporary file first, then ask user where to save
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            temp_path = tmp_file.name
        
        # Show the plot
        plt.show()
        
        # Ask user if they want to save the plot
        reply = QMessageBox.question(
            self, 
            'Save Plot?', 
            'Do you want to save this plot to a file?',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Hide the main window temporarily to show file dialog properly
            self.hide()
            
            # Ask user where to save
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Plot As",
                "inpainting_processing_times.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
            )
            
            # Show the main window again
            self.show()
            
            if file_path:
                try:
                    # Re-create and save the plot to the chosen location
                    plt.figure(figsize=(12, 8))
                    ax = sns.barplot(
                        data=df, 
                        x='Configuration', 
                        y='Processing Time (seconds)',
                        hue='Implementation',
                        palette=colors,
                        dodge=False
                    )
                    plt.title('Inpainting Processing Times by Configuration', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Configuration (Patch Size, P-Value, Implementation)', fontsize=12, fontweight='bold')
                    plt.ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels
                    for i, v in enumerate(df['Processing Time (seconds)']):
                        ax.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
                    
                    plt.grid(axis='y', alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    
                    # Save with high quality
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    plt.close()  # Close the figure to free memory
                    
                    QMessageBox.information(
                        self,
                        "Plot Saved",
                        f"The plot has been saved to:\n{file_path}"
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Save Error",
                        f"Could not save the plot:\n{str(e)}"
                    )
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    @Slot(int)
    def on_progress_update(self, value):
        """Handle progress updates"""
        self.progress_bar.setValue(value)
    
    @Slot(str)
    def on_status_update(self, message):
        """Handle status updates"""
        self.status_label.setText(message)
    
    @Slot(BatchResult)
    def on_combination_complete(self, result):
        """Handle completion of one combination"""
        # Store result
        self.results.append(result)
        
        # Create thumbnail
        thumbnail = ResultThumbnail(result)
        thumbnail.clicked.connect(self.show_result_detail)
        thumbnail.selection_changed.connect(self.update_selection_status)
        
        # Add to grid
        row = (len(self.results) - 1) // 5  # Keep 5 columns due to increased thumbnail height
        col = (len(self.results) - 1) % 5
        self.results_layout.addWidget(thumbnail, row, col)
        
        # Update selection status
        self.update_selection_status()
    
    @Slot()
    def on_all_complete(self):
        """Handle completion of all combinations"""
        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.set_configuration_enabled(True)
        
        # Show completion message
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        
        msg = f"Research completed!\n\n"
        msg += f"Total combinations: {len(self.results)}\n"
        msg += f"Successful: {successful}\n"
        msg += f"Failed: {failed}"
        
        QMessageBox.information(self, "Research Complete", msg)
    
    @Slot(BatchResult)
    def show_result_detail(self, result):
        """Show detailed view of a result"""
        # TODO: Implement a detailed view dialog
        # For now, just show a message box with parameters
        if result.success:
            msg = f"Parameters:\n"
            msg += f"- Patch Size: {result.parameters['patch_size']}\n"
            msg += f"- P-Value: {result.parameters['p_value']}\n"
            msg += f"- Implementation: {result.parameters['implementation']}\n"
            msg += f"- Processing Time: {result.processing_time:.2f} seconds"
        else:
            msg = f"This combination failed:\n{result.error_message}"
        
        QMessageBox.information(self, "Result Details", msg)
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        # Stop batch thread if running
        if self.batch_thread and self.batch_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Research in Progress",
                "Research is still running. Stop and close?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.batch_thread.stop()
                self.batch_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()