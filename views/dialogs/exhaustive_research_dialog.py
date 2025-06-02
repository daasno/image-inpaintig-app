"""
Exhaustive Research Dialog for running multiple parameter combinations
"""
import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QCheckBox, QLabel, QProgressBar,
                              QScrollArea, QWidget, QGridLayout, QFrame,
                              QSplitter, QListWidget, QListWidgetItem,
                              QMessageBox, QSpinBox)
from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QPixmap, QPainter, QFont

import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from models.inpaint_worker import InpaintWorker
from config.settings import AppConstants


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
                
                # Run synchronously (we're already in a thread)
                worker.moveToThread(self.thread())
                worker.run()
                
                # Get result
                result_image = worker.result_image
                processing_time = time.time() - start_time
                
                # Create result object
                result = BatchResult(
                    image=result_image,
                    parameters=params.copy(),
                    processing_time=processing_time,
                    success=True
                )
                
            except Exception as e:
                # Handle errors
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
    
    def stop(self):
        """Stop the batch processing"""
        self.should_stop = True


class ResultThumbnail(QFrame):
    """Widget for displaying a single result thumbnail"""
    
    clicked = Signal(BatchResult)
    
    def __init__(self, result: BatchResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the thumbnail UI"""
        self.setFrameStyle(QFrame.Box)
        self.setFixedSize(200, 250)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(190, 190)
        
        if self.result.success and self.result.image is not None:
            # Convert numpy array to QPixmap
            pixmap = self._numpy_to_pixmap(self.result.image)
            self.image_label.setPixmap(pixmap.scaled(190, 190, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        params_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(params_label)
        
        # Style based on success
        if self.result.success:
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
        self.resize(1200, 800)
        
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
        splitter.setSizes([400, 800])
        
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
        
        self.patch_checkboxes = {}
        for size in range(3, 22, 2):  # 3, 5, 7, ..., 21
            checkbox = QCheckBox(f"Size {size}")
            if size in [7, 9, 11]:  # Default selections
                checkbox.setChecked(True)
            self.patch_checkboxes[size] = checkbox
            patch_layout.addWidget(checkbox)
        
        layout.addWidget(patch_group)
        
        # P-values group
        p_group = QGroupBox("Minkowski Orders (P-values)")
        p_layout = QVBoxLayout(p_group)
        
        self.p_checkboxes = {}
        p_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        for p in p_values:
            checkbox = QCheckBox(f"P = {p}")
            if p in [1.0, 2.0]:  # Default selections
                checkbox.setChecked(True)
            self.p_checkboxes[p] = checkbox
            p_layout.addWidget(checkbox)
        
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
        self.summary_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.update_summary()
        layout.addWidget(self.summary_label)
        
        # Connect signals to update summary
        for checkbox in self.patch_checkboxes.values():
            checkbox.toggled.connect(self.update_summary)
        for checkbox in self.p_checkboxes.values():
            checkbox.toggled.connect(self.update_summary)
        self.cpu_checkbox.toggled.connect(self.update_summary)
        self.gpu_checkbox.toggled.connect(self.update_summary)
        
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
        
        layout.addStretch()
        
        return widget
    
    def create_results_panel(self):
        """Create the results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Results Gallery")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(10)
        
        scroll_area.setWidget(self.results_widget)
        layout.addWidget(scroll_area)
        
        return widget
    
    def update_summary(self):
        """Update the summary label with current selections"""
        # Count selections
        patch_count = sum(1 for cb in self.patch_checkboxes.values() if cb.isChecked())
        p_count = sum(1 for cb in self.p_checkboxes.values() if cb.isChecked())
        impl_count = sum(1 for cb in [self.cpu_checkbox, self.gpu_checkbox] if cb.isChecked() and cb.isEnabled())
        
        total_combinations = patch_count * p_count * impl_count
        
        # Estimate time (rough estimate: 3 seconds per combination)
        estimated_time = total_combinations * 3
        time_str = f"{estimated_time // 60}m {estimated_time % 60}s" if estimated_time >= 60 else f"{estimated_time}s"
        
        summary = f"<b>Total Combinations:</b> {total_combinations}<br>"
        summary += f"<b>Estimated Time:</b> ~{time_str}<br>"
        summary += f"<b>Selected:</b> {patch_count} patch sizes × {p_count} p-values × {impl_count} implementations"
        
        self.summary_label.setText(summary)
        
        # Enable/disable start button
        self.start_btn.setEnabled(total_combinations > 0)
    
    def get_selected_combinations(self):
        """Get all selected parameter combinations"""
        combinations = []
        
        # Get selected values
        selected_patches = [size for size, cb in self.patch_checkboxes.items() if cb.isChecked()]
        selected_p_values = [p for p, cb in self.p_checkboxes.items() if cb.isChecked()]
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
        for checkbox in self.patch_checkboxes.values():
            checkbox.setEnabled(enabled)
        for checkbox in self.p_checkboxes.values():
            checkbox.setEnabled(enabled)
        self.cpu_checkbox.setEnabled(enabled)
        if InpaintWorker.check_gpu_availability():
            self.gpu_checkbox.setEnabled(enabled)
    
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
        
        # Add to grid
        row = (len(self.results) - 1) // 4
        col = (len(self.results) - 1) % 4
        self.results_layout.addWidget(thumbnail, row, col)
    
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