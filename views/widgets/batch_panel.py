"""
Batch processing panel widget
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, 
    QProgressBar, QTextEdit, QScrollArea, QFrame, QGroupBox, QFileDialog,
    QMessageBox, QSplitter, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette

from models.batch_data import BatchData, ImagePair
from models.inpaint_worker import InpaintWorker
from config.settings import AppConstants


class BatchPanel(QWidget):
    """Panel for batch processing operations"""
    
    # Signals
    folders_changed = Signal()
    start_batch_requested = Signal()
    stop_batch_requested = Signal()
    exhaustive_research_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.batch_data = BatchData()
        
        # Parameter defaults
        self.current_implementation = "CPU"
        self.current_patch_size = 9
        self.current_p_value = 1.0
        
        self.setup_ui()
        self.update_ui_state()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Folder selection section
        self.create_folder_selection_section(layout)
        
        # Matched pairs section
        self.create_pairs_section(layout)
        
        # Parameters section
        self.create_parameters_section(layout)
        
        # Control section
        self.create_control_section(layout)
        
        # Progress section
        self.create_progress_section(layout)
    
    def create_folder_selection_section(self, parent_layout):
        """Create folder selection controls"""
        group = QGroupBox("📁 Folder Selection")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout = QGridLayout(group)
        
        # Images folder
        self.images_folder_label = QLabel("No folder selected")
        self.images_folder_label.setStyleSheet("color: #888; font-style: italic;")
        images_btn = QPushButton("📷 Select Images Folder")
        images_btn.clicked.connect(self.select_images_folder)
        
        layout.addWidget(QLabel("Images Folder:"), 0, 0)
        layout.addWidget(self.images_folder_label, 0, 1)
        layout.addWidget(images_btn, 0, 2)
        
        # Masks folder
        self.masks_folder_label = QLabel("No folder selected")
        self.masks_folder_label.setStyleSheet("color: #888; font-style: italic;")
        masks_btn = QPushButton("🎭 Select Masks Folder")
        masks_btn.clicked.connect(self.select_masks_folder)
        
        layout.addWidget(QLabel("Masks Folder:"), 1, 0)
        layout.addWidget(self.masks_folder_label, 1, 1)
        layout.addWidget(masks_btn, 1, 2)
        
        # Results folder
        self.results_folder_label = QLabel("No folder selected")
        self.results_folder_label.setStyleSheet("color: #888; font-style: italic;")
        results_btn = QPushButton("💾 Select Results Folder")
        results_btn.clicked.connect(self.select_results_folder)
        
        layout.addWidget(QLabel("Results Folder:"), 2, 0)
        layout.addWidget(self.results_folder_label, 2, 1)
        layout.addWidget(results_btn, 2, 2)
        
        parent_layout.addWidget(group)
    
    def create_pairs_section(self, parent_layout):
        """Create matched pairs display section"""
        group = QGroupBox("🔗 Matched Image Pairs")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout = QVBoxLayout(group)
        
        # Summary label
        self.pairs_summary_label = QLabel("No pairs found")
        self.pairs_summary_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.pairs_summary_label)
        
        # Pairs list
        self.pairs_scroll = QScrollArea()
        self.pairs_widget = QWidget()
        self.pairs_layout = QVBoxLayout(self.pairs_widget)
        self.pairs_scroll.setWidget(self.pairs_widget)
        self.pairs_scroll.setWidgetResizable(True)
        self.pairs_scroll.setMaximumHeight(200)
        
        layout.addWidget(self.pairs_scroll)
        
        # Validation issues
        self.validation_text = QTextEdit()
        self.validation_text.setMaximumHeight(100)
        self.validation_text.setPlaceholderText("Validation issues will appear here...")
        layout.addWidget(self.validation_text)
        
        parent_layout.addWidget(group)
    
    def create_parameters_section(self, parent_layout):
        """Create processing parameters section"""
        group = QGroupBox("⚙️ Processing Parameters")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout = QGridLayout(group)
        
        # Implementation selection
        impl_label = QLabel("Implementation:")
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        
        # ADD YOUR CUSTOM ALGORITHM RADIO BUTTON HERE:
        # Example:
        # self.custom_radio = QRadioButton("My Custom Algorithm")
        
        self.impl_group = QButtonGroup(self)
        self.impl_group.addButton(self.cpu_radio)
        self.impl_group.addButton(self.gpu_radio)
        # self.impl_group.addButton(self.custom_radio)  # Add your custom radio button
        
        # Check GPU availability
        gpu_available = InpaintWorker.check_gpu_availability()
        if gpu_available:
            self.gpu_radio.setChecked(True)
            self.current_implementation = "GPU"
            self.gpu_radio.setToolTip("GPU acceleration available")
        else:
            self.cpu_radio.setChecked(True)
            self.current_implementation = "CPU"
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip("GPU not available - requires CUDA")
        
        # Connect signals
        self.cpu_radio.toggled.connect(self._on_implementation_changed)
        self.gpu_radio.toggled.connect(self._on_implementation_changed)
        
        layout.addWidget(impl_label, 0, 0)
        layout.addWidget(self.cpu_radio, 0, 1)
        layout.addWidget(self.gpu_radio, 0, 2)
        # layout.addWidget(self.custom_radio, 0, 3)  # Add your custom radio button to layout
        
        # Patch size
        patch_label = QLabel("Patch Size:")
        self.patch_size_spinner = QSpinBox()
        self.patch_size_spinner.setRange(AppConstants.MIN_PATCH_SIZE, AppConstants.MAX_PATCH_SIZE)
        self.patch_size_spinner.setValue(self.current_patch_size)
        self.patch_size_spinner.setSingleStep(2)
        self.patch_size_spinner.setToolTip("Size of the patch used for matching (must be odd)")
        self.patch_size_spinner.valueChanged.connect(self._on_patch_size_changed)
        
        layout.addWidget(patch_label, 1, 0)
        layout.addWidget(self.patch_size_spinner, 1, 1)
        
        # P-value
        p_label = QLabel("Minkowski Order (p):")
        self.p_value_spinner = QDoubleSpinBox()
        self.p_value_spinner.setRange(AppConstants.MIN_P_VALUE, AppConstants.MAX_P_VALUE)
        self.p_value_spinner.setValue(self.current_p_value)
        self.p_value_spinner.setSingleStep(0.1)
        self.p_value_spinner.setDecimals(1)
        self.p_value_spinner.setToolTip("Distance metric: 1.0=Manhattan, 2.0=Euclidean")
        self.p_value_spinner.valueChanged.connect(self._on_p_value_changed)
        
        layout.addWidget(p_label, 2, 0)
        layout.addWidget(self.p_value_spinner, 2, 1)
        
        parent_layout.addWidget(group)
    
    def _on_implementation_changed(self):
        """Handle implementation change"""
        if self.cpu_radio.isChecked():
            self.current_implementation = "CPU"
        elif self.gpu_radio.isChecked():
            self.current_implementation = "GPU"
    
    def _on_patch_size_changed(self, value):
        """Handle patch size change"""
        # Force odd numbers only
        if value % 2 == 0:
            self.patch_size_spinner.setValue(value + 1)
            return
        self.current_patch_size = value
    
    def _on_p_value_changed(self, value):
        """Handle p-value change"""
        self.current_p_value = value
    
    def get_parameters(self) -> dict:
        """Get current processing parameters"""
        return {
            'implementation': self.current_implementation,
            'patch_size': self.current_patch_size,
            'p_value': self.current_p_value
        }
    
    def create_control_section(self, parent_layout):
        """Create processing control buttons"""
        group = QGroupBox("🎮 Batch Processing Controls")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout = QHBoxLayout(group)
        
        # Start button
        self.start_button = QPushButton("▶️ Start Batch Processing")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1b5e20;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.start_button.clicked.connect(self.start_batch_requested.emit)
        
        # Stop button
        self.stop_button = QPushButton("⏹️ Stop Processing")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.stop_button.clicked.connect(self.stop_batch_requested.emit)
        self.stop_button.setEnabled(False)
        
        # Exhaustive research button
        self.exhaustive_button = QPushButton("🔬 Batch Exhaustive Research")
        self.exhaustive_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.exhaustive_button.clicked.connect(self.exhaustive_research_requested.emit)
        
        # Refresh button
        refresh_button = QPushButton("🔄 Refresh Pairs")
        refresh_button.clicked.connect(self.refresh_pairs)
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.exhaustive_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(refresh_button)
        layout.addStretch()
        
        parent_layout.addWidget(group)
    
    def create_progress_section(self, parent_layout):
        """Create progress tracking section"""
        group = QGroupBox("📊 Processing Progress")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout = QVBoxLayout(group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Current status
        self.status_label = QLabel("Ready to start batch processing")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)
        
        # Results log
        self.results_log = QTextEdit()
        self.results_log.setMaximumHeight(150)
        self.results_log.setPlaceholderText("Processing results will appear here...")
        layout.addWidget(self.results_log)
        
        parent_layout.addWidget(group)
    
    def select_images_folder(self):
        """Select images folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Images Folder", "", QFileDialog.ShowDirsOnly
        )
        if folder:
            if self.batch_data.set_images_folder(folder):
                self.images_folder_label.setText(os.path.basename(folder))
                self.images_folder_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
                self.update_pairs_display()
                self.folders_changed.emit()
            else:
                QMessageBox.warning(self, "Error", "Invalid folder selected")
    
    def select_masks_folder(self):
        """Select masks folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Masks Folder", "", QFileDialog.ShowDirsOnly
        )
        if folder:
            if self.batch_data.set_masks_folder(folder):
                self.masks_folder_label.setText(os.path.basename(folder))
                self.masks_folder_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
                self.update_pairs_display()
                self.folders_changed.emit()
            else:
                QMessageBox.warning(self, "Error", "Invalid folder selected")
    
    def select_results_folder(self):
        """Select results folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Results Folder", "", QFileDialog.ShowDirsOnly
        )
        if folder:
            if self.batch_data.set_results_folder(folder):
                self.results_folder_label.setText(os.path.basename(folder))
                self.results_folder_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
                self.folders_changed.emit()
            else:
                QMessageBox.warning(self, "Error", "Invalid folder selected")
    
    def refresh_pairs(self):
        """Refresh the pairs display"""
        self.batch_data._scan_and_match_files()
        self.update_pairs_display()
    
    def update_pairs_display(self):
        """Update the display of matched pairs"""
        # Clear previous pairs
        for i in reversed(range(self.pairs_layout.count())):
            item = self.pairs_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        pairs = self.batch_data.image_pairs
        total_pairs = len(pairs)
        
        # Update summary
        if total_pairs == 0:
            self.pairs_summary_label.setText("No matching pairs found")
            self.pairs_summary_label.setStyleSheet("font-weight: bold; color: #d32f2f;")
        else:
            self.pairs_summary_label.setText(f"Found {total_pairs} matching pairs")
            self.pairs_summary_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
        
        # Add pair widgets
        for pair in pairs[:10]:  # Show first 10 pairs
            pair_widget = self.create_pair_widget(pair)
            self.pairs_layout.addWidget(pair_widget)
        
        if total_pairs > 10:
            more_label = QLabel(f"... and {total_pairs - 10} more pairs")
            more_label.setStyleSheet("color: #666; font-style: italic;")
            self.pairs_layout.addWidget(more_label)
        
        # Show validation issues
        self.show_validation_issues()
        
        self.pairs_layout.addStretch()
    
    def create_pair_widget(self, pair: ImagePair) -> QWidget:
        """Create a widget to display a single pair"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 4px; padding: 4px; }")
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Pair info
        pair_label = QLabel(f"✅ {pair.image_filename} ↔ {pair.mask_filename} → {pair.result_name}.jpg")
        pair_label.setStyleSheet("color: #2e7d32;")
        
        layout.addWidget(pair_label)
        layout.addStretch()
        
        return widget
    
    def show_validation_issues(self):
        """Show validation issues in the text area"""
        issues = []
        
        # Unmatched images
        unmatched_images = self.batch_data.get_unmatched_images()
        if unmatched_images:
            issues.append("Unmatched Images (no corresponding mask):")
            for img in unmatched_images[:5]:  # Show first 5
                issues.append(f"  ❌ {img}")
            if len(unmatched_images) > 5:
                issues.append(f"  ... and {len(unmatched_images) - 5} more")
        
        # Unmatched masks
        unmatched_masks = self.batch_data.get_unmatched_masks()
        if unmatched_masks:
            issues.append("\nUnmatched Masks (no corresponding image):")
            for mask in unmatched_masks[:5]:  # Show first 5
                issues.append(f"  ❌ {mask}")
            if len(unmatched_masks) > 5:
                issues.append(f"  ... and {len(unmatched_masks) - 5} more")
        
        if not issues:
            issues.append("✅ All files are properly matched!")
        
        self.validation_text.setText("\n".join(issues))
    
    def update_ui_state(self):
        """Update UI state based on current data"""
        can_start = self.batch_data.is_ready_for_processing
        self.start_button.setEnabled(can_start)
        self.exhaustive_button.setEnabled(can_start)
    
    def set_processing_state(self, is_processing: bool):
        """Update UI for processing state"""
        self.start_button.setEnabled(not is_processing and self.batch_data.is_ready_for_processing)
        self.exhaustive_button.setEnabled(not is_processing and self.batch_data.is_ready_for_processing)
        self.stop_button.setEnabled(is_processing)
        self.progress_bar.setVisible(is_processing)
        
        if not is_processing:
            self.progress_bar.setValue(0)
            self.status_label.setText("Ready to start batch processing")
    
    def update_progress(self, current: int, total: int):
        """Update progress bar"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"Processing pair {current + 1} of {total}")
    
    def add_result_log(self, message: str, is_error: bool = False):
        """Add a message to the results log"""
        color = "#d32f2f" if is_error else "#2e7d32"
        self.results_log.append(f'<span style="color: {color};">{message}</span>')
    
    def clear_results_log(self):
        """Clear the results log"""
        self.results_log.clear()
    
    def get_batch_data(self) -> BatchData:
        """Get the batch data object"""
        return self.batch_data 