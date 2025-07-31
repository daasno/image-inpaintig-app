"""
Batch Exhaustive Research Dialog
Allows users to run exhaustive parameter research on multiple image pairs
"""
import os
import time
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel, 
    QProgressBar, QTextEdit, QLineEdit, QCheckBox, QComboBox, QSpinBox,
    QMessageBox, QSplitter, QScrollArea, QWidget, QGridLayout, QFrame,
    QFileDialog, QTabWidget
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from models.batch_data import BatchData
from models.batch_exhaustive_worker import BatchExhaustiveWorker, ExhaustiveResult, PairResults
from models.inpaint_worker import InpaintWorker


class BatchExhaustiveDialog(QDialog):
    """Dialog for configuring and running batch exhaustive research"""
    
    def __init__(self, parent, batch_data: BatchData):
        super().__init__(parent)
        self.batch_data = batch_data
        self.exhaustive_worker = None
        self.all_results: List[PairResults] = []
        
        self.setWindowTitle("Batch Exhaustive Research - Image Inpainting")
        self.setModal(True)
        self.resize(1000, 700)
        
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
        
        # Right panel - Results and Progress
        results_widget = self.create_results_panel()
        splitter.addWidget(results_widget)
        
        # Set splitter sizes (40% config, 60% results)
        splitter.setSizes([400, 600])
        
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
        title = QLabel("Batch Exhaustive Research Configuration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Batch info
        self.create_batch_info_section(layout)
        
        # Parameter configuration
        self.create_parameter_config_section(layout)
        
        # Research options
        self.create_research_options_section(layout)
        
        # Summary and start
        self.create_summary_section(layout)
        
        layout.addStretch()
        
        return widget
    
    def create_batch_info_section(self, layout):
        """Create batch information section with pair selection"""
        group = QGroupBox("📁 Batch Information & Pair Selection")
        group_layout = QVBoxLayout(group)
        
        # Number of pairs
        pairs_count = self.batch_data.total_pairs
        pairs_label = QLabel(f"Available Image Pairs: {pairs_count}")
        pairs_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
        group_layout.addWidget(pairs_label)
        
        # Selection strategy
        strategy_label = QLabel("Research Strategy:")
        group_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Research All Pairs",
            "Research First N Pairs",
            "Research Selected Pairs"
        ])
        self.strategy_combo.setToolTip(
            "Research All Pairs: Test all available image pairs\n"
            "Research First N Pairs: Test only the first N pairs (good for quick testing)\n"
            "Research Selected Pairs: Manually select specific pairs to research"
        )
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        group_layout.addWidget(self.strategy_combo)
        
        # First N pairs selection (hidden by default)
        self.first_n_widget = QWidget()
        first_n_layout = QHBoxLayout(self.first_n_widget)
        first_n_layout.setContentsMargins(0, 0, 0, 0)
        
        first_n_label = QLabel("Number of pairs:")
        first_n_layout.addWidget(first_n_label)
        
        self.first_n_spinner = QSpinBox()
        self.first_n_spinner.setRange(1, pairs_count)
        self.first_n_spinner.setValue(min(5, pairs_count))
        self.first_n_spinner.valueChanged.connect(self.update_summary)
        first_n_layout.addWidget(self.first_n_spinner)
        
        first_n_layout.addStretch()
        group_layout.addWidget(self.first_n_widget)
        self.first_n_widget.setVisible(False)
        
        # Pair selection area (hidden by default)
        self.pair_selection_widget = QWidget()
        pair_selection_layout = QVBoxLayout(self.pair_selection_widget)
        pair_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        # Selection controls
        selection_controls = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_pairs)
        select_all_btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        selection_controls.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_pairs)
        deselect_all_btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        selection_controls.addWidget(deselect_all_btn)
        
        selection_controls.addStretch()
        pair_selection_layout.addLayout(selection_controls)
        
        # Scrollable list of pairs
        self.pairs_scroll = QScrollArea()
        self.pairs_scroll.setMaximumHeight(150)
        self.pairs_scroll.setWidgetResizable(True)
        
        self.pairs_list_widget = QWidget()
        self.pairs_list_layout = QVBoxLayout(self.pairs_list_widget)
        
        # Create checkboxes for each pair
        self.pair_checkboxes = []
        for i, pair in enumerate(self.batch_data.image_pairs):
            checkbox = QCheckBox(f"{pair.image_filename} ↔ {pair.mask_filename}")
            checkbox.setStyleSheet("font-size: 11px;")
            checkbox.toggled.connect(self.update_summary)
            self.pair_checkboxes.append(checkbox)
            self.pairs_list_layout.addWidget(checkbox)
        
        self.pairs_list_layout.addStretch()
        self.pairs_scroll.setWidget(self.pairs_list_widget)
        pair_selection_layout.addWidget(self.pairs_scroll)
        
        group_layout.addWidget(self.pair_selection_widget)
        self.pair_selection_widget.setVisible(False)
        
        layout.addWidget(group)
    
    def create_parameter_config_section(self, layout):
        """Create parameter configuration section"""
        group = QGroupBox("⚙️ Parameter Combinations")
        group_layout = QVBoxLayout(group)
        
        # Patch sizes
        patch_label = QLabel("Patch Sizes (comma-separated):")
        group_layout.addWidget(patch_label)
        
        self.patch_input = QLineEdit()
        self.patch_input.setPlaceholderText("e.g., 5,7,9,11,13")
        self.patch_input.setText("7,9,11")  # Default values
        self.patch_input.textChanged.connect(self.update_summary)
        group_layout.addWidget(self.patch_input)
        
        # P-values
        p_label = QLabel("P-values (comma-separated):")
        group_layout.addWidget(p_label)
        
        self.p_input = QLineEdit()
        self.p_input.setPlaceholderText("e.g., 0.5,1.0,2.0")
        self.p_input.setText("1.0,2.0")  # Default values
        self.p_input.textChanged.connect(self.update_summary)
        group_layout.addWidget(self.p_input)
        
        # Implementations
        impl_label = QLabel("Implementations:")
        group_layout.addWidget(impl_label)
        
        self.cpu_checkbox = QCheckBox("CPU")
        self.cpu_checkbox.setChecked(True)
        self.cpu_checkbox.toggled.connect(self.update_summary)
        group_layout.addWidget(self.cpu_checkbox)
        
        self.gpu_checkbox = QCheckBox("GPU (CUDA)")
        # Check GPU availability
        if InpaintWorker.check_gpu_availability():
            self.gpu_checkbox.setChecked(True)
        else:
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setToolTip("GPU not available")
        self.gpu_checkbox.toggled.connect(self.update_summary)
        group_layout.addWidget(self.gpu_checkbox)
        
        layout.addWidget(group)
    
    def create_research_options_section(self, layout):
        """Create research options section"""
        group = QGroupBox("🔬 Research Options")
        group_layout = QVBoxLayout(group)
        
        # Best result criteria
        criteria_label = QLabel("Best Result Criteria:")
        group_layout.addWidget(criteria_label)
        
        self.criteria_combo = QComboBox()
        self.criteria_combo.addItems([
            "Fastest Processing",
            "Best PSNR (if metrics enabled)",
            "Best SSIM (if metrics enabled)"
        ])
        self.criteria_combo.setCurrentText("Fastest Processing")
        group_layout.addWidget(self.criteria_combo)
        
        # Calculate metrics option
        self.metrics_checkbox = QCheckBox("Calculate PSNR/SSIM metrics (slower)")
        self.metrics_checkbox.setToolTip("Enable to calculate quality metrics for each result")
        group_layout.addWidget(self.metrics_checkbox)
        
        # Export options
        export_label = QLabel("Export Options:")
        group_layout.addWidget(export_label)
        
        self.export_csv_checkbox = QCheckBox("Export results to CSV")
        self.export_csv_checkbox.setChecked(True)
        group_layout.addWidget(self.export_csv_checkbox)
        
        self.export_best_images_checkbox = QCheckBox("Export best result images")
        self.export_best_images_checkbox.setChecked(True)
        group_layout.addWidget(self.export_best_images_checkbox)
        
        layout.addWidget(group)
    
    def create_summary_section(self, layout):
        """Create summary and start section"""
        group = QGroupBox("📊 Research Summary")
        group_layout = QVBoxLayout(group)
        
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        group_layout.addWidget(self.summary_label)
        
        # Start button
        self.start_btn = QPushButton("🚀 Start Batch Exhaustive Research")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self.start_research)
        group_layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹️ Stop Research")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_research)
        self.stop_btn.setEnabled(False)
        group_layout.addWidget(self.stop_btn)
        
        layout.addWidget(group)
        
        # Update summary initially
        self.update_summary()
    
    def create_results_panel(self):
        """Create the results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tab widget for different views
        self.results_tabs = QTabWidget()
        
        # Progress tab
        self.create_progress_tab()
        
        # Results tab
        self.create_results_tab()
        
        # Summary tab
        self.create_summary_tab()
        
        layout.addWidget(self.results_tabs)
        
        return widget
    
    def create_progress_tab(self):
        """Create progress monitoring tab"""
        progress_widget = QWidget()
        layout = QVBoxLayout(progress_widget)
        
        # Current status
        self.current_status_label = QLabel("Ready to start")
        self.current_status_label.setStyleSheet("font-weight: bold; color: #1976d2;")
        layout.addWidget(self.current_status_label)
        
        # Progress log
        self.progress_log = QTextEdit()
        self.progress_log.setPlaceholderText("Progress updates will appear here...")
        layout.addWidget(self.progress_log)
        
        self.results_tabs.addTab(progress_widget, "📊 Progress")
    
    def create_results_tab(self):
        """Create results display tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Results summary
        self.results_summary_label = QLabel("No results yet")
        layout.addWidget(self.results_summary_label)
        
        # Results list
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Results will appear here...")
        layout.addWidget(self.results_text)
        
        self.results_tabs.addTab(results_widget, "📋 Results")
    
    def create_summary_tab(self):
        """Create final summary tab"""
        summary_widget = QWidget()
        layout = QVBoxLayout(summary_widget)
        
        # Final summary
        self.final_summary_label = QLabel("Research not completed yet")
        layout.addWidget(self.final_summary_label)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_csv_btn = QPushButton("📊 Export CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)
        
        self.export_images_btn = QPushButton("🖼️ Export Best Images")
        self.export_images_btn.clicked.connect(self.export_best_images)
        self.export_images_btn.setEnabled(False)
        export_layout.addWidget(self.export_images_btn)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        layout.addStretch()
        
        self.results_tabs.addTab(summary_widget, "📈 Summary")
    
    def on_strategy_changed(self, strategy):
        """Handle strategy change"""
        self.first_n_widget.setVisible(strategy == "Research First N Pairs")
        self.pair_selection_widget.setVisible(strategy == "Research Selected Pairs")
        self.update_summary()
    
    def select_all_pairs(self):
        """Select all pairs"""
        for checkbox in self.pair_checkboxes:
            checkbox.setChecked(True)
    
    def deselect_all_pairs(self):
        """Deselect all pairs"""
        for checkbox in self.pair_checkboxes:
            checkbox.setChecked(False)
    
    def get_selected_pairs(self):
        """Get list of selected pairs based on current strategy"""
        strategy = self.strategy_combo.currentText()
        
        if strategy == "Research All Pairs":
            return self.batch_data.image_pairs
        elif strategy == "Research First N Pairs":
            n = self.first_n_spinner.value()
            return self.batch_data.image_pairs[:n]
        elif strategy == "Research Selected Pairs":
            selected_pairs = []
            for i, checkbox in enumerate(self.pair_checkboxes):
                if checkbox.isChecked():
                    selected_pairs.append(self.batch_data.image_pairs[i])
            return selected_pairs
        else:
            return []
    
    def update_summary(self):
        """Update the research summary"""
        # Parse inputs
        patch_sizes = self.parse_patch_sizes()
        p_values = self.parse_p_values()
        implementations = self.get_selected_implementations()
        
        # Calculate totals based on selected pairs
        selected_pairs = self.get_selected_pairs()
        pairs_count = len(selected_pairs)
        combinations_per_pair = len(patch_sizes) * len(p_values) * len(implementations)
        total_operations = pairs_count * combinations_per_pair
        
        # Estimate time (rough estimate: 3 seconds per operation)
        estimated_time = total_operations * 3
        if estimated_time >= 3600:
            time_str = f"{estimated_time // 3600}h {(estimated_time % 3600) // 60}m"
        elif estimated_time >= 60:
            time_str = f"{estimated_time // 60}m {estimated_time % 60}s"
        else:
            time_str = f"{estimated_time}s"
        
        # Check for errors
        has_errors = (len(patch_sizes) == 0 or len(p_values) == 0 or 
                     len(implementations) == 0 or pairs_count == 0)
        
        # Build summary
        strategy = self.strategy_combo.currentText()
        summary = f"<b>Batch Exhaustive Research Summary:</b><br>"
        summary += f"• Strategy: {strategy}<br>"
        summary += f"• Selected Pairs: {pairs_count} of {self.batch_data.total_pairs}<br>"
        summary += f"• Parameter Combinations per Pair: {combinations_per_pair}<br>"
        summary += f"• Total Operations: {total_operations}<br>"
        summary += f"• Estimated Time: ~{time_str}<br>"
        summary += f"• Patch Sizes: {patch_sizes}<br>"
        summary += f"• P-values: {p_values}<br>"
        summary += f"• Implementations: {implementations}"
        
        if has_errors:
            summary += "<br><font color='red'><b>⚠️ Configuration incomplete or invalid</b></font>"
        
        self.summary_label.setText(summary)
        self.start_btn.setEnabled(not has_errors)
    
    def parse_patch_sizes(self) -> List[int]:
        """Parse patch sizes from input"""
        try:
            text = self.patch_input.text().strip()
            if not text:
                return []
            return [int(x.strip()) for x in text.split(',') if x.strip() and 3 <= int(x.strip()) <= 50]
        except (ValueError, TypeError):
            return []
    
    def parse_p_values(self) -> List[float]:
        """Parse p-values from input"""
        try:
            text = self.p_input.text().strip()
            if not text:
                return []
            return [float(x.strip()) for x in text.split(',') if x.strip() and 0.1 <= float(x.strip()) <= 10.0]
        except (ValueError, TypeError):
            return []
    
    def get_selected_implementations(self) -> List[str]:
        """Get selected implementations"""
        implementations = []
        if self.cpu_checkbox.isChecked():
            implementations.append("CPU")
        if self.gpu_checkbox.isChecked() and self.gpu_checkbox.isEnabled():
            implementations.append("GPU")
        return implementations
    
    def get_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Get all parameter combinations"""
        combinations = []
        patch_sizes = self.parse_patch_sizes()
        p_values = self.parse_p_values()
        implementations = self.get_selected_implementations()
        
        for patch_size in patch_sizes:
            for p_value in p_values:
                for implementation in implementations:
                    combinations.append({
                        'patch_size': patch_size,
                        'p_value': p_value,
                        'implementation': implementation
                    })
        
        return combinations
    
    def get_best_criteria(self) -> str:
        """Get the best result criteria"""
        criteria_map = {
            "Fastest Processing": "fastest",
            "Best PSNR (if metrics enabled)": "best_psnr",
            "Best SSIM (if metrics enabled)": "best_ssim"
        }
        return criteria_map.get(self.criteria_combo.currentText(), "fastest")
    
    def start_research(self):
        """Start the batch exhaustive research"""
        # Get configurations
        combinations = self.get_parameter_combinations()
        best_criteria = self.get_best_criteria()
        selected_pairs = self.get_selected_pairs()
        
        if not combinations:
            QMessageBox.warning(self, "Invalid Configuration", 
                              "Please configure valid parameter combinations.")
            return
        
        if not selected_pairs:
            QMessageBox.warning(self, "No Pairs Selected", 
                              "Please select at least one image pair to research.")
            return
        
        # Confirm start
        total_ops = len(combinations) * len(selected_pairs)
        reply = QMessageBox.question(
            self,
            "Confirm Start",
            f"This will run {total_ops} inpainting operations.\n"
            f"This may take a very long time. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Clear previous results
        self.all_results.clear()
        self.progress_log.clear()
        self.results_text.clear()
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Switch to progress tab
        self.results_tabs.setCurrentIndex(0)
        
        # Create and start worker
        self.exhaustive_worker = BatchExhaustiveWorker(
            selected_pairs, combinations, best_criteria
        )
        
        # Enable metrics calculation if requested
        if self.metrics_checkbox.isChecked():
            self.exhaustive_worker.set_calculate_metrics(True)
        
        # Connect signals
        self.exhaustive_worker.progress_updated.connect(self.on_progress_updated)
        self.exhaustive_worker.pair_started.connect(self.on_pair_started)
        self.exhaustive_worker.combination_started.connect(self.on_combination_started)
        self.exhaustive_worker.combination_completed.connect(self.on_combination_completed)
        self.exhaustive_worker.pair_completed.connect(self.on_pair_completed)
        self.exhaustive_worker.batch_completed.connect(self.on_batch_completed)
        self.exhaustive_worker.error_occurred.connect(self.on_error_occurred)
        self.exhaustive_worker.status_update.connect(self.on_status_update)
        
        # Start worker
        self.exhaustive_worker.start()
    
    def stop_research(self):
        """Stop the research"""
        if self.exhaustive_worker and self.exhaustive_worker.isRunning():
            self.exhaustive_worker.stop_processing()
            self.progress_log.append("<span style='color: #d32f2f;'>⏹️ Stopping research...</span>")
    
    @Slot(int, int)
    def on_progress_updated(self, current_pair, total_pairs):
        """Handle progress update"""
        if total_pairs > 0:
            progress = int((current_pair / total_pairs) * 100)
            self.progress_bar.setValue(progress)
    
    @Slot(str, str)
    def on_pair_started(self, image_filename, mask_filename):
        """Handle pair started"""
        message = f"🔄 Starting pair: {image_filename} ↔ {mask_filename}"
        self.progress_log.append(f"<span style='color: #1976d2;'>{message}</span>")
        self.current_status_label.setText(f"Processing: {image_filename}")
    
    @Slot(str, str, dict)
    def on_combination_started(self, image_filename, mask_filename, parameters):
        """Handle combination started"""
        params_str = f"Patch={parameters['patch_size']}, P={parameters['p_value']}, {parameters['implementation']}"
        message = f"  ⚙️ Testing: {params_str}"
        self.progress_log.append(f"<span style='color: #666;'>{message}</span>")
    
    @Slot(str, object)
    def on_combination_completed(self, image_filename, result):
        """Handle combination completed"""
        if result.success:
            message = f"  ✅ Completed in {result.processing_time:.2f}s"
            color = "#2e7d32"
        else:
            message = f"  ❌ Failed: {result.error_message}"
            color = "#d32f2f"
        
        self.progress_log.append(f"<span style='color: {color};'>{message}</span>")
    
    @Slot(str, object)
    def on_pair_completed(self, image_filename, pair_results):
        """Handle pair completed"""
        self.all_results.append(pair_results)
        
        successful = len([r for r in pair_results.results if r.success])
        total = len(pair_results.results)
        
        message = f"📋 Pair completed: {successful}/{total} combinations successful"
        if pair_results.best_result:
            best_params = pair_results.best_result.parameters
            message += f" (Best: Patch={best_params['patch_size']}, P={best_params['p_value']}, {best_params['implementation']})"
        
        self.progress_log.append(f"<span style='color: #2e7d32; font-weight: bold;'>{message}</span>")
        
        # Update results tab
        self.update_results_display()
    
    @Slot(dict)
    def on_batch_completed(self, summary):
        """Handle batch completion"""
        # Update UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Show completion message
        total_pairs = summary['total_pairs']
        successful_pairs = summary['successful_pairs']
        total_ops = summary['total_combinations_tested']
        total_time = summary['total_processing_time']
        
        message = f"🎉 Batch exhaustive research completed!\n"
        message += f"Processed {successful_pairs}/{total_pairs} pairs successfully\n"
        message += f"Total operations: {total_ops}\n"
        message += f"Total time: {total_time:.1f} seconds"
        
        self.progress_log.append(f"<span style='color: #2e7d32; font-weight: bold;'>{message}</span>")
        self.current_status_label.setText("Research completed!")
        
        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_images_btn.setEnabled(True)
        
        # Update summary tab
        self.update_final_summary(summary)
        
        # Switch to summary tab
        self.results_tabs.setCurrentIndex(2)
        
        QMessageBox.information(self, "Research Complete", message)
    
    @Slot(str)
    def on_error_occurred(self, error_message):
        """Handle error"""
        self.progress_log.append(f"<span style='color: #d32f2f; font-weight: bold;'>❌ Error: {error_message}</span>")
    
    @Slot(str)
    def on_status_update(self, message):
        """Handle status update"""
        self.current_status_label.setText(message)
    
    def update_results_display(self):
        """Update the results display"""
        if not self.all_results:
            return
        
        results_text = []
        for pair_result in self.all_results:
            results_text.append(f"📁 {pair_result.image_pair.image_filename}:")
            
            for result in pair_result.results:
                if result.success:
                    params = result.parameters
                    status = "✅"
                    details = f"{result.processing_time:.2f}s"
                    if result.metrics:
                        details += f", PSNR: {result.metrics.get('psnr', 0):.2f}, SSIM: {result.metrics.get('ssim', 0):.4f}"
                else:
                    status = "❌"
                    details = result.error_message
                
                line = f"  {status} Patch={params['patch_size']}, P={params['p_value']}, {params['implementation']} - {details}"
                results_text.append(line)
            
            if pair_result.best_result:
                best = pair_result.best_result.parameters
                results_text.append(f"  🏆 Best: Patch={best['patch_size']}, P={best['p_value']}, {best['implementation']}")
            
            results_text.append("")
        
        self.results_text.setText("\n".join(results_text))
        
        # Update summary
        total_results = sum(len(pr.results) for pr in self.all_results)
        successful_results = sum(len([r for r in pr.results if r.success]) for pr in self.all_results)
        
        summary = f"Results: {successful_results}/{total_results} combinations successful across {len(self.all_results)} pairs"
        self.results_summary_label.setText(summary)
    
    def update_final_summary(self, summary):
        """Update the final summary tab"""
        text = f"<h3>Batch Exhaustive Research Summary</h3>"
        text += f"<b>Total Pairs Processed:</b> {summary['successful_pairs']}/{summary['total_pairs']}<br>"
        text += f"<b>Total Operations:</b> {summary['total_combinations_tested']}<br>"
        text += f"<b>Total Processing Time:</b> {summary['total_processing_time']:.1f} seconds<br>"
        text += f"<b>Average Time per Operation:</b> {summary['average_time_per_combination']:.2f} seconds<br>"
        text += f"<b>Best Result Criteria:</b> {summary['best_criteria']}<br>"
        
        if self.all_results:
            # Calculate some statistics
            all_successful = []
            for pair_result in self.all_results:
                all_successful.extend([r for r in pair_result.results if r.success])
            
            if all_successful:
                times = [r.processing_time for r in all_successful]
                text += f"<b>Fastest Operation:</b> {min(times):.2f}s<br>"
                text += f"<b>Slowest Operation:</b> {max(times):.2f}s<br>"
                
                if any(r.metrics for r in all_successful):
                    psnr_values = [r.metrics['psnr'] for r in all_successful if r.metrics and 'psnr' in r.metrics]
                    ssim_values = [r.metrics['ssim'] for r in all_successful if r.metrics and 'ssim' in r.metrics]
                    
                    if psnr_values:
                        text += f"<b>Best PSNR:</b> {max(psnr_values):.2f} dB<br>"
                    if ssim_values:
                        text += f"<b>Best SSIM:</b> {max(ssim_values):.4f}<br>"
        
        self.final_summary_label.setText(text)
    
    def export_to_csv(self):
        """Export results to CSV"""
        if not self.exhaustive_worker:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results to CSV", 
            "batch_exhaustive_results.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            if self.exhaustive_worker.export_results_to_csv(file_path):
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Export Failed", "Failed to export results to CSV.")
    
    def export_best_images(self):
        """Export best result images"""
        if not self.all_results:
            return
        
        directory = QFileDialog.getExistingDirectory(
            self, "Choose Directory for Best Images"
        )
        
        if directory:
            try:
                import cv2
                saved_count = 0
                
                for pair_result in self.all_results:
                    if pair_result.best_result and pair_result.best_result.success:
                        result = pair_result.best_result
                        
                        # Generate filename
                        base_name = os.path.splitext(result.image_pair.image_filename)[0]
                        params = result.parameters
                        filename = f"{base_name}_best_patch{params['patch_size']}_p{params['p_value']}_{params['implementation'].lower()}.png"
                        filepath = os.path.join(directory, filename)
                        
                        # Convert and save
                        image_bgr = cv2.cvtColor(result.result_image, cv2.COLOR_RGB2BGR)
                        if cv2.imwrite(filepath, image_bgr):
                            saved_count += 1
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {saved_count} best result images to:\n{directory}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export images:\n{str(e)}")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.exhaustive_worker and self.exhaustive_worker.isRunning():
            reply = QMessageBox.question(
                self, "Research in Progress",
                "Research is still running. Stop and close?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.exhaustive_worker.stop_processing()
                self.exhaustive_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept() 