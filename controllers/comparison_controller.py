"""
Comparison Controller
Handles comparison logic between original and inpainted images
"""
import os
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QPixmap

from config.settings import AppSettings, AppConstants
from models import ComparisonData


class ComparisonController(QObject):
    """Controller for image comparison functionality"""
    
    def __init__(self, comparison_panel):
        super().__init__()
        
        # Load settings
        self.settings = AppSettings.load()
        
        # Initialize data model
        self.comparison_data = ComparisonData()
        
        # Reference to the comparison panel
        self.comparison_panel = comparison_panel
        
        # Connect signals
        self.connect_signals()
        
        # Initialize UI state
        self.update_ui_state()
    
    def connect_signals(self):
        """Connect comparison panel signals to controller methods"""
        self.comparison_panel.load_original_requested.connect(self.load_original_image)
        self.comparison_panel.load_inpainted_requested.connect(self.load_inpainted_image)
        self.comparison_panel.calculate_metrics_requested.connect(self.calculate_metrics)
        self.comparison_panel.save_comparison_requested.connect(self.save_comparison_results)
    
    def update_ui_state(self):
        """Update UI state based on current data"""
        # Update calculate button state
        can_calculate = self.comparison_data.is_ready_for_comparison
        self.comparison_panel.set_calculate_button_enabled(can_calculate)
        
        # Update save button state
        can_save = self.comparison_data.metrics_available
        self.comparison_panel.set_save_button_enabled(can_save)
    
    @Slot()
    def load_original_image(self):
        """Load original image for comparison"""
        # Get initial directory from settings
        initial_dir = self.settings.last_image_directory or os.getcwd()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.comparison_panel,
            "Select Original Image",
            initial_dir,
            AppConstants.IMAGE_FORMATS
        )
        
        if file_path:
            # Try to load the image
            if self.comparison_data.load_original_image(file_path):
                # Update UI
                self.comparison_panel.set_original_image(self.comparison_data.original_image)
                
                # Update settings
                self.settings.last_image_directory = os.path.dirname(file_path)
                self.settings.save()
                
                # Update UI state
                self.update_ui_state()
                
                # Show success message
                filename = os.path.basename(file_path)
                self.show_info_message("Image Loaded", f"Original image loaded: {filename}")
                
            else:
                self.show_error_message(
                    "Error",
                    "Failed to load the selected original image. Please check the file format and try again."
                )
    
    @Slot()
    def load_inpainted_image(self):
        """Load inpainted image for comparison"""
        # Get initial directory from settings
        initial_dir = self.settings.last_image_directory or os.getcwd()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.comparison_panel,
            "Select Inpainted Image",
            initial_dir,
            AppConstants.IMAGE_FORMATS
        )
        
        if file_path:
            # Try to load the image
            if self.comparison_data.load_inpainted_image(file_path):
                # Update UI
                self.comparison_panel.set_inpainted_image(self.comparison_data.inpainted_image)
                
                # Update settings
                self.settings.last_image_directory = os.path.dirname(file_path)
                self.settings.save()
                
                # Update UI state
                self.update_ui_state()
                
                # Show success message
                filename = os.path.basename(file_path)
                self.show_info_message("Image Loaded", f"Inpainted image loaded: {filename}")
                
            else:
                self.show_error_message(
                    "Error",
                    "Failed to load the selected inpainted image. Please check the file format and try again."
                )
    
    @Slot()
    def calculate_metrics(self):
        """Calculate comparison metrics"""
        try:
            # Validate images first
            is_valid, error_message = self.comparison_data.validate_images()
            if not is_valid:
                self.show_warning_message("Validation Error", error_message)
                return
            
            # Calculate metrics
            if self.comparison_data.calculate_metrics():
                # Update UI with calculated metrics
                self.comparison_panel.update_metrics_display(self.comparison_data.metrics)
                
                # Update SSIM difference image if available
                ssim_diff = self.comparison_data.get_ssim_difference_image()
                if ssim_diff is not None:
                    self.comparison_panel.set_ssim_difference_image(ssim_diff)
                
                # Update UI state
                self.update_ui_state()
                
                # Show success message with summary
                metrics_summary = self.comparison_data.get_metrics_summary()
                quality_summary = self.comparison_data.get_quality_summary()
                
                self.show_info_message(
                    "Metrics Calculated",
                    f"Comparison metrics calculated successfully!\n\n"
                    f"{metrics_summary}\n\n"
                    f"Quality: {quality_summary}"
                )
                
            else:
                self.show_error_message(
                    "Calculation Error",
                    "Failed to calculate comparison metrics. Please ensure both images are loaded properly."
                )
                
        except Exception as e:
            self.show_error_message(
                "Error",
                f"An error occurred while calculating metrics:\n{str(e)}"
            )
    
    @Slot()
    def save_comparison_results(self):
        """Save comparison results to a file"""
        try:
            if not self.comparison_data.metrics_available:
                self.show_warning_message("Warning", "No metrics available to save. Please calculate metrics first.")
                return
            
            # Get initial directory from settings
            initial_dir = self.settings.last_save_directory or self.settings.last_image_directory or os.getcwd()
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self.comparison_panel,
                "Save Comparison Results",
                os.path.join(initial_dir, "comparison_results.txt"),
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                # Generate comparison report
                report = self.generate_comparison_report()
                
                # Save report to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                # Update settings
                self.settings.last_save_directory = os.path.dirname(file_path)
                self.settings.save()
                
                # Show success message
                filename = os.path.basename(file_path)
                self.show_info_message("Results Saved", f"Comparison results saved to: {filename}")
                
        except Exception as e:
            self.show_error_message(
                "Save Error",
                f"Failed to save comparison results:\n{str(e)}"
            )
    
    def generate_comparison_report(self) -> str:
        """Generate a detailed comparison report"""
        try:
            from datetime import datetime
            
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("IMAGE COMPARISON REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Image information
            report_lines.append("IMAGE INFORMATION:")
            report_lines.append("-" * 30)
            
            if self.comparison_data.original_path:
                report_lines.append(f"Original Image: {self.comparison_data.original_path}")
            else:
                report_lines.append("Original Image: Loaded from memory")
                
            if self.comparison_data.inpainted_path:
                report_lines.append(f"Inpainted Image: {self.comparison_data.inpainted_path}")
            else:
                report_lines.append("Inpainted Image: Loaded from memory")
            
            if self.comparison_data.original_image is not None:
                orig_shape = self.comparison_data.original_image.shape
                report_lines.append(f"Original Image Size: {orig_shape[1]}×{orig_shape[0]} pixels")
                if len(orig_shape) == 3:
                    report_lines.append(f"Original Image Channels: {orig_shape[2]}")
            
            if self.comparison_data.inpainted_image is not None:
                inp_shape = self.comparison_data.inpainted_image.shape
                report_lines.append(f"Inpainted Image Size: {inp_shape[1]}×{inp_shape[0]} pixels")
                if len(inp_shape) == 3:
                    report_lines.append(f"Inpainted Image Channels: {inp_shape[2]}")
            
            report_lines.append("")
            
            # Metrics
            report_lines.append("QUALITY METRICS:")
            report_lines.append("-" * 30)
            
            detailed_metrics = self.comparison_data.get_detailed_metrics()
            for metric_name, metric_value in detailed_metrics.items():
                report_lines.append(f"{metric_name}: {metric_value}")
            
            report_lines.append("")
            
            # Quality assessment
            report_lines.append("QUALITY ASSESSMENT:")
            report_lines.append("-" * 30)
            quality_summary = self.comparison_data.get_quality_summary()
            report_lines.append(quality_summary)
            report_lines.append("")
            
            # Metric explanations
            report_lines.append("METRIC EXPLANATIONS:")
            report_lines.append("-" * 30)
            report_lines.append("PSNR (Peak Signal-to-Noise Ratio):")
            report_lines.append("  • Measures pixel-level similarity")
            report_lines.append("  • Higher values indicate better quality")
            report_lines.append("  • >40 dB: Excellent, 30-40 dB: Good, 20-30 dB: Fair, <20 dB: Poor")
            report_lines.append("")
            report_lines.append("SSIM (Structural Similarity Index):")
            report_lines.append("  • Measures structural similarity")
            report_lines.append("  • Range: 0 to 1 (1 = identical)")
            report_lines.append("  • >0.95: Excellent, 0.85-0.95: Good, 0.7-0.85: Fair, <0.7: Poor")
            report_lines.append("")
            report_lines.append("MSE (Mean Squared Error):")
            report_lines.append("  • Measures average pixel-level differences")
            report_lines.append("  • Lower values indicate better quality")
            report_lines.append("")
            
            report_lines.append("=" * 60)
            report_lines.append("End of Report")
            report_lines.append("=" * 60)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def set_images_from_processing(self, original_image, inpainted_image):
        """
        Set images from the processing workflow
        
        Args:
            original_image: Original image numpy array
            inpainted_image: Inpainted result image numpy array
        """
        try:
            # Set images in the data model
            self.comparison_data.set_original_image_array(original_image)
            self.comparison_data.set_inpainted_image_array(inpainted_image)
            
            # Update UI
            self.comparison_panel.set_original_image(original_image)
            self.comparison_panel.set_inpainted_image(inpainted_image)
            
            # Update UI state
            self.update_ui_state()
            
            # Automatically calculate metrics
            self.calculate_metrics()
            
        except Exception as e:
            self.show_error_message(
                "Error",
                f"Failed to set images from processing:\n{str(e)}"
            )
    
    def reset(self):
        """Reset the comparison state"""
        try:
            # Reset data model
            self.comparison_data.reset()
            
            # Reset UI
            self.comparison_panel.reset()
            
            # Update UI state
            self.update_ui_state()
            
        except Exception as e:
            self.show_error_message(
                "Error",
                f"Failed to reset comparison:\n{str(e)}"
            )
    
    # Helper methods for dialogs
    def show_info_message(self, title, message):
        """Show info message"""
        QMessageBox.information(self.comparison_panel, title, message)
    
    def show_warning_message(self, title, message):
        """Show warning message"""
        QMessageBox.warning(self.comparison_panel, title, message)
    
    def show_error_message(self, title, message):
        """Show error message"""
        QMessageBox.critical(self.comparison_panel, title, message)
    
    def show_question_dialog(self, title, message):
        """Show question dialog"""
        reply = QMessageBox.question(
            self.comparison_panel, title, message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes
    
    # Public interface
    def get_comparison_data(self):
        """Get the comparison data model"""
        return self.comparison_data
    
    def has_comparison_results(self):
        """Check if comparison results are available"""
        return self.comparison_data.metrics_available 