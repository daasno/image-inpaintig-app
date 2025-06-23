"""
Enhanced application controller with batch processing support
"""
import os
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QObject, Slot, QTimer

from config.settings import AppSettings, AppConstants
from models import ImageData, InpaintWorker, BatchData, BatchInpaintWorker
from views.main_window_batch import BatchEnabledMainWindow


class BatchAppController(QObject):
    """Enhanced application controller supporting both single and batch processing"""
    
    def __init__(self):
        super().__init__()
        
        # Load settings
        self.settings = AppSettings.load()
        
        # Initialize data models
        self.image_data = ImageData()  # For single image processing
        self.batch_data = BatchData()  # For batch processing
        
        # Initialize UI
        self.main_window = BatchEnabledMainWindow()
        
        # Worker threads
        self.inpaint_worker = None
        self.batch_worker = None
        
        # Connect signals
        self.connect_signals()
        
        # Initialize UI state
        self.initialize_ui()
    
    def connect_signals(self):
        """Connect UI signals to controller methods"""
        # Single image processing signals
        self.main_window.load_image_requested.connect(self.load_image)
        self.main_window.load_mask_requested.connect(self.load_mask)
        self.main_window.create_mask_requested.connect(self.on_mask_created)
        self.main_window.save_result_requested.connect(self.save_result)
        self.main_window.run_inpainting_requested.connect(self.run_inpainting)
        self.main_window.reset_requested.connect(self.reset)
        self.main_window.exhaustive_research_requested.connect(self.open_exhaustive_research)
        
        # Batch processing signals
        self.main_window.batch_processing_requested.connect(self.start_batch_processing)
        self.main_window.batch_stop_requested.connect(self.stop_batch_processing)
        
        # Control panel signals (for single image mode)
        control_panel = self.main_window.get_control_panel()
        control_panel.implementation_changed.connect(self.on_implementation_changed)
        control_panel.patch_size_changed.connect(self.on_patch_size_changed)
        control_panel.p_value_changed.connect(self.on_p_value_changed)
        
        # Batch panel signals
        batch_panel = self.main_window.get_batch_panel()
        batch_panel.folders_changed.connect(self.on_batch_folders_changed)
        
        # Get batch data reference from the panel
        self.batch_data = batch_panel.get_batch_data()
    
    def initialize_ui(self):
        """Initialize UI with settings"""
        control_panel = self.main_window.get_control_panel()
        
        # Set control panel values from settings
        control_panel.set_implementation(self.settings.preferred_implementation)
        control_panel.set_patch_size(self.settings.default_patch_size)
        control_panel.set_p_value(self.settings.default_p_value)
        
        # Set window size from settings
        self.main_window.resize(self.settings.window_width, self.settings.window_height)
        
        # Update UI state
        self.update_ui_state()
    
    def update_ui_state(self):
        """Update UI state based on current data and mode"""
        current_mode = self.main_window.get_current_mode()
        
        if current_mode == "single":
            self.update_single_mode_ui()
        elif current_mode == "batch":
            self.update_batch_mode_ui()
    
    def update_single_mode_ui(self):
        """Update UI state for single image mode"""
        # Update run button state
        can_run = self.image_data.is_ready_for_processing
        self.main_window.set_run_button_enabled(can_run)
        
        # Update save button state
        can_save = self.image_data.has_result_image
        self.main_window.set_save_button_enabled(can_save)
        
        # Update status message
        if not self.image_data.has_input_image:
            self.main_window.set_status_message("Load an input image to begin")
        elif not self.image_data.has_mask_image:
            self.main_window.set_status_message("Load a mask image to proceed")
        elif can_run:
            self.main_window.set_status_message("Ready for inpainting")
        else:
            self.main_window.set_status_message("Ready")
    
    def update_batch_mode_ui(self):
        """Update UI state for batch mode"""
        batch_panel = self.main_window.get_batch_panel()
        batch_panel.update_ui_state()
        
        # Update status based on batch data
        if self.batch_data.total_pairs > 0:
            self.main_window.set_status_message(f"Batch mode: {self.batch_data.total_pairs} pairs ready")
        else:
            self.main_window.set_status_message("Batch mode: Select folders to find image pairs")
    
    # Single Image Processing Methods
    @Slot()
    def load_image(self):
        """Load input image"""
        # Get initial directory from settings
        initial_dir = self.settings.last_image_directory or os.getcwd()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Input Image",
            initial_dir,
            AppConstants.IMAGE_FORMATS
        )
        
        if file_path:
            # Try to load the image
            if self.image_data.load_input_image(file_path):
                # Update UI
                self.main_window.set_input_image(self.image_data.input_image)
                
                # Update settings
                self.settings.last_image_directory = os.path.dirname(file_path)
                self.settings.add_recent_image(file_path)
                self.settings.save()
                
                # Update status
                filename = os.path.basename(file_path)
                self.main_window.set_status_message(f"Loaded image: {filename}")
                
                # Update UI state
                self.update_ui_state()
                
            else:
                self.main_window.show_error_message(
                    "Error",
                    "Failed to load the selected image. Please check the file format and try again."
                )
    
    @Slot()
    def load_mask(self):
        """Load mask image"""
        # Get initial directory from settings
        initial_dir = self.settings.last_mask_directory or self.settings.last_image_directory or os.getcwd()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Mask Image",
            initial_dir,
            AppConstants.IMAGE_FORMATS
        )
        
        if file_path:
            # Try to load the mask
            if self.image_data.load_mask_image(file_path):
                # Update UI
                self.main_window.set_mask_image(self.image_data.mask_image)
                
                # Update settings
                self.settings.last_mask_directory = os.path.dirname(file_path)
                self.settings.add_recent_mask(file_path)
                self.settings.save()
                
                # Update status
                filename = os.path.basename(file_path)
                self.main_window.set_status_message(f"Loaded mask: {filename}")
                
                # Update UI state
                self.update_ui_state()
                
            else:
                self.main_window.show_error_message(
                    "Error",
                    "Failed to load the selected mask. Please check the file format and try again."
                )
    
    @Slot(object)  # np.ndarray
    def on_mask_created(self, mask_array):
        """Handle mask created from the mask editor"""
        try:
            # Set the mask in the data model
            self.image_data.set_mask_image_array(mask_array)
            
            # Update status
            self.main_window.set_status_message("Custom mask created - ready for inpainting")
            
            # Update UI state
            self.update_ui_state()
            
        except Exception as e:
            self.main_window.show_error_message(
                "Error",
                f"Failed to apply the created mask: {str(e)}"
            )
    
    @Slot()
    def save_result(self):
        """Save result image"""
        if not self.image_data.has_result_image:
            self.main_window.show_warning_message("Warning", "No result image to save")
            return
        
        # Get initial directory from settings
        initial_dir = self.settings.last_save_directory or self.settings.last_image_directory or os.getcwd()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save Result Image",
            initial_dir,
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            # Try to save the result
            if self.image_data.save_result_image(file_path):
                # Update settings
                self.settings.last_save_directory = os.path.dirname(file_path)
                self.settings.save()
                
                # Update status
                filename = os.path.basename(file_path)
                self.main_window.set_status_message(f"Saved result: {filename}")
                
            else:
                self.main_window.show_error_message(
                    "Error",
                    "Failed to save the result image. Please check the file path and try again."
                )
    
    @Slot()
    def run_inpainting(self):
        """Run inpainting process for single image"""
        if not self.image_data.is_ready_for_processing:
            self.main_window.show_warning_message("Warning", "Please load both image and mask first")
            return
        
        # Get current settings
        control_panel = self.main_window.get_control_panel()
        patch_size = control_panel.get_current_patch_size()
        p_value = control_panel.get_current_p_value()
        implementation = control_panel.get_current_implementation()
        
        # Create and start worker
        self.inpaint_worker = InpaintWorker(
            implementation=implementation,
            image=self.image_data.input_image,
            mask=self.image_data.mask_image,
            patch_size=patch_size,
            p_value=p_value
        )
        
        # Connect worker signals
        self.inpaint_worker.progress_update.connect(self.on_progress_update)
        self.inpaint_worker.process_complete.connect(self.on_process_complete)
        self.inpaint_worker.error_occurred.connect(self.on_process_error)
        self.inpaint_worker.status_update.connect(self.on_status_update)
        
        # Update UI state
        self.main_window.set_processing_state(True)
        self.main_window.set_status_message("Processing...")
        
        # Start processing
        self.inpaint_worker.start()
    
    @Slot()
    def reset(self):
        """Reset application state"""
        # Stop any running workers
        if self.inpaint_worker and self.inpaint_worker.isRunning():
            self.inpaint_worker.quit()
            self.inpaint_worker.wait()
        
        # Reset data
        self.image_data = ImageData()
        
        # Reset UI
        self.main_window.set_processing_state(False)
        self.main_window.set_status_message("Reset complete - ready for new images")
        
        # Update UI state
        self.update_ui_state()
    
    @Slot()
    def open_exhaustive_research(self):
        """Open exhaustive research dialog"""
        # Import here to avoid circular imports
        from views.dialogs.exhaustive_research_dialog import ExhaustiveResearchDialog
        
        if not self.image_data.is_ready_for_processing:
            self.main_window.show_warning_message("Warning", "Please load both image and mask first")
            return
        
        dialog = ExhaustiveResearchDialog(self.main_window, self.image_data)
        dialog.exec()
    
    # Batch Processing Methods
    @Slot()
    def start_batch_processing(self):
        """Start batch processing"""
        if not self.batch_data.is_ready_for_processing:
            self.main_window.show_warning_message(
                "Warning", 
                "Please select folders with matching image and mask pairs first"
            )
            return
        
        # Get current settings for batch processing from batch panel
        batch_panel = self.main_window.get_batch_panel()
        settings = batch_panel.get_parameters()
        
        # Log the settings being used
        batch_panel.add_result_log(f"Using parameters: {settings['implementation']} implementation, "
                                 f"patch size {settings['patch_size']}, p-value {settings['p_value']}")
        
        # Create and start batch worker
        self.batch_worker = BatchInpaintWorker(self.batch_data, settings)
        
        # Connect batch worker signals
        self.batch_worker.progress_updated.connect(self.on_batch_progress_update)
        self.batch_worker.pair_started.connect(self.on_batch_pair_started)
        self.batch_worker.pair_completed.connect(self.on_batch_pair_completed)
        self.batch_worker.batch_completed.connect(self.on_batch_completed)
        self.batch_worker.error_occurred.connect(self.on_batch_error)
        
        # Update UI state
        self.main_window.set_processing_state(True)
        batch_panel.clear_results_log()
        batch_panel.add_result_log("Starting batch processing...")
        
        # Start processing
        self.batch_worker.start()
    
    @Slot()
    def stop_batch_processing(self):
        """Stop batch processing"""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop_processing()
            batch_panel = self.main_window.get_batch_panel()
            batch_panel.add_result_log("Stopping batch processing...", is_error=True)
    
    @Slot()
    def on_batch_folders_changed(self):
        """Handle batch folders changed"""
        self.update_batch_mode_ui()
    
    # Signal handlers for single image processing
    @Slot(int)
    def on_progress_update(self, value):
        """Handle progress update"""
        self.main_window.update_progress(value)
    
    @Slot(object)  # np.ndarray
    def on_process_complete(self, result_image):
        """Handle process completion"""
        try:
            # Set result in data model
            self.image_data.result_image = result_image
            
            # Update UI
            self.main_window.set_result_image(result_image)
            self.main_window.set_processing_state(False)
            self.main_window.set_status_message("Inpainting completed successfully")
            
            # Update UI state
            self.update_ui_state()
            
        except Exception as e:
            self.main_window.show_error_message("Error", f"Failed to display result: {str(e)}")
        
        finally:
            # Clean up worker
            if self.inpaint_worker:
                self.inpaint_worker.deleteLater()
                self.inpaint_worker = None
    
    @Slot(str)
    def on_process_error(self, error_message):
        """Handle process error"""
        self.main_window.set_processing_state(False)
        self.main_window.show_error_message("Processing Error", error_message)
        
        # Clean up worker
        if self.inpaint_worker:
            self.inpaint_worker.deleteLater()
            self.inpaint_worker = None
    
    @Slot(str)
    def on_status_update(self, message):
        """Handle status update"""
        self.main_window.set_status_message(message)
    
    # Signal handlers for batch processing
    @Slot(int, int)
    def on_batch_progress_update(self, current, total):
        """Handle batch progress update"""
        batch_panel = self.main_window.get_batch_panel()
        batch_panel.update_progress(current, total)
    
    @Slot(str, str)
    def on_batch_pair_started(self, image_filename, mask_filename):
        """Handle batch pair started"""
        batch_panel = self.main_window.get_batch_panel()
        batch_panel.add_result_log(f"Processing: {image_filename} with {mask_filename}")
    
    @Slot(str, bool, str)
    def on_batch_pair_completed(self, result_filename, success, message):
        """Handle batch pair completed"""
        batch_panel = self.main_window.get_batch_panel()
        if success:
            batch_panel.add_result_log(f"✅ Completed: {result_filename} - {message}")
        else:
            batch_panel.add_result_log(f"❌ Failed: {result_filename} - {message}", is_error=True)
    
    @Slot(dict)
    def on_batch_completed(self, summary):
        """Handle batch processing completion"""
        batch_panel = self.main_window.get_batch_panel()
        self.main_window.set_processing_state(False)
        
        # Show summary
        total = summary['total_pairs']
        successful = summary['successful']
        failed = summary['failed']
        
        batch_panel.add_result_log(f"\n📊 Batch Processing Complete!")
        batch_panel.add_result_log(f"Total pairs: {total}")
        batch_panel.add_result_log(f"Successful: {successful}")
        batch_panel.add_result_log(f"Failed: {failed}")
        
        if failed > 0:
            batch_panel.add_result_log(f"Success rate: {(successful/total)*100:.1f}%")
        
        self.main_window.set_status_message(f"Batch processing complete: {successful}/{total} successful")
        
        # Clean up worker
        if self.batch_worker:
            self.batch_worker.deleteLater()
            self.batch_worker = None
    
    @Slot(str)
    def on_batch_error(self, error_message):
        """Handle batch processing error"""
        batch_panel = self.main_window.get_batch_panel()
        batch_panel.add_result_log(f"❌ Batch Error: {error_message}", is_error=True)
        self.main_window.set_processing_state(False)
        
        # Clean up worker
        if self.batch_worker:
            self.batch_worker.deleteLater()
            self.batch_worker = None
    
    # Settings handlers
    @Slot(str)
    def on_implementation_changed(self, implementation):
        """Handle implementation change"""
        self.settings.preferred_implementation = implementation
        self.settings.save()
    
    @Slot(int)
    def on_patch_size_changed(self, patch_size):
        """Handle patch size change"""
        self.settings.default_patch_size = patch_size
        self.settings.save()
    
    @Slot(float)
    def on_p_value_changed(self, p_value):
        """Handle p-value change"""
        self.settings.default_p_value = p_value
        self.settings.save()
    
    def show(self):
        """Show the main window"""
        self.main_window.show()
    
    def close(self):
        """Close the application"""
        self.main_window.close() 