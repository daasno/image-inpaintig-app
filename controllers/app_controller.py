"""
Main application controller coordinating between UI and business logic
"""
import os
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QObject, Slot

from config.settings import AppSettings, AppConstants
from models import ImageData, InpaintWorker
from views import MainWindow


class AppController(QObject):
    """Main application controller"""
    
    def __init__(self):
        super().__init__()
        
        # Load settings
        self.settings = AppSettings.load()
        
        # Initialize data model
        self.image_data = ImageData()
        
        # Initialize UI
        self.main_window = MainWindow()
        
        # Worker thread
        self.inpaint_worker = None
        
        # Connect signals
        self.connect_signals()
        
        # Initialize UI state
        self.initialize_ui()
    
    def connect_signals(self):
        """Connect UI signals to controller methods"""
        # Main window signals
        self.main_window.load_image_requested.connect(self.load_image)
        self.main_window.load_mask_requested.connect(self.load_mask)
        self.main_window.create_mask_requested.connect(self.on_mask_created)
        self.main_window.save_result_requested.connect(self.save_result)
        self.main_window.run_inpainting_requested.connect(self.run_inpainting)
        self.main_window.reset_requested.connect(self.reset)
        self.main_window.exhaustive_research_requested.connect(self.open_exhaustive_research)
        
        # Control panel signals
        control_panel = self.main_window.get_control_panel()
        control_panel.implementation_changed.connect(self.on_implementation_changed)
        control_panel.patch_size_changed.connect(self.on_patch_size_changed)
        control_panel.p_value_changed.connect(self.on_p_value_changed)
    
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
        """Update UI state based on current data"""
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
                self.main_window.set_status_message(f"Result saved to: {filename}")
                
            else:
                self.main_window.show_error_message(
                    "Error",
                    "Failed to save the result image. Please check the file path and permissions."
                )
    
    @Slot()
    def run_inpainting(self):
        """Start the inpainting process"""
        # Validate images
        is_valid, error_message = self.image_data.validate_images()
        if not is_valid:
            self.main_window.show_warning_message("Warning", error_message)
            return
        
        # Get parameters from control panel
        control_panel = self.main_window.get_control_panel()
        is_valid, error_message = control_panel.validate_parameters()
        if not is_valid:
            self.main_window.show_warning_message("Parameter Error", error_message)
            return
        
        parameters = control_panel.get_parameters()
        
        # Check implementation availability at runtime
        implementation = parameters['implementation']
        if implementation == "GPU" and not InpaintWorker.check_gpu_availability():
            if self.main_window.show_question_dialog(
                "GPU Not Available",
                "GPU implementation is not available. Switch to CPU implementation?"
            ):
                control_panel.set_implementation("CPU")
                implementation = "CPU"
            else:
                return
        
        # Set UI to processing state
        self.main_window.set_processing_state(True)
        self.main_window.update_progress(0)
        
        # Create and start worker thread
        try:
            self.inpaint_worker = InpaintWorker(
                implementation=implementation,
                image=self.image_data.input_image,
                mask=self.image_data.mask_image,
                patch_size=parameters['patch_size'],
                p_value=parameters['p_value']
            )
            
            # Connect worker signals
            self.inpaint_worker.progress_update.connect(self.on_progress_update)
            self.inpaint_worker.process_complete.connect(self.on_process_complete)
            self.inpaint_worker.error_occurred.connect(self.on_process_error)
            self.inpaint_worker.status_update.connect(self.on_status_update)
            
            # Start the worker
            self.inpaint_worker.start()
            
        except Exception as e:
            self.main_window.set_processing_state(False)
            self.main_window.show_error_message("Error", f"Failed to start inpainting: {str(e)}")
    
    @Slot()
    def reset(self):
        """Reset the application state"""
        # Stop any running worker
        if self.inpaint_worker and self.inpaint_worker.isRunning():
            if self.main_window.show_question_dialog(
                "Confirm Reset",
                "Inpainting is currently running. Do you want to stop it and reset?"
            ):
                self.inpaint_worker.terminate()
                self.inpaint_worker.wait()
            else:
                return
        
        # Clear data
        self.image_data.reset()
        
        # Reset UI
        self.main_window.reset_ui()
        self.main_window.set_processing_state(False)
        
        # Update state
        self.update_ui_state()
    
    @Slot()
    def open_exhaustive_research(self):
        """Open the exhaustive research dialog"""
        # Validate that we have images loaded
        if not self.image_data.has_input_image:
            self.main_window.show_warning_message(
                "No Input Image",
                "Please load an input image before starting exhaustive research."
            )
            return
        
        if not self.image_data.has_mask_image:
            self.main_window.show_warning_message(
                "No Mask Image",
                "Please load a mask image before starting exhaustive research."
            )
            return
        
        # Import the dialog (we'll create this next)
        try:
            from views.dialogs.exhaustive_research_dialog import ExhaustiveResearchDialog
            
            # Create and show the dialog
            dialog = ExhaustiveResearchDialog(
                self.main_window,
                self.image_data.input_image,
                self.image_data.mask_image
            )
            dialog.exec()
            
        except ImportError:
            # If the dialog doesn't exist yet, show a message
            self.main_window.show_info_message(
                "Coming Soon",
                "The exhaustive research feature is being implemented. Stay tuned!"
            )
    
    # Worker thread signal handlers
    @Slot(int)
    def on_progress_update(self, value):
        """Handle progress updates from worker"""
        self.main_window.update_progress(value)
    
    @Slot(object)  # np.ndarray
    def on_process_complete(self, result_image):
        """Handle process completion"""
        try:
            # Set result image
            self.image_data.set_result_image(result_image)
            self.main_window.set_result_image(result_image)
            
            # Update UI state
            self.main_window.set_processing_state(False)
            self.main_window.set_status_message("Inpainting completed successfully")
            self.update_ui_state()
            
            # Show success message
            self.main_window.show_info_message("Success", "Inpainting completed successfully!")
            
        except Exception as e:
            self.on_process_error(f"Error processing result: {str(e)}")
    
    @Slot(str)
    def on_process_error(self, error_message):
        """Handle process errors"""
        # Update UI state
        self.main_window.set_processing_state(False)
        self.main_window.set_status_message("Inpainting failed")
        
        # Show error message
        if "CUDA" in error_message:
            self.main_window.show_error_message(
                "CUDA Error",
                f"{error_message}\n\nTry using the CPU implementation instead."
            )
        else:
            self.main_window.show_error_message("Error", f"Inpainting failed: {error_message}")
    
    @Slot(str)
    def on_status_update(self, message):
        """Handle status updates from worker"""
        self.main_window.set_status_message(message)
    
    # Control panel signal handlers
    @Slot(str)
    def on_implementation_changed(self, implementation):
        """Handle implementation changes"""
        self.settings.preferred_implementation = implementation
        self.settings.save()
    
    @Slot(int)
    def on_patch_size_changed(self, patch_size):
        """Handle patch size changes"""
        self.settings.default_patch_size = patch_size
        self.settings.save()
    
    @Slot(float)
    def on_p_value_changed(self, p_value):
        """Handle p-value changes"""
        self.settings.default_p_value = p_value
        self.settings.save()
    
    # Public interface
    def show(self):
        """Show the main window"""
        self.main_window.show()
    
    def close(self):
        """Close the application"""
        # Save current window size
        self.settings.window_width = self.main_window.width()
        self.settings.window_height = self.main_window.height()
        self.settings.save()
        
        # Stop any running worker
        if self.inpaint_worker and self.inpaint_worker.isRunning():
            self.inpaint_worker.terminate()
            self.inpaint_worker.wait()
        
        self.main_window.close() 