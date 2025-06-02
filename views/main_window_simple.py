"""
Simple Enhanced Main Window - Basic Version for Testing
"""
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QPushButton, QStatusBar, QMenuBar, QMenu,
                             QMessageBox, QProgressBar, QLabel, QFrame, QDockWidget,
                             QToolBar, QFileDialog, QDialog)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QIcon

from config.settings import AppSettings
from .widgets.image_label import ImageLabel
from .widgets.control_panel import ControlPanel
from .widgets.mask_editor import MaskEditorDialog
from .widgets.help_dialog import HelpDialog


class SimpleMainWindow(QMainWindow):
    """Simple enhanced main window with improved styling"""
    
    # Signals for user actions
    load_image_requested = Signal()
    load_mask_requested = Signal()
    create_mask_requested = Signal(object)  # Pass the mask array
    save_result_requested = Signal()
    run_inpainting_requested = Signal()
    reset_requested = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Settings
        self.settings = AppSettings.load()
        
        # Current image for mask editor
        self.current_input_image = None
        
        self.setup_ui()
        self.setup_menus()
        self.apply_modern_styling()
        
        # Apply settings
        self.resize(self.settings.window_width, self.settings.window_height)
    
    def setup_ui(self):
        """Setup the main UI"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Control panel (left side)
        self.control_panel = ControlPanel()
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #444;
            }
        """)
        control_frame.setMinimumWidth(280)
        control_frame.setMaximumWidth(350)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.addWidget(self.control_panel)
        
        # Action buttons
        self.setup_action_buttons(control_layout)
        
        splitter.addWidget(control_frame)
        
        # Image display area
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel)
        image_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #444;
            }
        """)
        image_layout = QVBoxLayout(image_frame)
        
        # Image tabs
        tabs_layout = QHBoxLayout()
        
        self.input_tab_btn = QPushButton("Input Image")
        self.mask_tab_btn = QPushButton("Mask Image")
        self.result_tab_btn = QPushButton("Result Image")
        
        for btn in [self.input_tab_btn, self.mask_tab_btn, self.result_tab_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 8px 16px;
                    border: 1px solid #555;
                    border-radius: 4px 4px 0 0;
                    background-color: #3a3a3a;
                    color: #cccccc;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background-color: #2b2b2b;
                    border-bottom: 1px solid #2b2b2b;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
            """)
        
        self.input_tab_btn.setChecked(True)
        self.input_tab_btn.clicked.connect(lambda: self.show_image_tab('input'))
        self.mask_tab_btn.clicked.connect(lambda: self.show_image_tab('mask'))
        self.result_tab_btn.clicked.connect(lambda: self.show_image_tab('result'))
        
        tabs_layout.addWidget(self.input_tab_btn)
        tabs_layout.addWidget(self.mask_tab_btn)
        tabs_layout.addWidget(self.result_tab_btn)
        tabs_layout.addStretch()
        
        image_layout.addLayout(tabs_layout)
        
        # Image display
        self.input_image_label = ImageLabel()
        self.mask_image_label = ImageLabel()
        self.result_image_label = ImageLabel()
        
        self.input_image_label.setStyleSheet("border: 1px solid #444; background-color: #1e1e1e;")
        self.mask_image_label.setStyleSheet("border: 1px solid #444; background-color: #1e1e1e;")
        self.result_image_label.setStyleSheet("border: 1px solid #444; background-color: #1e1e1e;")
        
        image_layout.addWidget(self.input_image_label)
        image_layout.addWidget(self.mask_image_label)
        image_layout.addWidget(self.result_image_label)
        
        # Hide mask and result initially
        self.mask_image_label.hide()
        self.result_image_label.hide()
        
        splitter.addWidget(image_frame)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        
        # Status bar
        self.setup_status_bar()
        
        # Window properties
        self.setWindowTitle("Image Inpainting Application - Enhanced Edition")
        self.setMinimumSize(1000, 700)
    
    def setup_action_buttons(self, layout):
        """Setup action buttons in control panel"""
        # Load buttons
        load_group = QFrame()
        load_layout = QVBoxLayout(load_group)
        
        self.load_image_btn = QPushButton("📁 Load Image")
        self.load_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        self.load_image_btn.clicked.connect(self.load_image_requested.emit)
        load_layout.addWidget(self.load_image_btn)
        
        self.load_mask_btn = QPushButton("🎭 Load Mask")
        self.load_mask_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
        """)
        self.load_mask_btn.clicked.connect(self.load_mask_requested.emit)
        load_layout.addWidget(self.load_mask_btn)
        
        self.create_mask_btn = QPushButton("✏️ Create Mask")
        self.create_mask_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #5a32a3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.create_mask_btn.clicked.connect(self.open_mask_editor)
        self.create_mask_btn.setEnabled(False)  # Disabled until image is loaded
        load_layout.addWidget(self.create_mask_btn)
        
        layout.addWidget(load_group)
        
        # Process button
        self.run_btn = QPushButton("▶️ Run Inpainting")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px;
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
        self.run_btn.clicked.connect(self.run_inpainting_requested.emit)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)
        
        # Save and reset buttons
        action_group = QFrame()
        action_layout = QHBoxLayout(action_group)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a32a3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.save_btn.clicked.connect(self.save_result_requested.emit)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        action_layout.addWidget(self.reset_btn)
        
        layout.addWidget(action_group)
        layout.addStretch()
    
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        load_image_action = QAction("&Load Image...", self)
        load_image_action.setShortcut(QKeySequence("Ctrl+O"))
        load_image_action.triggered.connect(self.load_image_requested.emit)
        file_menu.addAction(load_image_action)
        
        load_mask_action = QAction("Load &Mask...", self)
        load_mask_action.setShortcut(QKeySequence("Ctrl+M"))
        load_mask_action.triggered.connect(self.load_mask_requested.emit)
        file_menu.addAction(load_mask_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Result...", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_result_requested.emit)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Process menu
        process_menu = menubar.addMenu("&Process")
        
        run_action = QAction("&Run Inpainting", self)
        run_action.setShortcut(QKeySequence("F5"))
        run_action.triggered.connect(self.run_inpainting_requested.emit)
        process_menu.addAction(run_action)
        
        reset_action = QAction("&Reset All", self)
        reset_action.setShortcut(QKeySequence("Ctrl+R"))
        reset_action.triggered.connect(self.reset_requested.emit)
        process_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        help_action = QAction("&Help", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(help_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        self.status_progress.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.status_progress)
    
    def apply_modern_styling(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QFrame {
                background-color: #2b2b2b;
                border-radius: 6px;
                color: #cccccc;
            }
            QStatusBar {
                background-color: #3a3a3a;
                border-top: 1px solid #555;
                color: #cccccc;
            }
            QMenuBar {
                background-color: #3a3a3a;
                border-bottom: 1px solid #555;
                color: #cccccc;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
                color: #cccccc;
            }
            QMenuBar::item:selected {
                background-color: #4a4a4a;
            }
            QMenu {
                background-color: #3a3a3a;
                border: 1px solid #555;
                color: #cccccc;
            }
            QMenu::item:selected {
                background-color: #4a4a4a;
            }
        """)
    
    def show_image_tab(self, tab_name):
        """Show specific image tab"""
        # Update tab buttons
        self.input_tab_btn.setChecked(tab_name == 'input')
        self.mask_tab_btn.setChecked(tab_name == 'mask')
        self.result_tab_btn.setChecked(tab_name == 'result')
        
        # Show/hide image labels
        self.input_image_label.setVisible(tab_name == 'input')
        self.mask_image_label.setVisible(tab_name == 'mask')
        self.result_image_label.setVisible(tab_name == 'result')
    
    # Image methods
    def set_input_image(self, image):
        """Set input image"""
        self.input_image_label.setImage(image)
        # Enable create mask button when image is loaded
        self.create_mask_btn.setEnabled(image is not None)
        # Store image for mask editor
        self.current_input_image = image
    
    def set_mask_image(self, image):
        """Set mask image"""
        self.mask_image_label.setImage(image)
    
    def set_result_image(self, image):
        """Set result image"""
        self.result_image_label.setImage(image)
    
    # UI state methods
    def set_status_message(self, message):
        """Set status message"""
        self.status_label.setText(message)
    
    def update_progress(self, value):
        """Update progress"""
        if not self.status_progress.isVisible():
            self.status_progress.setVisible(True)
        self.status_progress.setValue(value)
        
        if value >= 100:
            QTimer.singleShot(2000, lambda: self.status_progress.setVisible(False))
    
    def set_processing_state(self, processing):
        """Set processing state"""
        self.load_image_btn.setEnabled(not processing)
        self.load_mask_btn.setEnabled(not processing)
        self.create_mask_btn.setEnabled(not processing and self.current_input_image is not None)
        self.run_btn.setEnabled(not processing)
        self.save_btn.setEnabled(not processing)
    
    def set_run_button_enabled(self, enabled):
        """Enable/disable run button"""
        self.run_btn.setEnabled(enabled)
    
    def set_save_button_enabled(self, enabled):
        """Enable/disable save button"""
        self.save_btn.setEnabled(enabled)
    
    def reset_ui(self):
        """Reset UI"""
        self.input_image_label.clear()
        self.mask_image_label.clear()
        self.result_image_label.clear()
        self.current_input_image = None
        self.create_mask_btn.setEnabled(False)
        self.show_image_tab('input')
        self.set_status_message("Ready")
    
    # Dialog methods
    def show_info_message(self, title, message):
        """Show info message"""
        QMessageBox.information(self, title, message)
    
    def show_warning_message(self, title, message):
        """Show warning message"""
        QMessageBox.warning(self, title, message)
    
    def show_error_message(self, title, message):
        """Show error message"""
        QMessageBox.critical(self, title, message)
    
    def show_question_dialog(self, title, message):
        """Show question dialog"""
        reply = QMessageBox.question(
            self, title, message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes
    
    def show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Image Inpainting Application",
            """
            <h2>Image Inpainting Application</h2>
            <p>Version 1.1.0 - Enhanced Edition</p>
            <p>Professional image restoration and object removal tool</p>
            <br>
            <p>For detailed help and documentation, press F1 or use Help menu.</p>
            """
        )
    
    def show_help_dialog(self):
        """Show comprehensive help dialog"""
        help_dialog = HelpDialog(self)
        help_dialog.exec()
    
    def open_mask_editor(self):
        """Open the mask editor dialog"""
        if self.current_input_image is None:
            self.show_warning_message("No Image", "Please load an image first before creating a mask.")
            return
        
        try:
            # Open mask editor dialog
            mask_editor = MaskEditorDialog(self.current_input_image, self)
            mask_editor.mask_created.connect(self.on_mask_created)
            
            # Show dialog
            if mask_editor.exec() == QDialog.Accepted:
                self.set_status_message("Mask created successfully")
            
        except Exception as e:
            self.show_error_message("Mask Editor Error", f"Failed to open mask editor:\n{str(e)}")
    
    def on_mask_created(self, mask_array):
        """Handle mask created from editor"""
        try:
            # Set the created mask
            self.set_mask_image(mask_array)
            
            # Switch to mask tab to show the result
            self.show_image_tab('mask')
            
            # Emit signal to notify controller
            self.create_mask_requested.emit(mask_array)
            
            self.set_status_message("Custom mask created and applied")
            
        except Exception as e:
            self.show_error_message("Mask Creation Error", f"Failed to apply created mask:\n{str(e)}")
    
    # Control panel access
    def get_control_panel(self):
        """Get control panel widget"""
        return self.control_panel
    
    # Window events
    def closeEvent(self, event):
        """Handle window close event"""
        self.settings.window_width = self.width()
        self.settings.window_height = self.height()
        self.settings.save()
        event.accept()


# For testing purposes
MainWindow = SimpleMainWindow 