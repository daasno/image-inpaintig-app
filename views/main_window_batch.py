"""
Main Window with Batch Processing Support
Enhanced version that includes both single image and batch processing modes
"""
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QSplitter, QPushButton, QStatusBar, QMenuBar, QMenu,
    QMessageBox, QProgressBar, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QIcon

from config.settings import AppSettings
from .main_window import EnhancedMainWindow
from .widgets.batch_panel import BatchPanel
from .widgets.comparison_panel import ComparisonPanel
from controllers.comparison_controller import ComparisonController


class BatchEnabledMainWindow(QMainWindow):
    """Main window with both single and batch processing capabilities"""
    
    # Single image processing signals (forwarded from single mode)
    load_image_requested = Signal()
    load_mask_requested = Signal()
    create_mask_requested = Signal(object)
    save_result_requested = Signal()
    run_inpainting_requested = Signal()
    reset_requested = Signal()
    exhaustive_research_requested = Signal()
    
    # Batch processing signals
    batch_processing_requested = Signal()
    batch_stop_requested = Signal()
    
    # Comparison signals
    comparison_mode_requested = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Settings
        self.settings = AppSettings.load()
        
        # Current mode
        self.current_mode = "single"  # "single", "batch", or "comparison"
        
        self.setup_ui()
        self.setup_menus()
        self.connect_signals()
        
        # Apply settings
        self.apply_settings()
    
    def setup_ui(self):
        """Setup the tabbed interface"""
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 8px 16px;
                margin-right: 2px;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #007acc;
                border-color: #007acc;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4a4a4a;
            }
        """)
        layout.addWidget(self.tab_widget)
        
        # Single image processing tab
        self.single_window = EnhancedMainWindow()
        self.tab_widget.addTab(self.single_window, "🖼️ Single Image")
        
        # Batch processing tab
        self.batch_panel = BatchPanel()
        batch_container = QWidget()
        batch_layout = QVBoxLayout(batch_container)
        batch_layout.addWidget(self.batch_panel)
        self.tab_widget.addTab(batch_container, "📁 Batch Processing")
        
        # Comparison tab
        self.comparison_panel = ComparisonPanel()
        self.comparison_controller = ComparisonController(self.comparison_panel)
        self.tab_widget.addTab(self.comparison_panel, "📊 Comparison")
        
        # Connect tab change
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select a processing mode")
        
        # Window properties
        self.setWindowTitle("Image Inpainting Application - Professional Edition with Batch Processing")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def setup_menus(self):
        """Setup the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Single mode actions
        load_image_action = QAction("&Load Image", self)
        load_image_action.setShortcut(QKeySequence.Open)
        load_image_action.triggered.connect(self.load_image_requested.emit)
        file_menu.addAction(load_image_action)
        
        load_mask_action = QAction("Load &Mask", self)
        load_mask_action.setShortcut("Ctrl+M")
        load_mask_action.triggered.connect(self.load_mask_requested.emit)
        file_menu.addAction(load_mask_action)
        
        file_menu.addSeparator()
        
        save_result_action = QAction("&Save Result", self)
        save_result_action.setShortcut(QKeySequence.Save)
        save_result_action.triggered.connect(self.save_result_requested.emit)
        file_menu.addAction(save_result_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Processing menu
        process_menu = menubar.addMenu("&Processing")
        
        run_action = QAction("&Run Inpainting", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.run_inpainting_requested.emit)
        process_menu.addAction(run_action)
        
        research_action = QAction("&Exhaustive Research", self)
        research_action.setShortcut("Ctrl+R")
        research_action.triggered.connect(self.exhaustive_research_requested.emit)
        process_menu.addAction(research_action)
        
        process_menu.addSeparator()
        
        batch_action = QAction("Start &Batch Processing", self)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self.batch_processing_requested.emit)
        process_menu.addAction(batch_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        single_tab_action = QAction("&Single Image Mode", self)
        single_tab_action.setShortcut("Ctrl+1")
        single_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(single_tab_action)
        
        batch_tab_action = QAction("&Batch Processing Mode", self)
        batch_tab_action.setShortcut("Ctrl+2")
        batch_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        view_menu.addAction(batch_tab_action)
        
        comparison_tab_action = QAction("&Comparison Mode", self)
        comparison_tab_action.setShortcut("Ctrl+3")
        comparison_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        view_menu.addAction(comparison_tab_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def connect_signals(self):
        """Connect signals between components"""
        # Forward single window signals
        self.single_window.load_image_requested.connect(self.load_image_requested.emit)
        self.single_window.load_mask_requested.connect(self.load_mask_requested.emit)
        self.single_window.create_mask_requested.connect(self.create_mask_requested.emit)
        self.single_window.save_result_requested.connect(self.save_result_requested.emit)
        self.single_window.run_inpainting_requested.connect(self.run_inpainting_requested.emit)
        self.single_window.reset_requested.connect(self.reset_requested.emit)
        self.single_window.exhaustive_research_requested.connect(self.exhaustive_research_requested.emit)
        
        # Connect batch panel signals
        self.batch_panel.start_batch_requested.connect(self.batch_processing_requested.emit)
        self.batch_panel.stop_batch_requested.connect(self.batch_stop_requested.emit)
        self.batch_panel.folders_changed.connect(self.on_batch_folders_changed)
    
    def on_tab_changed(self, index):
        """Handle tab change"""
        if index == 0:
            self.current_mode = "single"
            self.status_bar.showMessage("Single image processing mode")
        elif index == 1:
            self.current_mode = "batch"
            self.status_bar.showMessage("Batch processing mode")
            self.batch_panel.update_ui_state()
        elif index == 2:
            self.current_mode = "comparison"
            self.status_bar.showMessage("Image comparison mode")
    
    def on_batch_folders_changed(self):
        """Handle batch folders changed"""
        batch_data = self.batch_panel.get_batch_data()
        if batch_data.total_pairs > 0:
            self.status_bar.showMessage(f"Batch mode: {batch_data.total_pairs} pairs ready")
        else:
            self.status_bar.showMessage("Batch mode: Select folders to find image pairs")
    
    def apply_settings(self):
        """Apply saved settings"""
        # Set window size
        self.resize(self.settings.window_width, self.settings.window_height)
        
        # Apply to single window
        if hasattr(self.single_window, 'apply_settings'):
            self.single_window.apply_settings()
    
    def apply_dark_theme(self):
        """Apply consistent dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #3c3c3c;
                color: #ffffff;
                border-bottom: 1px solid #555;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #007acc;
            }
            QMenu {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #007acc;
            }
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
                border-top: 1px solid #555;
            }
        """)
    
    def show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
            "Image Inpainting Application - Professional Edition with Batch Processing\n\n"
            "Features:\n"
            "• Single image inpainting with advanced algorithms\n"
            "• Batch processing for multiple image pairs\n"
            "• GPU acceleration support\n"
            "• Professional dark theme interface\n"
            "• Exhaustive parameter research\n\n"
            "Version 1.3.0 - Batch Processing Edition")
    
    # Delegate methods to current active mode
    def set_input_image(self, image):
        """Set input image for single mode"""
        if self.current_mode == "single":
            self.single_window.set_input_image(image)
    
    def set_mask_image(self, image):
        """Set mask image for single mode"""
        if self.current_mode == "single":
            self.single_window.set_mask_image(image)
    
    def set_result_image(self, image):
        """Set result image for single mode"""
        if self.current_mode == "single":
            self.single_window.set_result_image(image)
    
    def set_status_message(self, message):
        """Set status message"""
        self.status_bar.showMessage(message)
    
    def set_run_button_enabled(self, enabled):
        """Enable/disable run button for single mode"""
        if self.current_mode == "single":
            self.single_window.set_run_button_enabled(enabled)
    
    def set_save_button_enabled(self, enabled):
        """Enable/disable save button for single mode"""
        if self.current_mode == "single":
            self.single_window.set_save_button_enabled(enabled)
    
    def set_processing_state(self, processing):
        """Set processing state"""
        if self.current_mode == "single":
            self.single_window.set_processing_state(processing)
        elif self.current_mode == "batch":
            self.batch_panel.set_processing_state(processing)
    
    def update_progress(self, value):
        """Update progress for single mode"""
        if self.current_mode == "single":
            self.single_window.update_progress(value)
    
    def show_error_message(self, title, message):
        """Show error message"""
        QMessageBox.critical(self, title, message)
    
    def show_warning_message(self, title, message):
        """Show warning message"""
        QMessageBox.warning(self, title, message)
    
    def show_info_message(self, title, message):
        """Show info message"""
        QMessageBox.information(self, title, message)
    
    def get_control_panel(self):
        """Get control panel for single mode"""
        return self.single_window.get_control_panel()
    
    def get_batch_panel(self):
        """Get batch panel"""
        return self.batch_panel
    
    def get_current_mode(self):
        """Get current processing mode"""
        return self.current_mode
    
    def switch_to_single_mode(self):
        """Switch to single processing mode"""
        self.tab_widget.setCurrentIndex(0)
    
    def switch_to_batch_mode(self):
        """Switch to batch processing mode"""
        self.tab_widget.setCurrentIndex(1)
    
    def switch_to_comparison_mode(self):
        """Switch to comparison mode"""
        self.tab_widget.setCurrentIndex(2)
    
    def get_comparison_controller(self):
        """Get comparison controller"""
        return self.comparison_controller
    
    def closeEvent(self, event):
        """Handle close event"""
        # Save settings
        self.settings.window_width = self.width()
        self.settings.window_height = self.height()
        self.settings.save()
        
        # Close single window properly
        if hasattr(self.single_window, 'closeEvent'):
            self.single_window.closeEvent(event)
        
        event.accept()


# Alias for backward compatibility
MainWindow = BatchEnabledMainWindow 