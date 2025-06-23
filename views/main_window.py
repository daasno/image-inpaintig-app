"""
Enhanced Main Window with Improved UX/UI Features
"""
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QPushButton, QStatusBar, QMenuBar, QMenu,
                             QMessageBox, QProgressBar, QLabel, QFrame, QDockWidget,
                             QToolBar, QFileDialog, QDialog)
from PySide6.QtCore import Qt, Signal, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QIcon

from config.settings import AppSettings
from .widgets.image_label import ImageLabel
from .widgets.control_panel import ControlPanel
from .widgets.mask_editor import MaskEditorDialog
from .widgets.batch_panel import BatchPanel

# Enhanced widgets - import only what exists
try:
    from .widgets.enhanced_image_label import ImageViewerWidget, ComparisonViewWidget
    ENHANCED_VIEWER_AVAILABLE = True
except ImportError:
    ENHANCED_VIEWER_AVAILABLE = False

try:
    from .widgets.enhanced_progress_dialog import EnhancedProgressDialog
    ENHANCED_PROGRESS_AVAILABLE = True
except ImportError:
    ENHANCED_PROGRESS_AVAILABLE = False

try:
    from .widgets.recent_files_menu import RecentFilesMenu, RecentFilesPanel, ImageMetadataWidget
    RECENT_FILES_AVAILABLE = True
except ImportError:
    RECENT_FILES_AVAILABLE = False

try:
    from .welcome_dialog import WelcomeDialog
    WELCOME_DIALOG_AVAILABLE = True
except ImportError:
    WELCOME_DIALOG_AVAILABLE = False


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with improved UX/UI features"""
    
    # Signals for user actions
    load_image_requested = Signal()
    load_mask_requested = Signal()
    create_mask_requested = Signal(object)  # Pass the mask array
    save_result_requested = Signal()
    run_inpainting_requested = Signal()
    reset_requested = Signal()
    exhaustive_research_requested = Signal()  # New signal for exhaustive research
    
    # Batch processing signals
    batch_processing_requested = Signal()
    batch_stop_requested = Signal()
    
    # UX enhancement signals
    show_welcome_requested = Signal()
    show_recent_files_requested = Signal()
    tutorial_requested = Signal()
    
    def __init__(self):
        super().__init__()
        
        # State
        self.current_input_pixmap = None
        self.current_mask_pixmap = None
        self.current_result_pixmap = None
        self.current_input_image = None  # For mask editor
        self.current_mask_array = None  # Original binary mask for inpainting
        self.progress_dialog = None
        self.welcome_dialog = None
        
        # Settings
        self.settings = AppSettings.load()
        
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_keyboard_shortcuts()
        self.setup_dock_widgets()
        
        # Apply settings
        self.apply_settings()
        
        # Welcome dialog is now disabled
        # if self.settings.show_welcome_dialog:
        #     QTimer.singleShot(500, self.show_welcome_dialog)
    
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
        control_frame.setMinimumWidth(280)
        control_frame.setMaximumWidth(350)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.addWidget(self.control_panel)
        
        # Action buttons
        self.setup_action_buttons(control_layout)
        
        splitter.addWidget(control_frame)
        
        # Image display area (center/right)
        self.setup_image_display_area(splitter)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])  # Control panel smaller, image area larger
        
        # Status bar
        self.setup_status_bar()
        
        # Window properties
        self.setWindowTitle("Image Inpainting Application - Professional Edition")
        self.setMinimumSize(1000, 700)
        
        # Apply modern styling
        self.apply_modern_styling()
    
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
        
        # Exhaustive Research button
        self.research_btn = QPushButton("🔬 Exhaustive Research")
        self.research_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.research_btn.clicked.connect(self.exhaustive_research_requested.emit)
        self.research_btn.setEnabled(False)
        self.research_btn.setToolTip("Run multiple parameter combinations to find optimal settings")
        layout.addWidget(self.research_btn)
        
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
    
    def setup_image_display_area(self, splitter):
        """Setup the image display area with enhanced viewers"""
        # Create tabbed interface for different view modes
        self.image_stack = QWidget()
        image_layout = QVBoxLayout(self.image_stack)
        
        # Image tabs selector
        tabs_controls = QFrame()
        tabs_layout = QHBoxLayout(tabs_controls)
        
        tabs_label = QLabel("View:")
        tabs_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        tabs_layout.addWidget(tabs_label)
        
        # Image tab buttons
        self.input_tab_btn = QPushButton("📷 Input")
        self.mask_tab_btn = QPushButton("🎭 Mask")
        self.result_tab_btn = QPushButton("✨ Result")
        
        # Tab styling
        tab_style = """
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #3a3a3a;
                color: #cccccc;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: white;
                border-color: #007acc;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:checked:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
                border-color: #333;
            }
        """
        
        for btn in [self.input_tab_btn, self.mask_tab_btn, self.result_tab_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet(tab_style)
        
        self.input_tab_btn.setChecked(True)
        self.input_tab_btn.clicked.connect(lambda: self.show_image_tab('input'))
        self.mask_tab_btn.clicked.connect(lambda: self.show_image_tab('mask'))
        self.result_tab_btn.clicked.connect(lambda: self.show_image_tab('result'))
        
        # Initially disable mask and result tabs
        self.mask_tab_btn.setEnabled(False)
        self.result_tab_btn.setEnabled(False)
        
        tabs_layout.addWidget(self.input_tab_btn)
        tabs_layout.addWidget(self.mask_tab_btn)
        tabs_layout.addWidget(self.result_tab_btn)
        tabs_layout.addStretch()
        
        # View mode selector
        view_controls = QFrame()
        view_layout = QHBoxLayout(view_controls)
        
        view_label = QLabel("View Mode:")
        view_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        view_layout.addWidget(view_label)
        
        # View mode buttons
        self.single_view_btn = QPushButton("Single View")
        self.comparison_view_btn = QPushButton("Side by Side")
        
        for btn in [self.single_view_btn, self.comparison_view_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    border: 1px solid #555;
                    border-radius: 4px;
                    background-color: #3a3a3a;
                    color: #cccccc;
                }
                QPushButton:checked {
                    background-color: #007acc;
                    color: white;
                    border-color: #007acc;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:checked:hover {
                    background-color: #005a9e;
                }
            """)
        
        self.single_view_btn.setChecked(True)
        self.single_view_btn.clicked.connect(self.show_single_view)
        self.comparison_view_btn.clicked.connect(self.show_comparison_view)
        
        view_layout.addWidget(self.single_view_btn)
        view_layout.addWidget(self.comparison_view_btn)
        view_layout.addStretch()
        
        image_layout.addWidget(tabs_controls)
        image_layout.addWidget(view_controls)
        
        # Single view (default)
        if ENHANCED_VIEWER_AVAILABLE:
            self.single_viewer = ImageViewerWidget("Image Display")
        else:
            self.single_viewer = ImageLabel()
        image_layout.addWidget(self.single_viewer)
        
        # Comparison view (hidden initially)
        if ENHANCED_VIEWER_AVAILABLE:
            self.comparison_viewer = ComparisonViewWidget()
            self.comparison_viewer.hide()
            image_layout.addWidget(self.comparison_viewer)
        else:
            self.comparison_viewer = None
        
        splitter.addWidget(self.image_stack)
    
    def setup_menus(self):
        """Setup enhanced menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Load actions
        load_image_action = QAction("&Load Image...", self)
        load_image_action.setShortcut(QKeySequence("Ctrl+O"))
        load_image_action.triggered.connect(self.load_image_requested.emit)
        file_menu.addAction(load_image_action)
        
        load_mask_action = QAction("Load &Mask...", self)
        load_mask_action.setShortcut(QKeySequence("Ctrl+M"))
        load_mask_action.triggered.connect(self.load_mask_requested.emit)
        file_menu.addAction(load_mask_action)
        
        file_menu.addSeparator()
        
        # Recent files menus (only if available)
        if RECENT_FILES_AVAILABLE:
            self.recent_images_menu = RecentFilesMenu("Recent Images", self)
            self.recent_images_menu.file_selected.connect(self.on_recent_image_selected)
            file_menu.addMenu(self.recent_images_menu)
            
            self.recent_masks_menu = RecentFilesMenu("Recent Masks", self)
            self.recent_masks_menu.file_selected.connect(self.on_recent_mask_selected)
            file_menu.addMenu(self.recent_masks_menu)
            
            file_menu.addSeparator()
        else:
            self.recent_images_menu = None
            self.recent_masks_menu = None
        
        # Save actions
        save_action = QAction("&Save Result...", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_result_requested.emit)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit
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
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        single_view_action = QAction("&Single View", self)
        single_view_action.setShortcut(QKeySequence("Ctrl+1"))
        single_view_action.triggered.connect(self.show_single_view)
        view_menu.addAction(single_view_action)
        
        comparison_view_action = QAction("&Side by Side", self)
        comparison_view_action.setShortcut(QKeySequence("Ctrl+2"))
        comparison_view_action.triggered.connect(self.show_comparison_view)
        view_menu.addAction(comparison_view_action)
        
        view_menu.addSeparator()
        
        # Zoom actions
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence("Ctrl+="))
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        zoom_fit_action = QAction("Zoom to &Fit", self)
        zoom_fit_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit_action.triggered.connect(self.zoom_to_fit)
        view_menu.addAction(zoom_fit_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        welcome_action = QAction("&Welcome Guide", self)
        welcome_action.triggered.connect(self.show_welcome_dialog)
        help_menu.addAction(welcome_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup toolbar with common actions"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        
        # Load actions
        load_image_action = QAction("📁", self)
        load_image_action.setToolTip("Load Image (Ctrl+O)")
        load_image_action.triggered.connect(self.load_image_requested.emit)
        toolbar.addAction(load_image_action)
        
        load_mask_action = QAction("🎭", self)
        load_mask_action.setToolTip("Load Mask (Ctrl+M)")
        load_mask_action.triggered.connect(self.load_mask_requested.emit)
        toolbar.addAction(load_mask_action)
        
        toolbar.addSeparator()
        
        # Process action
        run_action = QAction("▶️", self)
        run_action.setToolTip("Run Inpainting (F5)")
        run_action.triggered.connect(self.run_inpainting_requested.emit)
        toolbar.addAction(run_action)
        
        toolbar.addSeparator()
        
        # View actions
        zoom_in_action = QAction("🔍+", self)
        zoom_in_action.setToolTip("Zoom In (Ctrl+=)")
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("🔍-", self)
        zoom_out_action.setToolTip("Zoom Out (Ctrl+-)")
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        zoom_fit_action = QAction("📏", self)
        zoom_fit_action.setToolTip("Zoom to Fit (Ctrl+0)")
        zoom_fit_action.triggered.connect(self.zoom_to_fit)
        toolbar.addAction(zoom_fit_action)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        # These are already handled in menu actions
        pass
    
    def setup_dock_widgets(self):
        """Setup dock widgets for advanced features"""
        if not RECENT_FILES_AVAILABLE:
            return
            
        # Recent files dock
        recent_dock = QDockWidget("Recent Files", self)
        recent_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.recent_files_panel = RecentFilesPanel("Recent Images")
        self.recent_files_panel.file_selected.connect(self.on_recent_image_selected)
        recent_dock.setWidget(self.recent_files_panel)
        
        self.addDockWidget(Qt.RightDockWidgetArea, recent_dock)
        recent_dock.hide()  # Hidden by default
        
        # Image metadata dock
        metadata_dock = QDockWidget("Image Information", self)
        metadata_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.metadata_widget = ImageMetadataWidget()
        metadata_dock.setWidget(self.metadata_widget)
        
        self.addDockWidget(Qt.RightDockWidgetArea, metadata_dock)
        metadata_dock.hide()  # Hidden by default
        
        # Add dock visibility actions to View menu
        view_menu = self.menuBar().findChild(QMenu, "&View")
        if view_menu:
            view_menu.addSeparator()
            view_menu.addAction(recent_dock.toggleViewAction())
            view_menu.addAction(metadata_dock.toggleViewAction())
    
    def setup_status_bar(self):
        """Setup enhanced status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        self.status_progress.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.status_progress)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        self.status_bar.addPermanentWidget(self.image_info_label)
    
    def apply_modern_styling(self):
        """Apply dark theme styling to match the original"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QFrame {
                background-color: #2b2b2b;
                border-radius: 6px;
                color: #cccccc;
            }
            QDockWidget {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
                color: #cccccc;
            }
            QDockWidget::title {
                background-color: #3a3a3a;
                padding: 8px;
                border-bottom: 1px solid #555;
                font-weight: bold;
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
            QToolBar {
                background-color: #3a3a3a;
                border-bottom: 1px solid #555;
                padding: 4px;
                color: #cccccc;
            }
            QToolBar QToolButton {
                padding: 6px;
                border-radius: 4px;
                margin: 2px;
                color: #cccccc;
            }
            QToolBar QToolButton:hover {
                background-color: #4a4a4a;
            }
            QLabel {
                color: #cccccc;
            }
        """)
        
        # Also set dark background for image display area
        if hasattr(self, 'single_viewer'):
            if not ENHANCED_VIEWER_AVAILABLE:
                # For basic ImageLabel
                self.single_viewer.setStyleSheet("border: 1px solid #444; background-color: #1e1e1e;")
            # Enhanced viewers already have proper styling
    
    def apply_settings(self):
        """Apply settings to the window"""
        # Window size and position
        if self.settings.window_x >= 0 and self.settings.window_y >= 0:
            self.move(self.settings.window_x, self.settings.window_y)
        
        self.resize(self.settings.window_width, self.settings.window_height)
        
        # Update recent files
        self.update_recent_files()
    
    def update_recent_files(self):
        """Update recent files displays"""
        if not RECENT_FILES_AVAILABLE:
            return
            
        # Update menus
        if self.recent_images_menu:
            self.recent_images_menu.update_recent_files(self.settings.recent_images)
        if self.recent_masks_menu:
            self.recent_masks_menu.update_recent_files(self.settings.recent_masks)
        
        # Update dock panel
        if hasattr(self, 'recent_files_panel'):
            self.recent_files_panel.update_recent_files(self.settings.recent_images)
    
    # Image tab and view mode methods
    def show_image_tab(self, tab_name):
        """Show specific image tab"""
        # Update tab button states
        self.input_tab_btn.setChecked(tab_name == 'input')
        self.mask_tab_btn.setChecked(tab_name == 'mask')
        self.result_tab_btn.setChecked(tab_name == 'result')
        
        # Display appropriate image
        if tab_name == 'input' and self.current_input_pixmap:
            self._display_image(self.current_input_pixmap, "Input Image")
        elif tab_name == 'mask' and self.current_mask_pixmap:
            self._display_image(self.current_mask_pixmap, "Mask Image")
        elif tab_name == 'result' and self.current_result_pixmap:
            self._display_image(self.current_result_pixmap, "Result Image")
        else:
            # Clear display if no image available
            if ENHANCED_VIEWER_AVAILABLE and hasattr(self.single_viewer, 'set_image'):
                self.single_viewer.set_image(None)
            else:
                self.single_viewer.clear()
    
    def _display_image(self, pixmap, title):
        """Helper method to display an image pixmap"""
        if ENHANCED_VIEWER_AVAILABLE and hasattr(self.single_viewer, 'set_image'):
            self.single_viewer.set_image(pixmap)
        else:
            scaled_pixmap = pixmap.scaled(
                self.single_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.single_viewer.setPixmap(scaled_pixmap)

    def show_single_view(self):
        """Show single image view"""
        self.single_view_btn.setChecked(True)
        self.comparison_view_btn.setChecked(False)
        self.single_viewer.show()
        self.comparison_viewer.hide()
    
    def show_comparison_view(self):
        """Show side-by-side comparison view"""
        if not ENHANCED_VIEWER_AVAILABLE or not self.comparison_viewer:
            return
            
        self.single_view_btn.setChecked(False)
        self.comparison_view_btn.setChecked(True)
        self.single_viewer.hide()
        self.comparison_viewer.show()
        
        # Update comparison viewer with current images
        if self.current_input_pixmap:
            self.comparison_viewer.set_original_image(self.current_input_pixmap)
        if self.current_result_pixmap:
            self.comparison_viewer.set_result_image(self.current_result_pixmap)
    
    # Zoom methods
    def zoom_in(self):
        """Zoom in current view"""
        if ENHANCED_VIEWER_AVAILABLE and hasattr(self.single_viewer, 'zoom_in') and self.single_viewer.isVisible():
            self.single_viewer.zoom_in()
    
    def zoom_out(self):
        """Zoom out current view"""
        if ENHANCED_VIEWER_AVAILABLE and hasattr(self.single_viewer, 'zoom_out') and self.single_viewer.isVisible():
            self.single_viewer.zoom_out()
    
    def zoom_to_fit(self):
        """Zoom to fit current view"""
        if ENHANCED_VIEWER_AVAILABLE and hasattr(self.single_viewer, 'zoom_to_fit') and self.single_viewer.isVisible():
            self.single_viewer.zoom_to_fit()
    
    # Image setting methods (maintaining compatibility)
    def set_input_image(self, image):
        """Set input image"""
        # Store the numpy array for mask editor
        self.current_input_image = image.copy() if image is not None else None
        
        pixmap = self.numpy_to_pixmap(image)
        self.current_input_pixmap = pixmap
        
        # Display the input image
        self._display_image(pixmap, "Input Image")
        
        if ENHANCED_VIEWER_AVAILABLE and self.comparison_viewer:
            self.comparison_viewer.set_original_image(pixmap)
        
        # Enable/disable create mask button based on image availability
        self.create_mask_btn.setEnabled(image is not None)
        
        # Update metadata
        if hasattr(self, 'metadata_widget') and hasattr(self, 'current_input_path'):
            self.metadata_widget.update_metadata(self.current_input_path)
        
        self.update_image_info("Input image loaded")
    
    def set_mask_image(self, image):
        """Set mask image"""
        # Store the original binary mask for inpainting
        self.current_mask_array = image.copy()
        
        # Create enhanced visualization for better mask visibility
        pixmap = self.create_enhanced_mask_pixmap(image)
        self.current_mask_pixmap = pixmap
        
        # Enable mask tab and switch to it to show the mask
        self.mask_tab_btn.setEnabled(True)
        self.show_image_tab('mask')
        
        # Update image info
        self.update_image_info("Mask image loaded - Red areas will be inpainted")
    
    def set_result_image(self, image):
        """Set result image"""
        pixmap = self.numpy_to_pixmap(image)
        self.current_result_pixmap = pixmap
        
        # Enable result tab and switch to it to show the result
        self.result_tab_btn.setEnabled(True)
        self.show_image_tab('result')
        
        if ENHANCED_VIEWER_AVAILABLE and self.comparison_viewer:
            self.comparison_viewer.set_result_image(pixmap)
        
        self.update_image_info("Result image ready")
    
    def numpy_to_pixmap(self, image):
        """Convert numpy array to QPixmap"""
        from PySide6.QtGui import QImage
        import cv2
        
        if len(image.shape) == 3:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            if channel == 3:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return QPixmap.fromImage(q_image)
    
    def create_enhanced_mask_pixmap(self, mask_array):
        """Create an enhanced visualization of the mask for better visibility"""
        import cv2
        import numpy as np
        
        # Create a colorized version of the mask for better visibility
        # White areas (to be inpainted) will be shown in bright red
        # Black areas (preserved) will be shown in dark gray
        
        enhanced_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        
        # Set background (preserved areas) to dark gray
        enhanced_mask[:, :] = [64, 64, 64]  # Dark gray background
        
        # Set mask areas (to be inpainted) to bright red
        mask_areas = mask_array > 127
        enhanced_mask[mask_areas] = [255, 0, 0]  # Bright red for mask
        
        return self.numpy_to_pixmap(enhanced_mask)
    
    def update_image_info(self, message):
        """Update image info in status bar"""
        if self.current_input_pixmap:
            size_info = f"{self.current_input_pixmap.width()}×{self.current_input_pixmap.height()}"
            self.image_info_label.setText(f"{message} | {size_info}")
        else:
            self.image_info_label.setText(message)
    
    # Progress and status methods
    def set_status_message(self, message):
        """Set status bar message"""
        self.status_label.setText(message)
    
    def update_progress(self, value):
        """Update progress bar"""
        if not self.status_progress.isVisible():
            self.status_progress.setVisible(True)
        self.status_progress.setValue(value)
        
        if value >= 100:
            QTimer.singleShot(2000, lambda: self.status_progress.setVisible(False))
    
    def set_processing_state(self, processing):
        """Set processing state"""
        # Disable/enable buttons during processing
        self.load_image_btn.setEnabled(not processing)
        self.load_mask_btn.setEnabled(not processing)
        self.create_mask_btn.setEnabled(not processing and self.current_input_image is not None)
        self.run_btn.setEnabled(not processing)
        self.research_btn.setEnabled(not processing)  # Also disable research button
        self.save_btn.setEnabled(not processing and self.current_result_pixmap is not None)
        
        if processing:
            self.show_enhanced_progress_dialog()
        else:
            self.hide_progress_dialog()
    
    def show_enhanced_progress_dialog(self):
        """Show enhanced progress dialog"""
        # For now, use simple status bar progress instead of complex dialog
        # to avoid overlay issues
        self.status_progress.setVisible(True)
        self.status_progress.setValue(0)
        self.set_status_message("Processing...")
        
        # Uncomment below to re-enable enhanced dialog after fixing
        # if ENHANCED_PROGRESS_AVAILABLE:
        #     if not self.progress_dialog:
        #         self.progress_dialog = EnhancedProgressDialog("Image Inpainting", self)
        #         self.progress_dialog.cancel_requested.connect(self.on_progress_cancel)
        #     
        #     self.progress_dialog.show()
        #     self.progress_dialog.start_processing()
    
    def hide_progress_dialog(self):
        """Hide progress dialog"""
        # Hide simple progress bar
        self.status_progress.setVisible(False)
        self.set_status_message("Ready")
        
        # Uncomment below when re-enabling enhanced dialog
        # if self.progress_dialog:
        #     self.progress_dialog.finish_processing(True)
        #     QTimer.singleShot(1000, lambda: self.progress_dialog.hide())
    
    def on_progress_cancel(self):
        """Handle progress dialog cancel"""
        # This would be connected to cancel the actual processing
        pass
    
    # UI state methods
    def set_run_button_enabled(self, enabled):
        """Enable/disable run button"""
        self.run_btn.setEnabled(enabled)
        # Enable research button when run button is enabled
        self.research_btn.setEnabled(enabled)
    
    def set_save_button_enabled(self, enabled):
        """Enable/disable save button"""
        self.save_btn.setEnabled(enabled)
    
    def reset_ui(self):
        """Reset UI to initial state"""
        self.current_input_pixmap = None
        self.current_mask_pixmap = None
        self.current_result_pixmap = None
        self.current_input_image = None  # Clear input image for mask editor
        self.current_mask_array = None  # Clear binary mask
        
        # Reset tabs - disable mask and result tabs, switch to input
        self.mask_tab_btn.setEnabled(False)
        self.result_tab_btn.setEnabled(False)
        self.show_image_tab('input')
        
        # Disable create mask button
        self.create_mask_btn.setEnabled(False)
        
        if ENHANCED_VIEWER_AVAILABLE and hasattr(self.single_viewer, 'set_image'):
            self.single_viewer.set_image(None)
        else:
            self.single_viewer.clear()
        
        if ENHANCED_VIEWER_AVAILABLE and self.comparison_viewer:
            self.comparison_viewer.set_original_image(None)
            self.comparison_viewer.set_result_image(None)
        
        if hasattr(self, 'metadata_widget'):
            self.metadata_widget.clear_metadata()
        self.update_image_info("No image loaded")
        self.set_status_message("Ready")
    
    # Dialog methods
    def show_welcome_dialog(self):
        """Show welcome dialog"""
        if not WELCOME_DIALOG_AVAILABLE:
            return
            
        if not self.welcome_dialog:
            self.welcome_dialog = WelcomeDialog(self)
            self.welcome_dialog.tutorial_requested.connect(self.tutorial_requested.emit)
        
        result = self.welcome_dialog.exec()
        
        # Update settings based on user choice
        if self.welcome_dialog.should_show_again():
            self.settings.show_welcome_dialog = True
        else:
            self.settings.show_welcome_dialog = False
        self.settings.save()
    
    def show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Image Inpainting Application",
            f"""
            <h2>Image Inpainting Application</h2>
            <p>Version 1.1.0</p>
            <p>Professional image restoration and object removal tool</p>
            <br>
            <p><b>Features:</b></p>
            <ul>
                <li>Advanced image inpainting algorithms</li>
                <li>GPU acceleration support</li>
                <li>Professional workflow tools</li>
                <li>Enhanced user experience</li>
            </ul>
            <br>
            <p><b>Keyboard Shortcuts:</b></p>
            <ul>
                <li>Ctrl+O: Load Image</li>
                <li>Ctrl+M: Load Mask</li>
                <li>Ctrl+S: Save Result</li>
                <li>F5: Run Inpainting</li>
                <li>Ctrl+R: Reset All</li>
            </ul>
            """
        )
    
    # Recent files handlers
    def on_recent_image_selected(self, file_path):
        """Handle recent image selection"""
        if file_path:  # Empty string means clear
            self.current_input_path = file_path
            self.load_image_requested.emit()  # This will be handled by controller
        else:
            # Clear recent images
            self.settings.recent_images.clear()
            self.settings.save()
            self.update_recent_files()
    
    def on_recent_mask_selected(self, file_path):
        """Handle recent mask selection"""
        if file_path:  # Empty string means clear
            self.current_mask_path = file_path
            self.load_mask_requested.emit()  # This will be handled by controller
        else:
            # Clear recent masks
            self.settings.recent_masks.clear()
            self.settings.save()
            self.update_recent_files()
    
    # Message dialogs
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
    
    # Control panel access
    def get_control_panel(self):
        """Get control panel widget"""
        return self.control_panel
    
    # Mask editor methods
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
            # Debug: Check mask values
            import numpy as np
            unique_values = np.unique(mask_array)
            white_pixels = np.sum(mask_array == 255)
            black_pixels = np.sum(mask_array == 0)
            print(f"Mask created - Unique values: {unique_values}")
            print(f"White pixels (target areas): {white_pixels}")
            print(f"Black pixels (source areas): {black_pixels}")
            
            # Set the created mask
            self.set_mask_image(mask_array)
            
            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Mask Created",
                "Your mask has been created and is now displayed in the Mask tab.\n\n"
                "Red areas will be inpainted (what you drew), gray areas will be preserved.\n\n"
                "Are you satisfied with this mask?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # User is satisfied - emit signal to notify controller
                self.create_mask_requested.emit(mask_array)
                self.set_status_message("Custom mask created and ready for inpainting")
            else:
                # User wants to recreate the mask
                self.set_status_message("Click 'Create Mask' to draw a new mask")
                return
            
        except Exception as e:
            self.show_error_message("Mask Creation Error", f"Failed to apply created mask:\n{str(e)}")
    
    # Window events
    def closeEvent(self, event):
        """Handle window close event"""
        # Save window position
        self.settings.window_x = self.x()
        self.settings.window_y = self.y()
        self.settings.window_width = self.width()
        self.settings.window_height = self.height()
        self.settings.save()
        
        event.accept()


# Maintain compatibility with existing code
MainWindow = EnhancedMainWindow 