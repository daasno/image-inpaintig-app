"""
Modern Welcome Dialog with Streamlined User Experience
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QWidget, QTextEdit, QCheckBox, 
                             QFrame, QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QPixmap, QIcon
import webbrowser


class WelcomeDialog(QDialog):
    """Modern, streamlined welcome dialog"""
    
    tutorial_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Image Inpainting App")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        
        # Make dialog responsive to screen size
        self.setup_responsive_size()
        self.setup_ui()
        self.apply_modern_styling()
        
    def setup_responsive_size(self):
        """Setup responsive dialog size based on screen"""
        # Get screen geometry properly
        from PySide6.QtWidgets import QApplication
        
        if self.parent():
            screen_rect = self.parent().screen().availableGeometry()
        else:
            screen_rect = QApplication.primaryScreen().availableGeometry()
            
        # Use 40% of screen width, max 600px, min 500px
        width = max(500, min(600, int(screen_rect.width() * 0.4)))
        # Use 60% of screen height, max 700px, min 400px  
        height = max(400, min(700, int(screen_rect.height() * 0.6)))
        
        self.resize(width, height)
        
        # Center on parent or screen
        if self.parent():
            parent_geo = self.parent().geometry()
            x = parent_geo.x() + (parent_geo.width() - width) // 2
            y = parent_geo.y() + (parent_geo.height() - height) // 2
            self.move(x, y)
        else:
            # Center on screen
            x = screen_rect.x() + (screen_rect.width() - width) // 2
            y = screen_rect.y() + (screen_rect.height() - height) // 2
            self.move(x, y)
        
    def setup_ui(self):
        """Setup the streamlined welcome dialog UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 15)
        layout.setSpacing(15)
        
        # Header section
        header_widget = self.create_header()
        layout.addWidget(header_widget)
        
        # Main content in scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = self.create_main_content()
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # Footer section
        footer_widget = self.create_footer()
        layout.addWidget(footer_widget)
        
    def create_header(self):
        """Create modern header section"""
        header = QFrame()
        header.setObjectName("header")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # App title
        title = QLabel("🎨 Image Inpainting Application")
        title.setObjectName("title")
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        # Subtitle
        subtitle = QLabel("Professional image restoration and object removal")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle, alignment=Qt.AlignCenter)
        
        return header
        
    def create_main_content(self):
        """Create main content area"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)
        
        # Quick start guide
        quick_start = self.create_quick_start_section()
        layout.addWidget(quick_start)
        
        # Key features
        features = self.create_features_section()
        layout.addWidget(features)
        
        # Getting started actions
        actions = self.create_actions_section()
        layout.addWidget(actions)
        
        layout.addStretch()
        return widget
        
    def create_quick_start_section(self):
        """Create streamlined quick start section"""
        section = QFrame()
        section.setObjectName("section")
        layout = QVBoxLayout(section)
        
        # Section title
        title = QLabel("🚀 Quick Start Guide")
        title.setObjectName("section-title")
        layout.addWidget(title)
        
        # Steps in a clean format
        steps_text = """
<div style="line-height: 1.4; color: #e0e0e0;">
<p><b>1. Load Image</b> - Click "📁 Load Image" to select your image</p>
<p><b>2. Create/Load Mask</b> - Use "✏️ Create Mask" or "🎭 Load Mask"</p>
<p><b>3. Adjust Settings</b> - Fine-tune patch size and p-value if needed</p>
<p><b>4. Run Inpainting</b> - Click "▶️ Run Inpainting" to process</p>
<p><b>5. Save Result</b> - Use "💾 Save" when satisfied with the result</p>
</div>
        """
        
        steps_label = QLabel(steps_text)
        steps_label.setWordWrap(True)
        steps_label.setTextFormat(Qt.RichText)
        layout.addWidget(steps_label)
        
        return section
        
    def create_features_section(self):
        """Create key features section"""
        section = QFrame()
        section.setObjectName("section")
        layout = QVBoxLayout(section)
        
        # Section title
        title = QLabel("✨ Key Features")
        title.setObjectName("section-title")
        layout.addWidget(title)
        
        # Features in two columns
        features_layout = QHBoxLayout()
        
        # Left column
        left_features = QLabel("""
<div style="line-height: 1.4; color: #e0e0e0;">
<p>• <b>Built-in Mask Editor</b><br/>Draw masks directly in the app</p>
<p>• <b>GPU Acceleration</b><br/>Fast processing with CUDA support</p>
<p>• <b>Real-time Preview</b><br/>See your mask before processing</p>
</div>
        """)
        left_features.setWordWrap(True)
        left_features.setTextFormat(Qt.RichText)
        features_layout.addWidget(left_features)
        
        # Right column
        right_features = QLabel("""
<div style="line-height: 1.4; color: #e0e0e0;">
<p>• <b>Parameter Research</b><br/>Find optimal settings automatically</p>
<p>• <b>Multiple Formats</b><br/>Support for PNG, JPG, TIFF, etc.</p>
<p>• <b>Professional UI</b><br/>Clean, intuitive interface</p>
</div>
        """)
        right_features.setWordWrap(True)
        right_features.setTextFormat(Qt.RichText)
        features_layout.addWidget(right_features)
        
        layout.addLayout(features_layout)
        return section
        
    def create_actions_section(self):
        """Create getting started actions"""
        section = QFrame()
        section.setObjectName("section")
        layout = QVBoxLayout(section)
        
        # Section title
        title = QLabel("🎯 Getting Started")
        title.setObjectName("section-title")
        layout.addWidget(title)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Load sample button
        sample_btn = QPushButton("📁 Load Sample Image")
        sample_btn.setObjectName("action-button")
        sample_btn.clicked.connect(self.load_sample_image)
        sample_btn.setToolTip("Load a sample image to try the application")
        buttons_layout.addWidget(sample_btn)
        
        # View help button  
        help_btn = QPushButton("❓ View Help")
        help_btn.setObjectName("action-button")
        help_btn.clicked.connect(self.open_help)
        help_btn.setToolTip("Open help documentation")
        buttons_layout.addWidget(help_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Tips
        tips_label = QLabel("""
<div style="line-height: 1.3; color: #c0c0c0; font-size: 11px;">
<p><b>💡 Pro Tips:</b></p>
<p>• Start with default settings (patch size 9, p-value 1.0)</p>
<p>• White areas in mask = regions to inpaint, black = preserve</p>
<p>• Use GPU mode for faster processing on supported systems</p>
<p>• Try Exhaustive Research to find optimal parameters automatically</p>
</div>
        """)
        tips_label.setWordWrap(True)
        tips_label.setTextFormat(Qt.RichText)
        layout.addWidget(tips_label)
        
        return section
        
    def create_footer(self):
        """Create footer with options"""
        footer = QFrame()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 10, 0, 0)
        
        # Don't show again checkbox
        self.dont_show_checkbox = QCheckBox("Don't show this welcome dialog again")
        self.dont_show_checkbox.setObjectName("checkbox")
        layout.addWidget(self.dont_show_checkbox)
        
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Get Started")
        close_btn.setObjectName("primary-button")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        close_btn.setMinimumWidth(120)
        layout.addWidget(close_btn)
        
        return footer
        
    def apply_modern_styling(self):
        """Apply modern dark theme styling"""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                border: 1px solid #404040;
                color: #ffffff;
            }
            
            QFrame#header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4a90e2, stop:1 #357abd);
                border-radius: 8px;
                margin-bottom: 5px;
            }
            
            QLabel#title {
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 5px;
            }
            
            QLabel#subtitle {
                color: #e6f2ff;
                font-size: 13px;
                padding-bottom: 5px;
            }
            
            QFrame#section {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 15px;
                margin: 5px 0px;
            }
            
            QLabel#section-title {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                margin-bottom: 8px;
            }
            
            QLabel {
                color: #e0e0e0;
            }
            
            QPushButton#action-button {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            
            QPushButton#action-button:hover {
                background-color: #5a6268;
            }
            
            QPushButton#primary-button {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            
            QPushButton#primary-button:hover {
                background-color: #218838;
            }
            
            QCheckBox#checkbox {
                color: #b0b0b0;
                font-size: 12px;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
    def load_sample_image(self):
        """Handle loading sample image"""
        from PySide6.QtWidgets import QMessageBox
        
        # Check if sample files exist
        import os
        sample_paths = [
            "examples/sample_image.jpg",
            "examples/sample_image.png", 
            "sample_image.jpg",
            "test_image.jpg"
        ]
        
        found_sample = None
        for path in sample_paths:
            if os.path.exists(path):
                found_sample = path
                break
                
        if found_sample:
            QMessageBox.information(
                self, 
                "Sample Image", 
                f"Sample image found at: {found_sample}\n\n"
                "Close this dialog and use 'Load Image' to select this file."
            )
        else:
            QMessageBox.information(
                self, 
                "No Sample Found", 
                "No sample images found in the application directory.\n\n"
                "You can download sample images from the project repository\n"
                "or use your own images to get started."
            )
    
    def open_help(self):
        """Open help documentation"""
        from PySide6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Open Help",
            "This will open the help documentation in your web browser.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Try to open a local help file or online documentation
                help_url = "https://github.com/your-repo/image-inpainting-app/wiki"
                webbrowser.open(help_url)
            except Exception:
                QMessageBox.information(
                    self,
                    "Help",
                    "Help documentation:\n\n"
                    "• Press F1 in the main application for context help\n"
                    "• Check the README.md file in the application folder\n"
                    "• Visit the project repository for detailed documentation"
                )
    
    def should_show_again(self):
        """Check if dialog should be shown again"""
        return not self.dont_show_checkbox.isChecked() 