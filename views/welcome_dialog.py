"""
Welcome Dialog with User Onboarding and Guidance
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTabWidget, QWidget, QTextEdit,
                             QCheckBox, QScrollArea, QFrame, QGridLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QIcon


class WelcomeDialog(QDialog):
    """Welcome dialog with user guidance and onboarding"""
    
    tutorial_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Image Inpainting Application")
        self.setFixedSize(700, 500)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the welcome dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_frame = self.create_header()
        layout.addWidget(header_frame)
        
        # Tabs
        tab_widget = QTabWidget()
        
        # Quick Start Tab
        quick_start_tab = self.create_quick_start_tab()
        tab_widget.addTab(quick_start_tab, "Quick Start")
        
        # Tutorial Tab
        tutorial_tab = self.create_tutorial_tab()
        tab_widget.addTab(tutorial_tab, "Tutorial")
        
        # Tips & Tricks Tab
        tips_tab = self.create_tips_tab()
        tab_widget.addTab(tips_tab, "Tips & Tricks")
        
        # What's New Tab
        whats_new_tab = self.create_whats_new_tab()
        tab_widget.addTab(whats_new_tab, "What's New")
        
        layout.addWidget(tab_widget)
        
        # Footer with options
        footer_frame = self.create_footer()
        layout.addWidget(footer_frame)
        
    def create_header(self):
        """Create the header section"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4a90e2, stop:1 #357abd);
                border-radius: 8px;
                margin: 5px;
            }
        """)
        frame.setFixedHeight(80)
        
        layout = QHBoxLayout(frame)
        
        # App icon (if available)
        icon_label = QLabel()
        icon_label.setFixedSize(64, 64)
        icon_label.setStyleSheet("background: transparent;")
        # You can set an actual icon here if you have one
        layout.addWidget(icon_label)
        
        # Title and subtitle
        text_layout = QVBoxLayout()
        
        title = QLabel("Image Inpainting Application")
        title.setStyleSheet("color: white; font-size: 24px; font-weight: bold; background: transparent;")
        text_layout.addWidget(title)
        
        subtitle = QLabel("Professional image restoration and object removal tool")
        subtitle.setStyleSheet("color: #e6f2ff; font-size: 14px; background: transparent;")
        text_layout.addWidget(subtitle)
        
        text_layout.addStretch()
        layout.addLayout(text_layout)
        layout.addStretch()
        
        return frame
        
    def create_quick_start_tab(self):
        """Create the quick start guide tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Quick start steps
        steps = [
            ("1. Load Input Image", "Click 'Load Image' to select the image you want to edit", "🖼️"),
            ("2. Load Mask Image", "Click 'Load Mask' to select the mask (white areas will be inpainted)", "🎭"),
            ("3. Adjust Parameters", "Fine-tune patch size and p-value for best results", "⚙️"),
            ("4. Choose Implementation", "Select CPU for compatibility or GPU for speed", "💻"),
            ("5. Run Inpainting", "Click 'Run Inpainting' to start the process", "▶️"),
            ("6. Save Result", "Save your restored image when satisfied with the result", "💾")
        ]
        
        for step_title, step_desc, emoji in steps:
            step_frame = self.create_step_frame(step_title, step_desc, emoji)
            layout.addWidget(step_frame)
        
        layout.addStretch()
        
        # Quick action buttons
        button_layout = QHBoxLayout()
        
        tutorial_btn = QPushButton("Start Interactive Tutorial")
        tutorial_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        tutorial_btn.clicked.connect(self.start_tutorial)
        
        examples_btn = QPushButton("View Examples")
        examples_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        examples_btn.clicked.connect(self.show_examples)
        
        button_layout.addWidget(tutorial_btn)
        button_layout.addWidget(examples_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        return widget
    
    def create_step_frame(self, title, description, emoji):
        """Create a frame for a single step"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin: 2px;
                padding: 8px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        # Emoji
        emoji_label = QLabel(emoji)
        emoji_label.setStyleSheet("font-size: 24px; background: transparent; color: #4a90e2;")
        emoji_label.setFixedWidth(40)
        layout.addWidget(emoji_label)
        
        # Text
        text_layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; background: transparent;")
        text_layout.addWidget(title_label)
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #6c757d; background: transparent;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        layout.addLayout(text_layout)
        
        return frame
    
    def create_tutorial_tab(self):
        """Create the tutorial tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tutorial sections
        tutorial_text = QTextEdit()
        tutorial_text.setReadOnly(True)
        tutorial_text.setHtml("""
        <h2>Image Inpainting Tutorial</h2>
        
        <h3>What is Image Inpainting?</h3>
        <p>Image inpainting is a technique used to restore damaged or remove unwanted objects from images. 
        The algorithm fills in missing or masked areas by analyzing the surrounding pixels and textures.</p>
        
        <h3>Creating Effective Masks</h3>
        <ul>
            <li><b>White areas</b> in the mask indicate regions to be inpainted</li>
            <li><b>Black areas</b> are preserved from the original image</li>
            <li>Use image editing software like GIMP, Photoshop, or Paint.NET to create masks</li>
            <li>Keep mask edges smooth for better blending results</li>
            <li>Smaller masked areas generally produce better results</li>
        </ul>
        
        <h3>Parameter Guidelines</h3>
        <ul>
            <li><b>Patch Size (3-21):</b> Controls the size of texture patches used for inpainting
                <ul>
                    <li>Smaller values (3-7): Good for fine details and small objects</li>
                    <li>Medium values (9-13): Balanced approach for most images</li>
                    <li>Larger values (15-21): Better for large areas and smooth textures</li>
                </ul>
            </li>
            <li><b>P-Value (0.1-10.0):</b> Controls the blending strength
                <ul>
                    <li>Lower values: More conservative, preserves structure</li>
                    <li>Higher values: More aggressive filling, may blur details</li>
                    <li>Start with 1.0 and adjust based on results</li>
                </ul>
            </li>
        </ul>
        
        <h3>Best Practices</h3>
        <ul>
            <li>Start with default parameters and make small adjustments</li>
            <li>Use GPU implementation for faster processing when available</li>
            <li>For large areas, consider multiple passes with different parameters</li>
            <li>Save your work frequently during experimentation</li>
        </ul>
        """)
        
        layout.addWidget(tutorial_text)
        
        return widget
    
    def create_tips_tab(self):
        """Create the tips and tricks tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        tips = [
            ("💡 Pro Tip", "Use keyboard shortcuts", "Ctrl+O: Load Image, Ctrl+M: Load Mask, Ctrl+S: Save Result, F5: Run Inpainting"),
            ("🎯 Accuracy", "Precise mask creation", "Use a graphics tablet for more precise mask creation, especially for complex shapes"),
            ("⚡ Performance", "GPU acceleration", "GPU implementation can be 10-20x faster than CPU for large images"),
            ("🔍 Quality", "Multiple passes", "For complex inpainting, try multiple passes with different parameters"),
            ("📁 Organization", "File management", "Keep original images, masks, and results in organized folders"),
            ("🖥️ Display", "Monitor calibration", "Use a calibrated monitor for accurate color representation"),
            ("💾 Backup", "Save frequently", "Save your work frequently, especially when experimenting with parameters"),
            ("🎨 Creative use", "Artistic effects", "Try creative masking for artistic effects beyond just object removal")
        ]
        
        for icon, title, description in tips:
            tip_frame = self.create_tip_frame(icon, title, description)
            layout.addWidget(tip_frame)
        
        layout.addStretch()
        
        return widget
    
    def create_tip_frame(self, icon, title, description):
        """Create a frame for a single tip"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 6px;
                margin: 3px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px; background: transparent;")
        icon_label.setFixedWidth(30)
        layout.addWidget(icon_label)
        
        # Content
        content_layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #856404; background: transparent;")
        content_layout.addWidget(title_label)
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #856404; background: transparent;")
        desc_label.setWordWrap(True)
        content_layout.addWidget(desc_label)
        
        layout.addLayout(content_layout)
        
        return frame
    
    def create_whats_new_tab(self):
        """Create the what's new tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        whats_new = QTextEdit()
        whats_new.setReadOnly(True)
        whats_new.setHtml("""
        <h2>What's New in Version 1.0</h2>
        
        <h3>🎉 New Features</h3>
        <ul>
            <li><b>Modular Architecture:</b> Clean, maintainable code structure</li>
            <li><b>Enhanced Settings:</b> Persistent user preferences</li>
            <li><b>GPU Acceleration:</b> CUDA support for faster processing</li>
            <li><b>Better Error Handling:</b> More informative error messages</li>
            <li><b>Improved UI:</b> Modern, intuitive interface design</li>
        </ul>
        
        <h3>🔧 Improvements</h3>
        <ul>
            <li>Better memory management for large images</li>
            <li>More responsive UI during processing</li>
            <li>Enhanced parameter validation</li>
            <li>Improved file format support</li>
            <li>Better progress reporting</li>
        </ul>
        
        <h3>🐛 Bug Fixes</h3>
        <ul>
            <li>Fixed memory leaks during processing</li>
            <li>Resolved UI freezing issues</li>
            <li>Improved mask handling for edge cases</li>
            <li>Better error recovery</li>
        </ul>
        
        <h3>🔮 Coming Soon</h3>
        <ul>
            <li>Batch processing capabilities</li>
            <li>Built-in mask editor</li>
            <li>Additional inpainting algorithms</li>
            <li>Preview modes</li>
            <li>Undo/redo functionality</li>
        </ul>
        """)
        
        layout.addWidget(whats_new)
        
        return widget
    
    def create_footer(self):
        """Create the footer with options"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Don't show again checkbox
        self.dont_show_checkbox = QCheckBox("Don't show this dialog again")
        self.dont_show_checkbox.setStyleSheet("QCheckBox { color: #6c757d; }")
        layout.addWidget(self.dont_show_checkbox)
        
        layout.addStretch()
        
        # Buttons
        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.show_help)
        
        close_btn = QPushButton("Get Started")
        close_btn.setDefault(True)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        close_btn.clicked.connect(self.accept)
        
        layout.addWidget(help_btn)
        layout.addWidget(close_btn)
        
        return frame
    
    def start_tutorial(self):
        """Start the interactive tutorial"""
        self.tutorial_requested.emit()
        self.accept()
    
    def show_examples(self):
        """Show example images and results"""
        # For now, just show a message
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, 
            "Examples", 
            "Example images and masks can be found in the 'examples' folder.\n\n"
            "Try loading 'sample_image.jpg' and 'sample_mask.png' to get started!"
        )
    
    def show_help(self):
        """Show help documentation"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, 
            "Help", 
            "For detailed help and documentation, please visit:\n"
            "https://github.com/yourusername/image-inpainting-app\n\n"
            "Or press F1 in the main application for context-sensitive help."
        )
    
    def should_show_again(self):
        """Check if dialog should be shown again"""
        return not self.dont_show_checkbox.isChecked() 