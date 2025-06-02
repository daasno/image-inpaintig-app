"""
Help Dialog Widget - Comprehensive help and documentation
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTabWidget, QWidget, QTextEdit,
                             QScrollArea, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class HelpDialog(QDialog):
    """Comprehensive help dialog with multiple sections"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help - Image Inpainting Application")
        self.setModal(True)
        self.resize(700, 600)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #3a3a3a;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #4a4a4a;
                color: #cccccc;
                border: 1px solid #666;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                border-bottom: 1px solid #3a3a3a;
            }
            QTabBar::tab:hover {
                background-color: #5a5a5a;
            }
            QTextEdit {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #cccccc;
                padding: 8px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: #cccccc;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QLabel {
                color: #cccccc;
                background: transparent;
            }
        """)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the help dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Image Inpainting Application - Help & Documentation")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)
        
        # Tabs
        tab_widget = QTabWidget()
        
        # Getting Started Tab
        getting_started_tab = self.create_getting_started_tab()
        tab_widget.addTab(getting_started_tab, "Getting Started")
        
        # Parameters Tab
        parameters_tab = self.create_parameters_tab()
        tab_widget.addTab(parameters_tab, "Parameters")
        
        # Mask Creation Tab
        mask_creation_tab = self.create_mask_creation_tab()
        tab_widget.addTab(mask_creation_tab, "Mask Creation")
        
        # Troubleshooting Tab
        troubleshooting_tab = self.create_troubleshooting_tab()
        tab_widget.addTab(troubleshooting_tab, "Troubleshooting")
        
        # Keyboard Shortcuts Tab
        shortcuts_tab = self.create_shortcuts_tab()
        tab_widget.addTab(shortcuts_tab, "Shortcuts")
        
        layout.addWidget(tab_widget)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_getting_started_tab(self):
        """Create the getting started tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Getting Started with Image Inpainting</h2>
        
        <h3>What is Image Inpainting?</h3>
        <p>Image inpainting is a technique used to restore damaged areas or remove unwanted objects from images. 
        The algorithm analyzes the surrounding pixels and fills in the masked areas with plausible content.</p>
        
        <h3>Basic Workflow</h3>
        <ol>
            <li><b>Load Input Image:</b> Click "Load Image" or use Ctrl+O to select your image</li>
            <li><b>Create or Load Mask:</b> 
                <ul>
                    <li>Click "Create Mask" to draw directly on your image</li>
                    <li>Or click "Load Mask" to use an existing mask file</li>
                </ul>
            </li>
            <li><b>Adjust Parameters:</b> Fine-tune patch size and p-value for optimal results</li>
            <li><b>Choose Implementation:</b> Select CPU for compatibility or GPU for speed</li>
            <li><b>Run Inpainting:</b> Click "Run Inpainting" or press F5 to start processing</li>
            <li><b>Save Result:</b> Save your restored image when satisfied</li>
        </ol>
        
        <h3>Tips for Best Results</h3>
        <ul>
            <li>Use high-quality input images for better results</li>
            <li>Keep masked areas relatively small compared to the total image</li>
            <li>Ensure good contrast between areas to be inpainted and surrounding regions</li>
            <li>Experiment with different parameter values for optimal results</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        return widget
    
    def create_parameters_tab(self):
        """Create the parameters explanation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Parameter Guide</h2>
        
        <h3>Patch Size (3-21)</h3>
        <p><b>What it does:</b> Controls the size of texture patches used for inpainting analysis.</p>
        <p><b>Effects:</b></p>
        <ul>
            <li><b>Smaller values (3-7):</b> Good for fine details and small objects. Faster processing but may miss larger patterns.</li>
            <li><b>Medium values (9-13):</b> Balanced approach suitable for most images. Good compromise between quality and speed.</li>
            <li><b>Larger values (15-21):</b> Better for large areas and smooth textures. Slower processing but captures larger patterns.</li>
        </ul>
        <p><b>Performance Impact:</b> Larger patch sizes result in smoother results but significantly slower processing.</p>
        <p><b>Recommendation:</b> Start with 9 and adjust based on your image content and quality requirements.</p>
        
        <h3>Minkowski Order (p-value: 0.1-10.0)</h3>
        <p><b>What it does:</b> Controls the distance metric used for patch matching.</p>
        <p><b>Common Values:</b></p>
        <ul>
            <li><b>p = 1.0 (Manhattan distance):</b> More conservative, preserves structure better. Good for images with clear edges.</li>
            <li><b>p = 2.0 (Euclidean distance):</b> Standard distance metric. Balanced approach for most images.</li>
            <li><b>p < 1.0:</b> More aggressive matching, may blur fine details but fills large areas well.</li>
            <li><b>p > 2.0:</b> Very conservative, preserves details but may leave artifacts in large areas.</li>
        </ul>
        <p><b>Recommendation:</b> Use 1.0 for Manhattan distance (good default) or 2.0 for Euclidean distance.</p>
        
        <h3>Implementation Choice</h3>
        <p><b>CPU Implementation:</b></p>
        <ul>
            <li>Compatible with all systems</li>
            <li>Slower processing, especially for large images</li>
            <li>Uses system RAM</li>
        </ul>
        <p><b>GPU Implementation (CUDA):</b></p>
        <ul>
            <li>Requires NVIDIA GPU with CUDA support</li>
            <li>10-20x faster processing for large images</li>
            <li>Uses GPU memory (VRAM)</li>
            <li>Automatically falls back to CPU if unavailable</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        return widget
    
    def create_mask_creation_tab(self):
        """Create the mask creation help tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Mask Creation Guide</h2>
        
        <h3>Understanding Masks</h3>
        <p>A mask is a black and white image that tells the algorithm which areas to inpaint:</p>
        <ul>
            <li><b>White areas:</b> Will be inpainted (filled in)</li>
            <li><b>Black areas:</b> Will be preserved from the original image</li>
        </ul>
        
        <h3>Built-in Mask Editor</h3>
        <p>Click "Create Mask" to open the interactive mask editor:</p>
        
        <h4>Tools Available:</h4>
        <ul>
            <li><b>Brush Tool:</b> Draw white areas (regions to inpaint)</li>
            <li><b>Eraser Tool:</b> Remove mask areas (make them black)</li>
            <li><b>Size Slider:</b> Adjust brush/eraser size (5-100 pixels)</li>
            <li><b>Opacity Slider:</b> Control mask overlay visibility</li>
            <li><b>Clear All:</b> Remove entire mask and start over</li>
        </ul>
        
        <h4>Usage Tips:</h4>
        <ul>
            <li>Masked areas appear as red overlay on your image</li>
            <li>Use smaller brush sizes for precise work</li>
            <li>Use larger brush sizes for filling large areas quickly</li>
            <li>Adjust opacity to see the underlying image clearly</li>
            <li>The "Apply Mask" button is only enabled when you have drawn something</li>
        </ul>
        
        <h3>External Mask Creation</h3>
        <p>You can also create masks in external image editors:</p>
        <ul>
            <li><b>GIMP:</b> Use paintbrush tool with white color on black background</li>
            <li><b>Photoshop:</b> Create selection and fill with white on black background</li>
            <li><b>Paint.NET:</b> Use brush tool with white color</li>
        </ul>
        
        <h3>Mask Quality Tips</h3>
        <ul>
            <li>Keep mask edges smooth for better blending</li>
            <li>Avoid very thin or disconnected mask regions</li>
            <li>Smaller masked areas generally produce better results</li>
            <li>Ensure sufficient surrounding context for the algorithm to work with</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        return widget
    
    def create_troubleshooting_tab(self):
        """Create the troubleshooting tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Troubleshooting</h2>
        
        <h3>Common Issues and Solutions</h3>
        
        <h4>GPU Not Available</h4>
        <p><b>Problem:</b> "GPU implementation is not available" message</p>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Ensure you have an NVIDIA GPU with CUDA support</li>
            <li>Install CUDA toolkit and drivers</li>
            <li>Install numba with CUDA support: <code>pip install numba[cuda]</code></li>
            <li>Use CPU implementation as fallback</li>
        </ul>
        
        <h4>Poor Inpainting Results</h4>
        <p><b>Problem:</b> Results look blurry or don't match surrounding areas</p>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Try different patch sizes (start with 9, then try 7 or 13)</li>
            <li>Adjust p-value (try 1.0 for Manhattan distance)</li>
            <li>Ensure mask doesn't cover too large an area</li>
            <li>Check that surrounding areas have sufficient texture/detail</li>
        </ul>
        
        <h4>Slow Processing</h4>
        <p><b>Problem:</b> Inpainting takes too long</p>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Use GPU implementation if available</li>
            <li>Reduce patch size (try 7 or 5)</li>
            <li>Resize image to smaller dimensions before processing</li>
            <li>Reduce the size of masked areas</li>
        </ul>
        
        <h4>Memory Issues</h4>
        <p><b>Problem:</b> Out of memory errors</p>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Reduce image size before processing</li>
            <li>Use smaller patch sizes</li>
            <li>Close other applications to free memory</li>
            <li>Switch from GPU to CPU implementation</li>
        </ul>
        
        <h4>File Loading Issues</h4>
        <p><b>Problem:</b> Cannot load images or masks</p>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Ensure file format is supported (PNG, JPG, BMP, TIFF)</li>
            <li>Check file permissions</li>
            <li>Try converting to PNG format</li>
            <li>Ensure file is not corrupted</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        return widget
    
    def create_shortcuts_tab(self):
        """Create the keyboard shortcuts tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Keyboard Shortcuts</h2>
        
        <h3>File Operations</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <tr><td><b>Ctrl+O</b></td><td>Load Input Image</td></tr>
            <tr><td><b>Ctrl+M</b></td><td>Load Mask Image</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>Save Result Image</td></tr>
            <tr><td><b>Ctrl+Q</b></td><td>Exit Application</td></tr>
        </table>
        
        <h3>Processing</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <tr><td><b>F5</b></td><td>Run Inpainting</td></tr>
            <tr><td><b>Ctrl+R</b></td><td>Reset All</td></tr>
        </table>
        
        <h3>View Controls</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <tr><td><b>Ctrl+1</b></td><td>Single View Mode</td></tr>
            <tr><td><b>Ctrl+2</b></td><td>Side by Side View</td></tr>
            <tr><td><b>Ctrl+=</b></td><td>Zoom In</td></tr>
            <tr><td><b>Ctrl+-</b></td><td>Zoom Out</td></tr>
            <tr><td><b>Ctrl+0</b></td><td>Zoom to Fit</td></tr>
        </table>
        
        <h3>Mask Editor (when open)</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <tr><td><b>Left Click + Drag</b></td><td>Draw/Erase mask</td></tr>
            <tr><td><b>Mouse Wheel</b></td><td>Adjust brush size</td></tr>
            <tr><td><b>B</b></td><td>Switch to Brush tool</td></tr>
            <tr><td><b>E</b></td><td>Switch to Eraser tool</td></tr>
            <tr><td><b>Ctrl+A</b></td><td>Clear all mask</td></tr>
            <tr><td><b>Enter</b></td><td>Apply mask</td></tr>
            <tr><td><b>Escape</b></td><td>Cancel mask editor</td></tr>
        </table>
        
        <h3>General</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <tr><td><b>F1</b></td><td>Show this Help dialog</td></tr>
            <tr><td><b>Alt+F4</b></td><td>Close window</td></tr>
        </table>
        """)
        
        layout.addWidget(help_text)
        return widget 