"""
Comparison Panel Widget
Provides side-by-side image comparison with quality metrics display
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, 
    QFrame, QSplitter, QGroupBox, QScrollArea, QFileDialog, QTextEdit,
    QSizePolicy, QSpacerItem, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor

from config.settings import AppConstants
from .enhanced_image_label import ImageViewerWidget


class ComparisonPanel(QWidget):
    """Widget for comparing original and inpainted images with metrics"""
    
    # Signals
    load_original_requested = Signal()
    load_inpainted_requested = Signal()
    calculate_metrics_requested = Signal()
    save_comparison_requested = Signal()
    
    def __init__(self):
        super().__init__()
        
        self.original_image = None
        self.inpainted_image = None
        self.ssim_diff_image = None
        
        self.setup_ui()
        self.apply_styling()
    
    def setup_ui(self):
        """Setup the comparison panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Image Quality Comparison")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Control buttons
        self.setup_control_buttons(main_layout)
        
        # Main content area with splitter
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter, 1)
        
        # Left side: Images
        self.setup_image_comparison(content_splitter)
        
        # Right side: Metrics and information
        self.setup_metrics_panel(content_splitter)
        
        # Set splitter proportions (70% images, 30% metrics)
        content_splitter.setStretchFactor(0, 7)
        content_splitter.setStretchFactor(1, 3)
    
    def setup_control_buttons(self, layout):
        """Setup control buttons"""
        button_layout = QHBoxLayout()
        
        # Load original button
        self.load_original_btn = QPushButton("📁 Load Original")
        self.load_original_btn.setToolTip("Load original image for comparison")
        self.load_original_btn.clicked.connect(self.load_original_requested.emit)
        button_layout.addWidget(self.load_original_btn)
        
        # Load inpainted button
        self.load_inpainted_btn = QPushButton("📁 Load Inpainted")
        self.load_inpainted_btn.setToolTip("Load inpainted image for comparison")
        self.load_inpainted_btn.clicked.connect(self.load_inpainted_requested.emit)
        button_layout.addWidget(self.load_inpainted_btn)
        
        # Calculate metrics button
        self.calculate_btn = QPushButton("📊 Calculate Metrics")
        self.calculate_btn.setToolTip("Calculate PSNR, SSIM, LPIPS, and MSE metrics")
        self.calculate_btn.clicked.connect(self.calculate_metrics_requested.emit)
        self.calculate_btn.setEnabled(False)
        button_layout.addWidget(self.calculate_btn)
        
        # Spacer
        button_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Save comparison button
        self.save_comparison_btn = QPushButton("💾 Save Comparison")
        self.save_comparison_btn.setToolTip("Save comparison results")
        self.save_comparison_btn.clicked.connect(self.save_comparison_requested.emit)
        self.save_comparison_btn.setEnabled(False)
        button_layout.addWidget(self.save_comparison_btn)
        
        layout.addLayout(button_layout)
    
    def setup_image_comparison(self, parent):
        """Setup the image comparison area"""
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tab widget for different views
        self.image_tabs = QTabWidget()
        image_layout.addWidget(self.image_tabs)
        
        # Side-by-side comparison tab
        self.setup_side_by_side_tab()
        
        # SSIM difference tab
        self.setup_ssim_diff_tab()
        
        parent.addWidget(image_widget)
    
    def setup_side_by_side_tab(self):
        """Setup side-by-side comparison tab"""
        side_by_side_widget = QWidget()
        side_by_side_layout = QHBoxLayout(side_by_side_widget)
        side_by_side_layout.setContentsMargins(5, 5, 5, 5)
        
        # Original image panel
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        
        self.original_image_label = ImageViewerWidget("Original")
        self.original_image_label.setMinimumSize(300, 300)
        original_layout.addWidget(self.original_image_label)
        
        # Original image info
        self.original_info_label = QLabel("No image loaded")
        self.original_info_label.setAlignment(Qt.AlignCenter)
        self.original_info_label.setStyleSheet("color: #888888; font-size: 11px;")
        original_layout.addWidget(self.original_info_label)
        
        side_by_side_layout.addWidget(original_group)
        
        # Inpainted image panel
        inpainted_group = QGroupBox("Inpainted Image")
        inpainted_layout = QVBoxLayout(inpainted_group)
        
        self.inpainted_image_label = ImageViewerWidget("Inpainted")
        self.inpainted_image_label.setMinimumSize(300, 300)
        inpainted_layout.addWidget(self.inpainted_image_label)
        
        # Inpainted image info
        self.inpainted_info_label = QLabel("No image loaded")
        self.inpainted_info_label.setAlignment(Qt.AlignCenter)
        self.inpainted_info_label.setStyleSheet("color: #888888; font-size: 11px;")
        inpainted_layout.addWidget(self.inpainted_info_label)
        
        side_by_side_layout.addWidget(inpainted_group)
        
        self.image_tabs.addTab(side_by_side_widget, "Side by Side")
    
    def setup_ssim_diff_tab(self):
        """Setup SSIM difference visualization tab"""
        ssim_widget = QWidget()
        ssim_layout = QVBoxLayout(ssim_widget)
        ssim_layout.setContentsMargins(5, 5, 5, 5)
        
        # SSIM difference image
        ssim_group = QGroupBox("SSIM Difference Map")
        ssim_group_layout = QVBoxLayout(ssim_group)
        
        self.ssim_diff_label = ImageViewerWidget("SSIM Difference")
        self.ssim_diff_label.setMinimumSize(400, 400)
        ssim_group_layout.addWidget(self.ssim_diff_label)
        
        # SSIM difference info
        ssim_info = QLabel("Darker areas indicate higher structural differences")
        ssim_info.setAlignment(Qt.AlignCenter)
        ssim_info.setStyleSheet("color: #888888; font-size: 11px;")
        ssim_group_layout.addWidget(ssim_info)
        
        ssim_layout.addWidget(ssim_group)
        
        self.image_tabs.addTab(ssim_widget, "SSIM Difference")
    
    def setup_metrics_panel(self, parent):
        """Setup the metrics display panel"""
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        
        # Metrics title
        metrics_title = QLabel("Quality Metrics")
        metrics_title.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_title.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(metrics_title)
        
        # Metrics display area
        self.setup_metrics_display(metrics_layout)
        
        # Quality interpretation area
        self.setup_quality_interpretation(metrics_layout)
        
        # Additional information area
        self.setup_additional_info(metrics_layout)
        
        parent.addWidget(metrics_widget)
    
    def setup_metrics_display(self, layout):
        """Setup metrics display area"""
        metrics_group = QGroupBox("Calculated Metrics")
        metrics_group_layout = QVBoxLayout(metrics_group)
        
        # PSNR display
        self.psnr_label = QLabel("PSNR: Not calculated")
        self.psnr_label.setFont(QFont("Courier", 10))
        metrics_group_layout.addWidget(self.psnr_label)
        
        # SSIM display
        self.ssim_label = QLabel("SSIM: Not calculated")
        self.ssim_label.setFont(QFont("Courier", 10))
        metrics_group_layout.addWidget(self.ssim_label)
        
        # LPIPS display
        self.lpips_label = QLabel("LPIPS: Not calculated")
        self.lpips_label.setFont(QFont("Courier", 10))
        metrics_group_layout.addWidget(self.lpips_label)
        
        # MSE display
        self.mse_label = QLabel("MSE: Not calculated")
        self.mse_label.setFont(QFont("Courier", 10))
        metrics_group_layout.addWidget(self.mse_label)
        
        layout.addWidget(metrics_group)
    
    def setup_quality_interpretation(self, layout):
        """Setup quality interpretation area"""
        quality_group = QGroupBox("Quality Assessment")
        quality_group_layout = QVBoxLayout(quality_group)
        
        self.quality_label = QLabel("Load both images and calculate metrics")
        self.quality_label.setWordWrap(True)
        self.quality_label.setAlignment(Qt.AlignTop)
        quality_group_layout.addWidget(self.quality_label)
        
        layout.addWidget(quality_group)
    
    def setup_additional_info(self, layout):
        """Setup additional information area"""
        info_group = QGroupBox("Information")
        info_group_layout = QVBoxLayout(info_group)
        
        # Create scrollable text area for detailed info
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText(
            "About the Metrics:\n\n"
            "PSNR (Peak Signal-to-Noise Ratio):\n"
            "• Higher values = better quality\n"
            "• >40 dB: Excellent\n"
            "• 30-40 dB: Good\n"
            "• 20-30 dB: Fair\n"
            "• <20 dB: Poor\n\n"
            "SSIM (Structural Similarity Index):\n"
            "• Range: 0 to 1\n"
            "• Higher values = more similar\n"
            "• >0.95: Excellent\n"
            "• 0.85-0.95: Good\n"
            "• 0.7-0.85: Fair\n"
            "• <0.7: Poor\n\n"
            "LPIPS (Learned Perceptual Similarity):\n"
            "• Range: 0 to 1+\n"
            "• Lower values = more perceptually similar\n"
            "• <0.1: Excellent\n"
            "• 0.1-0.3: Good\n"
            "• 0.3-0.5: Fair\n"
            "• >0.5: Poor"
        )
        info_group_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Add spacer to push everything up
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    def apply_styling(self):
        """Apply modern dark styling"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2b2b2b;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            
            QPushButton {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: bold;
                min-width: 100px;
            }
            
            QPushButton:hover {
                background-color: #505050;
                border-color: #707070;
            }
            
            QPushButton:pressed {
                background-color: #353535;
            }
            
            QPushButton:disabled {
                background-color: #2a2a2a;
                border-color: #404040;
                color: #808080;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                font-family: "Consolas", "Monaco", monospace;
                font-size: 10px;
            }
            
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            
            QTabBar::tab {
                background-color: #404040;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
                color: #ffffff;
            }
            
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                border-bottom-color: #2b2b2b;
            }
            
            QTabBar::tab:hover {
                background-color: #505050;
            }
        """)
    
    # Public interface methods
    def set_original_image(self, image_array):
        """Set the original image"""
        try:
            self.original_image = image_array
            
            # Convert numpy array to QPixmap and display
            if image_array is not None:
                pixmap = self.array_to_pixmap(image_array)
                self.original_image_label.set_image(pixmap)
                
                # Update info
                h, w = image_array.shape[:2]
                channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
                self.original_info_label.setText(f"{w}×{h}, {channels} channel(s)")
            else:
                self.original_image_label.set_image(None)
                self.original_info_label.setText("No image loaded")
                
            self.update_ui_state()
            
        except Exception as e:
            print(f"Error setting original image: {e}")
    
    def set_inpainted_image(self, image_array):
        """Set the inpainted image"""
        try:
            self.inpainted_image = image_array
            
            # Convert numpy array to QPixmap and display
            if image_array is not None:
                pixmap = self.array_to_pixmap(image_array)
                self.inpainted_image_label.set_image(pixmap)
                
                # Update info
                h, w = image_array.shape[:2]
                channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
                self.inpainted_info_label.setText(f"{w}×{h}, {channels} channel(s)")
            else:
                self.inpainted_image_label.set_image(None)
                self.inpainted_info_label.setText("No image loaded")
                
            self.update_ui_state()
            
        except Exception as e:
            print(f"Error setting inpainted image: {e}")
    
    def set_ssim_difference_image(self, ssim_diff_array):
        """Set the SSIM difference image"""
        try:
            self.ssim_diff_image = ssim_diff_array
            
            if ssim_diff_array is not None:
                # Convert SSIM difference to displayable format
                # SSIM difference is usually in range [0, 1], convert to [0, 255]
                display_array = (ssim_diff_array * 255).astype('uint8')
                
                # Convert to 3-channel if needed
                if len(display_array.shape) == 2:
                    import numpy as np
                    display_array = np.stack([display_array] * 3, axis=-1)
                
                pixmap = self.array_to_pixmap(display_array)
                self.ssim_diff_label.set_image(pixmap)
            else:
                self.ssim_diff_label.set_image(None)
                
        except Exception as e:
            print(f"Error setting SSIM difference image: {e}")
    
    def update_metrics_display(self, metrics_dict):
        """Update the metrics display"""
        try:
            if not metrics_dict:
                self.psnr_label.setText("PSNR: Not calculated")
                self.ssim_label.setText("SSIM: Not calculated")
                self.lpips_label.setText("LPIPS: Not calculated")
                self.mse_label.setText("MSE: Not calculated")
                self.quality_label.setText("Load both images and calculate metrics")
                return
            
            # Update individual metrics
            if 'psnr' in metrics_dict:
                psnr_val = metrics_dict['psnr']
                self.psnr_label.setText(f"PSNR: {psnr_val:.2f} dB")
            
            if 'ssim' in metrics_dict:
                ssim_val = metrics_dict['ssim']
                self.ssim_label.setText(f"SSIM: {ssim_val:.4f}")
            
            if 'lpips' in metrics_dict:
                lpips_val = metrics_dict['lpips']
                if lpips_val is not None:
                    self.lpips_label.setText(f"LPIPS: {lpips_val:.4f}")
                else:
                    self.lpips_label.setText("LPIPS: Not available")
            else:
                self.lpips_label.setText("LPIPS: Not calculated")
            
            if 'mse' in metrics_dict:
                mse_val = metrics_dict['mse']
                self.mse_label.setText(f"MSE: {mse_val:.2f}")
            
            # Update quality assessment
            from models.metrics import MetricsComparison
            quality_summary = MetricsComparison.get_quality_summary(metrics_dict)
            self.quality_label.setText(f"Overall Quality: {quality_summary}")
            
            self.update_ui_state()
            
        except Exception as e:
            print(f"Error updating metrics display: {e}")
    
    def update_ui_state(self):
        """Update UI state based on current data"""
        has_both_images = (self.original_image is not None and 
                          self.inpainted_image is not None)
        
        self.calculate_btn.setEnabled(has_both_images)
        
        # Enable save button if metrics are calculated
        # This will be controlled by the parent controller
    
    def set_calculate_button_enabled(self, enabled):
        """Enable/disable calculate button"""
        self.calculate_btn.setEnabled(enabled)
    
    def set_save_button_enabled(self, enabled):
        """Enable/disable save button"""
        self.save_comparison_btn.setEnabled(enabled)
    
    def array_to_pixmap(self, image_array):
        """Convert numpy array to QPixmap"""
        try:
            import numpy as np
            from PySide6.QtGui import QImage
            
            # Ensure the array is in the right format
            if image_array.dtype != np.uint8:
                # Normalize to 0-255 range
                image_array = ((image_array - image_array.min()) / 
                              (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            height, width = image_array.shape[:2]
            
            if len(image_array.shape) == 3:
                # Color image
                bytes_per_line = 3 * width
                q_image = QImage(image_array.data, width, height, 
                               bytes_per_line, QImage.Format_RGB888)
            else:
                # Grayscale image
                bytes_per_line = width
                q_image = QImage(image_array.data, width, height, 
                               bytes_per_line, QImage.Format_Grayscale8)
            
            return QPixmap.fromImage(q_image)
            
        except Exception as e:
            print(f"Error converting array to pixmap: {e}")
            return QPixmap()
    
    def reset(self):
        """Reset the comparison panel"""
        self.original_image = None
        self.inpainted_image = None
        self.ssim_diff_image = None
        
        self.original_image_label.set_image(None)
        self.inpainted_image_label.set_image(None)
        self.ssim_diff_label.set_image(None)
        
        self.original_info_label.setText("No image loaded")
        self.inpainted_info_label.setText("No image loaded")
        
        self.psnr_label.setText("PSNR: Not calculated")
        self.ssim_label.setText("SSIM: Not calculated")
        self.mse_label.setText("MSE: Not calculated")
        self.quality_label.setText("Load both images and calculate metrics")
        
        self.update_ui_state() 