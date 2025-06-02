"""
Enhanced Image Label with Zoom, Pan, and Comparison Features
"""
import math
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QSlider, QComboBox, QFrame, QToolBar,
                             QSizePolicy, QScrollArea, QButtonGroup, QCheckBox)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont, QCursor
import cv2
import numpy as np


class ZoomableImageLabel(QLabel):
    """Enhanced image label with zoom and pan functionality"""
    
    # Signals
    zoom_changed = Signal(float)
    position_changed = Signal(QPoint)
    mouse_moved = Signal(QPoint)  # For coordinate display
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Image data
        self.original_pixmap = None
        self.current_pixmap = None
        
        # Zoom and pan
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_offset = QPoint(0, 0)
        
        # Mouse interaction
        self.last_mouse_pos = QPoint()
        self.dragging = False
        
        # Setup
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #444;
                background-color: #1e1e1e;
                background-image: 
                    linear-gradient(45deg, #2a2a2a 25%, transparent 25%),
                    linear-gradient(-45deg, #2a2a2a 25%, transparent 25%),
                    linear-gradient(45deg, transparent 75%, #2a2a2a 75%),
                    linear-gradient(-45deg, transparent 75%, #2a2a2a 75%);
                background-size: 20px 20px;
                background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            }
        """)
        
        # Mouse tracking
        self.setMouseTracking(True)
        
    def set_image(self, pixmap):
        """Set the image to display"""
        if pixmap and not pixmap.isNull():
            self.original_pixmap = pixmap
            self.reset_view()
        else:
            self.original_pixmap = None
            self.current_pixmap = None
            self.clear()
    
    def reset_view(self):
        """Reset zoom and pan to fit image"""
        if not self.original_pixmap:
            return
        
        # Calculate zoom to fit
        widget_size = self.size()
        image_size = self.original_pixmap.size()
        
        zoom_x = widget_size.width() / image_size.width()
        zoom_y = widget_size.height() / image_size.height()
        
        self.zoom_factor = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%
        self.pan_offset = QPoint(0, 0)
        
        self.update_display()
    
    def zoom_in(self):
        """Zoom in"""
        self.set_zoom(self.zoom_factor * 1.2)
    
    def zoom_out(self):
        """Zoom out"""
        self.set_zoom(self.zoom_factor / 1.2)
    
    def zoom_to_fit(self):
        """Zoom to fit the widget"""
        self.reset_view()
    
    def zoom_to_actual_size(self):
        """Zoom to actual size (100%)"""
        self.set_zoom(1.0)
    
    def set_zoom(self, zoom):
        """Set specific zoom level"""
        zoom = max(self.min_zoom, min(self.max_zoom, zoom))
        if zoom != self.zoom_factor:
            self.zoom_factor = zoom
            self.update_display()
            self.zoom_changed.emit(self.zoom_factor)
    
    def update_display(self):
        """Update the displayed image"""
        if not self.original_pixmap:
            return
        
        # Calculate scaled size
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        
        # Create scaled pixmap
        self.current_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Apply pan offset and set
        self.setPixmap(self.current_pixmap)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_pixmap:
            # Zoom factor
            zoom_delta = 1.2 if event.angleDelta().y() > 0 else 1/1.2
            
            # Get mouse position for zoom center
            mouse_pos = event.position().toPoint()
            
            # Calculate new zoom
            old_zoom = self.zoom_factor
            new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * zoom_delta))
            
            if new_zoom != old_zoom:
                # Adjust pan to zoom around mouse position
                zoom_ratio = new_zoom / old_zoom
                widget_center = self.rect().center()
                offset_from_center = mouse_pos - widget_center
                
                self.pan_offset = (self.pan_offset + offset_from_center) * zoom_ratio - offset_from_center
                
                self.set_zoom(new_zoom)
    
    def mousePressEvent(self, event):
        """Handle mouse press for panning"""
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.position().toPoint()
            self.dragging = True
            self.setCursor(QCursor(Qt.ClosedHandCursor))
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning and coordinate display"""
        current_pos = event.position().toPoint()
        
        # Emit mouse position for coordinate display
        self.mouse_moved.emit(current_pos)
        
        if self.dragging and self.original_pixmap:
            # Calculate pan delta
            delta = current_pos - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = current_pos
            
            # Update display
            self.update_display()
            self.position_changed.emit(self.pan_offset)
        else:
            # Show hand cursor when hovering over image
            if self.original_pixmap:
                self.setCursor(QCursor(Qt.OpenHandCursor))
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click to reset view"""
        if event.button() == Qt.LeftButton:
            self.reset_view()


class ImageViewerWidget(QWidget):
    """Complete image viewer with controls"""
    
    # Signals for comparison modes
    comparison_mode_changed = Signal(str)
    overlay_opacity_changed = Signal(float)
    
    def __init__(self, title="Image", parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()
        
        # Comparison data
        self.comparison_image = None
        self.overlay_opacity = 0.5
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title and controls
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # View mode combo
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Normal", "Side by Side", "Overlay", "Difference"])
        self.view_mode_combo.currentTextChanged.connect(self.comparison_mode_changed.emit)
        header_layout.addWidget(self.view_mode_combo)
        
        # Zoom controls
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedSize(30, 25)
        zoom_in_btn.setToolTip("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        header_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedSize(30, 25)
        zoom_out_btn.setToolTip("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        header_layout.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("📏")
        fit_btn.setFixedSize(30, 25)
        fit_btn.setToolTip("Fit to Window")
        fit_btn.clicked.connect(self.zoom_to_fit)
        header_layout.addWidget(fit_btn)
        
        actual_btn = QPushButton("1:1")
        actual_btn.setFixedSize(30, 25)
        actual_btn.setToolTip("Actual Size")
        actual_btn.clicked.connect(self.zoom_to_actual_size)
        header_layout.addWidget(actual_btn)
        
        layout.addLayout(header_layout)
        
        # Image display
        self.image_label = ZoomableImageLabel()
        self.image_label.zoom_changed.connect(self.on_zoom_changed)
        self.image_label.mouse_moved.connect(self.on_mouse_moved)
        
        # Scroll area for large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(scroll_area)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setStyleSheet("color: #666; font-size: 11px;")
        status_layout.addWidget(self.zoom_label)
        
        status_layout.addStretch()
        
        self.coords_label = QLabel("Position: (0, 0)")
        self.coords_label.setStyleSheet("color: #666; font-size: 11px;")
        status_layout.addWidget(self.coords_label)
        
        self.size_label = QLabel("Size: -")
        self.size_label.setStyleSheet("color: #666; font-size: 11px;")
        status_layout.addWidget(self.size_label)
        
        layout.addLayout(status_layout)
        
        # Overlay opacity slider (hidden by default)
        self.opacity_layout = QHBoxLayout()
        self.opacity_label = QLabel("Overlay Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        
        self.opacity_layout.addWidget(self.opacity_label)
        self.opacity_layout.addWidget(self.opacity_slider)
        
        # Create widget to hold opacity controls
        self.opacity_widget = QWidget()
        self.opacity_widget.setLayout(self.opacity_layout)
        self.opacity_widget.hide()
        
        layout.addWidget(self.opacity_widget)
    
    def set_image(self, pixmap):
        """Set the main image"""
        self.image_label.set_image(pixmap)
        if pixmap and not pixmap.isNull():
            self.size_label.setText(f"Size: {pixmap.width()}×{pixmap.height()}")
        else:
            self.size_label.setText("Size: -")
    
    def set_comparison_image(self, pixmap):
        """Set image for comparison modes"""
        self.comparison_image = pixmap
    
    def zoom_in(self):
        """Zoom in"""
        self.image_label.zoom_in()
    
    def zoom_out(self):
        """Zoom out"""
        self.image_label.zoom_out()
    
    def zoom_to_fit(self):
        """Zoom to fit"""
        self.image_label.zoom_to_fit()
    
    def zoom_to_actual_size(self):
        """Zoom to actual size"""
        self.image_label.zoom_to_actual_size()
    
    def on_zoom_changed(self, zoom):
        """Handle zoom change"""
        self.zoom_label.setText(f"Zoom: {zoom*100:.0f}%")
    
    def on_mouse_moved(self, pos):
        """Handle mouse move for coordinate display"""
        # Convert widget coordinates to image coordinates
        if self.image_label.original_pixmap:
            # Account for zoom and centering
            image_pos = self.widget_to_image_coords(pos)
            self.coords_label.setText(f"Position: ({image_pos.x()}, {image_pos.y()})")
        else:
            self.coords_label.setText("Position: (0, 0)")
    
    def on_opacity_changed(self, value):
        """Handle opacity slider change"""
        self.overlay_opacity = value / 100.0
        self.overlay_opacity_changed.emit(self.overlay_opacity)
    
    def widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates"""
        if not self.image_label.original_pixmap:
            return QPoint(0, 0)
        
        # Get image center in widget
        widget_center = self.image_label.rect().center()
        
        # Calculate offset from center
        offset = widget_pos - widget_center - self.image_label.pan_offset
        
        # Scale by zoom factor
        image_offset = offset / self.image_label.zoom_factor
        
        # Add to image center
        image_center = QPoint(
            self.image_label.original_pixmap.width() // 2,
            self.image_label.original_pixmap.height() // 2
        )
        
        image_pos = image_center + image_offset
        
        # Clamp to image bounds
        image_pos.setX(max(0, min(self.image_label.original_pixmap.width() - 1, image_pos.x())))
        image_pos.setY(max(0, min(self.image_label.original_pixmap.height() - 1, image_pos.y())))
        
        return image_pos
    
    def show_comparison_controls(self, mode):
        """Show/hide comparison controls based on mode"""
        if mode == "Overlay":
            self.opacity_widget.show()
        else:
            self.opacity_widget.hide()
    
    def get_current_view_mode(self):
        """Get current view mode"""
        return self.view_mode_combo.currentText()


class ComparisonViewWidget(QWidget):
    """Widget for side-by-side image comparison"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup comparison view UI"""
        layout = QHBoxLayout(self)
        
        # Original image viewer
        self.original_viewer = ImageViewerWidget("Original")
        layout.addWidget(self.original_viewer)
        
        # Result image viewer
        self.result_viewer = ImageViewerWidget("Result")
        layout.addWidget(self.result_viewer)
        
        # Sync zoom and pan between viewers
        self.original_viewer.image_label.zoom_changed.connect(self.sync_zoom_to_result)
        self.result_viewer.image_label.zoom_changed.connect(self.sync_zoom_to_original)
        
        self.original_viewer.image_label.position_changed.connect(self.sync_pan_to_result)
        self.result_viewer.image_label.position_changed.connect(self.sync_pan_to_original)
        
        self._syncing = False  # Prevent infinite sync loops
        
    def sync_zoom_to_result(self, zoom):
        """Sync zoom from original to result"""
        if not self._syncing:
            self._syncing = True
            self.result_viewer.image_label.set_zoom(zoom)
            self._syncing = False
    
    def sync_zoom_to_original(self, zoom):
        """Sync zoom from result to original"""
        if not self._syncing:
            self._syncing = True
            self.original_viewer.image_label.set_zoom(zoom)
            self._syncing = False
    
    def sync_pan_to_result(self, pos):
        """Sync pan from original to result"""
        if not self._syncing:
            self._syncing = True
            self.result_viewer.image_label.pan_offset = pos
            self.result_viewer.image_label.update_display()
            self._syncing = False
    
    def sync_pan_to_original(self, pos):
        """Sync pan from result to original"""
        if not self._syncing:
            self._syncing = True
            self.original_viewer.image_label.pan_offset = pos
            self.original_viewer.image_label.update_display()
            self._syncing = False
    
    def set_original_image(self, pixmap):
        """Set original image"""
        self.original_viewer.set_image(pixmap)
    
    def set_result_image(self, pixmap):
        """Set result image"""
        self.result_viewer.set_image(pixmap) 