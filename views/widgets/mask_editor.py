"""
Mask Editor Widget - Interactive mask creation tool
"""
import cv2
import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSlider, QFrame, QButtonGroup,
                             QToolBar, QSizePolicy, QScrollArea, QWidget)
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QPixmap, QImage, 
                          QCursor, QPainterPath, QIcon)


class DrawableImageLabel(QLabel):
    """Image label that allows drawing mask regions"""
    
    mask_updated = Signal(np.ndarray)  # Emits the current mask
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Drawing state
        self.drawing = False
        self.brush_size = 20
        self.eraser_mode = False
        self.last_point = QPoint()
        
        # Images
        self.original_image = None
        self.display_pixmap = None
        self.mask_array = None
        self.overlay_opacity = 0.5
        
        # Setup
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1e1e1e;
            }
        """)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Custom cursor
        self.update_cursor()
    
    def set_image(self, image_array):
        """Set the image to edit (numpy array)"""
        if image_array is None:
            return
        
        self.original_image = image_array.copy()
        
        # Initialize mask (all black = no mask)
        if len(image_array.shape) == 3:
            h, w = image_array.shape[:2]
        else:
            h, w = image_array.shape
        
        self.mask_array = np.zeros((h, w), dtype=np.uint8)
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the displayed image with mask overlay"""
        if self.original_image is None:
            return
        
        # Convert original image to display format
        if len(self.original_image.shape) == 3:
            # Color image - convert BGR to RGB
            display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale - convert to RGB
            display_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        
        # Create mask overlay (red semi-transparent)
        mask_overlay = np.zeros_like(display_image)
        mask_overlay[self.mask_array > 0] = [255, 0, 0]  # Red for mask areas
        
        # Blend original image with mask overlay
        blended = cv2.addWeighted(
            display_image, 1.0 - self.overlay_opacity,
            mask_overlay, self.overlay_opacity,
            0
        )
        
        # Convert to QPixmap
        h, w, ch = blended.shape
        bytes_per_line = ch * w
        q_image = QImage(blended.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.display_pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label
        scaled_pixmap = self.display_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def set_brush_size(self, size):
        """Set brush size"""
        self.brush_size = max(1, min(100, size))
        self.update_cursor()
    
    def set_eraser_mode(self, enabled):
        """Set eraser mode"""
        self.eraser_mode = enabled
        self.update_cursor()
    
    def set_overlay_opacity(self, opacity):
        """Set mask overlay opacity"""
        self.overlay_opacity = max(0.0, min(1.0, opacity))
        self.update_display()
    
    def update_cursor(self):
        """Update cursor based on current tool"""
        if self.eraser_mode:
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CrossCursor))
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.LeftButton and self.original_image is not None:
            self.drawing = True
            self.last_point = self.map_to_image_coords(event.position().toPoint())
            self.draw_at_point(self.last_point)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move"""
        if self.drawing and self.original_image is not None:
            current_point = self.map_to_image_coords(event.position().toPoint())
            self.draw_line(self.last_point, current_point)
            self.last_point = current_point
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.drawing = False
            # Emit updated mask
            self.mask_updated.emit(self.mask_array.copy())
    
    def map_to_image_coords(self, widget_point):
        """Map widget coordinates to image coordinates"""
        if self.display_pixmap is None:
            return QPoint(0, 0)
        
        # Get the displayed pixmap rect
        pixmap_rect = self.pixmap().rect() if self.pixmap() else QRect()
        if pixmap_rect.isEmpty():
            return QPoint(0, 0)
        
        # Calculate the position of the pixmap within the label
        label_rect = self.rect()
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2
        
        # Map to pixmap coordinates
        pixmap_x = widget_point.x() - x_offset
        pixmap_y = widget_point.y() - y_offset
        
        # Scale to original image coordinates
        scale_x = self.original_image.shape[1] / pixmap_rect.width()
        scale_y = self.original_image.shape[0] / pixmap_rect.height()
        
        image_x = int(pixmap_x * scale_x)
        image_y = int(pixmap_y * scale_y)
        
        # Clamp to image bounds
        image_x = max(0, min(self.original_image.shape[1] - 1, image_x))
        image_y = max(0, min(self.original_image.shape[0] - 1, image_y))
        
        return QPoint(image_x, image_y)
    
    def draw_at_point(self, point):
        """Draw at a specific point"""
        if self.mask_array is None:
            return
        
        x, y = point.x(), point.y()
        radius = self.brush_size // 2
        
        # Create circular brush
        y_coords, x_coords = np.ogrid[:self.mask_array.shape[0], :self.mask_array.shape[1]]
        mask_circle = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius ** 2
        
        if self.eraser_mode:
            # Erase (set to black)
            self.mask_array[mask_circle] = 0
        else:
            # Draw (set to white)
            self.mask_array[mask_circle] = 255
        
        self.update_display()
    
    def draw_line(self, start_point, end_point):
        """Draw a line between two points"""
        if self.mask_array is None:
            return
        
        # Use Bresenham's line algorithm to get points along the line
        x0, y0 = start_point.x(), start_point.y()
        x1, y1 = end_point.x(), end_point.y()
        
        points = self.get_line_points(x0, y0, x1, y1)
        
        for x, y in points:
            self.draw_at_point(QPoint(x, y))
    
    def get_line_points(self, x0, y0, x1, y1):
        """Get points along a line using Bresenham's algorithm"""
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
            
            if error2 < dx:
                error += dx
                y += y_step
        
        return points
    
    def clear_mask(self):
        """Clear the entire mask"""
        if self.mask_array is not None:
            self.mask_array.fill(0)
            self.update_display()
            self.mask_updated.emit(self.mask_array.copy())
    
    def get_mask(self):
        """Get the current mask as numpy array"""
        return self.mask_array.copy() if self.mask_array is not None else None
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        if self.original_image is not None:
            self.update_display()


class MaskEditorDialog(QDialog):
    """Dialog for interactive mask creation"""
    
    mask_created = Signal(np.ndarray)  # Emits the final mask
    
    def __init__(self, image_array, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Editor")
        self.setModal(True)
        self.resize(800, 700)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QFrame {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: #cccccc;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QSlider::groove:horizontal {
                border: 1px solid #666;
                height: 6px;
                background: #3a3a3a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                border: 1px solid #005a9e;
                width: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QLabel {
                color: #cccccc;
                background: transparent;
            }
        """)
        
        self.setup_ui()
        
        # Set the image
        self.image_editor.set_image(image_array)
    
    def setup_ui(self):
        """Setup the mask editor UI"""
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar_frame = QFrame()
        toolbar_layout = QHBoxLayout(toolbar_frame)
        
        # Tool selection
        tool_label = QLabel("Tool:")
        toolbar_layout.addWidget(tool_label)
        
        self.brush_btn = QPushButton("🖌️ Brush")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.clicked.connect(lambda: self.set_tool_mode(False))
        toolbar_layout.addWidget(self.brush_btn)
        
        self.eraser_btn = QPushButton("🧽 Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self.set_tool_mode(True))
        toolbar_layout.addWidget(self.eraser_btn)
        
        # Tool button group
        self.tool_group = QButtonGroup()
        self.tool_group.addButton(self.brush_btn)
        self.tool_group.addButton(self.eraser_btn)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        # Brush size
        size_label = QLabel("Size:")
        toolbar_layout.addWidget(size_label)
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(5, 100)
        self.size_slider.setValue(20)
        self.size_slider.setFixedWidth(100)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        toolbar_layout.addWidget(self.size_slider)
        
        self.size_label = QLabel("20")
        self.size_label.setFixedWidth(30)
        toolbar_layout.addWidget(self.size_label)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        # Opacity
        opacity_label = QLabel("Opacity:")
        toolbar_layout.addWidget(opacity_label)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 90)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        toolbar_layout.addWidget(self.opacity_slider)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.clear_mask)
        toolbar_layout.addWidget(clear_btn)
        
        toolbar_layout.addStretch()
        
        layout.addWidget(toolbar_frame)
        
        # Image editor
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_editor = DrawableImageLabel()
        self.image_editor.mask_updated.connect(self.on_mask_updated)
        
        scroll_area.setWidget(self.image_editor)
        layout.addWidget(scroll_area)
        
        # Instructions
        instructions = QLabel(
            "Instructions: Use the brush to draw mask areas (shown in red). "
            "Use the eraser to remove mask areas. Adjust brush size and opacity as needed."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #999; font-size: 11px; padding: 5px;")
        layout.addWidget(instructions)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply Mask")
        self.apply_btn.setStyleSheet("""
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
        self.apply_btn.clicked.connect(self.apply_mask)
        self.apply_btn.setEnabled(False)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
    
    def set_tool_mode(self, eraser_mode):
        """Set the current tool mode"""
        self.image_editor.set_eraser_mode(eraser_mode)
        
        # Update button states
        self.brush_btn.setChecked(not eraser_mode)
        self.eraser_btn.setChecked(eraser_mode)
    
    def on_size_changed(self, value):
        """Handle brush size change"""
        self.image_editor.set_brush_size(value)
        self.size_label.setText(str(value))
    
    def on_opacity_changed(self, value):
        """Handle opacity change"""
        opacity = value / 100.0
        self.image_editor.set_overlay_opacity(opacity)
    
    def clear_mask(self):
        """Clear the entire mask"""
        self.image_editor.clear_mask()
    
    def on_mask_updated(self, mask_array):
        """Handle mask updates"""
        # Enable apply button if mask has content
        has_content = np.any(mask_array > 0)
        self.apply_btn.setEnabled(has_content)
    
    def apply_mask(self):
        """Apply the created mask"""
        mask = self.image_editor.get_mask()
        if mask is not None:
            self.mask_created.emit(mask)
            self.accept()
    
    def get_mask(self):
        """Get the current mask"""
        return self.image_editor.get_mask() 