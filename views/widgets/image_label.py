"""
Custom image display widget with proper scaling and enhanced features
"""
import cv2
import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QFrame, QSizePolicy
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with proper scaling and enhanced features"""

    def __init__(self, title="", min_size=(300, 300)):
        super().__init__()
        self.title_text = title
        self.min_size = min_size
        
        # Configure the label
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*min_size)
        self.setFrameShape(QFrame.Box)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)

        # Add a title label
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-weight: bold; margin: 5px;")

        # Create a layout for the image and title
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.title, 0, Qt.AlignTop)
        self.layout.addStretch()

        # Initialize with placeholder
        self.current_image = None
        self.original_pixmap = None
        self.setPlaceholder()

    def setPlaceholder(self):
        """Set a placeholder when no image is loaded"""
        self.clear()
        self.setText("No Image")
        self.setStyleSheet("""
            QLabel { 
                background-color: #2b2b2b; 
                color: #cccccc; 
                border: 2px dashed #555;
                font-size: 14px;
            }
        """)
        self.current_image = None
        self.original_pixmap = None

    def setImage(self, image):
        """
        Set an image to display (accepts cv2/numpy image)
        
        Args:
            image: numpy array representing the image
        """
        if image is None:
            self.setPlaceholder()
            return

        try:
            # Store the current image for potential future operations
            self.current_image = image

            # Convert cv2 image (BGR) to QImage (RGB)
            if len(image.shape) == 3:  # Color image
                if image.shape[2] == 3:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = image_rgb.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                elif image.shape[2] == 4:
                    # Handle RGBA
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    # Unsupported channel count
                    self.setPlaceholder()
                    return
            else:  # Grayscale image
                h, w = image.shape
                bytes_per_line = w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

            # Create pixmap and store original
            self.original_pixmap = QPixmap.fromImage(q_image)
            
            # Set the scaled pixmap
            self.setPixmap(self.scaled_pixmap(self.original_pixmap))
            
            # Set dark background for better image visibility
            self.setStyleSheet("""
                QLabel { 
                    background-color: #1e1e1e; 
                    border: 1px solid #444;
                }
            """)
            
            # Update title with image info
            if self.title_text:
                dimensions = f"({w}x{h})"
                self.title.setText(f"{self.title_text} {dimensions}")

        except Exception as e:
            print(f"Error setting image: {e}")
            self.setPlaceholder()

    def scaled_pixmap(self, pixmap):
        """Scale the pixmap to fit the label while preserving aspect ratio"""
        if pixmap is None:
            return None
            
        return pixmap.scaled(
            self.width() - 10, self.height() - 30,  # Account for margins and title
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )

    def resizeEvent(self, event):
        """Handle resize events to scale the image appropriately"""
        super().resizeEvent(event)
        
        if self.original_pixmap is not None:
            self.setPixmap(self.scaled_pixmap(self.original_pixmap))

    def getCurrentImage(self):
        """Get the currently displayed image as numpy array"""
        return self.current_image

    def hasImage(self):
        """Check if an image is currently loaded"""
        return self.current_image is not None

    def getImageDimensions(self):
        """Get the dimensions of the current image"""
        if self.current_image is not None:
            if len(self.current_image.shape) == 3:
                h, w, c = self.current_image.shape
                return w, h, c
            else:
                h, w = self.current_image.shape
                return w, h, 1
        return None

    def setTitle(self, title):
        """Update the title of the image label"""
        self.title_text = title
        if self.hasImage():
            w, h = self.getImageDimensions()[:2]
            self.title.setText(f"{title} ({w}x{h})")
        else:
            self.title.setText(title) 