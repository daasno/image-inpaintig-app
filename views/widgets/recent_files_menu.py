"""
Recent Files Menu and Advanced Image Management
"""
import os
from pathlib import Path
from PySide6.QtWidgets import (QMenu, QAction, QLabel, QHBoxLayout, QVBoxLayout,
                             QWidget, QFrame, QScrollArea, QGridLayout, 
                             QPushButton, QFileDialog, QMessageBox,
                             QListWidget, QListWidgetItem, QAbstractItemView)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QFileInfo
from PySide6.QtGui import QPixmap, QIcon, QFont, QCursor


class ThumbnailWidget(QWidget):
    """Widget to display image thumbnail with info"""
    
    clicked = Signal(str)  # Emits file path
    
    def __init__(self, file_path, thumbnail_size=64, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.thumbnail_size = thumbnail_size
        
        self.setup_ui()
        self.load_thumbnail()
        
        # Make it clickable
        self.setFixedSize(thumbnail_size + 40, thumbnail_size + 50)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
    def setup_ui(self):
        """Setup the thumbnail UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Thumbnail label
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
        """)
        layout.addWidget(self.thumbnail_label, alignment=Qt.AlignCenter)
        
        # File name label
        file_name = os.path.basename(self.file_path)
        if len(file_name) > 12:
            file_name = file_name[:9] + "..."
        
        self.name_label = QLabel(file_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #333;
                background: transparent;
                border: none;
            }
        """)
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)
        
    def load_thumbnail(self):
        """Load and display thumbnail"""
        try:
            if os.path.exists(self.file_path):
                pixmap = QPixmap(self.file_path)
                if not pixmap.isNull():
                    # Scale to thumbnail size
                    scaled_pixmap = pixmap.scaled(
                        self.thumbnail_size, self.thumbnail_size,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.thumbnail_label.setPixmap(scaled_pixmap)
                else:
                    self.set_error_thumbnail("Invalid image")
            else:
                self.set_error_thumbnail("File not found")
        except Exception as e:
            self.set_error_thumbnail(f"Error: {str(e)[:20]}")
    
    def set_error_thumbnail(self, error_text):
        """Set error thumbnail"""
        self.thumbnail_label.setText("❌")
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                border: 1px solid #dc3545;
                border-radius: 4px;
                background-color: #f8d7da;
                color: #721c24;
                font-size: 24px;
            }
        """)
        self.setToolTip(f"{self.file_path}\n{error_text}")
    
    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.LeftButton:
            if os.path.exists(self.file_path):
                self.clicked.emit(self.file_path)
            else:
                QMessageBox.warning(
                    self, 
                    "File Not Found", 
                    f"The file '{self.file_path}' could not be found.\n\n"
                    "It may have been moved or deleted."
                )
    
    def enterEvent(self, event):
        """Handle mouse enter (hover)"""
        self.setStyleSheet("""
            ThumbnailWidget {
                background-color: #e3f2fd;
                border: 1px solid #2196f3;
                border-radius: 4px;
            }
        """)
    
    def leaveEvent(self, event):
        """Handle mouse leave"""
        self.setStyleSheet("")


class RecentFilesPanel(QWidget):
    """Panel to display recent files with thumbnails"""
    
    file_selected = Signal(str)
    
    def __init__(self, title="Recent Files", parent=None):
        super().__init__(parent)
        self.title = title
        self.max_items = 12
        self.thumbnail_size = 64
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the recent files panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        clear_button.clicked.connect(self.clear_recent_files)
        header_layout.addWidget(clear_button)
        
        layout.addLayout(header_layout)
        
        # Scroll area for thumbnails
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
        """)
        
        # Container for thumbnails
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_container)
        self.thumbnail_layout.setSpacing(8)
        
        scroll_area.setWidget(self.thumbnail_container)
        layout.addWidget(scroll_area)
        
        # No files message
        self.no_files_label = QLabel("No recent files")
        self.no_files_label.setAlignment(Qt.AlignCenter)
        self.no_files_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-style: italic;
                padding: 20px;
                background: transparent;
            }
        """)
        self.no_files_label.hide()
        layout.addWidget(self.no_files_label)
        
    def update_recent_files(self, file_paths):
        """Update the display with recent files"""
        # Clear existing thumbnails
        self.clear_thumbnails()
        
        if not file_paths:
            self.show_no_files_message()
            return
        
        self.hide_no_files_message()
        
        # Add thumbnails (limit to max_items)
        files_to_show = file_paths[:self.max_items]
        
        columns = 4  # Number of columns in grid
        for i, file_path in enumerate(files_to_show):
            row = i // columns
            col = i % columns
            
            thumbnail = ThumbnailWidget(file_path, self.thumbnail_size)
            thumbnail.clicked.connect(self.file_selected.emit)
            
            self.thumbnail_layout.addWidget(thumbnail, row, col)
        
        # Add stretch to fill remaining space
        self.thumbnail_layout.setRowStretch(
            (len(files_to_show) // columns) + 1, 1
        )
    
    def clear_thumbnails(self):
        """Clear all thumbnail widgets"""
        while self.thumbnail_layout.count():
            child = self.thumbnail_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def show_no_files_message(self):
        """Show no files message"""
        self.no_files_label.show()
        self.thumbnail_container.hide()
    
    def hide_no_files_message(self):
        """Hide no files message"""
        self.no_files_label.hide()
        self.thumbnail_container.show()
    
    def clear_recent_files(self):
        """Clear all recent files"""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            f"Are you sure you want to clear all {self.title.lower()}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.clear_thumbnails()
            self.show_no_files_message()
            # Signal that recent files should be cleared
            self.file_selected.emit("")  # Empty string signals clear


class RecentFilesMenu(QMenu):
    """Enhanced recent files menu with thumbnails"""
    
    file_selected = Signal(str)
    
    def __init__(self, title="Recent Files", parent=None):
        super().__init__(title, parent)
        self.max_items = 10
        
        # Submenu for detailed view
        self.detailed_action = QAction("Show Recent Files Panel...", self)
        self.detailed_action.triggered.connect(self.show_detailed_panel)
        self.addAction(self.detailed_action)
        
        self.addSeparator()
        
        # Clear action
        self.clear_action = QAction("Clear Recent Files", self)
        self.clear_action.triggered.connect(self.clear_recent_files)
        
    def update_recent_files(self, file_paths):
        """Update menu with recent files"""
        # Remove existing file actions (keep detailed and clear actions)
        actions_to_remove = []
        for action in self.actions():
            if (action != self.detailed_action and 
                action != self.clear_action and 
                not action.isSeparator()):
                actions_to_remove.append(action)
        
        for action in actions_to_remove:
            self.removeAction(action)
        
        if not file_paths:
            # Add "No recent files" action
            no_files_action = QAction("No recent files", self)
            no_files_action.setEnabled(False)
            self.insertAction(self.clear_action, no_files_action)
            return
        
        # Add file actions (insert before clear action)
        files_to_show = file_paths[:self.max_items]
        
        for file_path in reversed(files_to_show):  # Reverse to maintain order
            file_name = os.path.basename(file_path)
            
            # Truncate long names
            if len(file_name) > 30:
                display_name = file_name[:27] + "..."
            else:
                display_name = file_name
            
            action = QAction(display_name, self)
            action.setToolTip(file_path)
            
            # Check if file exists and set icon accordingly
            if os.path.exists(file_path):
                action.triggered.connect(lambda checked, path=file_path: self.file_selected.emit(path))
            else:
                action.setText(f"{display_name} (missing)")
                action.setEnabled(False)
            
            self.insertAction(self.clear_action, action)
        
        # Ensure clear action is present
        if self.clear_action not in self.actions():
            self.addSeparator()
            self.addAction(self.clear_action)
    
    def show_detailed_panel(self):
        """Show detailed recent files panel"""
        # This will be implemented by the main window
        pass
    
    def clear_recent_files(self):
        """Clear recent files"""
        reply = QMessageBox.question(
            self.parent(),
            "Clear Recent Files",
            "Are you sure you want to clear all recent files?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.file_selected.emit("")  # Empty string signals clear


class ImageMetadataWidget(QWidget):
    """Widget to display image metadata and information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup metadata display UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Image Information")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        layout.addWidget(title_label)
        
        # Metadata frame
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
                padding: 8px;
            }
        """)
        
        self.metadata_layout = QVBoxLayout(frame)
        layout.addWidget(frame)
        
        # Initially empty
        self.clear_metadata()
        
    def update_metadata(self, file_path):
        """Update metadata display for given image"""
        self.clear_metadata()
        
        if not os.path.exists(file_path):
            self.add_metadata_item("Status", "File not found", "#dc3545")
            return
        
        try:
            # Basic file info
            file_info = QFileInfo(file_path)
            self.add_metadata_item("File Name", file_info.fileName())
            self.add_metadata_item("File Path", file_path)
            self.add_metadata_item("File Size", self.format_file_size(file_info.size()))
            self.add_metadata_item("Modified", file_info.lastModified().toString("yyyy-MM-dd hh:mm:ss"))
            
            # Image info
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.add_metadata_item("Dimensions", f"{pixmap.width()} × {pixmap.height()}")
                self.add_metadata_item("Aspect Ratio", f"{pixmap.width() / pixmap.height():.2f}")
                
                # Calculate megapixels
                megapixels = (pixmap.width() * pixmap.height()) / 1_000_000
                self.add_metadata_item("Megapixels", f"{megapixels:.1f} MP")
                
                # Color info
                if pixmap.depth() == 32:
                    color_info = "RGBA (32-bit)"
                elif pixmap.depth() == 24:
                    color_info = "RGB (24-bit)"
                elif pixmap.depth() == 8:
                    color_info = "Grayscale (8-bit)"
                else:
                    color_info = f"{pixmap.depth()}-bit"
                
                self.add_metadata_item("Color Format", color_info)
            else:
                self.add_metadata_item("Status", "Invalid image file", "#dc3545")
                
        except Exception as e:
            self.add_metadata_item("Error", str(e), "#dc3545")
    
    def add_metadata_item(self, key, value, color="#333"):
        """Add a metadata item"""
        item_layout = QHBoxLayout()
        
        key_label = QLabel(f"{key}:")
        key_label.setStyleSheet(f"font-weight: bold; color: #666; background: transparent;")
        key_label.setFixedWidth(100)
        item_layout.addWidget(key_label)
        
        value_label = QLabel(str(value))
        value_label.setStyleSheet(f"color: {color}; background: transparent;")
        value_label.setWordWrap(True)
        item_layout.addWidget(value_label, 1)
        
        # Create container widget
        container = QWidget()
        container.setLayout(item_layout)
        self.metadata_layout.addWidget(container)
    
    def clear_metadata(self):
        """Clear all metadata items"""
        while self.metadata_layout.count():
            child = self.metadata_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add empty message
        empty_label = QLabel("No image selected")
        empty_label.setStyleSheet("color: #999; font-style: italic; background: transparent;")
        empty_label.setAlignment(Qt.AlignCenter)
        self.metadata_layout.addWidget(empty_label)
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB" 