"""
Enhanced Progress Dialog with Detailed Feedback and Visual Improvements
"""
import time
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTextEdit, QFrame,
                             QGridLayout, QWidget, QGroupBox)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor, QPalette, QMovie


class AnimatedProgressBar(QProgressBar):
    """Enhanced progress bar with smooth animations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Animation for smooth progress updates
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(300)  # 300ms for smooth transition
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Custom styling
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
                background-color: #f8f9fa;
                color: #333;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #45a049, stop:1 #4CAF50);
                border-radius: 6px;
                margin: 1px;
            }
        """)
    
    def set_animated_value(self, value):
        """Set value with smooth animation"""
        self.animation.setStartValue(self.value())
        self.animation.setEndValue(value)
        self.animation.start()


class ProcessingStatsWidget(QWidget):
    """Widget to display processing statistics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Statistics
        self.start_time = None
        self.last_update_time = None
        self.processed_pixels = 0
        self.total_pixels = 0
        self.processing_speed = 0.0
        
    def setup_ui(self):
        """Setup the statistics UI"""
        layout = QGridLayout(self)
        
        # Time elapsed
        layout.addWidget(QLabel("Time Elapsed:"), 0, 0)
        self.time_elapsed_label = QLabel("00:00")
        self.time_elapsed_label.setStyleSheet("font-weight: bold; color: #007acc;")
        layout.addWidget(self.time_elapsed_label, 0, 1)
        
        # Estimated time remaining
        layout.addWidget(QLabel("Time Remaining:"), 0, 2)
        self.time_remaining_label = QLabel("--:--")
        self.time_remaining_label.setStyleSheet("font-weight: bold; color: #ff9500;")
        layout.addWidget(self.time_remaining_label, 0, 3)
        
        # Processing speed
        layout.addWidget(QLabel("Speed:"), 1, 0)
        self.speed_label = QLabel("-- pixels/sec")
        self.speed_label.setStyleSheet("font-weight: bold; color: #28a745;")
        layout.addWidget(self.speed_label, 1, 1)
        
        # Pixels processed
        layout.addWidget(QLabel("Progress:"), 1, 2)
        self.pixels_label = QLabel("0 / 0 pixels")
        self.pixels_label.setStyleSheet("font-weight: bold; color: #6f42c1;")
        layout.addWidget(self.pixels_label, 1, 3)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(1000)  # Update every second
        
    def start_processing(self, total_pixels):
        """Start processing with given total pixels"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processed_pixels = 0
        self.total_pixels = total_pixels
        self.processing_speed = 0.0
        
    def update_progress(self, current_pixels):
        """Update progress with current pixel count"""
        if self.start_time is None:
            return
        
        current_time = time.time()
        self.processed_pixels = current_pixels
        
        # Calculate processing speed
        if self.last_update_time and current_time > self.last_update_time:
            time_delta = current_time - self.last_update_time
            if time_delta > 0:
                # Smooth the speed calculation
                new_speed = current_pixels / (current_time - self.start_time)
                self.processing_speed = 0.7 * self.processing_speed + 0.3 * new_speed
        
        self.last_update_time = current_time
        
    def update_stats(self):
        """Update the displayed statistics"""
        if self.start_time is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Format elapsed time
        elapsed_str = self.format_time(elapsed_time)
        self.time_elapsed_label.setText(elapsed_str)
        
        # Update pixels display
        self.pixels_label.setText(f"{self.processed_pixels:,} / {self.total_pixels:,} pixels")
        
        # Update speed
        if self.processing_speed > 0:
            self.speed_label.setText(f"{self.processing_speed:.0f} pixels/sec")
            
            # Calculate remaining time
            remaining_pixels = self.total_pixels - self.processed_pixels
            if remaining_pixels > 0:
                remaining_time = remaining_pixels / self.processing_speed
                remaining_str = self.format_time(remaining_time)
                self.time_remaining_label.setText(remaining_str)
            else:
                self.time_remaining_label.setText("00:00")
        else:
            self.speed_label.setText("-- pixels/sec")
            self.time_remaining_label.setText("--:--")
    
    def format_time(self, seconds):
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


class EnhancedProgressDialog(QDialog):
    """Enhanced progress dialog with detailed feedback"""
    
    # Signals
    cancel_requested = Signal()
    details_toggled = Signal(bool)
    
    def __init__(self, title="Processing", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(500, 350)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint)
        
        # State
        self.is_cancellable = True
        self.show_details = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the progress dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header with title and status
        header_frame = self.create_header()
        layout.addWidget(header_frame)
        
        # Main progress section
        progress_frame = self.create_progress_section()
        layout.addWidget(progress_frame)
        
        # Statistics section
        stats_frame = self.create_stats_section()
        layout.addWidget(stats_frame)
        
        # Details section (initially hidden)
        self.details_frame = self.create_details_section()
        self.details_frame.hide()
        layout.addWidget(self.details_frame)
        
        # Button section
        button_frame = self.create_button_section()
        layout.addWidget(button_frame)
        
    def create_header(self):
        """Create the header section"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
                color: white;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        
        # Main title
        self.title_label = QLabel("Processing Image")
        self.title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: white; 
            background: transparent;
        """)
        layout.addWidget(self.title_label)
        
        # Status subtitle
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            font-size: 13px; 
            color: #e6f2ff; 
            background: transparent;
        """)
        layout.addWidget(self.status_label)
        
        return frame
    
    def create_progress_section(self):
        """Create the progress bar section"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        
        # Progress label
        self.progress_label = QLabel("0% Complete")
        self.progress_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        # Progress bar
        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Sub-progress info
        self.sub_progress_label = QLabel("Preparing...")
        self.sub_progress_label.setStyleSheet("color: #666; font-size: 11px;")
        self.sub_progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sub_progress_label)
        
        return frame
    
    def create_stats_section(self):
        """Create the statistics section"""
        group_box = QGroupBox("Processing Statistics")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #f8f9fa;
            }
        """)
        
        self.stats_widget = ProcessingStatsWidget()
        
        layout = QVBoxLayout(group_box)
        layout.addWidget(self.stats_widget)
        
        return group_box
    
    def create_details_section(self):
        """Create the details section"""
        group_box = QGroupBox("Processing Details")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(group_box)
        
        # Details text area
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.details_text)
        
        return group_box
    
    def create_button_section(self):
        """Create the button section"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Details toggle button
        self.details_button = QPushButton("Show Details")
        self.details_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.details_button.clicked.connect(self.toggle_details)
        layout.addWidget(self.details_button)
        
        layout.addStretch()
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.cancel_button)
        
        return frame
    
    def set_title(self, title):
        """Set the dialog title"""
        self.title_label.setText(title)
    
    def set_status(self, status):
        """Set the status message"""
        self.status_label.setText(status)
    
    def set_progress(self, value):
        """Set progress value (0-100)"""
        self.progress_bar.set_animated_value(value)
        self.progress_label.setText(f"{value}% Complete")
    
    def set_sub_progress(self, text):
        """Set sub-progress text"""
        self.sub_progress_label.setText(text)
    
    def add_detail(self, detail):
        """Add a detail message"""
        current_time = time.strftime("%H:%M:%S")
        formatted_detail = f"[{current_time}] {detail}"
        self.details_text.append(formatted_detail)
        
        # Auto-scroll to bottom
        from PySide6.QtGui import QTextCursor
        cursor = self.details_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.details_text.setTextCursor(cursor)
    
    def start_processing(self, total_pixels=0):
        """Start processing"""
        if total_pixels > 0:
            self.stats_widget.start_processing(total_pixels)
        self.add_detail("Processing started")
    
    def update_progress_pixels(self, current_pixels, total_pixels=None):
        """Update progress based on pixel count"""
        if total_pixels:
            self.stats_widget.total_pixels = total_pixels
        
        self.stats_widget.update_progress(current_pixels)
        
        # Update main progress bar
        if self.stats_widget.total_pixels > 0:
            progress_percent = (current_pixels / self.stats_widget.total_pixels) * 100
            self.set_progress(min(100, int(progress_percent)))
    
    def set_cancellable(self, cancellable):
        """Set whether the operation can be cancelled"""
        self.is_cancellable = cancellable
        self.cancel_button.setEnabled(cancellable)
    
    def toggle_details(self):
        """Toggle details visibility"""
        self.show_details = not self.show_details
        
        if self.show_details:
            self.details_frame.show()
            self.details_button.setText("Hide Details")
            self.setFixedSize(500, 500)  # Expand dialog
        else:
            self.details_frame.hide()
            self.details_button.setText("Show Details")
            self.setFixedSize(500, 350)  # Contract dialog
        
        self.details_toggled.emit(self.show_details)
    
    def finish_processing(self, success=True, message=""):
        """Finish processing"""
        if success:
            self.set_progress(100)
            self.set_status("Processing completed successfully!")
            self.add_detail("Processing completed successfully")
        else:
            self.set_status(f"Processing failed: {message}")
            self.add_detail(f"Processing failed: {message}")
        
        # Change cancel button to close
        self.cancel_button.setText("Close")
        self.cancel_button.setStyleSheet("""
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