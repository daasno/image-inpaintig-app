"""
Views package - UI components and widgets
Enhanced with UX/UI improvements
"""

# Main window - using batch-enabled version
from .main_window_batch import BatchEnabledMainWindow as MainWindow
from .main_window import EnhancedMainWindow

# Original widgets
from .widgets.image_label import ImageLabel
from .widgets.control_panel import ControlPanel

# Enhanced UX widgets - conditional imports
enhanced_widgets = []

try:
    from .widgets.enhanced_image_label import ImageViewerWidget, ComparisonViewWidget, ZoomableImageLabel
    enhanced_widgets.extend(['ImageViewerWidget', 'ComparisonViewWidget', 'ZoomableImageLabel'])
except ImportError:
    pass

try:
    from .widgets.enhanced_progress_dialog import EnhancedProgressDialog, AnimatedProgressBar, ProcessingStatsWidget
    enhanced_widgets.extend(['EnhancedProgressDialog', 'AnimatedProgressBar', 'ProcessingStatsWidget'])
except ImportError:
    pass

try:
    from .widgets.recent_files_menu import (RecentFilesMenu, RecentFilesPanel, 
                                           ThumbnailWidget, ImageMetadataWidget)
    enhanced_widgets.extend(['RecentFilesMenu', 'RecentFilesPanel', 'ThumbnailWidget', 'ImageMetadataWidget'])
except ImportError:
    pass

try:
    from .widgets.batch_panel import BatchPanel
    enhanced_widgets.append('BatchPanel')
except ImportError:
    pass

try:
    from .welcome_dialog import WelcomeDialog
    enhanced_widgets.append('WelcomeDialog')
except ImportError:
    pass

# Export all available widgets
__all__ = [
    # Main interface
    'MainWindow',
    'EnhancedMainWindow',
    
    # Core widgets
    'ImageLabel',
    'ControlPanel',
] + enhanced_widgets 