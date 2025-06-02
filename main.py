"""
Image Inpainting Application - Main Entry Point
Modular version with clean architecture
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import AppConstants
from controllers import AppController


def main():
    """Main application entry point"""
    try:
        # Create the application instance
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName(AppConstants.APP_NAME)
        app.setApplicationVersion(AppConstants.APP_VERSION)
        app.setOrganizationName("Image Processing Lab")
        app.setOrganizationDomain("imageprocessing.lab")
        
        # Set application icon if it exists
        icon_path = "logo.png"
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        
        # Apply Fusion style for consistent cross-platform appearance
        app.setStyle("Fusion")
        
        # Create and setup the main controller
        controller = AppController()
        
        # Show the main window
        controller.show()
        
        # Run the application event loop
        return app.exec()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 