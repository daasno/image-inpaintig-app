"""
Controllers for the Image Inpainting Application
"""

from .app_controller import AppController
from .batch_controller import BatchAppController

# Use batch controller as default
AppController = BatchAppController

__all__ = ['AppController', 'BatchAppController'] 