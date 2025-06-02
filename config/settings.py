"""
Application Configuration and Settings Management with Enhanced Features
"""
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from enum import Enum


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MemoryMode(Enum):
    """Memory management modes"""
    CONSERVATIVE = "conservative"  # Strict limits, auto-resize
    BALANCED = "balanced"         # Moderate limits, warnings
    PERFORMANCE = "performance"   # Higher limits, manual control


@dataclass
class AppSettings:
    """Application settings with enhanced configuration management"""
    # Configuration metadata
    config_version: str = "1.1.0"
    last_updated: str = ""
    
    # Processing parameters
    default_patch_size: int = 9
    default_p_value: float = 1.0
    preferred_implementation: str = "GPU"  # "CPU" or "GPU"
    
    # UI preferences
    window_width: int = 1200
    window_height: int = 600
    window_x: int = -1  # -1 means center
    window_y: int = -1  # -1 means center
    last_image_directory: str = ""
    last_mask_directory: str = ""
    last_save_directory: str = ""
    
    # Recent files (up to 10)
    recent_images: List[str] = field(default_factory=list)
    recent_masks: List[str] = field(default_factory=list)
    
    # Advanced processing settings
    max_image_size: int = 2048  # Max dimension for memory management
    memory_mode: str = MemoryMode.BALANCED.value
    auto_resize_large_images: bool = True
    gpu_memory_fraction: float = 0.8  # Use 80% of available GPU memory
    enable_memory_monitoring: bool = True
    
    # UI/UX settings
    show_tooltips: bool = True
    show_welcome_dialog: bool = True
    auto_save_settings: bool = True
    confirm_destructive_actions: bool = True
    show_progress_details: bool = True
    
    # Logging configuration
    log_level: str = LogLevel.INFO.value
    log_to_file: bool = True
    log_file_max_size_mb: int = 10
    log_file_backup_count: int = 5
    log_console_output: bool = True
    
    # Performance settings
    num_worker_threads: int = 1  # For future batch processing
    enable_performance_monitoring: bool = False
    cache_enabled: bool = True
    cache_size_mb: int = 100
    
    # Debug and development
    debug_mode: bool = False
    profile_performance: bool = False
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        if not self.recent_images:
            self.recent_images = []
        if not self.recent_masks:
            self.recent_masks = []
        
        # Set last updated timestamp
        if not self.last_updated:
            from datetime import datetime
            self.last_updated = datetime.now().isoformat()
        
        # Validate settings
        self._validate_and_fix_settings()
    
    def _validate_and_fix_settings(self):
        """Validate and auto-fix invalid settings"""
        # Clamp numeric values to valid ranges
        self.default_patch_size = max(3, min(21, self.default_patch_size))
        if self.default_patch_size % 2 == 0:
            self.default_patch_size += 1  # Ensure odd
        
        self.default_p_value = max(0.1, min(10.0, self.default_p_value))
        self.window_width = max(800, min(3840, self.window_width))
        self.window_height = max(600, min(2160, self.window_height))
        self.max_image_size = max(512, min(8192, self.max_image_size))
        self.gpu_memory_fraction = max(0.1, min(1.0, self.gpu_memory_fraction))
        
        # Validate enum values
        if self.memory_mode not in [mode.value for mode in MemoryMode]:
            self.memory_mode = MemoryMode.BALANCED.value
        
        if self.log_level not in [level.value for level in LogLevel]:
            self.log_level = LogLevel.INFO.value
        
        # Clean up recent files (remove non-existent files)
        self.recent_images = [f for f in self.recent_images if os.path.exists(f)][:10]
        self.recent_masks = [f for f in self.recent_masks if os.path.exists(f)][:10]
        
        # Validate directories
        for dir_attr in ['last_image_directory', 'last_mask_directory', 'last_save_directory']:
            dir_path = getattr(self, dir_attr)
            if dir_path and not os.path.exists(dir_path):
                setattr(self, dir_attr, "")
    
    @classmethod
    def get_settings_path(cls) -> Path:
        """Get the path to the settings file"""
        # Use user's app data directory
        if os.name == 'nt':  # Windows
            app_data = os.getenv('APPDATA', os.path.expanduser('~'))
        else:  # Linux/Mac
            app_data = os.path.expanduser('~/.config')
        
        settings_dir = Path(app_data) / 'ImageInpaintingApp'
        settings_dir.mkdir(exist_ok=True)
        return settings_dir / 'settings.json'
    
    @classmethod
    def get_logs_directory(cls) -> Path:
        """Get the logs directory"""
        settings_path = cls.get_settings_path()
        logs_dir = settings_path.parent / 'logs'
        logs_dir.mkdir(exist_ok=True)
        return logs_dir
    
    @classmethod
    def get_cache_directory(cls) -> Path:
        """Get the cache directory"""
        settings_path = cls.get_settings_path()
        cache_dir = settings_path.parent / 'cache'
        cache_dir.mkdir(exist_ok=True)
        return cache_dir
    
    @classmethod
    def load(cls) -> 'AppSettings':
        """Load settings from file with migration and error recovery"""
        settings_file = cls.get_settings_path()
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle configuration migration
                settings = cls._migrate_config(data)
                
                # Validate loaded settings
                settings._validate_and_fix_settings()
                
                return settings
                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # Try to backup corrupted config and create new one
                backup_path = settings_file.with_suffix('.json.backup')
                try:
                    settings_file.rename(backup_path)
                    print(f"Warning: Corrupted settings file backed up to {backup_path}")
                except Exception:
                    pass
                
                print(f"Warning: Could not load settings ({e}). Using defaults.")
                
            except Exception as e:
                print(f"Unexpected error loading settings ({e}). Using defaults.")
        
        # Return default settings
        return cls()
    
    @classmethod
    def _migrate_config(cls, data: Dict[str, Any]) -> 'AppSettings':
        """Migrate configuration from older versions"""
        version = data.get('config_version', '1.0.0')
        
        # Migration logic for different versions
        if version < '1.1.0':
            # Add new fields with defaults
            data.setdefault('memory_mode', MemoryMode.BALANCED.value)
            data.setdefault('gpu_memory_fraction', 0.8)
            data.setdefault('enable_memory_monitoring', True)
            data.setdefault('log_level', LogLevel.INFO.value)
            data.setdefault('log_to_file', True)
            data.setdefault('window_x', -1)
            data.setdefault('window_y', -1)
            data.setdefault('num_worker_threads', 1)
            data.setdefault('debug_mode', False)
            
            # Update version
            data['config_version'] = '1.1.0'
        
        return cls(**data)
    
    def save(self):
        """Save current settings to file with error handling"""
        if not self.auto_save_settings:
            return
        
        settings_file = self.get_settings_path()
        
        try:
            # Update timestamp
            from datetime import datetime
            self.last_updated = datetime.now().isoformat()
            
            # Create backup of existing file
            if settings_file.exists():
                backup_path = settings_file.with_suffix('.json.bak')
                settings_file.rename(backup_path)
            
            # Write new settings
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            
            # Remove backup if write was successful
            backup_path = settings_file.with_suffix('.json.bak')
            if backup_path.exists():
                backup_path.unlink()
                
        except Exception as e:
            # Restore backup if write failed
            backup_path = settings_file.with_suffix('.json.bak')
            if backup_path.exists():
                backup_path.rename(settings_file)
            
            print(f"Warning: Could not save settings ({e})")
    
    def add_recent_image(self, file_path: str):
        """Add a file to recent images list"""
        if not os.path.exists(file_path):
            return
        
        if file_path in self.recent_images:
            self.recent_images.remove(file_path)
        
        self.recent_images.insert(0, file_path)
        self.recent_images = self.recent_images[:10]  # Keep only last 10
    
    def add_recent_mask(self, file_path: str):
        """Add a file to recent masks list"""
        if not os.path.exists(file_path):
            return
        
        if file_path in self.recent_masks:
            self.recent_masks.remove(file_path)
        
        self.recent_masks.insert(0, file_path)
        self.recent_masks = self.recent_masks[:10]  # Keep only last 10
    
    def get_memory_limits(self) -> Dict[str, int]:
        """Get memory limits based on current mode"""
        if self.memory_mode == MemoryMode.CONSERVATIVE.value:
            return {
                'max_image_pixels': 2048 * 2048,
                'max_dimension': 2048,
                'warning_threshold': 1024 * 1024
            }
        elif self.memory_mode == MemoryMode.PERFORMANCE.value:
            return {
                'max_image_pixels': 6144 * 6144,
                'max_dimension': 6144,
                'warning_threshold': 4096 * 4096
            }
        else:  # BALANCED
            return {
                'max_image_pixels': 4096 * 4096,
                'max_dimension': 4096,
                'warning_threshold': 2048 * 2048
            }
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        default_settings = AppSettings()
        for field_name, field_def in self.__dataclass_fields__.items():
            if hasattr(default_settings, field_name):
                setattr(self, field_name, getattr(default_settings, field_name))
        
        self._validate_and_fix_settings()
    
    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate imported data
            imported_settings = self._migrate_config(data)
            
            # Copy valid settings
            for field_name in self.__dataclass_fields__.keys():
                if hasattr(imported_settings, field_name):
                    setattr(self, field_name, getattr(imported_settings, field_name))
            
            self._validate_and_fix_settings()
            return True
            
        except Exception as e:
            print(f"Failed to import settings: {e}")
            return False


class AppConstants:
    """Application constants with enhanced configuration"""
    APP_NAME = "Image Inpainting Application"
    APP_VERSION = "1.1.0"
    
    # Supported file formats
    IMAGE_FORMATS = "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
    CONFIG_FORMATS = "JSON Files (*.json);;All Files (*)"
    
    # UI Constants
    MIN_PATCH_SIZE = 3
    MAX_PATCH_SIZE = 21
    MIN_P_VALUE = 0.1
    MAX_P_VALUE = 10.0
    
    # Memory constants (in bytes)
    BYTES_PER_PIXEL_RGB = 3
    BYTES_PER_PIXEL_GRAY = 1
    MEGABYTE = 1024 * 1024
    GIGABYTE = 1024 * MEGABYTE
    
    # Processing limits by memory mode
    MEMORY_LIMITS = {
        MemoryMode.CONSERVATIVE.value: {
            'max_pixels': 4 * MEGABYTE,    # 4MP
            'max_dimension': 2048,
            'warning_pixels': 2 * MEGABYTE  # 2MP
        },
        MemoryMode.BALANCED.value: {
            'max_pixels': 16 * MEGABYTE,   # 16MP
            'max_dimension': 4096,
            'warning_pixels': 8 * MEGABYTE  # 8MP
        },
        MemoryMode.PERFORMANCE.value: {
            'max_pixels': 64 * MEGABYTE,   # 64MP
            'max_dimension': 8192,
            'warning_pixels': 32 * MEGABYTE # 32MP
        }
    }
    
    # For backward compatibility with reverted code
    MAX_SAFE_IMAGE_PIXELS = 4096 * 4096  # 16MP
    
    # Default directories
    DEFAULT_DIRECTORIES = {
        'images': ['Pictures', 'Documents', 'Desktop'],
        'downloads': ['Downloads'],
    }
    
    # Styling with enhanced themes
    MAIN_STYLE = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QPushButton {
            padding: 5px 15px;
            border-radius: 3px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QPushButton:disabled {
            color: #999;
            background-color: #f5f5f5;
        }
        QProgressBar {
            border: 1px solid #ccc;
            border-radius: 3px;
            text-align: center;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #5c85d6;
            border-radius: 2px;
        }
        QToolTip {
            background-color: #ffffcc;
            border: 1px solid #999;
            padding: 5px;
            border-radius: 3px;
        }
    """
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S' 