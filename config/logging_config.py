"""
Enhanced Logging and Error Handling System
"""
import logging
import logging.handlers
import sys
import traceback
import functools
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime
from enum import Enum

from .settings import AppSettings, AppConstants


class LogCategory(Enum):
    """Log categories for better organization"""
    GENERAL = "general"
    UI = "ui"
    PROCESSING = "processing"
    MEMORY = "memory"
    CONFIG = "config"
    PERFORMANCE = "performance"
    ERROR = "error"


class AppLogger:
    """Centralized logging system for the application"""
    
    _loggers = {}
    _initialized = False
    _settings = None
    
    @classmethod
    def initialize(cls, settings: AppSettings):
        """Initialize the logging system"""
        if cls._initialized:
            return
        
        cls._settings = settings
        
        # Create logs directory
        logs_dir = settings.get_logs_directory()
        
        # Configure root logger
        root_logger = logging.getLogger('ImageInpainting')
        root_logger.setLevel(getattr(logging, settings.log_level))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            AppConstants.LOG_FORMAT,
            datefmt=AppConstants.LOG_DATE_FORMAT
        )
        
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler with rotation
        if settings.log_to_file:
            log_file = logs_dir / 'inpainting.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=settings.log_file_max_size_mb * 1024 * 1024,
                backupCount=settings.log_file_backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
        
        # Console handler
        if settings.log_console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, settings.log_level))
            root_logger.addHandler(console_handler)
        
        # Create category-specific log files
        cls._create_category_loggers(logs_dir, file_formatter)
        
        cls._initialized = True
        
        # Log initialization
        logger = cls.get_logger(LogCategory.GENERAL)
        logger.info(f"Logging system initialized - Level: {settings.log_level}")
        logger.info(f"Application version: {AppConstants.APP_VERSION}")
    
    @classmethod
    def _create_category_loggers(cls, logs_dir: Path, formatter: logging.Formatter):
        """Create specialized loggers for different categories"""
        for category in LogCategory:
            logger_name = f'ImageInpainting.{category.value}'
            logger = logging.getLogger(logger_name)
            
            if cls._settings.log_to_file:
                log_file = logs_dir / f'{category.value}.log'
                handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=5 * 1024 * 1024,  # 5MB per category
                    backupCount=3,
                    encoding='utf-8'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            cls._loggers[category] = logger
    
    @classmethod
    def get_logger(cls, category: LogCategory = LogCategory.GENERAL) -> logging.Logger:
        """Get a logger for a specific category"""
        if not cls._initialized:
            # Fallback to basic logging if not initialized
            return logging.getLogger(f'ImageInpainting.{category.value}')
        
        return cls._loggers.get(category, logging.getLogger('ImageInpainting'))
    
    @classmethod
    def log_exception(cls, exception: Exception, context: str = "", 
                     category: LogCategory = LogCategory.ERROR):
        """Log an exception with full traceback and context"""
        logger = cls.get_logger(category)
        
        error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        logger.error(f"EXCEPTION [{error_id}] {context}")
        logger.error(f"Exception type: {type(exception).__name__}")
        logger.error(f"Exception message: {str(exception)}")
        logger.error("Full traceback:")
        
        # Log each line of the traceback
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"  {line}")
        
        return error_id
    
    @classmethod
    def log_memory_usage(cls, operation: str, memory_mb: float):
        """Log memory usage for monitoring"""
        logger = cls.get_logger(LogCategory.MEMORY)
        logger.info(f"{operation}: {memory_mb:.2f} MB")
    
    @classmethod
    def log_performance(cls, operation: str, duration_seconds: float, details: str = ""):
        """Log performance metrics"""
        logger = cls.get_logger(LogCategory.PERFORMANCE)
        logger.info(f"{operation}: {duration_seconds:.3f}s {details}")
    
    @classmethod
    def log_user_action(cls, action: str, details: str = ""):
        """Log user actions for debugging"""
        logger = cls.get_logger(LogCategory.UI)
        logger.info(f"User action: {action} {details}")
    
    @classmethod
    def cleanup_old_logs(cls, days_to_keep: int = 30):
        """Clean up old log files"""
        if not cls._settings:
            return
        
        logs_dir = cls._settings.get_logs_directory()
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        cleaned_count = 0
        for log_file in logs_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger = cls.get_logger(LogCategory.GENERAL)
                    logger.warning(f"Failed to delete old log file {log_file}: {e}")
        
        if cleaned_count > 0:
            logger = cls.get_logger(LogCategory.GENERAL)
            logger.info(f"Cleaned up {cleaned_count} old log files")


def log_exceptions(category: LogCategory = LogCategory.ERROR):
    """Decorator to automatically log exceptions in functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"in function {func.__name__}"
                if args and hasattr(args[0], '__class__'):
                    context = f"in {args[0].__class__.__name__}.{func.__name__}"
                
                error_id = AppLogger.log_exception(e, context, category)
                
                # Re-raise the exception with additional context
                raise Exception(f"Error {error_id}: {str(e)}") from e
        
        return wrapper
    return decorator


def log_performance(category: LogCategory = LogCategory.PERFORMANCE):
    """Decorator to log function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                func_name = func.__name__
                if args and hasattr(args[0], '__class__'):
                    func_name = f"{args[0].__class__.__name__}.{func.__name__}"
                
                AppLogger.log_performance(func_name, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                AppLogger.log_performance(f"{func.__name__} (FAILED)", duration)
                raise
        
        return wrapper
    return decorator


class ErrorReporter:
    """User-friendly error reporting system"""
    
    @staticmethod
    def get_user_friendly_message(exception: Exception) -> str:
        """Convert technical exceptions to user-friendly messages"""
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        # Common error patterns and their user-friendly messages
        if "CUDA" in error_msg:
            return "GPU processing is not available. Please try using CPU mode or check your CUDA installation."
        
        elif "FileNotFoundError" in error_type:
            return "The requested file could not be found. Please check the file path and try again."
        
        elif "PermissionError" in error_type:
            return "Permission denied. Please check that you have the necessary permissions to access this file or directory."
        
        elif "MemoryError" in error_type or "out of memory" in error_msg.lower():
            return "Not enough memory to complete this operation. Try using a smaller image or switching to Conservative memory mode."
        
        elif "ValueError" in error_type and "image" in error_msg.lower():
            return "The image file appears to be corrupted or in an unsupported format. Please try a different image."
        
        elif "ConnectionError" in error_type or "network" in error_msg.lower():
            return "Network connection error. Please check your internet connection and try again."
        
        elif "timeout" in error_msg.lower():
            return "The operation timed out. This might be due to a large image or system performance. Please try again."
        
        else:
            # Generic error message
            return f"An unexpected error occurred: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}"
    
    @staticmethod
    def should_show_technical_details(settings: AppSettings) -> bool:
        """Determine if technical details should be shown to the user"""
        return settings.debug_mode
    
    @staticmethod
    def format_error_report(exception: Exception, error_id: str, settings: AppSettings) -> tuple[str, str]:
        """Format error report for user display"""
        user_message = ErrorReporter.get_user_friendly_message(exception)
        
        if ErrorReporter.should_show_technical_details(settings):
            technical_details = f"""
Technical Details:
Error ID: {error_id}
Error Type: {type(exception).__name__}
Error Message: {str(exception)}

This information has been logged for debugging purposes.
"""
        else:
            technical_details = f"""
Error ID: {error_id}

This error has been logged. If the problem persists, please report this error ID.
"""
        
        return user_message, technical_details


class MemoryMonitor:
    """Monitor and log memory usage"""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics"""
        import psutil
        import gc
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident memory
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual memory
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'total_mb': psutil.virtual_memory().total / (1024 * 1024)
            }
        except ImportError:
            # psutil not available, use basic info
            return {
                'gc_objects': len(gc.get_objects()),
                'gc_collections': sum(gc.get_stats())
            }
    
    @staticmethod
    def log_memory_status(operation: str = ""):
        """Log current memory status"""
        memory_info = MemoryMonitor.get_memory_usage()
        
        if 'rss_mb' in memory_info:
            AppLogger.log_memory_usage(
                f"{operation} - RSS", memory_info['rss_mb']
            )
            AppLogger.log_memory_usage(
                f"{operation} - Available", memory_info['available_mb']
            )
        else:
            AppLogger.get_logger(LogCategory.MEMORY).info(
                f"{operation} - GC Objects: {memory_info.get('gc_objects', 'unknown')}"
            )
    
    @staticmethod
    def check_memory_limits(image_size: tuple, settings: AppSettings) -> tuple[bool, str]:
        """Check if image size is within memory limits"""
        height, width = image_size[:2]
        channels = image_size[2] if len(image_size) > 2 else 1
        
        total_pixels = height * width
        memory_limits = settings.get_memory_limits()
        
        estimated_memory_mb = (total_pixels * channels * 4) / (1024 * 1024)  # 4 bytes per pixel (float32)
        
        if total_pixels > memory_limits['max_image_pixels']:
            return False, f"Image too large: {width}x{height} ({total_pixels:,} pixels). Maximum allowed: {memory_limits['max_image_pixels']:,} pixels."
        
        if total_pixels > memory_limits['warning_threshold']:
            return True, f"Large image warning: {width}x{height} will use approximately {estimated_memory_mb:.1f} MB of memory."
        
        return True, "" 