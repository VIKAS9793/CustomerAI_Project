import logging
import os
import sys
import traceback
import json
from datetime import datetime

class Logger:
    """
    Custom logger for CustomerAI Insights Platform
    
    Features:
    - Configurable log levels
    - Console and file output
    - JSON structured logging for production
    - Request tracing via correlation IDs
    - Integration with monitoring systems
    """
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __init__(self, name, log_level=None, log_file=None, json_format=False):
        """
        Initialize logger with name and configuration
        
        Args:
            name (str): Logger name (typically module name)
            log_level (int, optional): Logging level. Defaults to INFO or env setting.
            log_file (str, optional): Path to log file. Defaults to None (console only).
            json_format (bool, optional): Use JSON format for logs. Defaults to False.
        """
        self.logger = logging.getLogger(name)
        
        # Get log level from environment or use default
        self.log_level = log_level or self._get_log_level_from_env() or logging.INFO
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates when reusing loggers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # File handler if log_file is specified
        handlers = [console_handler]
        if log_file:
            # Make sure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            handlers.append(file_handler)
        
        # Set formatter based on format type
        if json_format:
            formatter = self._json_formatter
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Apply formatter to all handlers
        for handler in handlers:
            if json_format and callable(formatter):
                handler.setFormatter(logging.Formatter('%(message)s'))
            else:
                handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Store settings for later use
        self.json_format = json_format
        self.correlation_id = None
    
    def _get_log_level_from_env(self):
        """Get log level from environment variable"""
        env_level = os.environ.get('LOG_LEVEL', '').upper()
        if env_level == 'DEBUG':
            return logging.DEBUG
        elif env_level == 'INFO':
            return logging.INFO
        elif env_level == 'WARNING':
            return logging.WARNING
        elif env_level == 'ERROR':
            return logging.ERROR
        elif env_level == 'CRITICAL':
            return logging.CRITICAL
        return None
    
    def _json_formatter(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add correlation ID if set
        if self.correlation_id:
            log_data['correlation_id'] = self.correlation_id
        
        return json.dumps(log_data)
    
    def set_correlation_id(self, correlation_id):
        """
        Set correlation ID for request tracing
        
        Args:
            correlation_id (str): Unique identifier for request tracing
        """
        self.correlation_id = correlation_id
    
    def debug(self, message, *args, **kwargs):
        """Log debug message"""
        if self.json_format:
            self._log_json(logging.DEBUG, message, *args, **kwargs)
        else:
            self.logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log info message"""
        if self.json_format:
            self._log_json(logging.INFO, message, *args, **kwargs)
        else:
            self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message"""
        if self.json_format:
            self._log_json(logging.WARNING, message, *args, **kwargs)
        else:
            self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log error message"""
        if self.json_format:
            self._log_json(logging.ERROR, message, *args, **kwargs)
        else:
            self.logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Log critical message"""
        if self.json_format:
            self._log_json(logging.CRITICAL, message, *args, **kwargs)
        else:
            self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message, *args, **kwargs):
        """Log exception with traceback"""
        if self.json_format:
            self._log_json(logging.ERROR, message, exc_info=True, *args, **kwargs)
        else:
            self.logger.exception(message, *args, **kwargs)
    
    def _log_json(self, level, message, *args, **kwargs):
        """Log message in JSON format"""
        # Format message with args if any
        if args:
            message = message % args
        
        # Extract extra fields
        extra_fields = kwargs.get('extra', {})
        
        # Build log data
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': logging.getLevelName(level),
            'name': self.logger.name,
            'message': message
        }
        
        # Add correlation ID if set
        if self.correlation_id:
            log_data['correlation_id'] = self.correlation_id
        
        # Add extra fields
        log_data.update(extra_fields)
        
        # Add exception info if present
        if kwargs.get('exc_info'):
            exc_info = kwargs['exc_info']
            if exc_info is True:
                exc_info = sys.exc_info()
            if exc_info:
                log_data['exception'] = {
                    'type': exc_info[0].__name__,
                    'message': str(exc_info[1]),
                    'traceback': traceback.format_exception(*exc_info)
                }
        
        # Log the JSON data
        self.logger.log(level, json.dumps(log_data))

# Create a default logger for import
default_logger = Logger('customerai', log_file='logs/application.log')

def get_logger(name, **kwargs):
    """
    Get a configured logger instance
    
    Args:
        name (str): Logger name
        **kwargs: Additional configuration (log_level, log_file, json_format)
    
    Returns:
        Logger: Configured logger instance
    """
    return Logger(name, **kwargs) 