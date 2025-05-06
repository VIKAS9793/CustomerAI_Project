import json
import os
from typing import Any, Dict, Optional


class Config:
    """Base configuration class"""

    # Application settings
    APP_NAME = "CustomerAI Insights Platform"
    APP_VERSION = "1.0.0"
    DEBUG = False
    TESTING = False

    # API settings
    API_PREFIX = "/api/v1"
    API_RATE_LIMIT = 100  # requests per hour

    # Database settings
    DATABASE_URI = "sqlite:///customerai.db"
    DATABASE_POOL_SIZE = 5
    DATABASE_MAX_OVERFLOW = 10

    # AI settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    AI_REQUEST_TIMEOUT = 30  # seconds
    AI_MAX_TOKENS = 500
    AI_TEMPERATURE = 0.7

    # Security settings
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
    JWT_EXPIRY = 86400  # 24 hours in seconds
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
    CORS_ORIGINS = ["*"]

    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "standard"  # standard or json
    LOG_FILE = "logs/application.log"

    # Performance settings
    WORKER_THREADS = 4
    WORKER_CONNECTIONS = 1000
    KEEP_ALIVE = 65

    # Feature flags
    FEATURES = {
        "human_in_loop": True,
        "bias_detection": True,
        "privacy_features": True,
        "advanced_analytics": True,
        "export_reports": True,
    }

    # Paths
    DATA_DIR = "data"
    EXPORT_DIR = "exports"

    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("_") and k.isupper()}


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    LOG_LEVEL = "DEBUG"

    # Database settings
    DATABASE_URI = "sqlite:///customerai_dev.db"

    # AI settings
    AI_TEMPERATURE = 0.8

    # Security settings - relaxed for development
    CORS_ORIGINS = ["*"]

    # Performance settings - reduced for development
    WORKER_THREADS = 2
    WORKER_CONNECTIONS = 500


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    DEBUG = True
    LOG_LEVEL = "DEBUG"

    # Database settings
    DATABASE_URI = "sqlite:///customerai_test.db"

    # AI settings - use mock for testing
    AI_MOCK = True

    # Security settings - disabled for testing
    # For testing only; in production, set via environment variables
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", None)
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", None)

    # Feature flags - enable all for testing
    FEATURES = {
        "human_in_loop": True,
        "bias_detection": True,
        "privacy_features": True,
        "advanced_analytics": True,
        "export_reports": True,
    }


class ProductionConfig(Config):
    """Production configuration"""

    # Security settings - strict for production
    CORS_ORIGINS = [
        "https://customerai.example.com",
        "https://api.customerai.example.com",
    ]

    # Logging settings
    LOG_LEVEL = "WARNING"
    LOG_FORMAT = "json"

    # Performance settings - optimized for production
    WORKER_THREADS = 8
    WORKER_CONNECTIONS = 2000

    # Database settings - use environment variables
    DATABASE_URI = os.environ.get("DATABASE_URI", Config.DATABASE_URI)
    DATABASE_POOL_SIZE = 20
    DATABASE_MAX_OVERFLOW = 20

    # AI settings - production limits
    AI_REQUEST_TIMEOUT = 15  # seconds


class DockerConfig(ProductionConfig):
    """Docker configuration (extends Production)"""

    # Database settings
    DATABASE_URI = os.environ.get(
        "DATABASE_URI",
        "postgresql://customerai:password@db:5432/customerai # pragma: allowlist secret",
    )

    # Logging settings
    LOG_FILE = "/var/log/customerai/application.log"

    # Paths
    DATA_DIR = "/data"
    EXPORT_DIR = "/exports"


# Environment-based configuration mapping
ENV_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "docker": DockerConfig,
}


def get_config() -> Any:
    """
    Get configuration based on environment

    Returns:
        Config: Configuration class for current environment
    """
    env = os.environ.get("ENVIRONMENT", "development").lower()
    return ENV_CONFIG_MAP.get(env, DevelopmentConfig)


def load_config_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON file

    Args:
        file_path (str): Path to configuration file

    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary or None if file not found
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration file: {e}")
        return None


# Current configuration
current_config = get_config()

# Override with file configuration if available
config_file = os.environ.get("CONFIG_FILE")
if config_file:
    file_config = load_config_file(config_file)
    if file_config:
        for key, value in file_config.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
