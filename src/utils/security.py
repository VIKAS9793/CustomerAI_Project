import re
import hashlib
import secrets
import time
import threading
import uuid
import json
import jwt
import os
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Constants
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DELTA = timedelta(hours=24)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

# Ensure secrets are set
if not JWT_SECRET_KEY:
    logger.warning("JWT_SECRET_KEY not set! Using random value - this will invalidate all tokens on restart")
    JWT_SECRET_KEY = secrets.token_hex(32)

if not ENCRYPTION_KEY:
    logger.warning("ENCRYPTION_KEY not set! Using random value - this will break existing encrypted data")
    ENCRYPTION_KEY = base64.urlsafe_b64encode(secrets.token_bytes(32))

# Initialize encryption
try:
    # Create Fernet cipher for encryption
    fernet = Fernet(ENCRYPTION_KEY)
except Exception as e:
    logger.error(f"Error initializing encryption: {str(e)}")
    # Fallback to a new key if there's an issue
    fernet = Fernet(Fernet.generate_key())

class InputValidator:
    """
    Validates and sanitizes user inputs to prevent injection attacks
    """
    
    # Regex patterns for common input validations
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone': r'^(\+\d{1,3})?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
        'name': r'^[a-zA-Z\s\'-]{2,50}$',
        'account_number': r'^\d{8,17}$',
        'alphanumeric': r'^[a-zA-Z0-9\s\-_]+$',
        'numeric': r'^\d+$',
        'float': r'^-?\d+(\.\d+)?$',
        'date': r'^\d{4}-\d{2}-\d{2}$',
        'password': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    }
    
    @staticmethod
    def validate_pattern(value: str, pattern_name: str) -> bool:
        """
        Validate string against a predefined pattern
        
        Args:
            value (str): Value to validate
            pattern_name (str): Pattern name from PATTERNS dictionary
            
        Returns:
            bool: True if valid, False otherwise
        """
        if pattern_name not in InputValidator.PATTERNS:
            logger.warning(f"Unknown pattern name: {pattern_name}")
            return False
            
        pattern = InputValidator.PATTERNS[pattern_name]
        return bool(re.match(pattern, value))
    
    @staticmethod
    def validate_custom(value: str, pattern: str) -> bool:
        """
        Validate string against a custom regex pattern
        
        Args:
            value (str): Value to validate
            pattern (str): Custom regex pattern
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            return bool(re.match(pattern, value))
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return False
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """
        Sanitize string by removing HTML tags
        
        Args:
            value (str): Value to sanitize
            
        Returns:
            str: Sanitized string
        """
        # Remove HTML tags
        clean = re.sub(r'<[^>]*>', '', value)
        # Replace potentially dangerous characters
        clean = clean.replace('&', '&amp;')
        clean = clean.replace('<', '&lt;')
        clean = clean.replace('>', '&gt;')
        clean = clean.replace('"', '&quot;')
        clean = clean.replace("'", '&#x27;')
        return clean
    
    @staticmethod
    def validate_length(value: str, min_length: int = 0, max_length: int = 1000) -> bool:
        """
        Validate string length is within bounds
        
        Args:
            value (str): Value to validate
            min_length (int): Minimum allowed length
            max_length (int): Maximum allowed length
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(value, str):
            return False
        return min_length <= len(value) <= max_length
    
    @staticmethod
    def validate_enum(value: Any, valid_values: List[Any]) -> bool:
        """
        Validate value is one of the valid options
        
        Args:
            value (Any): Value to validate
            valid_values (List[Any]): List of valid values
            
        Returns:
            bool: True if valid, False otherwise
        """
        return value in valid_values
    
    @staticmethod
    def validate_json(value: str) -> bool:
        """
        Validate string is valid JSON
        
        Args:
            value (str): Value to validate
            
        Returns:
            bool: True if valid JSON, False otherwise
        """
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False


class RateLimiter:
    """
    Rate limiting implementation to prevent abuse
    """
    
    def __init__(self, limit: int = 100, window: int = 3600, by_ip: bool = True):
        """
        Initialize rate limiter
        
        Args:
            limit (int): Maximum requests allowed in window
            window (int): Time window in seconds
            by_ip (bool): Track requests by IP address (True) or user (False)
        """
        self.limit = limit
        self.window = window
        self.by_ip = by_ip
        self.requests = {}  # {key: [(timestamp, count), ...]}
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed for key
        
        Args:
            key (str): Identifier (IP address or user ID)
            
        Returns:
            bool: True if allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Clean up old requests
            if key in self.requests:
                self.requests[key] = [r for r in self.requests[key] if now - r[0] < self.window]
            else:
                self.requests[key] = []
            
            # Count requests in current window
            count = sum(r[1] for r in self.requests[key])
            
            # Check if limit is reached
            if count >= self.limit:
                logger.warning(f"Rate limit exceeded for {key}: {count}/{self.limit}")
                return False
            
            # Record request
            if self.requests[key] and self.requests[key][-1][0] == int(now):
                # Increment count for current second
                self.requests[key][-1] = (int(now), self.requests[key][-1][1] + 1)
            else:
                # New second
                self.requests[key].append((int(now), 1))
            
            return True
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests for key
        
        Args:
            key (str): Identifier (IP address or user ID)
            
        Returns:
            int: Number of remaining requests
        """
        with self.lock:
            now = time.time()
            
            # Clean up old requests
            if key in self.requests:
                self.requests[key] = [r for r in self.requests[key] if now - r[0] < self.window]
                count = sum(r[1] for r in self.requests[key])
                return max(0, self.limit - count)
            else:
                return self.limit


class Authentication:
    """
    Authentication and authorization functionality
    """
    
    def __init__(self, secret_key: Optional[str] = None, token_expiry: int = 86400):
        """
        Initialize authentication module
        
        Args:
            secret_key (str, optional): Secret key for JWT. Defaults to environment variable or random.
            token_expiry (int, optional): Token expiry time in seconds. Defaults to 24 hours.
        """
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY') or secrets.token_hex(32)
        self.token_expiry = token_expiry
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Hash password with salt using SHA-256
        
        Args:
            password (str): Password to hash
            salt (str, optional): Salt for hashing. Defaults to random.
            
        Returns:
            Dict[str, str]: Dictionary with hash and salt
        """
        if not salt:
            salt = secrets.token_hex(16)
        
        # Hash password with salt
        hash_obj = hashlib.sha256((password + salt).encode())
        password_hash = hash_obj.hexdigest()
        
        return {
            'hash': password_hash,
            'salt': salt
        }
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        Verify password against stored hash
        
        Args:
            password (str): Password to verify
            stored_hash (str): Stored password hash
            salt (str): Salt used for hashing
            
        Returns:
            bool: True if password matches, False otherwise
        """
        # Hash provided password with stored salt
        hash_result = self.hash_password(password, salt)
        
        # Compare hashes using constant-time comparison
        return secrets.compare_digest(hash_result['hash'], stored_hash)
    
    def generate_token(self, user_id: str, roles: List[str] = None, 
                       additional_data: Dict[str, Any] = None) -> str:
        """
        Generate JWT token for user
        
        Args:
            user_id (str): User ID
            roles (List[str], optional): User roles. Defaults to None.
            additional_data (Dict[str, Any], optional): Additional data for token. Defaults to None.
            
        Returns:
            str: JWT token
        """
        now = datetime.utcnow()
        
        # Prepare payload
        payload = {
            'sub': str(user_id),
            'iat': now,
            'exp': now + timedelta(seconds=self.token_expiry),
            'jti': str(uuid.uuid4())
        }
        
        # Add roles if provided
        if roles:
            payload['roles'] = roles
        
        # Add additional data if provided
        if additional_data:
            for key, value in additional_data.items():
                if key not in payload:
                    payload[key] = value
        
        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        logger.info(f"Generated token for user {user_id}")
        
        return token
    
    def verify_token(self, token: str) -> Union[Dict[str, Any], None]:
        """
        Verify JWT token
        
        Args:
            token (str): JWT token
            
        Returns:
            Union[Dict[str, Any], None]: Token payload if valid, None if invalid
        """
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            logger.debug(f"Verified token for user {payload.get('sub')}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def has_role(self, token: str, required_roles: List[str]) -> bool:
        """
        Check if token has required roles
        
        Args:
            token (str): JWT token
            required_roles (List[str]): Required roles
            
        Returns:
            bool: True if token has required roles, False otherwise
        """
        payload = self.verify_token(token)
        
        if not payload:
            return False
        
        # Get roles from token
        user_roles = payload.get('roles', [])
        
        # Check if user has admin role
        if 'admin' in user_roles:
            return True
        
        # Check if user has all required roles
        return all(role in user_roles for role in required_roles)


class DeviceSecurityValidator:
    """
    Validates device security status to prevent access from compromised devices
    such as rooted phones, devices with unlocked bootloaders, or other security risks.
    """
    
    def __init__(self):
        """Initialize the device security validator."""
        self.logger = logging.getLogger(__name__)
    
    def is_device_secure(self, request_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if the requesting device meets security requirements.
        
        Args:
            request_data: Dictionary containing device information
            
        Returns:
            Tuple of (is_secure, reason)
        """
        # Extract device info
        user_agent = request_data.get("user_agent", "")
        device_fingerprint = request_data.get("device_fingerprint", {})
        device_id = request_data.get("device_id", "")
        
        # Check for root/jailbreak indicators
        if self._is_device_rooted(device_fingerprint):
            self.logger.warning(f"Rejected request from rooted device: {device_id}")
            return False, "Device appears to be rooted or jailbroken"
        
        # Check for unlocked bootloader
        if self._has_unlocked_bootloader(device_fingerprint):
            self.logger.warning(f"Rejected request from device with unlocked bootloader: {device_id}")
            return False, "Device has an unlocked bootloader"
            
        # Check for emulator/virtual device
        if self._is_emulator(user_agent, device_fingerprint):
            self.logger.warning(f"Rejected request from emulator: {device_id}")
            return False, "Access from emulators is not permitted"
            
        # Check for suspicious modifications
        if self._has_security_risks(device_fingerprint):
            self.logger.warning(f"Rejected request from device with security risks: {device_id}")
            return False, "Device has security risks that prevent access"
            
        return True, ""
    
    def _is_device_rooted(self, fingerprint: Dict[str, Any]) -> bool:
        """
        Check if device is rooted (Android) or jailbroken (iOS).
        
        Args:
            fingerprint: Device fingerprint data
            
        Returns:
            True if device appears to be rooted, False otherwise
        """
        # Get platform
        platform = fingerprint.get("platform", "").lower()
        
        if "android" in platform:
            # Android root detection
            root_indicators = fingerprint.get("root_indicators", [])
            if root_indicators and any(root_indicators):
                return True
                
            # Check for SU binary paths
            su_paths = [
                "/system/bin/su", 
                "/system/xbin/su", 
                "/sbin/su", 
                "/system/app/Superuser.apk",
                "/system/app/SuperSU.apk"
            ]
            installed_paths = fingerprint.get("installed_paths", [])
            if any(path in installed_paths for path in su_paths):
                return True
                
            # Check for root management apps
            root_apps = [
                "com.noshufou.android.su",
                "com.thirdparty.superuser",
                "eu.chainfire.supersu",
                "com.topjohnwu.magisk"
            ]
            installed_apps = fingerprint.get("installed_apps", [])
            if any(app in installed_apps for app in root_apps):
                return True
                
        elif "ios" in platform:
            # iOS jailbreak detection
            jailbreak_indicators = fingerprint.get("jailbreak_indicators", [])
            if jailbreak_indicators and any(jailbreak_indicators):
                return True
                
            # Check for jailbreak apps
            jailbreak_apps = [
                "Cydia", "Sileo", "Zebra", "Installer"
            ]
            installed_apps = fingerprint.get("installed_apps", [])
            if any(app in installed_apps for app in jailbreak_apps):
                return True
                
            # Check for suspicious file paths
            jailbreak_paths = [
                "/Applications/Cydia.app",
                "/Library/MobileSubstrate/MobileSubstrate.dylib",
                "/private/var/lib/apt"
            ]
            installed_paths = fingerprint.get("installed_paths", [])
            if any(path in installed_paths for path in jailbreak_paths):
                return True
                
        return False
    
    def _has_unlocked_bootloader(self, fingerprint: Dict[str, Any]) -> bool:
        """
        Check if device has an unlocked bootloader.
        
        Args:
            fingerprint: Device fingerprint data
            
        Returns:
            True if bootloader appears to be unlocked, False otherwise
        """
        platform = fingerprint.get("platform", "").lower()
        
        if "android" in platform:
            bootloader_state = fingerprint.get("bootloader_state", "")
            if bootloader_state.lower() in ["unlocked", "modified"]:
                return True
                
            # Check for custom ROM indicators
            custom_rom = fingerprint.get("custom_rom", False)
            if custom_rom:
                return True
                
            # Check for system integrity
            verified_boot = fingerprint.get("verified_boot", "")
            if verified_boot.lower() in ["orange", "yellow", "red"]:
                return True
        
        return False
    
    def _is_emulator(self, user_agent: str, fingerprint: Dict[str, Any]) -> bool:
        """
        Check if device is an emulator.
        
        Args:
            user_agent: User agent string
            fingerprint: Device fingerprint data
            
        Returns:
            True if device appears to be an emulator, False otherwise
        """
        # Check user agent for emulator indicators
        emulator_indicators = [
            "sdk_gphone", "emulator", "android sdk", "generic",
            "sdk", "simulator"
        ]
        
        if any(indicator in user_agent.lower() for indicator in emulator_indicators):
            return True
            
        # Check fingerprint data
        is_emulator = fingerprint.get("is_emulator", False)
        if is_emulator:
            return True
            
        # Check hardware details for emulator tells
        model = fingerprint.get("model", "").lower()
        manufacturer = fingerprint.get("manufacturer", "").lower()
        
        if "emulator" in model or "virtual" in model or "sdk" in model:
            return True
            
        if "genymotion" in manufacturer or "bluestacks" in manufacturer:
            return True
            
        return False
    
    def _has_security_risks(self, fingerprint: Dict[str, Any]) -> bool:
        """
        Check for other security risks.
        
        Args:
            fingerprint: Device fingerprint data
            
        Returns:
            True if security risks detected, False otherwise
        """
        # Check if device encryption is disabled
        encryption_status = fingerprint.get("encryption_status", "").lower()
        if encryption_status in ["disabled", "off", "none"]:
            return True
            
        # Check if device has developer mode enabled
        developer_mode = fingerprint.get("developer_mode", False)
        usb_debugging = fingerprint.get("usb_debugging", False)
        
        # Only flag if both developer mode and USB debugging are enabled
        if developer_mode and usb_debugging:
            return True
            
        # Check for suspicious apps or security tools that could be used maliciously
        suspicious_apps = [
            "com.guoshi.httpcanary",  # Packet capture
            "com.minhui.networkcapture",  # Network capture
            "com.github.megatronking.netbare",  # SSL bypass
            "de.robv.android.xposed",  # Xposed framework
            "com.saurik.substrate"  # Substrate framework
        ]
        
        installed_apps = fingerprint.get("installed_apps", [])
        if any(app in installed_apps for app in suspicious_apps):
            return True
            
        return False


# Initialize default instances for import
input_validator = InputValidator()
rate_limiter = RateLimiter()
authentication = Authentication()

def create_jwt_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT token with given data and expiration.
    
    Args:
        data: Payload to include in the token
        expires_delta: Custom expiration time (uses default if not provided)
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    expires = datetime.utcnow() + (expires_delta or JWT_EXPIRATION_DELTA)
    
    # Add standard claims
    to_encode.update({
        "exp": expires.timestamp(),
        "iat": datetime.utcnow().timestamp(),
        "nbf": datetime.utcnow().timestamp(),
        "jti": secrets.token_hex(16)  # Add unique token ID to prevent replay attacks
    })
    
    try:
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"Error creating JWT token: {str(e)}")
        raise

def validate_jwt_token(token: str) -> Dict[str, Any]:
    """
    Validate and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        Exception: If token is invalid, expired, or has been tampered with
    """
    try:
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM],
            options={"verify_signature": True, "verify_exp": True}
        )
        
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning(f"Expired token attempted: {token[:10]}...")
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise ValueError(f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        raise

def get_password_hash(password: str) -> str:
    """
    Generate a secure hash for a password using Argon2.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    try:
        import argon2
        ph = argon2.PasswordHasher(
            time_cost=3,       # Iterations
            memory_cost=65536, # 64MB memory usage
            parallelism=4,     # 4 parallel threads
            hash_len=32,       # 32 bytes output
            salt_len=16        # 16 bytes salt
        )
        return ph.hash(password)
    except ImportError:
        # Fallback to PBKDF2 if argon2 is not available
        logger.warning("Argon2 not available, using PBKDF2 (less secure)")
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        key = kdf.derive(password.encode())
        return f"pbkdf2_sha256$600000${salt.hex()}${key.hex()}"

def verify_password(hashed_password: str, plain_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        hashed_password: The hashed password
        plain_password: The plain text password to verify
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        if hashed_password.startswith("$argon2"):
            import argon2
            ph = argon2.PasswordHasher()
            return ph.verify(hashed_password, plain_password)
        elif hashed_password.startswith("pbkdf2_sha256"):
            # Parse the hash string
            algorithm, iterations, salt_hex, hash_hex = hashed_password.split('$')
            salt = bytes.fromhex(salt_hex)
            stored_key = bytes.fromhex(hash_hex)
            
            # Compute the key from the password and compare
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=int(iterations),
            )
            key = kdf.derive(plain_password.encode())
            return secrets.compare_digest(key, stored_key)
        else:
            logger.error(f"Unsupported hash format: {hashed_password[:10]}...")
            return False
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False

def encrypt_data(data: Union[str, bytes]) -> str:
    """
    Encrypt sensitive data.
    
    Args:
        data: Data to encrypt (string or bytes)
        
    Returns:
        Base64-encoded encrypted data
    """
    try:
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        logger.error(f"Error encrypting data: {str(e)}")
        raise

def decrypt_data(encrypted_data: str) -> bytes:
    """
    Decrypt encrypted data.
    
    Args:
        encrypted_data: Base64-encoded encrypted data
        
    Returns:
        Decrypted data as bytes
    """
    try:
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data)
        return fernet.decrypt(encrypted_bytes)
    except Exception as e:
        logger.error(f"Error decrypting data: {str(e)}")
        raise

def generate_secure_filename() -> str:
    """Generate a secure random filename to prevent path traversal."""
    return f"{secrets.token_hex(16)}-{int(time.time())}"

def sanitize_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_str: Input string to sanitize
        
    Returns:
        Sanitized string
    """
    if not input_str:
        return ""
        
    # Remove common dangerous characters
    dangerous_chars = ["../", "..\\", "<!--", "-->", "<script>", "</script>"]
    result = input_str
    
    for char in dangerous_chars:
        result = result.replace(char, "")
        
    return result 