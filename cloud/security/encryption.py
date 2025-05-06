"""
Encryption utilities for cloud services.

This module provides tools for encrypting data in transit and at rest
for cloud service operations.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import functools
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

# Try to import cryptography libraries, providing helpful error if not installed
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, padding
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
        load_pem_private_key,
        load_pem_public_key,
    )

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    FERNET = "fernet"  # Fernet is AES-128-CBC with HMAC
    RSA = "rsa"


class EncryptionManager:
    """
    Manager for encryption operations.

    This class provides methods for encrypting and decrypting data
    for cloud service operations.
    """

    _instance = None
    _lock = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            import threading

            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    if not CRYPTOGRAPHY_AVAILABLE:
                        raise ImportError(
                            "Encryption functionality requires the 'cryptography' package. "
                            "Please install it with 'pip install cryptography'."
                        )
                    cls._instance = super(EncryptionManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize encryption manager (only once for singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Key storage
        self.symmetric_keys: Dict[str, bytes] = {}
        self.key_pairs: Dict[str, Dict[str, bytes]] = (
            {}
        )  # {key_name: {"private": ..., "public": ...}}
        self.fernet_keys: Dict[str, Fernet] = {}

        # Configuration
        self.enabled = True
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM

        # Key rotation
        self.key_rotation_interval = 86400  # 24 hours in seconds
        self.last_rotation_time = time.time()
        self._key_rotation_initialized = False

        # Thread safety
        import threading

        self.lock = threading.RLock()

        self._initialized = True

    def _ensure_key_rotation_scheduled(self):
        """Ensure key rotation is scheduled."""
        if not self._key_rotation_initialized:
            import threading

            rotation_thread = threading.Thread(target=self._key_rotation_task, daemon=True)
            rotation_thread.start()
            self._key_rotation_initialized = True

    def _key_rotation_task(self):
        """Background task for key rotation."""
        while self.enabled:
            current_time = time.time()
            if current_time - self.last_rotation_time >= self.key_rotation_interval:
                try:
                    self.rotate_keys()
                except Exception as e:
                    logger.error(f"Error rotating encryption keys: {str(e)}")

            # Check again in 1 hour or 1/24 of the rotation interval, whichever is smaller
            sleep_time = min(3600, self.key_rotation_interval / 24)
            time.sleep(sleep_time)

    def generate_symmetric_key(
        self, key_name: str, algorithm: Optional[EncryptionAlgorithm] = None
    ) -> bytes:
        """
        Generate a new symmetric key.

        Args:
            key_name: Name to identify the key
            algorithm: Encryption algorithm to generate key for

        Returns:
            The generated key
        """
        with self.lock:
            algo = algorithm or self.default_algorithm

            # Generate key based on algorithm
            if algo == EncryptionAlgorithm.AES_256_GCM or algo == EncryptionAlgorithm.AES_256_CBC:
                # 256-bit key
                key = os.urandom(32)
            elif algo == EncryptionAlgorithm.FERNET:
                # Fernet key is 32 bytes, base64-encoded
                key = Fernet.generate_key()
                self.fernet_keys[key_name] = Fernet(key)
            else:
                raise ValueError(f"Unsupported algorithm for symmetric key: {algo.value}")

            # Store key
            self.symmetric_keys[key_name] = key

            # Ensure key rotation is scheduled
            self._ensure_key_rotation_scheduled()

            return key

    def generate_key_pair(self, key_name: str, key_size: int = 2048) -> Dict[str, bytes]:
        """
        Generate a new RSA key pair.

        Args:
            key_name: Name to identify the key pair
            key_size: RSA key size in bits

        Returns:
            Dictionary with 'private' and 'public' keys
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("RSA encryption requires the 'cryptography' package.")

        with self.lock:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

            # Get public key
            public_key = private_key.public_key()

            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption(),
            )

            public_pem = public_key.public_bytes(
                encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
            )

            # Store keys
            key_pair = {"private": private_pem, "public": public_pem}
            self.key_pairs[key_name] = key_pair

            # Ensure key rotation is scheduled
            self._ensure_key_rotation_scheduled()

            return key_pair

    def set_symmetric_key(self, key_name: str, key: bytes) -> None:
        """
        Set a symmetric key.

        Args:
            key_name: Name to identify the key
            key: The key to set
        """
        with self.lock:
            self.symmetric_keys[key_name] = key

            # Create Fernet instance if it's a valid Fernet key
            try:
                fernet = Fernet(key)
                self.fernet_keys[key_name] = fernet
            except Exception as e:
                # Not a valid Fernet key, which is fine, but log for debugging
                import logging

                logging.warning(f"Failed to create Fernet instance for key '{key_name}': {e}")

    def set_key_pair(self, key_name: str, private_key: bytes, public_key: bytes) -> None:
        """
        Set an RSA key pair.

        Args:
            key_name: Name to identify the key pair
            private_key: The private key PEM data
            public_key: The public key PEM data
        """
        with self.lock:
            self.key_pairs[key_name] = {"private": private_key, "public": public_key}

    def get_symmetric_key(self, key_name: str) -> bytes:
        """
        Get a symmetric key.

        Args:
            key_name: Name of the key

        Returns:
            The key

        Raises:
            KeyError: If key not found
        """
        with self.lock:
            return self.symmetric_keys[key_name]

    def get_key_pair(self, key_name: str) -> Dict[str, bytes]:
        """
        Get an RSA key pair.

        Args:
            key_name: Name of the key pair

        Returns:
            Dictionary with 'private' and 'public' keys

        Raises:
            KeyError: If key pair not found
        """
        with self.lock:
            return self.key_pairs[key_name]

    def rotate_keys(self) -> None:
        """
        Rotate all encryption keys.

        This generates new keys for all existing keys and keeps
        the old keys temporarily for decryption of existing data.
        """
        with self.lock:
            current_time = time.time()

            # Rotate symmetric keys
            for key_name in list(self.symmetric_keys.keys()):
                if key_name.endswith("_old"):
                    # Remove old keys from previous rotation
                    del self.symmetric_keys[key_name]
                    if key_name in self.fernet_keys:
                        del self.fernet_keys[key_name]
                else:
                    # Move current key to old key
                    old_key_name = f"{key_name}_old"
                    self.symmetric_keys[old_key_name] = self.symmetric_keys[key_name]
                    if key_name in self.fernet_keys:
                        self.fernet_keys[old_key_name] = self.fernet_keys[key_name]

                    # Generate new key
                    if key_name in self.fernet_keys:
                        # It's a Fernet key
                        new_key = Fernet.generate_key()
                        self.symmetric_keys[key_name] = new_key
                        self.fernet_keys[key_name] = Fernet(new_key)
                    else:
                        # Assume it's an AES key
                        self.symmetric_keys[key_name] = os.urandom(32)

            # Rotate key pairs
            for key_name in list(self.key_pairs.keys()):
                if key_name.endswith("_old"):
                    # Remove old keys from previous rotation
                    del self.key_pairs[key_name]
                else:
                    # Move current key to old key
                    old_key_name = f"{key_name}_old"
                    self.key_pairs[old_key_name] = self.key_pairs[key_name]

                    # Generate new key pair
                    private_key = rsa.generate_private_key(
                        public_exponent=65537, key_size=2048  # Default key size
                    )

                    public_key = private_key.public_key()

                    private_pem = private_key.private_bytes(
                        encoding=Encoding.PEM,
                        format=PrivateFormat.PKCS8,
                        encryption_algorithm=NoEncryption(),
                    )

                    public_pem = public_key.public_bytes(
                        encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
                    )

                    self.key_pairs[key_name] = {
                        "private": private_pem,
                        "public": public_pem,
                    }

            self.last_rotation_time = current_time
            logger.info(
                f"Rotated encryption keys: {len(self.symmetric_keys)} symmetric keys, {len(self.key_pairs)} key pairs"
            )

    def encrypt_data(
        self,
        data: Union[str, bytes],
        key_name: str,
        algorithm: Optional[EncryptionAlgorithm] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Encrypt data using a symmetric key.

        Args:
            data: Data to encrypt
            key_name: Name of the key to use
            algorithm: Encryption algorithm to use
            metadata: Additional metadata to include

        Returns:
            Encrypted data with metadata
        """
        if not self.enabled:
            if isinstance(data, str):
                return data.encode("utf-8")
            return data

        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data

            # Get the key and algorithm
            algo = algorithm or self.default_algorithm

            # Create metadata
            meta = metadata or {}
            meta.update(
                {
                    "algorithm": algo.value,
                    "key_name": key_name,
                    "timestamp": time.time(),
                }
            )

            # Encrypt based on algorithm
            if algo == EncryptionAlgorithm.FERNET:
                with self.lock:
                    if key_name not in self.fernet_keys:
                        if key_name in self.symmetric_keys:
                            # Try to create Fernet from existing key
                            try:
                                self.fernet_keys[key_name] = Fernet(self.symmetric_keys[key_name])
                            except Exception:
                                # Not a valid Fernet key, generate a new one
                                key = Fernet.generate_key()
                                self.symmetric_keys[key_name] = key
                                self.fernet_keys[key_name] = Fernet(key)
                        else:
                            # Generate new key
                            key = Fernet.generate_key()
                            self.symmetric_keys[key_name] = key
                            self.fernet_keys[key_name] = Fernet(key)

                # Encrypt with Fernet
                encrypted = self.fernet_keys[key_name].encrypt(data_bytes)

            elif algo == EncryptionAlgorithm.AES_256_GCM:
                with self.lock:
                    if key_name not in self.symmetric_keys:
                        # Generate new key
                        self.generate_symmetric_key(key_name, algo)

                    key = self.symmetric_keys[key_name]

                # Generate a random 96-bit IV
                iv = os.urandom(12)

                # Create encryptor
                encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()

                # Convert metadata to bytes
                meta_bytes = json.dumps(meta).encode("utf-8")

                # Update with associated data (metadata)
                encryptor.authenticate_additional_data(meta_bytes)

                # Encrypt data
                encrypted_data = encryptor.update(data_bytes) + encryptor.finalize()

                # Get the tag
                tag = encryptor.tag

                # Combine everything: metadata length (2 bytes) + metadata + iv + tag + encrypted data
                meta_len = len(meta_bytes)
                encrypted = (
                    meta_len.to_bytes(2, byteorder="big") + meta_bytes + iv + tag + encrypted_data
                )

            elif algo == EncryptionAlgorithm.AES_256_CBC:
                with self.lock:
                    if key_name not in self.symmetric_keys:
                        # Generate new key
                        self.generate_symmetric_key(key_name, algo)

                    key = self.symmetric_keys[key_name]

                # Generate a random 128-bit IV
                iv = os.urandom(16)

                # Add padding
                padder = padding.PKCS7(algorithms.AES.block_size).padder()
                padded_data = padder.update(data_bytes) + padder.finalize()

                # Create encryptor
                encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()

                # Encrypt data
                encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

                # Convert metadata to bytes
                meta_bytes = json.dumps(meta).encode("utf-8")

                # Combine everything: metadata length (2 bytes) + metadata + iv + encrypted data
                meta_len = len(meta_bytes)
                encrypted = meta_len.to_bytes(2, byteorder="big") + meta_bytes + iv + encrypted_data

            elif algo == EncryptionAlgorithm.RSA:
                with self.lock:
                    if key_name not in self.key_pairs:
                        # Generate new key pair
                        self.generate_key_pair(key_name)

                    key_pair = self.key_pairs[key_name]
                    public_key = load_pem_public_key(key_pair["public"])

                # RSA can only encrypt small data, so only use for small amounts of data
                if len(data_bytes) > 190:  # Adjust based on key size and padding
                    raise ValueError(
                        "Data too large for RSA encryption. Use symmetric encryption for large data."
                    )

                # Encrypt data
                encrypted_data = public_key.encrypt(
                    data_bytes,
                    asym_padding.OAEP(
                        mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

                # Convert metadata to bytes
                meta_bytes = json.dumps(meta).encode("utf-8")

                # Combine everything: metadata length (2 bytes) + metadata + encrypted data
                meta_len = len(meta_bytes)
                encrypted = meta_len.to_bytes(2, byteorder="big") + meta_bytes + encrypted_data

            else:
                raise ValueError(f"Unsupported encryption algorithm: {algo.value}")

            return encrypted

        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise

    def decrypt_data(
        self, encrypted_data: bytes, key_name: Optional[str] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Decrypt data.

        Args:
            encrypted_data: Encrypted data
            key_name: Optional key name override

        Returns:
            Tuple of (decrypted_data, metadata)
        """
        if not self.enabled:
            return encrypted_data, {}

        try:
            # Extract metadata
            meta_len = int.from_bytes(encrypted_data[:2], byteorder="big")
            meta_bytes = encrypted_data[2 : 2 + meta_len]
            meta = json.loads(meta_bytes.decode("utf-8"))

            # Get algorithm from metadata
            algo_str = meta.get("algorithm")
            data_key_name = meta.get("key_name")

            # Use provided key_name if specified, otherwise use from metadata
            used_key_name = key_name or data_key_name

            if not used_key_name:
                raise ValueError("Key name not found in metadata and not provided")

            try:
                algo = EncryptionAlgorithm(algo_str)
            except ValueError:
                raise ValueError(f"Unsupported encryption algorithm in metadata: {algo_str}")

            # Extract encrypted data
            data_start = 2 + meta_len

            # Decrypt based on algorithm
            if algo == EncryptionAlgorithm.FERNET:
                with self.lock:
                    # Try current key first
                    if used_key_name in self.fernet_keys:
                        try:
                            fernet = self.fernet_keys[used_key_name]
                            decrypted = fernet.decrypt(encrypted_data[data_start:])
                            return decrypted, meta
                        except Exception as e:
                            import logging

                            logging.warning(
                                f"Fernet decryption failed for key '{used_key_name}': {e}"
                            )
                            # Try old key if present
                            old_key_name = f"{used_key_name}_old"
                            if old_key_name in self.fernet_keys:
                                try:
                                    fernet = self.fernet_keys[old_key_name]
                                    decrypted = fernet.decrypt(encrypted_data[data_start:])
                                    return decrypted, meta
                                except Exception as e2:
                                    logging.warning(
                                        f"Fernet decryption failed for old key '{old_key_name}': {e2}"
                                    )

                    raise ValueError(f"Unable to decrypt data with Fernet key: {used_key_name}")

            elif algo == EncryptionAlgorithm.AES_256_GCM:
                # Extract IV (12 bytes) and tag (16 bytes)
                iv = encrypted_data[data_start : data_start + 12]
                tag = encrypted_data[data_start + 12 : data_start + 28]
                ciphertext = encrypted_data[data_start + 28 :]

                # Get key
                with self.lock:
                    if used_key_name in self.symmetric_keys:
                        key = self.symmetric_keys[used_key_name]
                    elif f"{used_key_name}_old" in self.symmetric_keys:
                        key = self.symmetric_keys[f"{used_key_name}_old"]
                    else:
                        raise ValueError(f"Key not found: {used_key_name}")

                # Create decryptor
                decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()

                # Set associated data
                decryptor.authenticate_additional_data(meta_bytes)

                # Decrypt data
                decrypted = decryptor.update(ciphertext) + decryptor.finalize()

                return decrypted, meta

            elif algo == EncryptionAlgorithm.AES_256_CBC:
                # Extract IV (16 bytes)
                iv = encrypted_data[data_start : data_start + 16]
                ciphertext = encrypted_data[data_start + 16 :]

                # Get key
                with self.lock:
                    if used_key_name in self.symmetric_keys:
                        key = self.symmetric_keys[used_key_name]
                    elif f"{used_key_name}_old" in self.symmetric_keys:
                        key = self.symmetric_keys[f"{used_key_name}_old"]
                    else:
                        raise ValueError(f"Key not found: {used_key_name}")

                # Create decryptor
                decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()

                # Decrypt data
                padded_data = decryptor.update(ciphertext) + decryptor.finalize()

                # Remove padding
                unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
                decrypted = unpadder.update(padded_data) + unpadder.finalize()

                return decrypted, meta

            elif algo == EncryptionAlgorithm.RSA:
                ciphertext = encrypted_data[data_start:]

                # Get key
                with self.lock:
                    if used_key_name in self.key_pairs:
                        private_key_pem = self.key_pairs[used_key_name]["private"]
                    elif f"{used_key_name}_old" in self.key_pairs:
                        private_key_pem = self.key_pairs[f"{used_key_name}_old"]["private"]
                    else:
                        raise ValueError(f"Key pair not found: {used_key_name}")

                    private_key = load_pem_private_key(private_key_pem, password=None)

                # Decrypt data
                decrypted = private_key.decrypt(
                    ciphertext,
                    asym_padding.OAEP(
                        mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

                return decrypted, meta

            else:
                raise ValueError(f"Unsupported encryption algorithm: {algo.value}")

        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise

    def enable(self) -> None:
        """Enable encryption."""
        self.enabled = True

    def disable(self) -> None:
        """Disable encryption."""
        self.enabled = False

    def set_key_rotation_interval(self, seconds: int) -> None:
        """Set the key rotation interval in seconds."""
        self.key_rotation_interval = max(3600, seconds)  # Minimum 1 hour


# Global encryption manager instance
_encryption_manager = EncryptionManager() if CRYPTOGRAPHY_AVAILABLE else None


def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance."""
    if _encryption_manager is None:
        raise ImportError(
            "Encryption functionality requires the 'cryptography' package. "
            "Please install it with 'pip install cryptography'."
        )
    return _encryption_manager


def encrypt_data(
    data: Union[str, bytes],
    key_name: str,
    algorithm: Optional[EncryptionAlgorithm] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Encrypt data (convenience function).

    Args:
        data: Data to encrypt
        key_name: Name of the key to use
        algorithm: Encryption algorithm to use
        metadata: Additional metadata to include

    Returns:
        Encrypted data with metadata
    """
    if _encryption_manager is None:
        if isinstance(data, str):
            return data.encode("utf-8")
        return data

    return _encryption_manager.encrypt_data(data, key_name, algorithm, metadata)


def decrypt_data(
    encrypted_data: bytes, key_name: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Decrypt data (convenience function).

    Args:
        encrypted_data: Encrypted data
        key_name: Optional key name override

    Returns:
        Tuple of (decrypted_data, metadata)
    """
    if _encryption_manager is None:
        return encrypted_data, {}

    return _encryption_manager.decrypt_data(encrypted_data, key_name)


def with_encryption(
    key_name: str,
    algorithm: Optional[EncryptionAlgorithm] = None,
    encrypt_result: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically encrypt function arguments and decrypt results.

    Args:
        key_name: Name of the key to use
        algorithm: Encryption algorithm to use
        encrypt_result: Whether to encrypt the function result

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Check if encryption is enabled
            if _encryption_manager is None or not _encryption_manager.enabled:
                return func(*args, **kwargs)

            # Get the encryption manager
            mgr = get_encryption_manager()

            # Encrypt any bytes or string arguments
            encrypted_args = []
            for arg in args:
                if isinstance(arg, (bytes, str)):
                    # Encrypt the argument
                    encrypted_arg = mgr.encrypt_data(arg, key_name, algorithm)
                    encrypted_args.append(encrypted_arg)
                else:
                    encrypted_args.append(arg)

            # Encrypt any bytes or string keyword arguments
            encrypted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (bytes, str)):
                    # Encrypt the argument
                    encrypted_value = mgr.encrypt_data(value, key_name, algorithm)
                    encrypted_kwargs[key] = encrypted_value
                else:
                    encrypted_kwargs[key] = value

            # Call the function with encrypted arguments
            result = func(*encrypted_args, **encrypted_kwargs)

            # Encrypt or decrypt the result if needed
            if encrypt_result:
                if isinstance(result, (bytes, str)):
                    return cast(T, mgr.encrypt_data(result, key_name, algorithm))

            return result

        return wrapper

    return decorator
