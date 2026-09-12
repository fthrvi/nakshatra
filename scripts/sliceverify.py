import hashlib
import os


def verify_slice(path: str, expected_sha256: str, expected_bytes: int = 0) -> tuple[bool, str]:
    try:
        # Check if file exists
        if not os.path.exists(path):
            return (False, "not found")
        
        # Check if path is a directory
        if os.path.isdir(path):
            return (False, "not found")
        
        # Check permissions by trying to open the file
        try:
            file = open(path, 'rb')
        except PermissionError:
            return (False, "permission denied")
        except OSError:
            return (False, "not found")
        
        # Get file size
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file.close()
            return (False, "not found")
        
        # Check size before hashing if expected_bytes is provided
        if expected_bytes > 0 and file_size != expected_bytes:
            file.close()
            return (False, f"size mismatch: expected {expected_bytes}, got {file_size}")
        
        # Validate expected_sha256 format
        if len(expected_sha256) != 64:
            file.close()
            return (False, f"malformed expectation: expected_sha256 must be 64 characters, got {len(expected_sha256)}")
        
        # Check if expected_sha256 is valid hex (case-insensitive for validation)
        try:
            int(expected_sha256, 16)
        except ValueError:
            file.close()
            return (False, f"malformed expectation: expected_sha256 must be hex")
        
        # Compute SHA256 hash in streaming chunks of 1 MiB
        sha256_hash = hashlib.sha256()
        chunk_size = 1024 * 1024  # 1 MiB
        
        try:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        except OSError:
            file.close()
            return (False, "not found")
        
        file.close()
        
        # Get the computed hash in lowercase hex
        computed_hash = sha256_hash.hexdigest().lower()
        
        # Normalize expected hash to lowercase for comparison
        expected_lower = expected_sha256.lower()
        
        # Compare hashes (case-insensitive on actual hash)
        if computed_hash == expected_lower:
            return (True, "ok")
        else:
            return (False, f"hash mismatch: expected {expected_sha256}, got {computed_hash}")
    
    except Exception:
        # Never raise, whatever the path or arguments
        return (False, "not found")