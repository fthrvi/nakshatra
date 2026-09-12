import os
import hashlib
import tempfile
from pathlib import Path
from sliceverify import verify_slice


def test_matching_file(tmp_path: Path):
    """Test with a file that matches the expected hash and size."""
    test_file = tmp_path / "test.txt"
    content = b"Hello, world!"
    test_file.write_bytes(content)
    
    expected_hash = hashlib.sha256(content).hexdigest()
    expected_size = len(content)
    
    ok, reason = verify_slice(str(test_file), expected_hash, expected_size)
    assert ok is True
    assert reason == "ok"


def test_wrong_hash(tmp_path: Path):
    """Test with a file that has a wrong hash."""
    test_file = tmp_path / "test.txt"
    content = b"Hello, world!"
    test_file.write_bytes(content)
    
    wrong_hash = "a" * 64  # Wrong hash
    expected_size = len(content)
    
    ok, reason = verify_slice(str(test_file), wrong_hash, expected_size)
    assert ok is False
    assert "hash mismatch" in reason


def test_size_mismatch_before_hashing(tmp_path: Path):
    """Test that size mismatch is caught before hashing."""
    test_file = tmp_path / "test.txt"
    content = b"Hello, world!"
    test_file.write_bytes(content)
    
    expected_hash = hashlib.sha256(content).hexdigest()
    wrong_size = 100  # Wrong size
    
    ok, reason = verify_slice(str(test_file), expected_hash, wrong_size)
    assert ok is False
    assert "size mismatch" in reason
    # Verify we didn't hash (the error should mention size, not hash)
    assert "hash" not in reason.lower()


def test_missing_file(tmp_path: Path):
    """Test with a missing file."""
    missing_path = tmp_path / "nonexistent.txt"
    
    ok, reason = verify_slice(str(missing_path), "a" * 64)
    assert ok is False
    assert reason == "not found"


def test_directory_passed_as_path(tmp_path: Path):
    """Test with a directory passed as the path."""
    ok, reason = verify_slice(str(tmp_path), "a" * 64)
    assert ok is False
    assert reason == "not found"


def test_malformed_expectation(tmp_path: Path):
    """Test with a malformed expected hash."""
    test_file = tmp_path / "test.txt"
    content = b"Hello, world!"
    test_file.write_bytes(content)
    
    malformed_hash = "xyz" * 20  # Not 64 chars, not hex
    
    ok, reason = verify_slice(str(test_file), malformed_hash)
    assert ok is False
    assert "malformed expectation" in reason


def test_empty_file_with_sha256_of_empty(tmp_path: Path):
    """Test with an empty file and the sha256 of empty."""
    test_file = tmp_path / "empty.txt"
    test_file.write_bytes(b"")
    
    # SHA256 of empty string
    expected_hash = hashlib.sha256(b"").hexdigest()
    expected_size = 0
    
    ok, reason = verify_slice(str(test_file), expected_hash, expected_size)
    assert ok is True
    assert reason == "ok"


def test_never_raises(tmp_path: Path):
    """Test that the function never raises an exception."""
    # Test with various invalid inputs
    invalid_paths = [
        "/nonexistent/path/to/file",
        str(tmp_path / "nonexistent"),
        "",
        None,  # This will be handled by the function
    ]
    
    for path in invalid_paths:
        try:
            ok, reason = verify_slice(path, "a" * 64) if path is not None else verify_slice(None, "a" * 64)  # type: ignore
            # Should not raise
        except Exception:
            assert False, "verify_slice should never raise"
    
    # Test with malformed hash
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"test")
    ok, reason = verify_slice(str(test_file), "invalid")
    assert ok is False
    
    # Test with uppercase hex (should still work for actual hash comparison)
    test_hash = hashlib.sha256(b"test").hexdigest().upper()
    ok, reason = verify_slice(str(test_file), test_hash)
    assert ok is True