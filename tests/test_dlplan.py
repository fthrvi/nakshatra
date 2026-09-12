import pytest
from dlplan import download_argv


def test_basic_https_url():
    """Test basic HTTPS URL with default parameters"""
    url = "https://example.com/file.bin"
    dest = "/tmp/file.bin"
    result = download_argv(url, dest)
    expected = [
        "curl", "-fSL", "--proto", "=https", "-o", dest,
        "--max-time", "1800", "--connect-timeout", "30",
        "-C", "-", url
    ]
    assert result == expected


def test_resume_false():
    """Test resume=False omits -C -"""
    url = "https://example.com/file.bin"
    dest = "/tmp/file.bin"
    result = download_argv(url, dest, resume=False)
    expected = [
        "curl", "-fSL", "--proto", "=https", "-o", dest,
        "--max-time", "1800", "--connect-timeout", "30",
        url
    ]
    assert result == expected
    assert "-C" not in result
    assert "-C -" not in result


def test_proto_flag_always_present():
    """Test --proto '=https' is always present"""
    url = "https://example.com/file.bin"
    dest = "/tmp/file.bin"
    result = download_argv(url, dest)
    assert "--proto" in result
    assert "=https" in result


def test_http_url_raises():
    """Test HTTP URL raises ValueError"""
    url = "http://example.com/file.bin"
    dest = "/tmp/file.bin"
    with pytest.raises(ValueError):
        download_argv(url, dest)


def test_credentialed_url_raises():
    """Test URL with embedded credentials raises ValueError"""
    url = "https://user:pass@example.com/file.bin"
    dest = "/tmp/file.bin"
    with pytest.raises(ValueError):
        download_argv(url, dest)


def test_empty_dest_raises():
    """Test empty dest raises ValueError"""
    url = "https://example.com/file.bin"
    dest = ""
    with pytest.raises(ValueError):
        download_argv(url, dest)


def test_non_positive_timeout_raises():
    """Test non-positive timeout raises ValueError"""
    url = "https://example.com/file.bin"
    dest = "/tmp/file.bin"
    with pytest.raises(ValueError):
        download_argv(url, dest, timeout_s=0)
    with pytest.raises(ValueError):
        download_argv(url, dest, timeout_s=-1)


def test_dest_with_space_survives_as_one_element():
    """Test dest containing space survives as one element"""
    url = "https://example.com/file.bin"
    dest = "/tmp/my file.bin"
    result = download_argv(url, dest)
    assert dest in result
    # Count occurrences of dest in result - should be exactly one
    assert result.count(dest) == 1
    # Verify no splitting occurred
    assert "/tmp/my" not in result
    assert "file.bin" not in result


def test_no_credentials_in_argv():
    """Test no element contains -u or :@ pattern"""
    url = "https://example.com/file.bin"
    dest = "/tmp/file.bin"
    result = download_argv(url, dest)
    for elem in result:
        assert "-u" not in elem, f"Found -u in element: {elem}"
        assert ":@" not in elem, f"Found :@ pattern in element: {elem}"


def test_custom_timeout():
    """Test custom timeout value"""
    url = "https://example.com/file.bin"
    dest = "/tmp/file.bin"
    result = download_argv(url, dest, timeout_s=3600)
    assert "--max-time" in result
    assert "3600" in result


def test_url_without_path():
    """Test URL without path (no slash after domain)"""
    url = "https://example.com"
    dest = "/tmp/file.bin"
    with pytest.raises(ValueError):
        download_argv(url, dest)


def test_url_with_credentials_in_path():
    """Test URL with @ in path (not credentials) is allowed"""
    url = "https://example.com/path@file.bin"
    dest = "/tmp/file.bin"
    result = download_argv(url, dest)
    # Should not raise - @ in path is not credentials
    assert url in result