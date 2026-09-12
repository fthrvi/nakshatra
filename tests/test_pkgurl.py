import pytest
from pkgurl import check_package_url


class TestCheckPackageUrl:
    def test_good_url(self):
        ok, why = check_package_url("https://example.com/path.gguf")
        assert ok is True
        assert why == "OK"
    
    def test_http_rejected(self):
        ok, why = check_package_url("http://example.com/path.gguf")
        assert ok is False
        assert "Scheme must be https" in why
    
    def test_file_rejected(self):
        ok, why = check_package_url("file:///etc/passwd")
        assert ok is False
        assert "Scheme must be https" in why
    
    def test_ftp_rejected(self):
        ok, why = check_package_url("ftp://example.com/path.gguf")
        assert ok is False
        assert "Scheme must be https" in why
    
    def test_credentials_rejected(self):
        ok, why = check_package_url("https://user:pass@example.com/path.gguf")
        assert ok is False
        assert "embedded credentials" in why
    
    def test_private_ip_10_0_0_0_8(self):
        ok, why = check_package_url("https://10.0.0.1/path.gguf")
        assert ok is False
        assert "private" in why.lower() or "reserved" in why.lower()
    
    def test_private_ip_172_16_0_0_12(self):
        ok, why = check_package_url("https://172.16.0.1/path.gguf")
        assert ok is False
        assert "private" in why.lower() or "reserved" in why.lower()
    
    def test_private_ip_192_168_0_0_16(self):
        ok, why = check_package_url("https://192.168.1.1/path.gguf")
        assert ok is False
        assert "private" in why.lower() or "reserved" in why.lower()
    
    def test_loopback_127_0_0_0_8(self):
        ok, why = check_package_url("https://127.0.0.1/path.gguf")
        assert ok is False
        assert "loopback" in why.lower() or "private" in why.lower() or "reserved" in why.lower()
    
    def test_link_local_169_254_0_0_16(self):
        ok, why = check_package_url("https://169.254.169.254/path.gguf")
        assert ok is False
        assert "link-local" in why.lower() or "private" in why.lower() or "reserved" in why.lower()
    
    def test_ipv6_loopback(self):
        ok, why = check_package_url("https://[::1]/path.gguf")
        assert ok is False
        assert "loopback" in why.lower() or "private" in why.lower() or "reserved" in why.lower()
    
    def test_ipv6_unique_local(self):
        ok, why = check_package_url("https://[fc00::1]/path.gguf")
        assert ok is False
        assert "unique local" in why.lower() or "private" in why.lower() or "reserved" in why.lower()
    
    def test_public_ip_accepted(self):
        ok, why = check_package_url("https://8.8.8.8/path.gguf")
        assert ok is True
        assert why == "OK"
    
    def test_allow_hosts_matching(self):
        ok, why = check_package_url("https://example.com/path.gguf", allow_hosts=["example.com"])
        assert ok is True
        assert why == "OK"
    
    def test_allow_hosts_not_matching(self):
        ok, why = check_package_url("https://evil.com/path.gguf", allow_hosts=["example.com"])
        assert ok is False
        assert "not in allowed hosts" in why
    
    def test_evil_example_com_not_matching_example_com(self):
        ok, why = check_package_url("https://evil-example.com/path.gguf", allow_hosts=["example.com"])
        assert ok is False
        assert "not in allowed hosts" in why
    
    def test_case_insensitive_host_match(self):
        ok, why = check_package_url("https://EXAMPLE.COM/path.gguf", allow_hosts=["example.com"])
        assert ok is True
        assert why == "OK"
    
    def test_empty_path_rejected(self):
        ok, why = check_package_url("https://example.com")
        assert ok is False
        assert "Path must not be empty" in why
    
    def test_slash_path_rejected(self):
        ok, why = check_package_url("https://example.com/")
        assert ok is False
        assert "Path must not be empty" in why
    
    def test_garbage_input(self):
        ok, why = check_package_url("not a url at all")
        assert ok is False
    
    def test_non_string_input(self):
        ok, why = check_package_url(12345)
        assert ok is False
        assert "URL must be a string" in why
    
    def test_never_raises_with_garbage(self):
        # Test various garbage inputs that shouldn't raise
        garbage_inputs = [
            None,
            123,
            ["list"],
            {"dict": "value"},
            b"bytes",
            "",
            "scheme://",
            "https://",
            "https:///",
            "https://host/",
            "http://",
            "ftp://",
            "file://",
        ]
        for inp in garbage_inputs:
            try:
                ok, why = check_package_url(inp)
                # Should return (False, reason) without raising
                assert isinstance(ok, bool)
                assert isinstance(why, str)
            except Exception:
                pytest.fail(f"check_package_url raised on input {inp!r}")