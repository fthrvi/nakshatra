import pytest
from errclass import classify


class TestErrclass:
    # Test transient cases
    def test_transient_timeout(self):
        result = classify(1, "Connection timeout occurred")
        assert result == ("transient", "timeout")
    
    def test_transient_temporary_failure(self):
        result = classify(1, "Temporary failure in name resolution")
        assert result == ("transient", "temporary failure")
    
    def test_transient_connection_reset(self):
        result = classify(1, "Connection reset by peer")
        assert result == ("transient", "connection reset")
    
    def test_transient_503(self):
        result = classify(1, "HTTP 503 Service Unavailable")
        assert result == ("transient", "503")
    
    def test_transient_try_again(self):
        result = classify(1, "Try again later")
        assert result == ("transient", "try again")
    
    def test_transient_network_unreachable(self):
        result = classify(1, "Network is unreachable")
        assert result == ("transient", "network is unreachable")
    
    # Test needs_operator cases
    def test_needs_operator_no_space(self):
        result = classify(1, "No space left on device")
        assert result == ("needs_operator", "no space left on device")
    
    def test_needs_operator_permission_denied(self):
        result = classify(1, "Permission denied")
        assert result == ("needs_operator", "permission denied")
    
    def test_needs_operator_command_not_found(self):
        result = classify(1, "Command not found")
        assert result == ("needs_operator", "command not found")
    
    def test_needs_operator_unable_to_locate_package(self):
        result = classify(1, "Unable to locate package")
        assert result == ("needs_operator", "unable to locate package")
    
    def test_needs_operator_disk_quota_exceeded(self):
        result = classify(1, "Disk quota exceeded")
        assert result == ("needs_operator", "disk quota exceeded")
    
    # Test fatal cases
    def test_fatal_404(self):
        result = classify(1, "404 Not Found")
        assert result == ("fatal", "404")
    
    def test_fatal_no_such_file_or_directory(self):
        result = classify(1, "No such file or directory")
        assert result == ("fatal", "no such file or directory")
    
    def test_fatal_unsupported_architecture(self):
        result = classify(1, "Unsupported architecture")
        assert result == ("fatal", "unsupported architecture")
    
    def test_fatal_checksum_mismatch(self):
        result = classify(1, "Checksum mismatch")
        assert result == ("fatal", "checksum mismatch")
    
    # Test exit_code 0 with scary stderr -> unknown
    def test_exit_code_zero_with_scary_stderr(self):
        result = classify(0, "No space left on device")
        assert result == ("unknown", "")
    
    # Test precedence: needs_operator > fatal > transient
    def test_precedence_needs_operator_over_transient(self):
        result = classify(1, "timeout and No space left on device")
        assert result == ("needs_operator", "no space left on device")
    
    def test_precedence_fatal_over_transient(self):
        result = classify(1, "timeout and 404 Not Found")
        assert result == ("fatal", "404")
    
    def test_precedence_needs_operator_over_fatal(self):
        result = classify(1, "No space left on device and 404 Not Found")
        assert result == ("needs_operator", "no space left on device")
    
    # Test case-insensitivity
    def test_case_insensitive_transient(self):
        result = classify(1, "TIMEOUT occurred")
        assert result == ("transient", "timeout")
    
    def test_case_insensitive_needs_operator(self):
        result = classify(1, "NO SPACE LEFT ON DEVICE")
        assert result == ("needs_operator", "no space left on device")
    
    def test_case_insensitive_fatal(self):
        result = classify(1, "404 NOT FOUND")
        assert result == ("fatal", "404")
    
    # Test empty stderr with nonzero exit -> unknown
    def test_empty_stderr_nonzero_exit(self):
        result = classify(1, "")
        assert result == ("unknown", "")
    
    # Test non-int exit_code
    def test_non_int_exit_code(self):
        result = classify("1", "error")
        assert result == ("unknown", "")
    
    def test_none_exit_code(self):
        result = classify(None, "error")
        assert result == ("unknown", "")
    
    # Test non-string stderr
    def test_non_string_stderr(self):
        result = classify(1, 123)
        assert result == ("unknown", "")
    
    def test_none_stderr(self):
        result = classify(1, None)
        assert result == ("unknown", "")
    
    # Test never raising (edge cases)
    def test_float_exit_code(self):
        result = classify(1.0, "error")
        assert result == ("unknown", "")
    
    def test_list_stderr(self):
        result = classify(1, ["error"])
        assert result == ("unknown", "")
    
    def test_dict_stderr(self):
        result = classify(1, {"error": "message"})
        assert result == ("unknown", "")