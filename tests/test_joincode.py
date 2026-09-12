import pytest
from joincode import encode_join, decode_join


class TestJoinCode:
    def test_round_trip_full(self):
        coordinator = "https://example.com"
        token = "secret-token-123"
        relay = "https://relay.example.com"
        expires_at = 9999999999

        code = encode_join(coordinator, token, relay=relay, expires_at=expires_at)
        result = decode_join(code, now=1000000000)

        assert result["coordinator"] == coordinator
        assert result["token"] == token
        assert result["relay"] == relay
        assert result["expires_at"] == expires_at

    def test_round_trip_minimal(self):
        coordinator = "https://example.com"
        token = "secret-token-123"

        code = encode_join(coordinator, token)
        result = decode_join(code, now=1000000000)

        assert result["coordinator"] == coordinator
        assert result["token"] == token
        assert result["relay"] == ""
        assert result["expires_at"] == 0

    def test_expired_code_raises(self):
        coordinator = "https://example.com"
        token = "secret-token-123"
        expires_at = 2000000000

        code = encode_join(coordinator, token, expires_at=expires_at)

        with pytest.raises(ValueError) as exc_info:
            decode_join(code, now=3000000000)

        assert "expired" in str(exc_info.value).lower()

    def test_valid_before_expiry(self):
        coordinator = "https://example.com"
        token = "secret-token-123"
        expires_at = 3000000000

        code = encode_join(coordinator, token, expires_at=expires_at)

        result = decode_join(code, now=2000000000)

        assert result["coordinator"] == coordinator
        assert result["token"] == token
        assert result["expires_at"] == expires_at

    def test_missing_c_raises(self):
        coordinator = "https://example.com"
        token = "secret-token-123"
        relay = "https://relay.example.com"
        expires_at = 9999999999

        code = encode_join(coordinator, token, relay=relay, expires_at=expires_at)
        # Manually remove 'c' from the encoded code by decoding, modifying, re-encoding
        import base64
        import json

        # Decode the code
        padding = 4 - (len(code) % 4)
        if padding != 4:
            code += "=" * padding
        decoded = base64.urlsafe_b64decode(code)
        data = json.loads(decoded.decode("utf-8"))
        del data["c"]
        modified_payload = json.dumps(data, separators=(",", ":"))
        modified_code = base64.urlsafe_b64encode(modified_payload.encode("utf-8")).decode("ascii").rstrip("=")

        with pytest.raises(ValueError) as exc_info:
            decode_join(modified_code, now=1000000000)

        assert "coordinator" in str(exc_info.value).lower()

    def test_missing_t_raises(self):
        coordinator = "https://example.com"
        token = "secret-token-123"
        relay = "https://relay.example.com"
        expires_at = 9999999999

        code = encode_join(coordinator, token, relay=relay, expires_at=expires_at)
        # Manually remove 't' from the encoded code
        import base64
        import json

        # Decode the code
        padding = 4 - (len(code) % 4)
        if padding != 4:
            code += "=" * padding
        decoded = base64.urlsafe_b64decode(code)
        data = json.loads(decoded.decode("utf-8"))
        del data["t"]
        modified_payload = json.dumps(data, separators=(",", ":"))
        modified_code = base64.urlsafe_b64encode(modified_payload.encode("utf-8")).decode("ascii").rstrip("=")

        with pytest.raises(ValueError) as exc_info:
            decode_join(modified_code, now=1000000000)

        assert "token" in str(exc_info.value).lower()

    def test_empty_t_raises(self):
        coordinator = "https://example.com"
        token = ""

        code = encode_join(coordinator, token)
        with pytest.raises(ValueError) as exc_info:
            decode_join(code, now=1000000000)

        assert "token" in str(exc_info.value).lower()

    def test_non_url_coordinator_raises(self):
        coordinator = "ftp://example.com"
        token = "secret-token-123"

        code = encode_join(coordinator, token)
        with pytest.raises(ValueError) as exc_info:
            decode_join(code, now=1000000000)

        assert "http" in str(exc_info.value).lower()

    def test_malformed_base64_raises(self):
        with pytest.raises(ValueError) as exc_info:
            decode_join("!!!invalid_base64!!!", now=1000000000)

        assert "base64" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_base64_of_json_array_raises(self):
        import base64
        import json

        arr = ["not", "an", "object"]
        payload = json.dumps(arr, separators=(",", ":"))
        code = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")

        with pytest.raises(ValueError) as exc_info:
            decode_join(code, now=1000000000)

        assert "object" in str(exc_info.value).lower()

    def test_unknown_key_raises(self):
        coordinator = "https://example.com"
        token = "secret-token-123"

        code = encode_join(coordinator, token)
        # Manually add an unknown key
        import base64
        import json

        # Decode the code
        padding = 4 - (len(code) % 4)
        if padding != 4:
            code += "=" * padding
        decoded = base64.urlsafe_b64decode(code)
        data = json.loads(decoded.decode("utf-8"))
        data["unknown"] = "value"
        modified_payload = json.dumps(data, separators=(",", ":"))
        modified_code = base64.urlsafe_b64encode(modified_payload.encode("utf-8")).decode("ascii").rstrip("=")

        with pytest.raises(ValueError) as exc_info:
            decode_join(modified_code, now=1000000000)

        assert "unknown" in str(exc_info.value).lower()

    def test_token_not_in_error_messages(self):
        coordinator = "https://example.com"
        token = "secret-token-123"

        # Test various error conditions and ensure token is not in error messages
        test_cases = [
            # Empty token
            lambda: encode_join(coordinator, ""),
            # Malformed base64
            lambda: decode_join("!!!invalid_base64!!!", now=1000000000),
            # JSON array instead of object
            lambda: (
                __import__("base64").urlsafe_b64encode(
                    __import__("json").dumps(["not", "an", "object"], separators=(",", ":")).encode("utf-8")
                ).decode("ascii").rstrip("=")
            ),
            # Unknown key
            lambda: (
                __import__("base64").urlsafe_b64encode(
                    __import__("json").dumps(
                        {"c": coordinator, "t": token, "unknown": "value"}, separators=(",", ":")
                    ).encode("utf-8")
                ).decode("ascii").rstrip("=")
            ),
        ]

        for i, test_case in enumerate(test_cases):
            try:
                result = test_case()
                if isinstance(result, str):
                    decode_join(result, now=1000000000)
            except ValueError as e:
                error_msg = str(e)
                assert token not in error_msg, f"Token found in error message for test case {i}: {error_msg}"
            except Exception:
                # Some test cases may raise other exceptions, which is fine
                pass

        # Test expired code
        code = encode_join(coordinator, token, expires_at=2000000000)
        try:
            decode_join(code, now=3000000000)
        except ValueError as e:
            assert token not in str(e), f"Token found in error message for expired code: {str(e)}"

        # Test non-URL coordinator
        code = encode_join("ftp://example.com", token)
        try:
            decode_join(code, now=1000000000)
        except ValueError as e:
            assert token not in str(e), f"Token found in error message for non-URL coordinator: {str(e)}"