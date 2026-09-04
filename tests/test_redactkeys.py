import pytest
from redactkeys import redact_keys


def test_private_key_redacted():
    """A 64-char hex private key should be redacted."""
    private_key = "a" * 64
    result = redact_keys(private_key)
    assert result == "«redacted:key»"


def test_sha256_redacted():
    """A 64-char hex SHA256 digest should be redacted."""
    sha256_hash = "b" * 64
    result = redact_keys(sha256_hash)
    assert result == "«redacted:key»"


def test_both_in_one_line():
    """Two 64-char hex strings on the same line should both be redacted."""
    text = "key1: " + "c" * 64 + " key2: " + "d" * 64
    result = redact_keys(text)
    expected = "key1: «redacted:key» key2: «redacted:key»"
    assert result == expected


def test_63_char_run_untouched():
    """A 63-char hex run should NOT be redacted."""
    text = "a" * 63
    result = redact_keys(text)
    assert result == text


def test_65_char_run_untouched():
    """A 65-char hex run should NOT be redacted."""
    text = "a" * 65
    result = redact_keys(text)
    assert result == text


def test_uppercase_hex_redacted():
    """Uppercase hex should be redacted."""
    text = "A" * 64
    result = redact_keys(text)
    assert result == "«redacted:key»"


def test_idempotency():
    """Applying redact_keys twice should give the same result."""
    text = "key: " + "a" * 64 + " more text"
    result1 = redact_keys(text)
    result2 = redact_keys(result1)
    assert result1 == result2


def test_line_count_preserved():
    """Line count should be preserved."""
    text = "line1\n" + "a" * 64 + "\nline3\n"
    result = redact_keys(text)
    assert result.count("\n") == text.count("\n")


def test_line_with_no_keys_byte_identical():
    """A line with no keys should be byte-identical."""
    text = "no keys here\n"
    result = redact_keys(text)
    assert result == text


def test_non_string_input():
    """Non-string input should return empty string."""
    assert redact_keys(123) == ""
    assert redact_keys(None) == ""
    assert redact_keys(["list"]) == ""
    assert redact_keys({"dict": "value"}) == ""


def test_empty_string():
    """Empty string should return empty string."""
    assert redact_keys("") == ""