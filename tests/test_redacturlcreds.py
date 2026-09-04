import pytest
from redacturlcreds import redact_url_creds


def test_user_pass_redacted_with_scheme_host_path_kept():
    """user:pass redacted with scheme/host/path kept"""
    text = "https://admin:secret123@example.com/path/to/resource"
    expected = "https://«redacted:token»@example.com/path/to/resource"
    assert redact_url_creds(text) == expected


def test_url_with_no_creds_unchanged():
    """A URL with no creds unchanged"""
    text = "https://example.com/x"
    expected = "https://example.com/x"
    assert redact_url_creds(text) == expected


def test_two_urls_on_one_line():
    """Two URLs on one line"""
    text = "Visit https://user1:pass1@host1.com and http://user2:pass2@host2.com/path"
    expected = "Visit https://«redacted:token»@host1.com and http://«redacted:token»@host2.com/path"
    assert redact_url_creds(text) == expected


def test_at_in_path_after_host_untouched():
    """An @ in a path after the host untouched"""
    text = "https://example.com/path@with@ats"
    expected = "https://example.com/path@with@ats"
    assert redact_url_creds(text) == expected


def test_bare_user_host_in_prose_untouched():
    """A bare user@host in prose untouched"""
    text = "Contact user@host for help"
    expected = "Contact user@host for help"
    assert redact_url_creds(text) == expected


def test_git_ssh_style_scheme():
    """git+ssh:// style scheme"""
    text = "git+ssh://deploy:key123@github.com/org/repo.git"
    expected = "git+ssh://«redacted:token»@github.com/org/repo.git"
    assert redact_url_creds(text) == expected


def test_idempotency():
    """Idempotency — a second pass must not redact inside the marker"""
    text1 = "https://admin:secret@example.com/path"
    redacted_once = redact_url_creds(text1)
    redacted_twice = redact_url_creds(redacted_once)
    assert redacted_once == redacted_twice


def test_line_count_preserved():
    """Line count preserved"""
    text = "Line 1: https://user:pass@host1.com\nLine 2: https://user2:pass2@host2.com"
    result = redact_url_creds(text)
    assert result.count('\n') == text.count('\n')


def test_clean_line_byte_identical():
    """A clean line byte-identical"""
    text = "https://example.com/path"
    result = redact_url_creds(text)
    assert result == text


def test_non_string():
    """Non-string returns empty string"""
    assert redact_url_creds(123) == ""
    assert redact_url_creds(None) == ""
    assert redact_url_creds([]) == ""
    assert redact_url_creds({}) == ""


def test_empty():
    """Empty string"""
    assert redact_url_creds("") == ""