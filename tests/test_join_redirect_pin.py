"""Fix 1 (SSRF/DNS-rebinding via HTTP redirect): a URL that resolves publicly must not be
able to smuggle a request to a private/link-local address through a redirect, on either the
curl path (`act.download`) or the `urlopen` path (`act.fetch_join_info`).

⚠️⚠️ WHY THIS FILE EXISTS. `_resolved_ips_are_safe()` checked the URL's host ONCE, then a
separate `curl -fSL` (or `urlopen()`) re-resolved the same hostname itself moments later with
no IP pinning (TOCTOU) and, worse, followed redirects unconditionally with no re-validation
of the Location host. A URL that resolves publicly could 302 to 169.254.169.254 or any
RFC1918 address and the transport would follow it. There was no test proving a redirect is
refused — which is why it shipped.
"""
import urllib.error

import pytest

from join import act


def test_pin_argv_pins_the_checked_ips_and_drops_redirect_following():
    """`_pin_argv` is what closes the TOCTOU + redirect gap on the curl path: the IPs already
    validated by `_resolved_ips_are_safe` get pinned via `--resolve` (curl's own DNS
    resolution never runs for this host), and `-fSL` loses its `L` — no redirect following."""
    argv = ["curl", "-fSL", "--proto", "=https", "-o", "/tmp/x", "https://cdn.example.com/m.gguf"]
    out = act._pin_argv(argv, "https://cdn.example.com/m.gguf", ["8.8.8.8"])
    assert "-fSL" not in out
    assert "-fS" in out
    assert "--resolve" in out
    i = out.index("--resolve")
    assert out[i + 1] == "cdn.example.com:443:8.8.8.8"
    # curl must never be left to resolve this host itself
    assert "-L" not in out


def test_pin_argv_pins_multiple_ips_comma_joined():
    argv = ["curl", "-fSL", "-o", "/tmp/x", "https://cdn.example.com/m.gguf"]
    out = act._pin_argv(argv, "https://cdn.example.com/m.gguf", ["8.8.8.8", "8.8.4.4"])
    i = out.index("--resolve")
    assert out[i + 1] == "cdn.example.com:443:8.8.8.8,8.8.4.4"


def test_download_pins_ips_into_the_actual_curl_invocation(monkeypatch):
    """End to end through `act.download`: the argv that reaches `subprocess.run` must be the
    pinned one, not the caller-supplied one re-resolving on its own."""
    seen = {}

    def fake_getaddrinfo(host, *a, **kw):
        assert host == "cdn.example.com"
        import socket as s
        return [(s.AF_INET, s.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    def fake_run(argv, **kw):
        seen["argv"] = argv
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(act._socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(act.subprocess, "run", fake_run)
    monkeypatch.setattr(act.os.path, "getsize", lambda p: 1)
    monkeypatch.setattr(act, "sha256_file", lambda p: "deadbeef")

    argv = ["curl", "-fSL", "-o", "/tmp/dest", "https://cdn.example.com/m.gguf"]
    out = act.download(argv, "/tmp/dest", url="https://cdn.example.com/m.gguf")

    assert out["download_exit_code"] == 0
    assert "--resolve" in seen["argv"]
    assert "-fSL" not in seen["argv"]
    assert "cdn.example.com:443:8.8.8.8" in seen["argv"]


def test_download_refuses_before_ever_building_a_pinned_argv_when_host_resolves_private(monkeypatch):
    def fake_getaddrinfo(host, *a, **kw):
        import socket as s
        return [(s.AF_INET, s.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]

    called = {"run": False}

    def fake_run(argv, **kw):
        called["run"] = True
        raise AssertionError("curl must never run against an unsafe host")

    monkeypatch.setattr(act._socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(act.subprocess, "run", fake_run)

    out = act.download(["curl", "-fSL", "-o", "/tmp/d", "https://evil.example.com/m.gguf"],
                        "/tmp/d", url="https://evil.example.com/m.gguf")
    assert out["download_exit_code"] == 126
    assert "refused" in out["download_stderr"]
    assert called["run"] is False


def test_safe_redirect_handler_refuses_a_hop_to_a_private_address():
    """The `urlopen` path (`fetch_join_info`) has the exact same class of gap as curl's `-L`:
    a redirect is followed with no re-check of where it now points. `_SafeRedirectHandler`
    is what closes it — assert it actually raises rather than silently following."""
    handler = act._SafeRedirectHandler()
    req = act.urllib.request.Request("https://coordinator.example.com/v1/join-info")
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(req, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/")


def test_safe_redirect_handler_refuses_a_hop_to_loopback():
    handler = act._SafeRedirectHandler()
    req = act.urllib.request.Request("https://coordinator.example.com/v1/join-info")
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:8080/admin")


def test_safe_redirect_handler_allows_a_hop_to_another_public_host(monkeypatch):
    """Proves the handler discriminates rather than refusing everything — a redirect to a
    second legitimately-public host must still be allowed to proceed."""
    monkeypatch.setattr(act, "_resolved_ips_are_safe",
                        lambda url: (True, "", ["9.9.9.9"]))
    handler = act._SafeRedirectHandler()
    req = act.urllib.request.Request("https://coordinator.example.com/v1/join-info")
    # super().redirect_request builds a new Request when allowed; no exception means "allowed"
    new_req = handler.redirect_request(req, None, 302, "Found", {}, "https://mirror.example.com/v1/join-info")
    assert new_req is not None
