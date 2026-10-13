"""Signed capacity registrar: local probes are fully isolated from the real node."""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest

import nakshatra_auth as auth
import registrar


def _config(tmp_path, **updates):
    env = {
        "NKS_NODE_ID": "test-node",
        "NKS_PILLAR_URL": "http://pillar.test:7777/",
        "NKS_NODE_ADDRESS": "127.0.0.1:5560",
        "NKS_GPU_BACKEND": "cpu",
        "MESH_COMPUTE_STATE": str(tmp_path / "mesh-compute"),
    }
    env.update(updates)
    return registrar.load_config(env, home=tmp_path, hostname="unused-host")


def test_load_config_has_no_import_time_environment_state(tmp_path):
    first = _config(tmp_path, NKS_NODE_ID="first", NKS_OWNER_RESERVE_GB="2")
    second = _config(tmp_path, NKS_NODE_ID="second", NKS_OWNER_RESERVE_GB="7")
    assert (first.node_id, first.reserve_gb) == ("first", 2.0)
    assert (second.node_id, second.reserve_gb) == ("second", 7.0)


def test_ollama_discovery_is_opt_in(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OLLAMA_URL="")
    monkeypatch.setattr(registrar, "_ollama_json", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not query Ollama")))
    assert registrar._ollama_inventory(config) == ([], [])


def test_whole_ollama_models_become_single_link_chains(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OLLAMA_URL="http://127.0.0.1:11434")

    def fake_json(_config, path, body=None):
        if path == "/api/tags":
            return {"models": [
                {"name": "qwen2.5-coder:14b", "size": 8_988_124_298,
                 "details": {"family": "qwen2"}, "capabilities": ["completion", "tools"]},
                {"name": "llama3.2:latest", "size": 2_019_393_189,
                 "details": {"family": "llama"}, "capabilities": ["completion"]},
            ]}
        assert path == "/api/show"
        if body["model"] == "qwen2.5-coder:14b":
            return {"details": {"family": "qwen2"}, "model_info": {"qwen2.block_count": 48}}
        return {"details": {"family": "llama"}, "model_info": {"llama.block_count": 28}}

    monkeypatch.setattr(registrar, "_ollama_json", fake_json)
    offerings, files = registrar._ollama_inventory(config)
    assert offerings == [
        {"model_id": "qwen2.5-coder:14b", "model_sha256": "", "layer_start": 0, "layer_end": 48},
        {"model_id": "llama3.2", "model_sha256": "", "layer_start": 0, "layer_end": 28},
    ]
    assert [item["size_bytes"] for item in files] == [8_988_124_298, 2_019_393_189]
    assert all(item["file_path"] == "" and item["model_sha256"] == "" for item in files)


def test_family_key_wins_over_an_unrelated_vision_tower():
    info = {"llama.block_count": 28, "clip.block_count": 24}
    assert registrar._ollama_block_count(info, "llama") == 28
    assert registrar._ollama_block_count(info) == 0


def test_inventory_filters_non_chat_and_allowlisted_models(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OLLAMA_URL="http://ollama", NKS_OLLAMA_MODELS="wanted")
    shown = []

    def fake_json(_config, path, body=None):
        if path == "/api/tags":
            return {"models": [
                {"name": "embed:latest", "capabilities": ["embedding"]},
                {"name": "other:latest", "capabilities": ["completion"]},
                {"name": "wanted:latest", "capabilities": ["completion"]},
            ]}
        shown.append(body["model"])
        return {"details": {"family": "x"}, "model_info": {"x.block_count": 7}}

    monkeypatch.setattr(registrar, "_ollama_json", fake_json)
    offerings, _ = registrar._ollama_inventory(config)
    assert shown == ["wanted:latest"]
    assert offerings[0]["model_id"] == "wanted"


def test_ollama_failure_does_not_suppress_capacity_payload(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OLLAMA_URL="http://ollama", NKS_OWNER_RESERVE_GB="4")
    monkeypatch.setattr(registrar, "_ollama_json", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(registrar, "_gpu", lambda _config: (12.0, 16.0, "GPU", "cuda"))
    payload = registrar.build_payload(config, "ab" * 32)
    assert payload["layer_offerings"] == [] and payload["cached_files"] == []
    assert payload["budget"]["vram_offered_gb"] == 8.0


def test_reserve_never_makes_offered_vram_negative(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OWNER_RESERVE_GB="20")
    monkeypatch.setattr(registrar, "_gpu", lambda _config: (12.0, 16.0, "GPU", "cuda"))
    monkeypatch.setattr(registrar, "_ollama_inventory", lambda _config: ([], []))
    assert registrar.build_payload(config, "ab" * 32)["budget"] == {"vram_offered_gb": 0.0}


def test_heartbeat_posts_inventory_and_a_signature_over_the_exact_body(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OWNER_RESERVE_GB="4")
    private_key, public_key_hex = auth.generate_keypair()
    offerings = [{"model_id": "llama3.2", "model_sha256": "", "layer_start": 0, "layer_end": 28}]
    files = [{**offerings[0], "size_bytes": 123, "file_path": ""}]
    monkeypatch.setattr(registrar, "_gpu", lambda _config: (10.0, 16.0, "GPU", "cuda"))
    monkeypatch.setattr(registrar, "_ollama_inventory", lambda _config: (offerings, files))
    monkeypatch.setattr(registrar, "_mesh_compute_blocks", lambda _config: {"compute": {"offered": True}})
    captured = {}

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return None

    def fake_urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return Response()

    monkeypatch.setattr(registrar.urllib.request, "urlopen", fake_urlopen)
    assert registrar.heartbeat_once(config, private_key, public_key_hex) == (200, 6.0, "GPU")
    request = captured["request"]
    body = json.loads(request.data)
    assert request.full_url == "http://pillar.test:7777/peer" and captured["timeout"] == 10
    assert body["layer_offerings"] == offerings and body["cached_files"] == files
    assert body["public_key_hex"] == public_key_hex
    header = request.get_header("Authorization")
    match = re.search(r'sig="([^"]+)",ts="(\d+)"', header)
    assert match and auth.verify_request(public_key_hex, "POST", "/peer", request.data,
                                         int(match.group(2)), match.group(1))


def test_gpu_and_network_are_mocked_in_heartbeat_test(tmp_path, monkeypatch):
    """A regression guard: the complete heartbeat path does not need this machine's GPU/network."""
    config = _config(tmp_path)
    monkeypatch.setattr(registrar, "_gpu", lambda _config: (1.0, 2.0, "fake", "cpu"))
    monkeypatch.setattr(registrar, "_ollama_inventory", lambda _config: ([], []))
    monkeypatch.setattr(registrar.na, "build_signed_envelope", lambda *args: ("signed", 1))

    class Response:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *args): return None

    monkeypatch.setattr(registrar.urllib.request, "urlopen", lambda *a, **k: Response())
    assert registrar.heartbeat_once(config, b"x" * 32, "pub") == (204, 1.0, "fake")


def test_invalid_health_fields_are_omitted_without_suppressing_heartbeat(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    config.mesh_compute_state.mkdir()
    (config.mesh_compute_state / "health.json").write_text(json.dumps({
        "state": ["healthy"],
        "reasons": None,
        "last_crash": "not-an-object",
        "checked_ts": ["not-a-timestamp"],
    }))
    monkeypatch.setattr(registrar, "_gpu", lambda _config: (1.0, 2.0, "fake", "cpu"))
    monkeypatch.setattr(registrar, "_ollama_inventory", lambda _config: ([], []))
    monkeypatch.setattr(registrar.na, "build_signed_envelope", lambda *args: ("signed", 1))
    captured = {}

    class Response:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *args): return None

    def fake_urlopen(request, **kwargs):
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(registrar.urllib.request, "urlopen", fake_urlopen)
    assert registrar.heartbeat_once(config, b"x" * 32, "pub") == (204, 1.0, "fake")
    assert captured["body"]["health"] == {}
    warnings = [line for line in capsys.readouterr().out.splitlines() if "warning:" in line]
    assert len(warnings) == 1
    assert "state, reasons, last_crash, checked_ts" in warnings[0]


def test_strict_worker_key_load_refuses_missing_malformed_and_open_mode(tmp_path):
    key = tmp_path / "worker.ed25519"
    with pytest.raises(FileNotFoundError, match="missing"):
        auth.load_worker_key(key)
    key.write_bytes(b"short")
    key.chmod(0o600)
    with pytest.raises(ValueError, match="32-byte"):
        auth.load_worker_key(key)
    key.write_bytes(os.urandom(32))
    key.chmod(0o644)
    with pytest.raises(PermissionError, match=r"chmod 600"):
        auth.load_worker_key(key)
    assert (key.stat().st_mode & 0o777) == 0o644  # loader refused; it did not repair custody


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_strict_worker_key_load_accepts_owner_only_modes(tmp_path, mode):
    key = tmp_path / "worker.ed25519"
    key.write_bytes(os.urandom(32))
    key.chmod(mode)
    private_key, public_key_hex = auth.load_worker_key(key)
    assert private_key == key.read_bytes()
    assert public_key_hex == auth.public_key_hex_from_private(private_key)


def test_strict_worker_key_load_reads_the_same_fd_it_validated(tmp_path, monkeypatch):
    key = tmp_path / "worker.ed25519"
    replacement = tmp_path / "replacement.ed25519"
    original_private, _ = auth.generate_keypair()
    replacement_private, _ = auth.generate_keypair()
    key.write_bytes(original_private)
    replacement.write_bytes(replacement_private)
    key.chmod(0o600)
    replacement.chmod(0o600)
    real_fstat = auth.os.fstat

    def swap_path_after_fstat(fd):
        key_stat = real_fstat(fd)
        replacement.replace(key)
        return key_stat

    monkeypatch.setattr(auth.os, "fstat", swap_path_after_fstat)
    private_key, public_key_hex = auth.load_worker_key(key)
    assert private_key == original_private
    assert public_key_hex == auth.public_key_hex_from_private(original_private)
    assert key.read_bytes() == replacement_private


def test_strict_worker_key_load_refuses_symlink_and_wrong_owner(tmp_path, monkeypatch):
    key = tmp_path / "worker.ed25519"
    key.write_bytes(os.urandom(32))
    key.chmod(0o600)
    symlink = tmp_path / "linked.ed25519"
    symlink.symlink_to(key)
    with pytest.raises(OSError):
        auth.load_worker_key(symlink)

    real_fstat = auth.os.fstat

    def foreign_fstat(fd):
        key_stat = real_fstat(fd)
        return SimpleNamespace(st_mode=key_stat.st_mode, st_uid=auth.os.getuid() + 1)

    monkeypatch.setattr(auth.os, "fstat", foreign_fstat)
    with pytest.raises(PermissionError, match="owned by uid"):
        auth.load_worker_key(key)


def test_an_absurd_ollama_size_never_suppresses_the_heartbeat(tmp_path, monkeypatch):
    config = _config(tmp_path, NKS_OLLAMA_URL="http://127.0.0.1:11434")

    def fake_json(_config, path, body=None):
        if path == "/api/tags":
            return {"models": [{"name": "huge:1", "size": float("inf"), "details": {"family": "llama"}}]}
        return {"details": {"family": "llama"}, "model_info": {"llama.block_count": 28}}

    monkeypatch.setattr(registrar, "_ollama_json", fake_json)
    offerings, files = registrar._ollama_inventory(config)
    assert [o["model_id"] for o in offerings] == ["huge:1"]
    assert files[0]["size_bytes"] == 0


def test_rocm_vram_falls_back_to_amdgpu_sysfs(tmp_path, monkeypatch):
    """No rocm-smi on PATH (the hub): VRAM comes from the amdgpu sysfs counters, largest card."""
    gib = 1024 ** 3
    for name, total, used in (("card1", 20 * gib, 19 * gib), ("card2", 16 * gib, 4 * gib),
                              ("card3", 2 * gib, 0)):
        dev = tmp_path / name / "device"
        dev.mkdir(parents=True)
        (dev / "mem_info_vram_total").write_text(f"{total}\n")
        (dev / "mem_info_vram_used").write_text(f"{used}\n")
    (tmp_path / "card0" / "device").mkdir(parents=True)  # a card without counters is skipped
    monkeypatch.setattr(registrar, "SYSFS_DRM", tmp_path)
    monkeypatch.setattr(registrar.shutil, "which", lambda _name: None)
    config = _config(tmp_path, NKS_GPU_BACKEND="rocm")
    free, total, _model, backend = registrar._gpu(config)
    assert backend == "rocm" and round(total) == 20 and round(free) == 1

