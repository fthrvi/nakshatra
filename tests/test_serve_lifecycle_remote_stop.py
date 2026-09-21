"""RemoteSshController: an explicit `stop` command for remote workers whose ssh login shell is not bash.

A Windows/WSL node (blackwell) lands in PowerShell over ssh, where the default reap - `pkill -f '<pattern>' 2>/dev/null;
pkill -f llama-nakshatra-worker ...` - does not exist. `stop` lets the config name the command; the pkill default stays
for every existing config.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_lifecycle as sl  # noqa: E402


class _Recorder(sl.RemoteSshController):
    def __init__(self, workers):
        super().__init__(workers, log=lambda *_: None)
        self.calls = []

    def _ssh(self, uh, remote_cmd, timeout=30.0):
        self.calls.append((uh, remote_cmd))
        return 0


def _w(**kw):
    base = dict(name="w", ssh="box", launch="start-it", probe=("127.0.0.1", 1), stop_match="worker.py --port 5572")
    base.update(kw)
    return sl.RemoteWorker(**base)


def test_the_default_reap_is_still_pkill_by_pattern():
    ctl = _Recorder([_w()])
    ctl.stop()
    assert ctl.calls == [("box", "pkill -f 'worker.py --port 5572' 2>/dev/null; pkill -f llama-nakshatra-worker 2>/dev/null; true")]


def test_an_explicit_stop_command_replaces_the_pkill_default():
    ctl = _Recorder([_w(stop="wsl -e bash /home/prithvi/nks-q3a.sh stop", stop_match="")])
    ctl.stop()
    assert ctl.calls == [("box", "wsl -e bash /home/prithvi/nks-q3a.sh stop")]


def test_json_loader_reads_stop_and_keeps_stop_match_optional_when_stop_is_given(tmp_path):
    cfg = tmp_path / "remote.json"
    cfg.write_text(json.dumps({"remote_workers": [
        {"name": "a", "ssh": "ijru", "launch": "l", "probe": "10.51.0.14:5572", "stop_match": "worker.py --port 5572"},
        {"name": "b", "ssh": "blackwell", "launch": "l2", "probe": "10.42.0.7:5562", "stop": "wsl -e bash x.sh stop"}]}))
    a, b = sl._remote_workers_from_json(str(cfg))
    assert (a.stop, a.stop_match) == ("", "worker.py --port 5572")
    assert (b.stop, b.stop_match, b.probe) == ("wsl -e bash x.sh stop", "", ("10.42.0.7", 5562))


def test_a_worker_with_neither_stop_nor_stop_match_is_refused_at_load_time(tmp_path):
    cfg = tmp_path / "remote.json"
    cfg.write_text(json.dumps({"remote_workers": [{"name": "c", "ssh": "h", "launch": "l", "probe": "h:1"}]}))
    with pytest.raises(ValueError, match="needs 'stop' or 'stop_match'"):
        sl._remote_workers_from_json(str(cfg))
