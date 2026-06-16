"""PackageSlicer — tested without real GGUFs (assemble_fn + manifest_reader injected).

Proves: slices are content-addressed by the package REVISION; a re-plan of the same range reuses the
cached file (no re-assembly); a model update (new revision) writes a fresh file; the signature policy
is passed through to the assembler fail-closed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_slicer as ps


class _FakeWorker:
    node_id = "w"


def _slicer(tmp, revision="revAAAAAAAAAAAA", calls=None, sign=False, trusted=None):
    calls = calls if calls is not None else []

    def manifest_reader(loc):
        return ("dsr1", revision, 32)

    def assemble_fn(loc, start, end, dest, *, require_signature, trusted_pubkeys):
        calls.append({"start": start, "end": end, "dest": dest,
                      "require_signature": require_signature, "trusted": trusted_pubkeys})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(f"slice[{start},{end})@{revision}")  # stand in for a real sub-GGUF
        return dest

    return ps.PackageSlicer("/pkg", cache_dir=str(tmp), require_signature=sign,
                            trusted_pubkeys=trusted, assemble_fn=assemble_fn,
                            manifest_reader=manifest_reader), calls


def test_content_addressed_dest_includes_revision(tmp_path):
    s, _ = _slicer(tmp_path, revision="deadbeefcafe00")
    dest = s.dest_for("dsr1", 0, 20)
    assert dest.name == "dsr1@deadbeefcafe-L0-20.gguf"


def test_assembles_then_reuses(tmp_path):
    s, calls = _slicer(tmp_path)
    p1 = s.slice_for(_FakeWorker(), 0, 20, "dsr1")
    p2 = s.slice_for(_FakeWorker(), 0, 20, "dsr1")   # same range -> cached, no re-assembly
    assert p1 == p2
    assert len(calls) == 1, "second call must hit the cache, not re-assemble"
    assert Path(p1).exists()


def test_distinct_ranges_distinct_slices(tmp_path):
    s, calls = _slicer(tmp_path)
    a = s.slice_for(_FakeWorker(), 0, 20, "dsr1")
    b = s.slice_for(_FakeWorker(), 20, 32, "dsr1")
    assert a != b and len(calls) == 2
    assert {(c["start"], c["end"]) for c in calls} == {(0, 20), (20, 32)}


def test_new_revision_writes_fresh_file(tmp_path):
    s1, _ = _slicer(tmp_path, revision="rev1AAAAAAAAAA")
    old = s1.slice_for(_FakeWorker(), 0, 16, "dsr1")
    s2, c2 = _slicer(tmp_path, revision="rev2BBBBBBBBBB")
    new = s2.slice_for(_FakeWorker(), 0, 16, "dsr1")
    assert old != new, "a new package revision must not reuse the old slice"
    assert len(c2) == 1


def test_signature_policy_passed_through(tmp_path):
    trusted = {"abc123"}
    s, calls = _slicer(tmp_path, sign=True, trusted=trusted)
    s.slice_for(_FakeWorker(), 0, 8, "dsr1")
    assert calls[0]["require_signature"] is True
    assert calls[0]["trusted"] == trusted


def test_registry_resolve(tmp_path):
    import yaml
    reg = tmp_path / "packages.yaml"
    reg.write_text(yaml.safe_dump({"packages": {"dsr1": "/home/x/.nakshatra/packages/dsr1"}}))
    assert ps.resolve_package_location("dsr1", registry_path=str(reg)) == "/home/x/.nakshatra/packages/dsr1"
    assert ps.resolve_package_location("nope", registry_path=str(reg)) is None
    assert ps.resolve_package_location("dsr1", registry_path=str(tmp_path / "absent.yaml")) is None


# tiny pytest-free runner (mirrors the other fabric tests)
def _run():
    import tempfile, shutil
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        d = Path(tempfile.mkdtemp(prefix="nks-slicer-test-"))
        try:
            fn(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"all package_slicer tests PASS ({len(fns)})")


if __name__ == "__main__":
    _run()
