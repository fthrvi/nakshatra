#!/usr/bin/env python3
"""Sthambha planner Step 4 — worker /slice endpoint integration test.

Tests the HTTP orchestration we added to worker.py's FileServerHandler:
  - POST /slice spawns a subprocess and returns 202 + task_id
  - GET /slice/<task_id> reports status
  - On completion: sha256 is computed, output is in _FILE_SERVER_DIR
  - Failed subprocess → status=failed with captured stderr
  - 400 for malformed input, 404 for unknown task_id

The actual partial_gguf.py invocation is not exercised here — it has been
in production use since v0.0 and its behaviour is orthogonal to the HTTP
orchestration we're testing. We point _PARTIAL_GGUF_PATH at a fake slicer
that takes the same CLI shape and writes deterministic content. This keeps
the test stdlib-only and runs in <1s.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

sys.path.insert(0, str(Path(__file__).parent))

import worker as W  # noqa: E402  — module under test


def _http_post_json(url: str, body: dict, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST",
                              headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urlerror.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        return e.code, {"_raw": body_text}


def _http_get_json(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urlerror.HTTPError as e:
        return e.code, {"_raw": e.read().decode("utf-8", errors="replace")}


def _make_fake_slicer(target_dir: Path) -> Path:
    """Write a fake partial_gguf.py that matches its CLI shape.

    The fake takes the same positional + flag args, validates them in the
    same shapes, and writes deterministic content keyed on (start, end).
    Setting NAKSHATRA_SLICER_FAIL=1 in the env makes it exit nonzero so
    the failure path can be tested.
    """
    script = target_dir / "fake_partial_gguf.py"
    script.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import argparse, os, sys
        ap = argparse.ArgumentParser()
        ap.add_argument("src")
        ap.add_argument("dst")
        ap.add_argument("--keep", type=int, default=None)
        ap.add_argument("--start", type=int, default=None)
        ap.add_argument("--end", type=int, default=None)
        ap.add_argument("--keep-token-embd", action="store_true")
        ap.add_argument("--keep-output", action="store_true")
        args = ap.parse_args()
        if os.environ.get("NAKSHATRA_SLICER_FAIL") == "1":
            sys.stderr.write("simulated slicer failure\\n")
            sys.exit(2)
        if not os.path.isfile(args.src):
            sys.stderr.write(f"fake slicer: src not found: {args.src}\\n")
            sys.exit(3)
        payload = f"slice s={args.start} e={args.end} embd={args.keep_token_embd} lm={args.keep_output}\\n"
        with open(args.dst, "wb") as f:
            f.write(payload.encode())
        sys.exit(0)
    """))
    script.chmod(0o755)
    return script


class SliceEndpointTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.serve_dir = Path(self.tmpdir.name) / "serve"
        self.serve_dir.mkdir()
        # Plausible source GGUF: any file qualifies for the path-exists check.
        # The fake slicer only verifies isfile(src), not GGUF-ness.
        self.src_gguf = Path(self.tmpdir.name) / "fake_full.gguf"
        self.src_gguf.write_bytes(b"FAKEGGUF" * 16)

        fake_script = _make_fake_slicer(Path(self.tmpdir.name))

        # Point worker module state at our test fixtures.
        W._FILE_SERVER_DIR = str(self.serve_dir.resolve())
        W._PARTIAL_GGUF_PATH = str(fake_script.resolve())
        W._SLICE_TASKS.clear()

        # Bind to port 0 → OS picks free port; record it for client URLs.
        self.server = HTTPServer(("127.0.0.1", 0), W.FileServerHandler)
        self.host, self.port = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://{self.host}:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmpdir.cleanup()
        os.environ.pop("NAKSHATRA_SLICER_FAIL", None)

    def _wait_terminal(self, task_id: str, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        snap = {}
        while time.time() < deadline:
            status, snap = _http_get_json(f"{self.base}/slice/{task_id}")
            self.assertEqual(status, 200, f"poll failed: {snap}")
            if snap["status"] in ("completed", "failed"):
                return snap
            time.sleep(0.05)
        self.fail(f"task {task_id} did not terminate within {timeout}s; last={snap}")

    # ── Happy path ──────────────────────────────────────────────────

    def test_post_then_poll_completed(self):
        body = {
            "model_id": "test-model-q4",
            "full_gguf_path": str(self.src_gguf),
            "layer_start": 0,
            "layer_end": 4,
        }
        status, resp = _http_post_json(f"{self.base}/slice", body)
        self.assertEqual(status, 202, resp)
        self.assertIn("task_id", resp)
        self.assertEqual(resp["output_filename"], "test-model-q4.l0-4.gguf")

        final = self._wait_terminal(resp["task_id"])
        self.assertEqual(final["status"], "completed", final)
        self.assertEqual(final["output_filename"], "test-model-q4.l0-4.gguf")
        self.assertIsNotNone(final["sha256"])
        self.assertEqual(len(final["sha256"]), 64)
        self.assertGreater(final["size_bytes"], 0)
        self.assertIsNone(final["error"])

        # Output landed in the file-server dir (so Phase-4 auto-fetch sees it)
        # AND the sha256 sidecar was written (so cache-scan skips re-hashing).
        output_path = self.serve_dir / final["output_filename"]
        self.assertTrue(output_path.is_file())
        sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
        self.assertTrue(sidecar.is_file())
        self.assertEqual(sidecar.read_text().strip().split()[0], final["sha256"])

    def test_pillar_derived_filename_no_path_injection(self):
        # Client cannot pick the filename — even attempting traversal in
        # model_id gets sanitised. Defense-in-depth alongside path checks.
        body = {
            "model_id": "../../etc/passwd",
            "full_gguf_path": str(self.src_gguf),
            "layer_start": 0,
            "layer_end": 4,
        }
        status, resp = _http_post_json(f"{self.base}/slice", body)
        self.assertEqual(status, 202, resp)
        # ".." sequences must NOT appear in the slug.
        self.assertNotIn("..", resp["output_filename"])
        self.assertNotIn("/", resp["output_filename"])
        # Wait for terminal state so the slicer thread doesn't outlive
        # this test and crash setUp() of the next one.
        self._wait_terminal(resp["task_id"])

    # ── Failure path ────────────────────────────────────────────────

    def test_subprocess_failure_reports_error(self):
        os.environ["NAKSHATRA_SLICER_FAIL"] = "1"
        body = {
            "model_id": "test",
            "full_gguf_path": str(self.src_gguf),
            "layer_start": 0,
            "layer_end": 4,
        }
        status, resp = _http_post_json(f"{self.base}/slice", body)
        self.assertEqual(status, 202)
        final = self._wait_terminal(resp["task_id"])
        self.assertEqual(final["status"], "failed")
        self.assertIn("simulated slicer failure", final["error"])
        self.assertIsNone(final["sha256"])

    # ── Input validation ────────────────────────────────────────────

    def test_missing_model_id(self):
        status, resp = _http_post_json(
            f"{self.base}/slice",
            {"full_gguf_path": str(self.src_gguf), "layer_start": 0, "layer_end": 4},
        )
        self.assertEqual(status, 400)

    def test_bad_range(self):
        status, _ = _http_post_json(
            f"{self.base}/slice",
            {"model_id": "x", "full_gguf_path": str(self.src_gguf),
             "layer_start": 5, "layer_end": 3},
        )
        self.assertEqual(status, 400)

    def test_nonexistent_full_gguf(self):
        status, _ = _http_post_json(
            f"{self.base}/slice",
            {"model_id": "x", "full_gguf_path": "/no/such/path.gguf",
             "layer_start": 0, "layer_end": 4},
        )
        self.assertEqual(status, 400)

    def test_unknown_task_id(self):
        status, _ = _http_get_json(f"{self.base}/slice/does-not-exist")
        self.assertEqual(status, 404)

    def test_post_to_wrong_path_404(self):
        status, _ = _http_post_json(f"{self.base}/other", {"x": 1})
        self.assertEqual(status, 404)

    def test_slicer_disabled_when_path_unset(self):
        # If the worker can't find partial_gguf.py, POST /slice should refuse
        # immediately rather than enqueue a task that's guaranteed to fail.
        original = W._PARTIAL_GGUF_PATH
        try:
            W._PARTIAL_GGUF_PATH = ""
            status, _ = _http_post_json(
                f"{self.base}/slice",
                {"model_id": "x", "full_gguf_path": str(self.src_gguf),
                 "layer_start": 0, "layer_end": 4},
            )
            self.assertEqual(status, 503)
        finally:
            W._PARTIAL_GGUF_PATH = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
