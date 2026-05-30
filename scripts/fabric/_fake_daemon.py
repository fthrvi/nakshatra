#!/usr/bin/env python3
"""Fake worker daemon — mirrors the real ``llama-nakshatra-worker``'s
shm wire protocol enough for ShmDaemonClient integration testing.

Used by ``tests/test_fabric_shm_daemon_client.py`` to validate the
Python A.2 wiring **without** requiring a built C++ binary. Phase A.3
will produce the real daemon; until then this script is the test
fixture that lets A.2 ship + be exercised on every commit.

Argv mirrors the real daemon:

  _fake_daemon.py <sub_gguf> <mode> <n_ctx> <n_threads> <n_gpu_layers>
                  --fabric-shm-req <req_path> --fabric-shm-resp <resp_path>

Wire format (must stay byte-identical to ShmDaemonClient + the
eventual C++ daemon):

  Request:  u32 cmd | u32 n_tokens | u32 start_pos | u32 flags
            | u32 payload_bytes | <payload>
  Response: u32 status | u32 payload_bytes | <payload>

What this fake does NOT model: actual model inference (we return
deterministic stub bytes), GPU offload, layer-loading time, the
stderr "loading..." log lines. It returns CMD_INFO with predictable
values and echoes decode payloads with a small transform so tests
can verify the right bytes flowed through.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

# Same scripts/ sys.path trick the rest of fabric/ uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabric.shm_ring import ShmRing


CMD_TOKEN_DECODE = 1
CMD_EMBD_DECODE = 2
CMD_INFO = 3

_REQ_HEADER_FMT = "<IIIII"
_REQ_HEADER_SIZE = struct.calcsize(_REQ_HEADER_FMT)
_RESP_HEADER_FMT = "<II"

_POLL_INTERVAL_S = 5e-5


def _pack_response(status: int, payload: bytes) -> bytes:
    return struct.pack(_RESP_HEADER_FMT, status, len(payload)) + payload


def _handle_request(
    cmd: int, n_tokens: int, start_pos: int, flags: int, payload: bytes,
    *, n_embd: int, mode: str,
) -> bytes:
    """Translate a parsed request into a response frame. Deterministic
    + fast — no real inference."""
    if cmd == CMD_INFO:
        # Match the real daemon's 24-byte info struct exactly
        # (layer_start, layer_end, n_embd, has_token_embd,
        # has_lm_head, n_vocab) so DaemonClient.info() parses it
        # without special-casing.
        body = struct.pack(
            "<6i",
            0,                      # layer_start
            14,                     # layer_end
            n_embd,
            int(mode == "first"),   # has_token_embd
            int(mode == "last"),    # has_lm_head
            32000,                  # n_vocab
        )
        return _pack_response(0, body)

    if cmd in (CMD_TOKEN_DECODE, CMD_EMBD_DECODE):
        # For decode requests, fake an n_tokens × n_embd × 4 fp32
        # output. Each byte = (byte_in ^ 0x55) so tests can assert
        # the bytes round-tripped through req → daemon → resp intact.
        # The real daemon prefixes its decode output with a 4-byte
        # rtype tag (worker.DaemonClient.call drops it); we mirror
        # that so the consumer code path is identical.
        rtype_prefix = struct.pack("<I", 0)
        if mode == "last" and cmd == CMD_TOKEN_DECODE:
            # Last worker on first-step decode returns a single
            # int32 token id. For tests we just emit a known one.
            return _pack_response(0, rtype_prefix + struct.pack("<i", 42))
        # Otherwise, echo-transform the payload.
        out = bytes((b ^ 0x55) for b in payload)
        return _pack_response(0, rtype_prefix + out)

    # Unknown command → status != 0
    return _pack_response(1, b"")


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("sub_gguf")
    ap.add_argument("mode", choices=("first", "middle", "last"))
    ap.add_argument("n_ctx", type=int)
    ap.add_argument("n_threads", type=int)
    ap.add_argument("n_gpu_layers", type=int)
    ap.add_argument("--fabric-shm-req", required=True)
    ap.add_argument("--fabric-shm-resp", required=True)
    # Fake-only knob — tests can ask the fake to inject a startup
    # delay (validates ShmDaemonClient's ready timeout).
    ap.add_argument("--fake-startup-delay-s", type=float, default=0.0)
    # Fake-only knob — n_embd reported in CMD_INFO + used for
    # decode-output sizing. Real daemon derives this from the model.
    ap.add_argument("--fake-n-embd", type=int, default=4)
    args = ap.parse_args()

    if args.fake_startup_delay_s > 0:
        time.sleep(args.fake_startup_delay_s)

    # Attach (not create) — the parent ShmDaemonClient owns + unlinks.
    req_ring = ShmRing.attach(args.fabric_shm_req)
    resp_ring = ShmRing.attach(args.fabric_shm_resp)

    # The real daemon prints "[daemon] ready" to stderr after model
    # load. We just announce immediately — ShmDaemonClient's ready
    # handshake polls CMD_INFO so the printed line is informative,
    # not load-bearing.
    sys.stderr.write("[fake-daemon] ready\n")
    sys.stderr.flush()

    try:
        while True:
            req_frame = req_ring.read_message()
            if req_frame is None:
                time.sleep(_POLL_INTERVAL_S)
                continue
            if len(req_frame) < _REQ_HEADER_SIZE:
                sys.stderr.write(
                    f"[fake-daemon] short request frame: "
                    f"{len(req_frame)} bytes\n")
                continue
            cmd, n_tokens, start_pos, flags, plen = struct.unpack(
                _REQ_HEADER_FMT, req_frame[:_REQ_HEADER_SIZE])
            payload = req_frame[_REQ_HEADER_SIZE:_REQ_HEADER_SIZE + plen]
            response = _handle_request(
                cmd, n_tokens, start_pos, flags, payload,
                n_embd=args.fake_n_embd, mode=args.mode,
            )
            # Spin on full resp ring — tests that don't consume
            # responses would otherwise wedge here, but that's the
            # intended back-pressure shape.
            while not resp_ring.write_message(response):
                time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"[fake-daemon] error: {e}\n")
        return 1
    finally:
        try:
            req_ring.close()
        except Exception:
            pass
        try:
            resp_ring.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
