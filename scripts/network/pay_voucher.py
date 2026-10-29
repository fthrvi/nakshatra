"""Cumulative inference vouchers for ``NAK_PAY_MODE=chain``.

The module has no Solana SDK dependency.  Its only non-stdlib dependency is
``cryptography`` for Ed25519.

Wire shape
==========

Paid inference reuses the existing ``InferenceStep.spend_claim_token`` string;
protobuf and legacy wire bytes do not change.  Values beginning ``nakpay1.``
are canonical compact JSON (the prefix is outside the JSON):

* first payer -> worker frame (the legacy token is an opaque correlation value
  only in chain mode; the worker does not claim it from the credit ledger).
  ``channel_base`` is the payer's on-chain lower bound, while
  ``max_signed_total`` is its durable exposure ceiling::

    {"channel":"<base58>","channel_base":<u64>,"claim_token":"<opaque>",
     "hold_id":"<scope>","kind":"init","max_output_tokens":<u64>,
     "max_prompt_tokens":<u64>,"max_signed_total":<u64>,"model_id":"<scope>",
     "program":"<base58>","session_id":"<scope>","v":1}

* checkpoint worker -> payer response::

    {"channel_base":<worker-derived u64>,"kind":"receipt",
     "receipt":{...usage receipt v2...},
     "signature":"<128 lowercase hex>","v":1}

* later payer -> worker frame::

    {"channel":"<base58>","kind":"voucher","program":"<base58>",
     "signature":"<base64 raw Ed25519 signature>","total":<u64>,"v":1}

The optional ``signer`` member is accepted for forward compatibility but is
never trusted.  Verification uses only ``Channel.voucher_signer`` or the
on-chain ``Escrow.owner``.  At normal completion, the consumer sends
``nak-pay-final-request-v1``; the worker returns its final receipt as
``nak-pay-final-receipt-v1`` and waits at most one 15-second checkpoint interval
for the covering ``nak-pay-control-v1`` voucher.  Control frames perform no
compute.
All of these shapes are active only when ``NAK_PAY_MODE`` is exactly ``chain``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import stat
import struct
import threading
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


VOUCHER_DOMAIN = b"NAK_PAY_VOUCHER1"
VOUCHER_MESSAGE_LEN = 88
RATE = 100
CHECKPOINT_MARGIN_S = 15
WIRE_PREFIX = "nakpay1."
CONTROL_REQUEST = b"nak-pay-control-v1"
CONTROL_OK = b"nak-pay-control-ok"
CONTROL_FINAL_REQUEST = b"nak-pay-final-request-v1"
CONTROL_FINAL_RECEIPT = b"nak-pay-final-receipt-v1"
DEVNET_GENESIS_HASH = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
MAINNET_GENESIS_HASH = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {char: index for index, char in enumerate(_B58)}
_CHANNEL_DISCRIMINATOR = hashlib.sha256(b"account:Channel").digest()[:8]
_ESCROW_DISCRIMINATOR = hashlib.sha256(b"account:Escrow").digest()[:8]
_U64_MAX = (1 << 64) - 1


class VoucherError(ValueError):
    """A voucher, channel, or chain control message was refused."""


class ChannelClosingError(VoucherError):
    """The channel cannot safely cover another inference checkpoint."""


def _u64(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _U64_MAX:
        raise VoucherError(f"{name} must be a u64")
    return value


def b58decode(value: str) -> bytes:
    """Decode canonical Bitcoin/Solana base58 without a third-party package."""
    if not isinstance(value, str) or not value:
        raise VoucherError("base58 value must be a non-empty string")
    number = 0
    try:
        for char in value:
            number = number * 58 + _B58_INDEX[char]
    except KeyError as exc:
        raise VoucherError("base58 value contains an invalid character") from exc
    zeroes = len(value) - len(value.lstrip("1"))
    body = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    raw = b"\0" * zeroes + body
    if b58encode(raw) != value:
        raise VoucherError("base58 value is not canonical")
    return raw


def b58encode(raw: bytes) -> str:
    data = bytes(raw)
    zeroes = len(data) - len(data.lstrip(b"\0"))
    number = int.from_bytes(data, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = _B58[remainder] + encoded
    return "1" * zeroes + encoded


def _pubkey_bytes(value: object, name: str) -> bytes:
    try:
        raw = b58decode(value) if isinstance(value, str) else bytes(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise VoucherError(f"invalid {name}: {exc}") from exc
    if len(raw) != 32:
        raise VoucherError(f"{name} must contain exactly 32 bytes")
    return raw


def _pubkey_text(value: object, name: str) -> str:
    return b58encode(_pubkey_bytes(value, name))


def voucher_message(program_id: object, channel: object,
                    cumulative_total_micro_usdc: int) -> bytes:
    message = (
        VOUCHER_DOMAIN
        + _pubkey_bytes(program_id, "program id")
        + _pubkey_bytes(channel, "channel")
        + struct.pack("<Q", _u64(cumulative_total_micro_usdc, "voucher total"))
    )
    assert len(message) == VOUCHER_MESSAGE_LEN
    return message


def _private_key(key: object) -> ed25519.Ed25519PrivateKey:
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key
    raw = bytes(key)  # type: ignore[arg-type]
    if len(raw) == 64:
        raw = raw[:32]
    if len(raw) != 32:
        raise VoucherError("Ed25519 private key must be a 32-byte seed or 64-byte keypair")
    return ed25519.Ed25519PrivateKey.from_private_bytes(raw)


def sign_voucher(key: object, program_id: object, channel: object,
                 cumulative_total_micro_usdc: int) -> bytes:
    return _private_key(key).sign(
        voucher_message(program_id, channel, cumulative_total_micro_usdc))


def verify_voucher(public_key: object, signature: bytes, program_id: object,
                   channel: object, cumulative_total_micro_usdc: int) -> bool:
    try:
        raw_key = _pubkey_bytes(public_key, "voucher signer")
        raw_signature = bytes(signature)
        if len(raw_signature) != 64:
            return False
        ed25519.Ed25519PublicKey.from_public_bytes(raw_key).verify(
            raw_signature,
            voucher_message(program_id, channel, cumulative_total_micro_usdc),
        )
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


def credits_to_total(channel_base: int, cumulative_credits: int) -> int:
    """Convert the cumulative receipt meter; never round checkpoint deltas."""
    base = _u64(channel_base, "channel base")
    credits = _u64(cumulative_credits, "cumulative credits")
    return _u64(base + credits // RATE, "voucher total")


def required_capacity_micro_usdc(max_cost_credits: int) -> int:
    credits = _u64(max_cost_credits, "maximum cost")
    return (credits + RATE - 1) // RATE


@dataclass(frozen=True)
class ChannelAccount:
    escrow: str
    generation: int
    seq: int
    worker: str
    voucher_signer: str
    delegate: str | None
    cap: int
    redeemed: int
    fee_bps_snapshot: int
    status: str
    close_at: int
    rent_payer: str
    bump: int


def require_open_channel(account: ChannelAccount, *, now: float | None = None,
                         checkpoint_margin_s: int = CHECKPOINT_MARGIN_S) -> None:
    """Refuse work/vouchers that cannot be redeemed after one checkpoint.

    The program permits redemption on ``Closing`` only until ``close_at``.
    Workers deliberately stop accepting new work as soon as closing begins,
    and retain one checkpoint of time margin for the final redeem transaction.
    """
    if account.status != "Open":
        raise ChannelClosingError("channel is not open")
    margin = max(0, int(checkpoint_margin_s))
    timestamp = time.time() if now is None else float(now)
    if account.close_at and account.close_at <= timestamp + margin:
        raise ChannelClosingError("channel close deadline is within one checkpoint")


def _take(data: memoryview, offset: int, size: int) -> tuple[bytes, int]:
    end = offset + size
    if end > len(data):
        raise VoucherError("account data is truncated")
    return bytes(data[offset:end]), end


def _number(data: memoryview, offset: int, fmt: str) -> tuple[int, int]:
    raw, offset = _take(data, offset, struct.calcsize(fmt))
    return int(struct.unpack(fmt, raw)[0]), offset


def _pubkey(data: memoryview, offset: int) -> tuple[str, int]:
    raw, offset = _take(data, offset, 32)
    return b58encode(raw), offset


def decode_channel(data: bytes) -> ChannelAccount:
    raw = bytes(data)
    if len(raw) < 8 or raw[:8] != _CHANNEL_DISCRIMINATOR:
        raise VoucherError("data is not a Channel account")
    view = memoryview(raw)[8:]
    offset = 0
    escrow, offset = _pubkey(view, offset)
    generation, offset = _number(view, offset, "<Q")
    seq, offset = _number(view, offset, "<Q")
    worker, offset = _pubkey(view, offset)
    signer, offset = _pubkey(view, offset)
    option, offset = _number(view, offset, "<B")
    if option == 0:
        delegate = None
    elif option == 1:
        delegate, offset = _pubkey(view, offset)
    else:
        raise VoucherError(f"invalid Borsh option tag {option}")
    cap, offset = _number(view, offset, "<Q")
    redeemed, offset = _number(view, offset, "<Q")
    fee_bps, offset = _number(view, offset, "<H")
    status_index, offset = _number(view, offset, "<B")
    if status_index not in {0, 1}:
        raise VoucherError(f"invalid ChannelStatus variant {status_index}")
    close_at, offset = _number(view, offset, "<q")
    rent_payer, offset = _pubkey(view, offset)
    bump, offset = _number(view, offset, "<B")
    if any(view[offset:]):
        raise VoucherError("account data contains unexpected trailing bytes")
    return ChannelAccount(
        escrow, generation, seq, worker, signer, delegate, cap, redeemed,
        fee_bps, ("Open", "Closing")[status_index], close_at, rent_payer, bump,
    )


def decode_escrow_owner(data: bytes) -> str:
    raw = bytes(data)
    if len(raw) < 40 or raw[:8] != _ESCROW_DISCRIMINATOR:
        raise VoucherError("data is not an Escrow account")
    return b58encode(raw[8:40])


class ChannelState(Protocol):
    def read_channel(self, channel: str, program_id: str) -> ChannelAccount: ...
    def read_escrow_owner(self, escrow: str, program_id: str) -> str: ...


class FakeChannelState:
    """In-memory authoritative account reader for tests and local simulations."""

    def __init__(self, channels: Mapping[str, ChannelAccount],
                 escrow_owners: Mapping[str, str], *, program_id: str):
        self.program_id = _pubkey_text(program_id, "program id")
        self.channels = dict(channels)
        self.escrow_owners = dict(escrow_owners)

    def _program(self, program_id: str) -> None:
        if _pubkey_text(program_id, "program id") != self.program_id:
            raise VoucherError("account is not owned by the configured program")

    def read_channel(self, channel: str, program_id: str) -> ChannelAccount:
        self._program(program_id)
        key = _pubkey_text(channel, "channel")
        try:
            return self.channels[key]
        except KeyError as exc:
            raise VoucherError("channel account does not exist") from exc

    def read_escrow_owner(self, escrow: str, program_id: str) -> str:
        self._program(program_id)
        key = _pubkey_text(escrow, "escrow")
        try:
            return _pubkey_text(self.escrow_owners[key], "escrow owner")
        except KeyError as exc:
            raise VoucherError("escrow account does not exist") from exc


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect could move a verified https:// endpoint to plaintext or another host.
        raise VoucherError("Solana RPC redirects are refused")


# An empty ProxyHandler ignores http_proxy/https_proxy: the https/loopback rule is about the endpoint
# actually reached, so no environment proxy may sit in between.
_RPC_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirect)
RPC_RESPONSE_MAX = 1 << 20  # a Channel/Escrow account reply is a few KiB


class JsonRpcChannelState:
    """Devnet/local ``getAccountInfo`` reader with strict owner checks."""

    def __init__(self, rpc_url: str, *, timeout: float = 5.0,
                 opener: Callable = _RPC_OPENER.open):
        if not isinstance(rpc_url, str) or not rpc_url:
            raise VoucherError("Solana RPC URL is required")
        parts = urllib.parse.urlsplit(rpc_url)
        host = (parts.hostname or "").lower()
        if parts.scheme == "https" and host:
            pass
        elif parts.scheme == "http" and host in _LOOPBACK_HOSTS:
            pass
        else:
            # Channel signers and owners are read from this endpoint; a plaintext remote RPC would let
            # anyone on the path forge them, so only TLS or a loopback test validator is trusted.
            raise VoucherError("Solana RPC must be https://, or http:// on loopback only")
        self.rpc_url = rpc_url
        self.timeout = float(timeout)
        self.opener = opener
        self._request_id = 0
        self._lock = threading.Lock()
        self._cluster_checked = False

    def _rpc(self, method: str, params: list[object]) -> object:
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
        body = json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method,
            "params": params,
        }, separators=(",", ":")).encode("ascii")
        request = urllib.request.Request(
            self.rpc_url, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                raw = response.read(RPC_RESPONSE_MAX + 1)
                if len(raw) > RPC_RESPONSE_MAX:
                    raise VoucherError(f"Solana RPC {method} response is too large")
                decoded = json.loads(raw)
            if (not isinstance(decoded, dict)
                    or decoded.get("jsonrpc") != "2.0"
                    or decoded.get("id") != request_id
                    or set(decoded) != {"jsonrpc", "id", "result"}):
                raise VoucherError(f"Solana RPC returned malformed {method} response")
            return decoded["result"]
        except VoucherError:
            raise
        except Exception as exc:
            raise VoucherError(
                f"Solana RPC {method} failed ({type(exc).__name__})") from exc

    def _ensure_cluster(self) -> None:
        with self._lock:
            if self._cluster_checked:
                return
        genesis = self._rpc("getGenesisHash", [])
        if not isinstance(genesis, str) or not genesis:
            raise VoucherError("Solana RPC returned malformed getGenesisHash response")
        host = (urllib.parse.urlsplit(self.rpc_url).hostname or "").lower()
        local = host in _LOOPBACK_HOSTS
        if genesis == MAINNET_GENESIS_HASH:
            raise VoucherError("Solana mainnet RPC is forbidden for Nakshatra payments")
        if genesis != DEVNET_GENESIS_HASH and not local:
            raise VoucherError(
                "Solana RPC cluster is neither devnet nor a loopback test validator")
        with self._lock:
            self._cluster_checked = True

    def _account(self, address: str, program_id: str) -> bytes:
        self._ensure_cluster()
        address = _pubkey_text(address, "account address")
        program_id = _pubkey_text(program_id, "program id")
        try:
            result = self._rpc("getAccountInfo", [
                address, {"commitment": "finalized", "encoding": "base64"},
            ])
            if (not isinstance(result, dict)
                    or set(result) != {"context", "value"}
                    or not isinstance(result["context"], dict)):
                raise VoucherError("RPC returned a malformed account result")
            context = result["context"]
            if (set(context) not in ({"slot"}, {"apiVersion", "slot"})
                    or isinstance(context.get("slot"), bool)
                    or not isinstance(context.get("slot"), int)
                    or context["slot"] < 0
                    or ("apiVersion" in context
                        and (not isinstance(context["apiVersion"], str)
                             or not context["apiVersion"]))):
                raise VoucherError("RPC returned a malformed account context")
            value = result["value"]
            if value is None:
                raise VoucherError(f"account does not exist: {address}")
            if (not isinstance(value, dict)
                    or set(value) != {
                        "data", "executable", "lamports", "owner",
                        "rentEpoch", "space",
                    }
                    or not isinstance(value["executable"], bool)
                    or isinstance(value["lamports"], bool)
                    or not isinstance(value["lamports"], int)
                    or value["lamports"] < 0
                    or isinstance(value["rentEpoch"], bool)
                    or not isinstance(value["rentEpoch"], int)
                    or value["rentEpoch"] < 0
                    or isinstance(value["space"], bool)
                    or not isinstance(value["space"], int)
                    or value["space"] < 0):
                raise VoucherError("RPC returned a malformed account value")
            if value["owner"] != program_id:
                raise VoucherError("account is not owned by the configured program")
            encoded = value["data"]
            if not isinstance(encoded, list) or len(encoded) != 2 or encoded[1] != "base64":
                raise VoucherError("RPC returned an unsupported account encoding")
            raw = base64.b64decode(encoded[0], validate=True)
            if len(raw) != value["space"]:
                raise VoucherError("RPC account data length differs from space")
            return raw
        except VoucherError:
            raise
        except Exception as exc:
            raise VoucherError(f"Solana RPC account read failed ({type(exc).__name__})") from exc

    def read_channel(self, channel: str, program_id: str) -> ChannelAccount:
        return decode_channel(self._account(channel, program_id))

    def read_escrow_owner(self, escrow: str, program_id: str) -> str:
        return decode_escrow_owner(self._account(escrow, program_id))


def _secure_db_path(path: Path) -> None:
    old_umask = os.umask(0o077)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        if os.path.lexists(path) and path.is_symlink():
            raise VoucherError("voucher database must not be a symlink")
        if path.exists():
            info = path.stat()
            if info.st_uid != os.geteuid() or not stat.S_ISREG(info.st_mode):
                raise VoucherError("voucher database must be a uid-owned regular file")
    finally:
        os.umask(old_umask)


class _Book:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        _secure_db_path(self.path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=5000")

    def _secure(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def close(self) -> None:
        self._secure()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


@dataclass(frozen=True)
class PayerChannel:
    program: str
    channel: str
    worker_key: str
    cap: int
    channel_base: int
    max_signed_total: int


class PayerChannelBook(_Book):
    """Mode-0600 channel pins, session bases, and signing exposure ceilings."""

    def __init__(self, path: str | Path | None = None):
        super().__init__(path or Path.home() / ".nakshatra" / "pay" / "channels.sqlite")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS channels("
            "program TEXT NOT NULL, channel TEXT PRIMARY KEY, worker_key TEXT NOT NULL UNIQUE, "
            "cap TEXT NOT NULL, channel_base TEXT NOT NULL, max_signed_total TEXT NOT NULL)"
        )
        columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(channels)")
        }
        if "max_signed_total" not in columns and "last_vouched_total" in columns:
            self._db.execute(
                "ALTER TABLE channels RENAME COLUMN last_vouched_total TO max_signed_total")
        self._secure()

    @staticmethod
    def _row(row: sqlite3.Row) -> PayerChannel:
        return PayerChannel(row["program"], row["channel"], row["worker_key"],
                            int(row["cap"]), int(row["channel_base"]),
                            int(row["max_signed_total"]))

    def put(self, *, program: str, channel: str, worker_key: str, cap: int,
            channel_base: int = 0, max_signed_total: int = 0) -> PayerChannel:
        base = _u64(channel_base, "channel base")
        ceiling = _u64(max_signed_total, "maximum signed total")
        if base > ceiling:
            raise VoucherError("channel base cannot exceed the payer signing ceiling")
        values = (
            _pubkey_text(program, "program id"), _pubkey_text(channel, "channel"),
            str(worker_key).lower(), str(_u64(cap, "channel cap")),
            str(base), str(ceiling),
        )
        if re.fullmatch(r"[0-9a-f]{64}", values[2]) is None:
            raise VoucherError("worker key must be 32-byte lowercase hex")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing = self._db.execute(
                    "SELECT * FROM channels WHERE channel=?", (values[1],)).fetchone()
                if existing is None:
                    self._db.execute(
                        "INSERT INTO channels(program,channel,worker_key,cap,channel_base,"
                        "max_signed_total) VALUES(?,?,?,?,?,?)", values)
                else:
                    if (existing["program"] != values[0]
                            or existing["worker_key"] != values[2]
                            or int(existing["cap"]) != int(values[3])):
                        raise VoucherError(
                            "channel program, worker, and cap are immutable")
                    if ceiling < int(existing["max_signed_total"]):
                        raise VoucherError("payer signing ceiling would decrease")
                    self._db.execute(
                        "UPDATE channels SET channel_base=?,max_signed_total=? "
                        "WHERE channel=?", (values[4], values[5], values[1]))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        self._secure()
        return self.for_worker(values[2])

    def for_worker(self, worker_key: str) -> PayerChannel:
        row = self._db.execute(
            "SELECT * FROM channels WHERE worker_key=?", (str(worker_key).lower(),)).fetchone()
        if row is None:
            raise VoucherError("payer has no channel for worker")
        return self._row(row)

    def begin_session(self, worker_key: str, *, program: str, state: ChannelState,
                      max_cost_credits: int) -> PayerChannel:
        configured = _pubkey_text(program, "program id")
        with self._lock:
            row = self.for_worker(worker_key)
            if row.program != configured:
                raise VoucherError("channel program differs from configured program")
            channel_state = state.read_channel(row.channel, configured)
            require_open_channel(channel_state)
            if channel_state.cap != row.cap:
                raise VoucherError("on-chain channel cap differs from pinned channel book")
            if row.max_signed_total < channel_state.redeemed:
                raise VoucherError(
                    "payer signing ceiling is below the on-chain redeemed total")
            available = channel_state.cap - row.max_signed_total
            if available < required_capacity_micro_usdc(max_cost_credits):
                raise VoucherError("channel has insufficient uncommitted capacity")
            self._db.execute(
                "UPDATE channels SET channel_base=? WHERE channel=?",
                (str(channel_state.redeemed), row.channel))
            row = PayerChannel(row.program, row.channel, row.worker_key, row.cap,
                               channel_state.redeemed, row.max_signed_total)
        self._secure()
        return row

    def adopt_session_base(self, channel: str, base: int, *,
                           on_chain_redeemed: int) -> PayerChannel:
        """Adopt the worker-derived base without expanding payer exposure."""
        key = _pubkey_text(channel, "channel")
        amount = _u64(base, "worker-derived channel base")
        redeemed = _u64(on_chain_redeemed, "on-chain redeemed total")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM channels WHERE channel=?", (key,)).fetchone()
                if row is None:
                    raise VoucherError("unknown payer channel")
                if amount < redeemed:
                    raise VoucherError(
                        "worker-derived base is below the on-chain redeemed total")
                if amount > int(row["max_signed_total"]):
                    raise VoucherError(
                        "worker-derived base is above the payer signing ceiling")
                self._db.execute(
                    "UPDATE channels SET channel_base=? WHERE channel=?",
                    (str(amount), key))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        self._secure()
        return self._row(self._db.execute(
            "SELECT * FROM channels WHERE channel=?", (key,)).fetchone())

    def record(self, channel: str, total: int) -> PayerChannel:
        key = _pubkey_text(channel, "channel")
        amount = _u64(total, "voucher total")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM channels WHERE channel=?", (key,)).fetchone()
                if row is None:
                    raise VoucherError("unknown payer channel")
                current = int(row["max_signed_total"])
                if amount < int(row["channel_base"]):
                    raise VoucherError("voucher total is below the session base")
                if amount > int(row["cap"]):
                    raise VoucherError("voucher total exceeds channel cap")
                if amount > current:
                    self._db.execute(
                        "UPDATE channels SET max_signed_total=? WHERE channel=?",
                        (str(amount), key))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        self._secure()
        updated = self._db.execute(
            "SELECT * FROM channels WHERE channel=?", (key,)).fetchone()
        return self._row(updated)


@dataclass(frozen=True)
class VoucherRecord:
    channel: str
    program: str
    total: int
    signature: bytes


def _process_start_time(pid: int) -> str | None:
    """Return Linux's non-recycled process start tick, or None if it is gone."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise VoucherError("cannot verify voucher reservation process") from exc
    try:
        # comm is parenthesized and may contain spaces; starttime is field 22,
        # which is tail index 19 after fields 1 and 2 are removed.
        tail = raw[raw.rindex(")") + 2:].split()
        start = tail[19]
        if not start.isdigit():
            raise ValueError
        return start
    except (IndexError, ValueError) as exc:
        raise VoucherError("cannot parse voucher reservation process") from exc


def _boot_id() -> str:
    """Return the Linux boot identity, failing closed if it is unavailable."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii").strip().lower()
    except OSError as exc:
        raise VoucherError("cannot verify voucher reservation boot") from exc
    if re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            value) is None:
        raise VoucherError("cannot parse voucher reservation boot")
    return value


def _reservation_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
        raise VoucherError("voucher reservation id is invalid")
    return value


class WorkerVoucherStore(_Book):
    """Latest verified voucher per channel, committed before work resumes."""

    def __init__(self, path: str | Path | None = None):
        super().__init__(path or Path.home() / ".nakshatra" / "pay" / "vouchers.sqlite")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vouchers("
            "channel TEXT PRIMARY KEY, program TEXT NOT NULL, total TEXT NOT NULL, "
            "signature BLOB NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS reservations("
            "channel TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, "
            "pid INTEGER NOT NULL, process_start TEXT NOT NULL, "
            "boot_id TEXT NOT NULL, acquired_at INTEGER NOT NULL)"
        )
        reservation_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(reservations)")
        }
        if "boot_id" not in reservation_columns:
            # An old row has no trustworthy boot identity.  The empty value is
            # deliberately treated as live below rather than reclaimed.
            self._db.execute(
                "ALTER TABLE reservations ADD COLUMN boot_id TEXT NOT NULL DEFAULT ''")
        self._secure()

    def acquire_reservation(self, channel: str, reservation_id: str) -> None:
        """Persist exclusive ownership, reclaiming only a dead process's row."""
        key = _pubkey_text(channel, "channel")
        owner = _reservation_id(reservation_id)
        pid = os.getpid()
        process_start = _process_start_time(pid)
        boot_id = _boot_id()
        if process_start is None:
            raise VoucherError("cannot identify voucher reservation process")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM reservations WHERE channel=?", (key,)).fetchone()
                if row is not None:
                    same_owner = (
                        row["reservation_id"] == owner
                        and int(row["pid"]) == pid
                        and row["process_start"] == process_start
                        and row["boot_id"] == boot_id)
                    if not same_owner:
                        stored_boot = row["boot_id"]
                        if not stored_boot:
                            raise VoucherError(
                                "cannot verify existing voucher reservation boot")
                        if stored_boot == boot_id and _process_start_time(
                                int(row["pid"])) == row["process_start"]:
                            raise VoucherError("channel busy")
                        self._db.execute(
                            "DELETE FROM reservations WHERE channel=?", (key,))
                self._db.execute(
                    "INSERT OR REPLACE INTO reservations("
                    "channel,reservation_id,pid,process_start,boot_id,acquired_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (key, owner, pid, process_start, boot_id, int(time.time())))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        self._secure()

    def _assert_reservation_row(self, channel: str, reservation_id: str) -> None:
        row = self._db.execute(
            "SELECT * FROM reservations WHERE channel=?", (channel,)).fetchone()
        process_start = _process_start_time(os.getpid())
        boot_id = _boot_id()
        if (row is None
                or row["reservation_id"] != reservation_id
                or int(row["pid"]) != os.getpid()
                or row["process_start"] != process_start
                or row["boot_id"] != boot_id):
            raise VoucherError("channel reservation is not held by this session")

    def assert_reservation(self, channel: str, reservation_id: str) -> None:
        key = _pubkey_text(channel, "channel")
        owner = _reservation_id(reservation_id)
        with self._lock:
            self._assert_reservation_row(key, owner)

    def release_reservation(self, channel: str, reservation_id: str) -> None:
        key = _pubkey_text(channel, "channel")
        owner = _reservation_id(reservation_id)
        process_start = _process_start_time(os.getpid())
        boot_id = _boot_id()
        with self._lock:
            self._db.execute(
                "DELETE FROM reservations WHERE channel=? AND reservation_id=? "
                "AND pid=? AND process_start=? AND boot_id=?",
                (key, owner, os.getpid(), process_start, boot_id))
        self._secure()

    def latest(self, channel: str) -> VoucherRecord | None:
        key = _pubkey_text(channel, "channel")
        row = self._db.execute(
            "SELECT * FROM vouchers WHERE channel=?", (key,)).fetchone()
        if row is None:
            return None
        return VoucherRecord(key, row["program"], int(row["total"]), bytes(row["signature"]))

    def accept(self, voucher: Mapping[str, object], *, configured_program: str,
               state: ChannelState, expected_channel: str | None = None,
               now: float | None = None,
               reservation_id: str | None = None) -> VoucherRecord:
        program = _pubkey_text(voucher.get("program"), "program id")
        configured = _pubkey_text(configured_program, "configured program id")
        if program != configured:
            raise VoucherError("voucher program differs from configured program")
        channel = _pubkey_text(voucher.get("channel"), "channel")
        if expected_channel is not None and channel != _pubkey_text(expected_channel, "channel"):
            raise VoucherError("voucher is for another channel")
        total = _u64(voucher.get("total"), "voucher total")
        try:
            signature = base64.b64decode(voucher.get("signature"), validate=True)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise VoucherError("voucher signature is not canonical base64") from exc
        if len(signature) != 64:
            raise VoucherError("voucher signature must contain 64 bytes")
        account = state.read_channel(channel, configured)
        require_open_channel(account, now=now)
        if total < account.redeemed:
            raise VoucherError("voucher total is below the on-chain redeemed total")
        if total > account.cap:
            raise VoucherError("voucher total exceeds channel cap")
        owner = state.read_escrow_owner(account.escrow, configured)
        # Deliberately ignore voucher.get("signer"): sender assertions have no authority.
        if not any(verify_voucher(signer, signature, configured, channel, total)
                   for signer in (account.voucher_signer, owner)):
            raise VoucherError("voucher is not signed by an on-chain accepted signer")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if reservation_id is not None:
                    self._assert_reservation_row(
                        channel, _reservation_id(reservation_id))
                previous = self._db.execute(
                    "SELECT * FROM vouchers WHERE channel=?", (channel,)).fetchone()
                if previous is not None:
                    if previous["program"] != configured:
                        raise VoucherError("stored voucher program differs")
                    old_total = int(previous["total"])
                    if total < old_total:
                        raise VoucherError("voucher total is below the stored total")
                    if total > old_total:
                        self._db.execute(
                            "UPDATE vouchers SET total=?,signature=? WHERE channel=?",
                            (str(total), signature, channel))
                else:
                    self._db.execute(
                        "INSERT INTO vouchers(channel,program,total,signature) VALUES(?,?,?,?)",
                        (channel, configured, str(total), signature))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        self._secure()
        return self.latest(channel)  # type: ignore[return-value]


def _wire(value: Mapping[str, object]) -> str:
    return WIRE_PREFIX + json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_wire(value: str) -> dict:
    if not isinstance(value, str) or not value.startswith(WIRE_PREFIX):
        raise VoucherError("value is not a Nakshatra pay control envelope")
    try:
        result = json.loads(value[len(WIRE_PREFIX):])
    except json.JSONDecodeError as exc:
        raise VoucherError("pay control envelope is invalid JSON") from exc
    if not isinstance(result, dict) or result.get("v") != 1:
        raise VoucherError("unknown pay control envelope")
    return result


def init_wire(claim_token: str, *, program: str, channel: str,
              channel_base: int, hold_id: str, session_id: str,
              model_id: str, max_prompt_tokens: int,
              max_output_tokens: int,
              max_signed_total: int | None = None) -> str:
    base = _u64(channel_base, "channel base")
    ceiling = _u64(
        base if max_signed_total is None else max_signed_total,
        "maximum signed total")
    if base > ceiling:
        raise VoucherError("channel base cannot exceed the payer signing ceiling")
    return _wire({
        "v": 1, "kind": "init", "claim_token": str(claim_token),
        "program": _pubkey_text(program, "program id"),
        "channel": _pubkey_text(channel, "channel"),
        "channel_base": base, "max_signed_total": ceiling,
        "hold_id": str(hold_id), "session_id": str(session_id),
        "model_id": str(model_id),
        "max_prompt_tokens": _u64(max_prompt_tokens, "maximum prompt tokens"),
        "max_output_tokens": _u64(max_output_tokens, "maximum output tokens"),
    })


def voucher_wire(*, program: str, channel: str, total: int,
                 signature: bytes, signer: str | None = None) -> str:
    value: dict[str, object] = {
        "v": 1, "kind": "voucher", "program": _pubkey_text(program, "program id"),
        "channel": _pubkey_text(channel, "channel"),
        "total": _u64(total, "voucher total"),
        "signature": base64.b64encode(bytes(signature)).decode("ascii"),
    }
    if signer is not None:
        value["signer"] = _pubkey_text(signer, "claimed signer")
    return _wire(value)


def receipt_wire(receipt: Mapping[str, object], signature_hex: str, *,
                 channel_base: int) -> str:
    if (not isinstance(signature_hex, str)
            or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None):
        raise VoucherError("receipt signature must be 64-byte lowercase hex")
    bytes.fromhex(signature_hex)
    return _wire({
        "v": 1, "kind": "receipt", "receipt": dict(receipt),
        "signature": signature_hex,
        "channel_base": _u64(channel_base, "worker-derived channel base"),
    })


def public_key_bytes(private_key: object) -> bytes:
    return _private_key(private_key).public_key().public_bytes_raw()


class ConsumerVoucherManager:
    """Checks delivered-token receipts and returns monotonic payer vouchers."""

    def __init__(self, *, book: PayerChannelBook, state: ChannelState,
                 program: str, private_key: bytes, plan: list[Mapping[str, object]],
                 max_cost_by_worker: Mapping[str, int],
                 session_id: str | None = None, hold_id: str | None = None,
                 model_id: str | None = None,
                 max_prompt_tokens: int | None = None,
                 max_output_tokens: int | None = None):
        self.book = book
        self.state = state
        self.program = _pubkey_text(program, "program id")
        self.private_key = bytes(private_key)
        self.session_id = session_id
        self.hold_id = hold_id
        self.model_id = model_id
        self.max_prompt_tokens = max_prompt_tokens
        self.max_output_tokens = max_output_tokens
        self.channels: dict[str, PayerChannel] = {}
        self.plan = {str(stage["worker_keyid"]): dict(stage) for stage in plan}
        self.receipts: dict[str, list[dict]] = {key: [] for key in self.plan}
        self.session_bases: dict[str, int | None] = {
            key: None for key in self.plan
        }
        self.pending: dict[str, str] = {}
        self.sent: dict[str, str] = {}
        for worker_key in self.plan:
            self.channels[worker_key] = book.begin_session(
                worker_key, program=self.program, state=state,
                max_cost_credits=int(max_cost_by_worker[worker_key]))
            account = state.read_channel(
                self.channels[worker_key].channel, self.program)
            owner = state.read_escrow_owner(account.escrow, self.program)
            signer = b58encode(public_key_bytes(self.private_key))
            if signer not in {account.voucher_signer, owner}:
                raise VoucherError(
                    "consumer node key is not an on-chain accepted voucher signer")

    def first_wire(self, worker_key: str, claim_token: str) -> str:
        channel = self.channels[worker_key]
        if (self.session_id is None or self.hold_id is None or self.model_id is None
                or self.max_prompt_tokens is None or self.max_output_tokens is None):
            raise VoucherError("chain session scope is incomplete")
        return init_wire(claim_token, program=self.program, channel=channel.channel,
                         channel_base=channel.channel_base,
                         max_signed_total=channel.max_signed_total,
                         hold_id=self.hold_id, session_id=self.session_id,
                         model_id=self.model_id,
                         max_prompt_tokens=self.max_prompt_tokens,
                         max_output_tokens=self.max_output_tokens)

    def outgoing(self, worker_key: str) -> str:
        return self.pending.get(worker_key, "")

    def mark_sent(self, worker_key: str, value: str) -> None:
        if value and self.pending.get(worker_key) == value:
            self.sent[worker_key] = value
            self.pending.pop(worker_key, None)

    def observe(self, worker_key: str, value: str) -> None:
        if not value:
            return
        control = parse_wire(value)
        if control.get("kind") != "receipt" or set(control) != {
                "v", "kind", "receipt", "signature", "channel_base"}:
            raise VoucherError("worker returned an invalid receipt control envelope")
        receipt = control["receipt"]
        if not isinstance(receipt, dict):
            raise VoucherError("worker receipt is not an object")
        signature_hex = control["signature"]
        try:
            signature = bytes.fromhex(signature_hex)
            public = bytes.fromhex(worker_key)
            ed25519.Ed25519PublicKey.from_public_bytes(public).verify(
                signature,
                json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8"))
        except (InvalidSignature, TypeError, ValueError) as exc:
            raise VoucherError("worker receipt signature is invalid") from exc
        channel = self.channels[worker_key]
        account = self.state.read_channel(channel.channel, self.program)
        base = _u64(control["channel_base"], "worker-derived channel base")
        previous_base = self.session_bases[worker_key]
        if previous_base is not None and base != previous_base:
            raise VoucherError("worker-derived channel base changed within the session")
        adopted = self.book.adopt_session_base(
            channel.channel, base, on_chain_redeemed=account.redeemed)
        self.channels[worker_key] = adopted
        self.session_bases[worker_key] = base
        self.receipts[worker_key].append(receipt)

    def acknowledge_delivery(self, *, prompt_tokens: int,
                             generated_tokens: int) -> None:
        for worker_key, queued in self.receipts.items():
            stage = self.plan[worker_key]
            channel = self.channels[worker_key]
            while queued:
                receipt = queued.pop(0)
                if (receipt.get("worker_keyid") != worker_key
                        or receipt.get("layer_start") != stage["layer_start"]
                        or receipt.get("layer_end") != stage["layer_end"]):
                    raise VoucherError("receipt worker or layer span differs from paid plan")
                expected_scope = {
                    "session_id": self.session_id,
                    "hold_id": self.hold_id,
                    "model_id": self.model_id,
                }
                if any(expected is not None and receipt.get(name) != expected
                       for name, expected in expected_scope.items()):
                    raise VoucherError("receipt scope differs from paid authorization")
                prompt = _u64(receipt.get("cumulative_prompt_tokens"), "receipt prompt tokens")
                generated = _u64(
                    receipt.get("cumulative_generated_tokens"), "receipt generated tokens")
                if prompt > prompt_tokens or generated > generated_tokens:
                    raise VoucherError("receipt claims tokens the consumer has not received")
                layers = int(stage["layer_end"]) - int(stage["layer_start"])
                price = _u64(receipt.get("cumulative_cost"), "receipt cumulative cost")
                expected = (prompt + generated) * layers if receipt.get("billable") else 0
                if price != expected:
                    raise VoucherError("receipt cumulative cost is inconsistent")
                total = credits_to_total(channel.channel_base, price)
                # Expand the durable exposure ceiling before producing a
                # signature.  A crash after this write may safely re-sign any
                # total at or below the ceiling in a later session.
                self.book.record(channel.channel, total)
                signature = sign_voucher(
                    self.private_key, self.program, channel.channel, total)
                self.pending[worker_key] = voucher_wire(
                    program=self.program, channel=channel.channel, total=total,
                    signature=signature)


def configured_program() -> str:
    value = os.environ.get("NAKSHATRA_PAY_PROGRAM_ID", "").strip()
    if not value:
        raise VoucherError("NAKSHATRA_PAY_PROGRAM_ID is required in chain mode")
    return _pubkey_text(value, "configured program id")


def configured_state() -> JsonRpcChannelState:
    url = os.environ.get("NAKSHATRA_PAY_RPC_URL", "").strip()
    if not url:
        raise VoucherError("NAKSHATRA_PAY_RPC_URL is required in chain mode")
    return JsonRpcChannelState(url)
