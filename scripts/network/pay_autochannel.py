"""Open consumer payment channels on demand under local soft limits."""
from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import stat
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

from network import pay_chain, pay_voucher


DEFAULT_LIMITS = {
    "cap_total": 5_000_000,
    "days": 30,
    "per_tx": 1_000_000,
    "per_day": 5_000_000,
}
CHANNEL_ACCOUNT_SPACE = 213
DAY_SECONDS = 24 * 60 * 60


def _amount(value: int) -> str:
    whole, fraction = divmod(int(value), 1_000_000)
    return str(whole) if not fraction else f"{whole}.{fraction:06d}".rstrip("0")


def _positive_u64(value: object, name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 0 < value < 2**64):
        raise pay_voucher.VoucherError(f"{name} must be a positive u64")
    return value


def load_limits(path: str | Path | None = None) -> dict[str, int]:
    """Read only the operator-managed mode-0600 limits file."""
    limits_path = Path(path or Path.home() / ".nakshatra" / "pay" / "limits.json")
    if not limits_path.exists():
        return dict(DEFAULT_LIMITS)
    try:
        info = limits_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise pay_voucher.VoucherError("pay limits must be a regular file")
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise pay_voucher.VoucherError("pay limits must be uid-owned mode 0600")
        values = json.loads(limits_path.read_text(encoding="utf-8"))
    except pay_voucher.VoucherError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise pay_voucher.VoucherError(f"cannot read pay limits: {exc}") from exc
    if not isinstance(values, dict) or set(values) != set(DEFAULT_LIMITS):
        raise pay_voucher.VoucherError(
            "pay limits must contain cap_total, days, per_tx, and per_day")
    parsed = {
        name: _positive_u64(values[name], f"pay limit {name}")
        for name in ("cap_total", "per_tx", "per_day")
    }
    days = values["days"]
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 3650:
        raise pay_voucher.VoucherError("pay limit days must be between 1 and 3650")
    parsed["days"] = days
    if parsed["per_tx"] > parsed["per_day"]:
        raise pay_voucher.VoucherError("pay limit per-tx cannot exceed per-day")
    if parsed["per_day"] > parsed["cap_total"]:
        raise pay_voucher.VoucherError(
            "pay limit per-day cannot exceed the on-chain cap")
    return parsed


class OpenLedger:
    """Mode-0600 conservative accounting for channel-open caps."""

    _STATUSES = frozenset({"pending", "sent", "finalized", "failed"})

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".nakshatra" / "pay" / "opens.sqlite")
        pay_voucher._secure_db_path(self.path)
        self._db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS opens("
            "id INTEGER PRIMARY KEY, opened_at REAL NOT NULL, cap TEXT NOT NULL, "
            "status TEXT NOT NULL, signature TEXT NOT NULL DEFAULT '')")
        self._secure()

    def _secure(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def total_since(self, now: float) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(CAST(cap AS INTEGER)),0) AS total FROM opens "
            "WHERE opened_at>? AND status!='failed'", (float(now) - DAY_SECONDS,)
        ).fetchone()
        return int(row["total"])

    def monotonic_time(self, chain_time: int) -> float:
        """Keep the accounting window from moving behind any stored row."""
        if isinstance(chain_time, bool) or not isinstance(chain_time, int) or chain_time <= 0:
            raise pay_voucher.VoucherError("chain time must be a positive integer")
        row = self._db.execute("SELECT MAX(opened_at) AS newest FROM opens").fetchone()
        newest = row["newest"]
        return max(float(chain_time), float(newest)) if newest is not None else float(chain_time)

    def reserve(self, cap: int, now: float) -> int:
        amount = _positive_u64(cap, "channel cap")
        cursor = self._db.execute(
            "INSERT INTO opens(opened_at,cap,status) VALUES(?,?,'pending')",
            (float(now), str(amount)))
        self._secure()
        return int(cursor.lastrowid)

    def set_status(self, row_id: int, status: str, signature: str = "") -> None:
        if status not in self._STATUSES:
            raise ValueError("invalid channel-open status")
        cursor = self._db.execute(
            "UPDATE opens SET status=?,signature=? WHERE id=?",
            (status, str(signature), int(row_id)))
        if cursor.rowcount != 1:
            raise pay_voucher.VoucherError("unknown channel-open reservation")
        self._secure()

    def close(self) -> None:
        self._secure()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class _OpenLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fd = -1

    def __enter__(self):
        old_umask = os.umask(0o077)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.path.parent, 0o700)
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            self.fd = os.open(self.path, flags, 0o600)
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise pay_voucher.VoucherError(
                    "channel-open lock must be a uid-owned regular file")
            os.fchmod(self.fd, 0o600)
        except pay_voucher.VoucherError:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise
        except OSError as exc:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise pay_voucher.VoucherError(f"cannot secure channel-open lock: {exc}") from exc
        finally:
            os.umask(old_umask)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.fd >= 0:
            fd, self.fd = self.fd, -1
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def pay_lock(path: str | Path | None = None) -> _OpenLock:
    """Return the one host-wide lock shared by opens and CLI ensure."""
    lock_path = Path(path or Path.home() / ".nakshatra" / "pay" / "channel-open.lock")
    return _OpenLock(lock_path)


def _account(rpc, address: str, *, owner: str, label: str) -> dict:
    info = rpc.account_info(address)
    if info is None:
        raise pay_voucher.VoucherError(f"{label} is not configured; run `nak pay ensure`")
    if info["owner"] != owner:
        raise pay_voucher.VoucherError(f"{label} is owned by an unexpected program")
    return info


def _limiting_refusal(required: int, candidates: Mapping[str, int]) -> str:
    cap = min(candidates.values())
    if candidates["per_tx"] == cap:
        return (f"required {_amount(required)} USDC exceeds the local per-transaction "
                f"channel limit {_amount(cap)} USDC; raise it with `nak pay limits`")
    if candidates["per_day"] == cap:
        return (f"required {_amount(required)} USDC exceeds the remaining local 24-hour "
                f"channel-open budget {_amount(cap)} USDC; raise it with `nak pay limits`")
    if candidates["delegate"] == cap:
        return (f"required {_amount(required)} USDC exceeds the delegate's remaining "
                f"on-chain cap {_amount(cap)} USDC; run `nak pay limits`, then `nak pay ensure`")
    return (f"required {_amount(required)} USDC exceeds uncommitted escrow collateral "
            f"{_amount(cap)} USDC; add funds with `nak pay deposit`")


def ensure_channel(worker_key: str, required_micro_usdc: int, *, owner: str,
                   book: pay_voucher.PayerChannelBook | None = None,
                   state: pay_voucher.ChannelState | None = None,
                   program: str | None = None, private_key: bytes | None = None,
                   rpc=None, limits_path: str | Path | None = None,
                   ledger_path: str | Path | None = None,
                   lock_path: str | Path | None = None,
                   clock: Callable[[], float] = time.time) -> pay_voucher.PayerChannel:
    """Return a usable worker channel, opening one with the node delegate if needed."""
    required = _positive_u64(required_micro_usdc, "required channel capacity")
    if re.fullmatch(r"[0-9a-f]{64}", str(worker_key)) is None:
        raise pay_voucher.VoucherError("worker key must be 32-byte lowercase hex")
    configured_program = (pay_voucher.configured_program() if program is None
                          else pay_chain.pubkey_text(program, "program id"))
    channel_book = book or pay_voucher.PayerChannelBook()
    channel_state = state or pay_voucher.configured_state()
    try:
        return channel_book.begin_session(
            worker_key, program=configured_program, state=channel_state,
            required_micro_usdc=required)
    except pay_voucher.NoUsableChannel:
        pass

    pay_dir = Path.home() / ".nakshatra" / "pay"
    with pay_lock(lock_path):
        # A different gateway process may have completed an open while this
        # process waited for the host-wide lock.
        try:
            return channel_book.begin_session(
                worker_key, program=configured_program, state=channel_state,
                required_micro_usdc=required)
        except pay_voucher.NoUsableChannel:
            pass

        if private_key is None:
            raise pay_voucher.VoucherError("node signing key is required to open a channel")
        node_seed = bytes(private_key)
        node_public = pay_voucher.public_key_bytes(node_seed)
        node_address = pay_chain.b58encode(node_public)
        owner_address = pay_chain.pubkey_text(owner, "escrow owner")
        worker_address = pay_chain.b58encode(bytes.fromhex(worker_key))
        tx_rpc = rpc
        if tx_rpc is None:
            rpc_url = pay_chain.configured_rpc_url(os.environ)
            if not rpc_url:
                raise pay_voucher.VoucherError("NAK_PAY_RPC is required to open a channel")
            tx_rpc = pay_chain.PayRpc(rpc_url)

        # This check precedes every signing path and keeps mainnet refused.
        tx_rpc.ensure_cluster()
        limits = load_limits(limits_path)
        ledger_file = Path(ledger_path or pay_dir / "opens.sqlite")
        with OpenLedger(ledger_file) as ledger:
            for attempt in range(2):
                # ``clock`` remains accepted for branch compatibility, but a
                # local wall clock never participates in channel-open policy.
                now = ledger.monotonic_time(tx_rpc.chain_time())
                local_day = max(0, limits["per_day"] - ledger.total_since(now))
                own_derived = pay_chain._derived(owner_address, configured_program)

                def source(delegate_address: str, delegate: dict, *, own: bool):
                    escrow_address = delegate["escrow"]
                    escrow_info = tx_rpc.account_info(escrow_address)
                    if escrow_info is None:
                        raise pay_voucher.VoucherError("delegate names a missing escrow")
                    if escrow_info["owner"] != configured_program:
                        raise pay_voucher.VoucherError(
                            "delegate escrow is owned by an unexpected program")
                    escrow = pay_chain.decode_escrow(escrow_info["data"])
                    canonical = pay_chain._derived(escrow["owner"], configured_program)
                    if canonical["escrow"] != escrow_address or canonical["vault"] != escrow["vault"]:
                        raise pay_voucher.VoucherError(
                            "delegate escrow is not canonical for its recorded owner")
                    if (delegate["key"] != node_address
                            or delegate["generation"] != escrow["generation"]):
                        return None, ("node delegate does not match the current escrow generation; "
                                      "run `nak pay ensure`" if own else "starter generation is stale")
                    if delegate["disabled"]:
                        return None, ("node delegate is disabled; run `nak pay ensure`"
                                      if own else "starter is disabled")
                    if now >= delegate["expires_at"]:
                        return None, ("node delegate is expired; run `nak pay ensure`"
                                      if own else "starter is expired")
                    if not delegate["flags"] & pay_chain.DELEGATE_OPEN_CHANNEL:
                        return None, ("node delegate lacks OPEN_CHANNEL; run `nak pay ensure`"
                                      if own else "starter lacks OPEN_CHANNEL")
                    vault_info = _account(
                        tx_rpc, escrow["vault"], owner=pay_chain.TOKEN_PROGRAM_ID,
                        label="escrow vault")
                    candidates = {
                        "per_tx": limits["per_tx"], "per_day": local_day,
                        "delegate": max(0, delegate["cap_total"] - delegate["spent"]),
                        "vault": max(
                            0, pay_chain.token_amount(vault_info["data"]) - escrow["committed"]),
                    }
                    cap = min(candidates.values())
                    if cap < required:
                        return None, _limiting_refusal(required, candidates)
                    return {
                        "owner": escrow["owner"], "escrow": escrow_address,
                        "delegate": delegate_address, "delegate_state": delegate,
                        "escrow_state": escrow, "cap": cap,
                    }, ""

                chosen_source = None
                own_reason = "pay escrow is not configured; run `nak pay ensure`"
                own_escrow_info = tx_rpc.account_info(own_derived["escrow"])
                if own_escrow_info is not None:
                    if own_escrow_info["owner"] != configured_program:
                        raise pay_voucher.VoucherError(
                            "pay escrow is owned by an unexpected program")
                    own_escrow = pay_chain.decode_escrow(own_escrow_info["data"])
                    if (own_escrow["owner"] != owner_address
                            or own_escrow["vault"] != own_derived["vault"]):
                        raise pay_voucher.VoucherError(
                            "pay escrow does not match the configured owner and vault")
                    own_delegate_address = pay_chain.delegate_pda(
                        own_derived["escrow"], node_public, configured_program)[0]
                    own_delegate_info = tx_rpc.account_info(own_delegate_address)
                    if own_delegate_info is None:
                        own_reason = "node delegate is not configured; run `nak pay ensure`"
                    elif own_delegate_info["owner"] != configured_program:
                        raise pay_voucher.VoucherError(
                            "node delegate is owned by an unexpected program")
                    else:
                        own_delegate = pay_chain.decode_delegate(own_delegate_info["data"])
                        chosen_source, own_reason = source(
                            own_delegate_address, own_delegate, own=True)

                if chosen_source is None:
                    lookup = getattr(tx_rpc, "delegates_for_key", None)
                    starter_delegates = (lookup(node_public, configured_program)
                                         if callable(lookup) else [])
                    starter_delegates = sorted(
                        (item for item in starter_delegates
                         if item["escrow"] != own_derived["escrow"]),
                        key=lambda item: (item["expires_at"], item["address"]))
                    starter_reason = ""
                    for starter in starter_delegates:
                        candidate, candidate_reason = source(
                            starter["address"], starter, own=False)
                        if candidate is not None:
                            chosen_source = candidate
                            break
                        if not starter_reason:
                            starter_reason = candidate_reason
                if chosen_source is None:
                    raise pay_voucher.VoucherError(starter_reason or own_reason)

                escrow = chosen_source["escrow_state"]
                escrow_address = chosen_source["escrow"]
                delegate_address = chosen_source["delegate"]
                source_owner = chosen_source["owner"]
                cap = chosen_source["cap"]

                seq = escrow["channel_seq"]
                generation = escrow["generation"]
                channel_address = pay_chain.channel_pda(
                    escrow_address, generation, seq, configured_program)[0]
                instructions = pay_chain.channel_open(
                    node_address, source_owner, worker_address, cap, generation, seq,
                    configured_program, delegate=delegate_address)
                blockhash = tx_rpc.latest_blockhash()
                message = pay_chain.compile_message(
                    instructions, node_address, blockhash)
                fee = tx_rpc.fee_for_message(message.data)
                rent = tx_rpc.rent_exemption(CHANNEL_ACCOUNT_SPACE)
                node_balance = tx_rpc.balance(node_address)
                if node_balance < rent + fee:
                    raise pay_voucher.VoucherError(
                        f"node key lacks SOL for channel rent and fee "
                        f"(needs {rent + fee} lamports, has {node_balance}); "
                        "run `nak pay ensure`")

                reservation = ledger.reserve(cap, now)
                try:
                    transaction = pay_chain.sign_transaction(
                        instructions, node_address, blockhash,
                        {node_address: node_seed})
                except Exception:
                    ledger.set_status(reservation, "failed")
                    raise
                try:
                    signature = tx_rpc.send_and_confirm(
                        transaction,
                        on_sent=lambda sent, row=reservation: ledger.set_status(
                            row, "sent", sent))
                except pay_chain.TransactionDefiniteFailure:
                    ledger.set_status(reservation, "failed", transaction.signature)
                    fresh_info = tx_rpc.account_info(escrow_address)
                    fresh_seq = (pay_chain.decode_escrow(fresh_info["data"])["channel_seq"]
                                 if fresh_info is not None else seq)
                    if attempt == 0 and fresh_seq != seq:
                        continue
                    raise
                # Any other exception deliberately leaves pending/sent charged:
                # the broadcast outcome is not known.
                ledger.set_status(reservation, "finalized", signature)
                account = tx_rpc.read_channel(channel_address, configured_program)
                if (account.escrow != escrow_address
                        or account.worker != worker_address
                        or account.cap != cap
                        or account.voucher_signer != node_address
                        or account.delegate != delegate_address
                        or account.status != "Open"):
                    raise pay_voucher.VoucherError(
                        "finalized channel does not match the requested open")
                channel_book.put(
                    program=configured_program, channel=channel_address,
                    worker_key=worker_key, cap=cap, channel_base=0,
                    max_signed_total=0)
                chosen = channel_book.begin_session(
                    worker_key, program=configured_program, state=tx_rpc,
                    required_micro_usdc=required, channel=channel_address)
                print(
                    f"[paid] opened channel for {worker_key[:12]} "
                    f"cap={_amount(cap)} USDC signature={signature}",
                    file=sys.stderr)
                return chosen
    raise pay_voucher.VoucherError("channel open retry was exhausted")
