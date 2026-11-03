"""Worker-owned voucher redemption and payer channel reclaim.

All policy time and all transaction-state decisions come from finalized RPC
state.  Signed attempts are committed durably before broadcast so a crash can
never cause an ambiguous transaction to be replaced while it can still land.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Callable, Iterable

from network import pay_chain, pay_voucher
from network.pay_autochannel import _OpenLock
from network.pay_wallet import load_wallet


DEFAULT_THRESHOLD = 1_000_000
DEFAULT_MAX_AGE_S = 24 * 60 * 60
DEFAULT_SAFETY_MARGIN_S = 600


def _positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise pay_voucher.VoucherError(f"{name} must be an integer >= {minimum}")
    return value


class AttemptBook:
    """Mode-0600 journal for signed payment transactions and channel clocks."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".nakshatra" / "pay" / "redeems.sqlite")
        pay_voucher._secure_db_path(self.path)
        self._db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS attempts("
            "id INTEGER PRIMARY KEY, channel TEXT NOT NULL, kind TEXT NOT NULL, "
            "signature TEXT NOT NULL UNIQUE, blockhash TEXT NOT NULL, "
            "last_valid_block_height INTEGER NOT NULL, total TEXT NOT NULL DEFAULT '0', "
            "created_chain INTEGER NOT NULL, resolution TEXT, error TEXT NOT NULL DEFAULT '', "
            "resolved_chain INTEGER NOT NULL DEFAULT 0)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS channels("
            "channel TEXT PRIMARY KEY, first_voucher_chain INTEGER NOT NULL, "
            "last_redeem_chain INTEGER NOT NULL DEFAULT 0)"
        )
        self._secure()

    def _secure(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def note_voucher(self, channel: str, chain_now: int) -> None:
        key = pay_chain.pubkey_text(channel, "channel")
        now = _positive(chain_now, "chain time")
        self._db.execute(
            "INSERT OR IGNORE INTO channels(channel,first_voucher_chain) VALUES(?,?)",
            (key, now))
        self._secure()

    def timing(self, channel: str) -> tuple[int, int]:
        key = pay_chain.pubkey_text(channel, "channel")
        row = self._db.execute(
            "SELECT first_voucher_chain,last_redeem_chain FROM channels WHERE channel=?",
            (key,)).fetchone()
        return (0, 0) if row is None else (
            int(row["first_voucher_chain"]), int(row["last_redeem_chain"]))

    def record(self, *, channel: str, kind: str, signature: str, blockhash: str,
               last_valid_block_height: int, chain_now: int, total: int = 0) -> int:
        if kind not in {"redeem", "close", "finalize"}:
            raise ValueError("invalid payment attempt kind")
        key = pay_chain.pubkey_text(channel, "channel")
        pay_chain.pubkey_bytes(blockhash, "recent blockhash")
        _positive(last_valid_block_height, "last valid block height", allow_zero=True)
        _positive(chain_now, "chain time")
        _positive(total, "attempt total", allow_zero=True)
        cursor = self._db.execute(
            "INSERT INTO attempts(channel,kind,signature,blockhash,"
            "last_valid_block_height,total,created_chain) VALUES(?,?,?,?,?,?,?)",
            (key, kind, str(signature), str(blockhash), int(last_valid_block_height),
             str(total), int(chain_now)))
        self._secure()
        return int(cursor.lastrowid)

    def unresolved(self, channel: str) -> list[dict]:
        key = pay_chain.pubkey_text(channel, "channel")
        return [dict(row) for row in self._db.execute(
            "SELECT * FROM attempts WHERE channel=? AND resolution IS NULL ORDER BY id",
            (key,))]

    def succeeded(self, channel: str, kind: str) -> bool:
        key = pay_chain.pubkey_text(channel, "channel")
        return self._db.execute(
            "SELECT 1 FROM attempts WHERE channel=? AND kind=? "
            "AND resolution='succeeded' LIMIT 1", (key, kind)).fetchone() is not None

    def resolve(self, attempt_id: int, resolution: str, chain_now: int,
                error: str = "") -> None:
        if resolution not in {"succeeded", "failed", "expired"}:
            raise ValueError("invalid payment attempt resolution")
        row = self._db.execute(
            "SELECT channel,kind FROM attempts WHERE id=?", (int(attempt_id),)).fetchone()
        if row is None:
            raise pay_voucher.VoucherError("unknown payment attempt")
        cursor = self._db.execute(
            "UPDATE attempts SET resolution=?,error=?,resolved_chain=? "
            "WHERE id=? AND resolution IS NULL",
            (resolution, str(error)[:500], int(chain_now), int(attempt_id)))
        if cursor.rowcount and resolution == "succeeded" and row["kind"] == "redeem":
            self._db.execute(
                "UPDATE channels SET last_redeem_chain=? WHERE channel=?",
                (int(chain_now), row["channel"]))
        self._secure()

    def close(self) -> None:
        self._secure()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()


def redeem_lock(path: str | Path | None = None) -> _OpenLock:
    return _OpenLock(path or Path.home() / ".nakshatra" / "pay" / "redeem.lock")


def resolve_attempts(book: AttemptBook, rpc, channel: str, chain_now: int,
                     log: Callable[[str], None] = print) -> bool:
    """Resolve every attempt; return True while any can still land."""
    unresolved = False
    height = None
    for attempt in book.unresolved(channel):
        status = pay_chain.validate_signature_status(
            rpc.signature_status(attempt["signature"]))
        if status is not None and status["confirmationStatus"] == "finalized":
            if status["err"] is None:
                book.resolve(attempt["id"], "succeeded", chain_now)
                log(f"{attempt['kind']} finalized for {channel}: {attempt['signature']}")
            else:
                book.resolve(attempt["id"], "failed", chain_now, str(status["err"]))
                log(f"{attempt['kind']} failed at finalized for {channel}: {status['err']}")
            continue
        if status is not None:
            # Processed/confirmed is an observation that the signature was
            # seen.  Neither a success nor an error is final until the rooted
            # fork says so, and blockhash expiry cannot erase that observation.
            unresolved = True
            continue
        if height is None:
            height = rpc.block_height(commitment="finalized")
        if height > int(attempt["last_valid_block_height"]):
            book.resolve(attempt["id"], "expired", chain_now)
            log(f"{attempt['kind']} expired before finalization for {channel}")
        else:
            unresolved = True
    return unresolved


def _account_exists(rpc, address: str, *, owner: str) -> bool:
    info = rpc.account_info(address, commitment="finalized")
    if info is None:
        return False
    if info["owner"] != owner:
        raise pay_voucher.VoucherError(
            f"account {address} is owned by an unexpected program")
    return True


def _signer_for(voucher: pay_voucher.VoucherRecord, channel: pay_voucher.ChannelAccount,
                owner: str) -> str:
    for candidate in (channel.voucher_signer, owner):
        if pay_voucher.verify_voucher(
                candidate, voucher.signature, voucher.program,
                voucher.channel, voucher.total):
            return candidate
    raise pay_voucher.VoucherError("stored voucher no longer verifies against chain signers")


def _prepare_and_send(*, rpc, book: AttemptBook, channel: str, kind: str,
                      instructions: list[pay_chain.Instruction], payer: str,
                      seed: bytes, chain_now: int, rent: int = 0, total: int = 0,
                      dry_run: bool = False,
                      on_recorded: Callable[[], None] | None = None,
                      log: Callable[[str], None] = print) -> bool:
    blockhash_info = rpc.latest_blockhash_info(commitment="finalized")
    message = pay_chain.compile_message(
        instructions, payer, blockhash_info["blockhash"])
    fee = rpc.fee_for_message(message.data, commitment="finalized")
    balance = rpc.balance(payer, commitment="finalized")
    needed = int(rent) + fee
    if balance < needed:
        log(f"skip {kind} {channel}: wallet lacks SOL (needs {needed} lamports, "
            f"has {balance}); nothing was sent")
        return False
    if dry_run:
        log(f"dry-run {kind} {channel}: {len(instructions)} instruction(s), "
            f"{needed} lamports required; nothing was sent")
        return True
    transaction = pay_chain.sign_transaction(
        instructions, payer, blockhash_info["blockhash"], {payer: seed})
    # Durability boundary: the exact signed intent is committed before the
    # first possible sendTransaction call.
    book.record(
        channel=channel, kind=kind, signature=transaction.signature,
        blockhash=blockhash_info["blockhash"],
        last_valid_block_height=blockhash_info["lastValidBlockHeight"],
        chain_now=chain_now, total=total)
    if on_recorded is not None:
        on_recorded()
    try:
        rpc.send_and_confirm(
            transaction,
            last_valid_block_height=blockhash_info["lastValidBlockHeight"])
    except (pay_chain.VoucherError, ValueError) as exc:
        # Even preflight/transport errors stay unresolved until a finalized
        # status or finalized block-height expiry proves the outcome.
        log(f"{kind} outcome unresolved for {channel}: {exc}")
        return False
    # send_and_confirm returns only for finalized success.  Resolve through the
    # same status path so tests and alternate RPC implementations share rules.
    return not resolve_attempts(book, rpc, channel, chain_now, log)


def _cooperative_finalize(*, rpc, book: AttemptBook, voucher: pay_voucher.VoucherRecord,
                          channel: pay_voucher.ChannelAccount, owner: str, wallet: dict,
                          chain_now: int, dry_run: bool,
                          log: Callable[[str], None]) -> bool:
    if channel.status != "Closing" or channel.redeemed != voucher.total:
        return False
    if resolve_attempts(book, rpc, voucher.channel, chain_now, log):
        return False
    instructions = pay_chain.channel_close_finalize(
        wallet["address"], owner, channel.generation, channel.seq,
        channel.rent_payer, channel.delegate, voucher.program)
    return _prepare_and_send(
        rpc=rpc, book=book, channel=voucher.channel, kind="finalize",
        instructions=instructions, payer=wallet["address"], seed=wallet["seed"],
        chain_now=chain_now, dry_run=dry_run, log=log)


def run_redeemer(*, store: pay_voucher.WorkerVoucherStore | None = None,
                 rpc=None, wallet: dict | None = None, program: str | None = None,
                 mint: str | None = None, channel: str | None = None,
                 threshold: int = DEFAULT_THRESHOLD,
                 max_age_s: int = DEFAULT_MAX_AGE_S,
                 safety_margin_s: int = DEFAULT_SAFETY_MARGIN_S,
                 dry_run: bool = False, attempts_path: str | Path | None = None,
                 lock_path: str | Path | None = None,
                 log: Callable[[str], None] = print) -> dict[str, int]:
    """Run one locked worker redemption cycle."""
    threshold = _positive(threshold, "redeem threshold")
    max_age_s = _positive(max_age_s, "maximum redeem age")
    safety_margin_s = _positive(safety_margin_s, "close safety margin", allow_zero=True)
    wanted = None if channel is None else pay_chain.pubkey_text(channel, "channel")
    owned_store = store is None
    vouchers = store or pay_voucher.WorkerVoucherStore()
    result = {"examined": 0, "sent": 0, "would_send": 0,
              "finalized": 0, "skipped": 0}
    try:
        with redeem_lock(lock_path), AttemptBook(attempts_path) as attempts:
            if rpc is None:
                rpc_url = pay_chain.configured_rpc_url(os.environ)
                if not rpc_url:
                    raise pay_voucher.VoucherError("NAK_PAY_RPC is required for redemption")
                rpc = pay_chain.PayRpc(rpc_url)
            rpc.ensure_cluster()
            configured_program = (pay_chain.configured_pubkey(
                os.environ, "NAK_PAY_PROGRAM") if program is None else
                pay_chain.pubkey_text(program, "program id"))
            configured_mint = (pay_chain.configured_pubkey(
                os.environ, "NAK_PAY_MINT") if mint is None else
                pay_chain.pubkey_text(mint, "mint"))
            worker_wallet = wallet or load_wallet(Path.home())
            if worker_wallet["custody"] != "file":
                raise pay_voucher.VoucherError(
                    "worker redeemer requires file-custody pay wallet")
            chain_now = rpc.chain_time()
            for voucher in vouchers.all_latest():
                if wanted is not None and voucher.channel != wanted:
                    continue
                result["examined"] += 1
                if voucher.program != configured_program:
                    log(f"skip {voucher.channel}: voucher program is not NAK_PAY_PROGRAM")
                    result["skipped"] += 1
                    continue
                attempts.note_voucher(voucher.channel, chain_now)
                if resolve_attempts(attempts, rpc, voucher.channel, chain_now, log):
                    log(f"skip {voucher.channel}: a prior transaction can still finalize")
                    result["skipped"] += 1
                    continue
                try:
                    state = rpc.read_channel(voucher.channel, configured_program)
                    owner = rpc.read_escrow_owner(state.escrow, configured_program)
                except pay_voucher.VoucherError as exc:
                    log(f"skip {voucher.channel}: finalized channel unavailable ({exc})")
                    result["skipped"] += 1
                    continue
                expected = pay_chain.channel_pda(
                    state.escrow, state.generation, state.seq, configured_program)[0]
                if expected != voucher.channel:
                    raise pay_voucher.VoucherError("channel is not its canonical generation/seq PDA")
                if state.worker != worker_wallet["address"]:
                    log(f"skip {voucher.channel}: channel worker is not this pay wallet")
                    result["skipped"] += 1
                    continue
                if voucher.total <= state.redeemed:
                    if _cooperative_finalize(
                            rpc=rpc, book=attempts, voucher=voucher, channel=state,
                            owner=owner, wallet=worker_wallet, chain_now=chain_now,
                            dry_run=dry_run, log=log):
                        result["finalized"] += 1
                    continue
                if state.status == "Closing" and chain_now >= state.close_at:
                    log(f"WARNING: voucher {voucher.channel} is UNREDEEMABLE: chain time "
                        f"{chain_now} reached close_at {state.close_at}; nothing was sent")
                    result["skipped"] += 1
                    continue
                unredeemed = voucher.total - state.redeemed
                first_seen, last_success = attempts.timing(voucher.channel)
                age_base = last_success or first_seen
                due = unredeemed >= threshold or chain_now - age_base >= max_age_s
                if state.status == "Closing":
                    due = True
                    remaining = state.close_at - chain_now
                    label = "inside safety margin" if remaining <= safety_margin_s else "urgent"
                    log(f"Closing channel {voucher.channel}: {label}, {remaining}s remain")
                if not due:
                    result["skipped"] += 1
                    continue
                signer = _signer_for(voucher, state, owner)
                instructions = pay_chain.channel_redeem(
                    worker_wallet["address"], owner, state.generation, state.seq,
                    state.worker, voucher.total, signer, voucher.signature,
                    configured_program, configured_mint)
                worker_ata = pay_chain.associated_token_address(
                    state.worker, configured_mint)[0]
                ata_missing = not _account_exists(
                    rpc, worker_ata, owner=pay_chain.TOKEN_PROGRAM_ID)
                rent = 0
                if ata_missing:
                    instructions.insert(0, pay_chain.ata_create_idempotent(
                        worker_wallet["address"], state.worker, configured_mint))
                    rent = rpc.rent_exemption(165, commitment="finalized")
                sent = _prepare_and_send(
                    rpc=rpc, book=attempts, channel=voucher.channel, kind="redeem",
                    instructions=instructions, payer=worker_wallet["address"],
                    seed=worker_wallet["seed"], chain_now=chain_now, rent=rent,
                    total=voucher.total, dry_run=dry_run, log=log)
                if sent:
                    result["would_send" if dry_run else "sent"] += 1
                    try:
                        fresh = rpc.read_channel(voucher.channel, configured_program)
                    except pay_voucher.VoucherError:
                        fresh = state
                    if _cooperative_finalize(
                            rpc=rpc, book=attempts, voucher=voucher, channel=fresh,
                            owner=owner, wallet=worker_wallet, chain_now=chain_now,
                            dry_run=dry_run, log=log):
                        result["finalized"] += 1
    finally:
        if owned_store:
            vouchers.close()
    return result


def request_close(*, channels: Iterable[str], book: pay_voucher.PayerChannelBook,
                  rpc, program: str, wallet: dict, node_seed: bytes,
                  node_public: bytes, attempts_path: str | Path | None = None,
                  lock_path: str | Path | None = None,
                  dry_run: bool = False,
                  log: Callable[[str], None] = print) -> dict[str, int]:
    """Request close for selected payer channels with the on-chain authority."""
    result = {"examined": 0, "sent": 0, "would_send": 0, "skipped": 0}
    node_address = pay_chain.b58encode(node_public)
    with redeem_lock(lock_path), AttemptBook(attempts_path) as attempts:
        rpc.ensure_cluster()
        chain_now = rpc.chain_time()
        for requested in channels:
            key = pay_chain.pubkey_text(requested, "channel")
            result["examined"] += 1
            if any(row["kind"] == "close" for row in attempts.unresolved(key)):
                # A prior process may have died after journaling the signed
                # close but before recording local Closing.  Fail closed for
                # new work while that signature can still land.
                book.mark_closing(key, 0)
            if resolve_attempts(attempts, rpc, key, chain_now, log):
                result["skipped"] += 1
                continue
            row = book.for_channel(key)
            state = rpc.read_channel(key, program)
            owner = rpc.read_escrow_owner(state.escrow, program)
            if state.status == "Closing":
                book.mark_closing(key, state.close_at)
                result["skipped"] += 1
                continue
            if state.voucher_signer == node_address:
                signer, seed = node_address, bytes(node_seed)
            elif state.voucher_signer == owner and owner == wallet["address"]:
                signer, seed = wallet["address"], wallet["seed"]
            else:
                raise pay_voucher.VoucherError(
                    f"no local close authority for channel {key}")
            instructions = pay_chain.channel_close_request(
                signer, owner, state.generation, state.seq, program)
            if _prepare_and_send(
                    rpc=rpc, book=attempts, channel=key, kind="close",
                    instructions=instructions, payer=signer, seed=seed,
                    chain_now=chain_now, dry_run=dry_run,
                    on_recorded=lambda key=key: book.mark_closing(key, 0), log=log):
                result["would_send" if dry_run else "sent"] += 1
                if not dry_run:
                    fresh = rpc.read_channel(key, program)
                    if fresh.status != "Closing" or fresh.close_at <= chain_now:
                        raise pay_voucher.VoucherError(
                            "finalized close request did not produce a future close_at")
                    book.mark_closing(key, fresh.close_at)
            elif row.status == "Closing":
                result["skipped"] += 1
    return result


def finalize_payer_channels(*, book: pay_voucher.PayerChannelBook | None = None,
                            rpc=None, program: str | None = None,
                            node_seed: bytes | None = None,
                            channel: str | None = None, all_channels: bool = False,
                            attempts_path: str | Path | None = None,
                            lock_path: str | Path | None = None,
                            dry_run: bool = False,
                            log: Callable[[str], None] = print) -> dict[str, int]:
    """Finalize this payer node's elapsed Closing channels using its node key."""
    owned_book = book is None
    payer_book = book or pay_voucher.PayerChannelBook()
    result = {"examined": 0, "sent": 0, "would_send": 0, "skipped": 0}
    try:
        if node_seed is None:
            from nakshatra_auth import load_worker_key
            node_seed, _node_hex = load_worker_key(
                Path.home() / ".nakshatra" / "keys" / "worker.ed25519")
        node_seed = bytes(node_seed)
        node_public = pay_voucher.public_key_bytes(node_seed)
        node_address = pay_chain.b58encode(node_public)
        if rpc is None:
            rpc_url = pay_chain.configured_rpc_url(os.environ)
            if not rpc_url:
                raise pay_voucher.VoucherError("NAK_PAY_RPC is required for finalize")
            rpc = pay_chain.PayRpc(rpc_url)
        configured_program = (pay_chain.configured_pubkey(
            os.environ, "NAK_PAY_PROGRAM") if program is None else
            pay_chain.pubkey_text(program, "program id"))
        wanted = None if channel is None else pay_chain.pubkey_text(channel, "channel")
        with redeem_lock(lock_path), AttemptBook(attempts_path) as attempts:
            rpc.ensure_cluster()
            chain_now = rpc.chain_time()
            rows = payer_book.list_channels()
            for row in rows:
                if wanted is not None and row.channel != wanted:
                    continue
                if wanted is None and not all_channels and row.status != "Closing":
                    continue
                result["examined"] += 1
                if resolve_attempts(attempts, rpc, row.channel, chain_now, log):
                    result["skipped"] += 1
                    continue
                try:
                    state = rpc.read_channel(row.channel, configured_program)
                except pay_voucher.VoucherError as exc:
                    if attempts.succeeded(row.channel, "finalize"):
                        payer_book.mark_closed(row.channel)
                        log(f"finalize already succeeded for {row.channel}; channel is closed")
                        continue
                    raise
                if state.status != "Closing":
                    result["skipped"] += 1
                    continue
                if chain_now < state.close_at:
                    log(f"skip finalize {row.channel}: close_at {state.close_at} has not elapsed")
                    result["skipped"] += 1
                    continue
                owner = rpc.read_escrow_owner(state.escrow, configured_program)
                instructions = pay_chain.channel_close_finalize(
                    node_address, owner, state.generation, state.seq,
                    state.rent_payer, state.delegate, configured_program)
                if _prepare_and_send(
                        rpc=rpc, book=attempts, channel=row.channel, kind="finalize",
                        instructions=instructions, payer=node_address, seed=node_seed,
                        chain_now=chain_now, dry_run=dry_run, log=log):
                    result["would_send" if dry_run else "sent"] += 1
                    if not dry_run:
                        payer_book.mark_closed(row.channel)
    finally:
        if owned_book:
            payer_book.close()
    return result
