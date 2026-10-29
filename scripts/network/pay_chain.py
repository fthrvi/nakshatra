"""Dependency-free Solana transaction support for ``nakshatra_pay``.

Only :mod:`cryptography` is used for Ed25519 signatures.  The module deliberately
does not import solders/solana: installed nodes contain neither package.
"""
from __future__ import annotations

import base64
import hashlib
import struct
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from network.pay_voucher import (
    DEVNET_GENESIS_HASH,
    MAINNET_GENESIS_HASH,
    JsonRpcChannelState,
    VoucherError,
    _private_key,
    b58decode,
    b58encode,
)


SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
# The deployed program currently defines OPEN_CHANNEL and OPEN_TASK.  Its
# DELEGATE_FLAGS_ALL is therefore 0b11 (the older design prose named four).
DELEGATE_FLAGS_ALL = 0x03
DELEGATE_OPEN_CHANNEL = 0x01

_PDA_MARKER = b"ProgramDerivedAddress"
_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def pubkey_bytes(value: object, name: str = "public key") -> bytes:
    try:
        raw = b58decode(value) if isinstance(value, str) else bytes(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if len(raw) != 32:
        raise ValueError(f"{name} must contain exactly 32 bytes")
    return raw


def pubkey_text(value: object, name: str = "public key") -> str:
    return b58encode(pubkey_bytes(value, name))


def _is_ed25519_point(encoded: bytes) -> bool:
    """Match curve25519-dalek's compressed Edwards-Y decompression check."""
    if len(encoded) != 32:
        return False
    value = int.from_bytes(encoded, "little")
    # curve25519-dalek's FieldElement decoder masks the sign bit and reduces
    # the remaining integer modulo p; it deliberately accepts non-canonical
    # encodings here.  Solana's Pubkey::is_on_curve uses that decompressor.
    y = (value & ((1 << 255) - 1)) % _P
    y2 = y * y % _P
    denominator = (_D * y2 + 1) % _P
    if denominator == 0:
        return False
    x2 = (y2 - 1) * pow(denominator, _P - 2, _P) % _P
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _I % _P
    if (x * x - x2) % _P:
        return False
    return True


def create_program_address(seeds: Sequence[bytes], program_id: object) -> bytes:
    if len(seeds) > 16:
        raise ValueError("a PDA may have at most 16 seeds")
    normalized = []
    for seed in seeds:
        raw = bytes(seed)
        if len(raw) > 32:
            raise ValueError("a PDA seed may contain at most 32 bytes")
        normalized.append(raw)
    candidate = hashlib.sha256(
        b"".join(normalized) + pubkey_bytes(program_id, "program id") + _PDA_MARKER
    ).digest()
    if _is_ed25519_point(candidate):
        raise ValueError("PDA candidate lies on the Ed25519 curve")
    return candidate


def find_program_address(seeds: Sequence[bytes], program_id: object) -> tuple[str, int]:
    if len(seeds) >= 16:
        raise ValueError("PDA bump would exceed the 16-seed limit")
    for bump in range(255, -1, -1):
        try:
            raw = create_program_address([*seeds, bytes([bump])], program_id)
        except ValueError as exc:
            if str(exc) == "PDA candidate lies on the Ed25519 curve":
                continue
            raise
        return b58encode(raw), bump
    raise ValueError("could not find an off-curve PDA")


def config_pda(program_id: object) -> tuple[str, int]:
    return find_program_address([b"config"], program_id)


def owner_state_pda(owner: object, program_id: object) -> tuple[str, int]:
    config = config_pda(program_id)[0]
    return find_program_address(
        [b"owner_state", pubkey_bytes(config), pubkey_bytes(owner)], program_id)


def escrow_pda(owner: object, program_id: object) -> tuple[str, int]:
    config = config_pda(program_id)[0]
    return find_program_address([b"escrow", pubkey_bytes(config), pubkey_bytes(owner)], program_id)


def vault_pda(escrow: object, program_id: object) -> tuple[str, int]:
    return find_program_address([b"vault", pubkey_bytes(escrow)], program_id)


def delegate_pda(escrow: object, key: object, program_id: object) -> tuple[str, int]:
    return find_program_address(
        [b"delegate", pubkey_bytes(escrow), pubkey_bytes(key)], program_id)


def channel_pda(escrow: object, generation: int, seq: int,
                program_id: object) -> tuple[str, int]:
    return find_program_address(
        [b"channel", pubkey_bytes(escrow), _u64(generation), _u64(seq)], program_id)


def event_authority_pda(program_id: object) -> tuple[str, int]:
    return find_program_address([b"__event_authority"], program_id)


def associated_token_address(owner: object, mint: object) -> tuple[str, int]:
    return find_program_address(
        [pubkey_bytes(owner), pubkey_bytes(TOKEN_PROGRAM_ID), pubkey_bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID,
    )


def _u64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError("value must be a u64")
    return struct.pack("<Q", value)


def _i64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not -(2**63) <= value < 2**63:
        raise ValueError("value must be an i64")
    return struct.pack("<q", value)


def anchor_discriminator(name: str) -> bytes:
    return hashlib.sha256(f"global:{name}".encode("ascii")).digest()[:8]


@dataclass(frozen=True)
class AccountMeta:
    pubkey: str
    is_signer: bool = False
    is_writable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pubkey", pubkey_text(self.pubkey))


@dataclass(frozen=True)
class Instruction:
    program_id: str
    data: bytes
    accounts: tuple[AccountMeta, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", pubkey_text(self.program_id, "program id"))
        object.__setattr__(self, "data", bytes(self.data))
        object.__setattr__(self, "accounts", tuple(self.accounts))


def _meta(key: object, signer: bool = False, writable: bool = False) -> AccountMeta:
    return AccountMeta(pubkey_text(key), signer, writable)


def _program_instruction(name: str, program_id: object, accounts: Iterable[AccountMeta],
                         args: bytes = b"") -> Instruction:
    return Instruction(pubkey_text(program_id), anchor_discriminator(name) + args, tuple(accounts))


def _derived(owner: object, program_id: object) -> dict[str, str]:
    program = pubkey_text(program_id, "program id")
    owner_key = pubkey_text(owner, "owner")
    config = config_pda(program)[0]
    escrow = escrow_pda(owner_key, program)[0]
    return {
        "program": program,
        "owner": owner_key,
        "config": config,
        "owner_state": owner_state_pda(owner_key, program)[0],
        "escrow": escrow,
        "vault": vault_pda(escrow, program)[0],
        "event": event_authority_pda(program)[0],
    }


def ata_create_idempotent(payer: object, owner: object, mint: object) -> Instruction:
    ata = associated_token_address(owner, mint)[0]
    return Instruction(ASSOCIATED_TOKEN_PROGRAM_ID, b"\x01", (
        _meta(payer, True, True), _meta(ata, False, True), _meta(owner), _meta(mint),
        _meta(SYSTEM_PROGRAM_ID), _meta(TOKEN_PROGRAM_ID),
    ))


def system_transfer(source: object, destination: object, lamports: int) -> Instruction:
    """Build SystemInstruction::Transfer (instruction index 2)."""
    return Instruction(SYSTEM_PROGRAM_ID, struct.pack("<I", 2) + _u64(lamports), (
        _meta(source, True, True), _meta(destination, False, True),
    ))


def escrow_init(owner: object, mint: object, program_id: object) -> list[Instruction]:
    d = _derived(owner, program_id)
    return [_program_instruction("escrow_init", d["program"], (
        _meta(d["owner"], True, True), _meta(d["config"]),
        _meta(d["owner_state"], False, True), _meta(mint),
        _meta(d["escrow"], False, True), _meta(d["vault"], False, True),
        _meta(TOKEN_PROGRAM_ID), _meta(SYSTEM_PROGRAM_ID), _meta(d["event"]),
        _meta(d["program"]),
    ))]


def escrow_deposit(owner: object, mint: object, amount: int,
                   program_id: object) -> list[Instruction]:
    d = _derived(owner, program_id)
    source = associated_token_address(owner, mint)[0]
    return [ata_create_idempotent(owner, owner, mint), _program_instruction(
        "escrow_deposit", d["program"], (
            _meta(d["owner"], True), _meta(d["config"]), _meta(mint),
            _meta(d["escrow"]), _meta(source, False, True), _meta(d["vault"], False, True),
            _meta(TOKEN_PROGRAM_ID), _meta(d["event"]), _meta(d["program"]),
        ), _u64(amount))]


def escrow_withdraw_request(owner: object, amount: int,
                            program_id: object) -> list[Instruction]:
    d = _derived(owner, program_id)
    return [_program_instruction("escrow_withdraw_request", d["program"], (
        _meta(d["owner"], True), _meta(d["config"]), _meta(d["escrow"], False, True),
        _meta(d["vault"]), _meta(d["event"]), _meta(d["program"]),
    ), _u64(amount))]


def escrow_withdraw_execute(owner: object, mint: object,
                            program_id: object) -> list[Instruction]:
    d = _derived(owner, program_id)
    destination = associated_token_address(owner, mint)[0]
    return [ata_create_idempotent(owner, owner, mint), _program_instruction(
        "escrow_withdraw_execute", d["program"], (
            _meta(d["owner"], True), _meta(d["config"]), _meta(mint),
            _meta(d["escrow"], False, True), _meta(d["vault"], False, True),
            _meta(destination, False, True), _meta(TOKEN_PROGRAM_ID),
            _meta(d["event"]), _meta(d["program"]),
        ))]


def delegate_set(owner: object, key: object, cap_total: int, expires_at: int,
                 flags: int, program_id: object) -> list[Instruction]:
    if isinstance(flags, bool) or not isinstance(flags, int) or not 0 <= flags <= 255:
        raise ValueError("flags must be a u8")
    d = _derived(owner, program_id)
    key_raw = pubkey_bytes(key, "delegate key")
    delegate = delegate_pda(d["escrow"], key_raw, d["program"])[0]
    return [_program_instruction("delegate_set", d["program"], (
        _meta(d["owner"], True, True), _meta(d["config"]), _meta(d["escrow"]),
        _meta(delegate, False, True), _meta(SYSTEM_PROGRAM_ID), _meta(d["event"]),
        _meta(d["program"]),
    ), key_raw + _u64(cap_total) + _i64(expires_at) + bytes([flags]))]


def delegate_revoke(owner: object, key: object, program_id: object) -> list[Instruction]:
    d = _derived(owner, program_id)
    key_raw = pubkey_bytes(key, "delegate key")
    delegate = delegate_pda(d["escrow"], key_raw, d["program"])[0]
    return [_program_instruction("delegate_revoke", d["program"], (
        _meta(d["owner"], True, True), _meta(d["config"]), _meta(d["escrow"]),
        _meta(delegate, False, True), _meta(d["event"]), _meta(d["program"]),
    ), key_raw)]


def channel_open(signer: object, owner: object, worker: object, cap: int,
                 generation: int, seq: int, program_id: object,
                 *, delegate: object | None = None) -> list[Instruction]:
    d = _derived(owner, program_id)
    delegate_key = d["program"] if delegate is None else pubkey_text(delegate, "delegate account")
    channel = channel_pda(d["escrow"], generation, seq, d["program"])[0]
    worker_raw = pubkey_bytes(worker, "worker")
    return [_program_instruction("channel_open", d["program"], (
        _meta(signer, True, True), _meta(d["config"]), _meta(d["escrow"], False, True),
        _meta(d["vault"]), _meta(delegate_key, False, True), _meta(channel, False, True),
        _meta(SYSTEM_PROGRAM_ID), _meta(d["event"]), _meta(d["program"]),
    ), worker_raw + _u64(cap))]


def channel_close_request(signer: object, owner: object, generation: int, seq: int,
                          program_id: object) -> list[Instruction]:
    d = _derived(owner, program_id)
    channel = channel_pda(d["escrow"], generation, seq, d["program"])[0]
    return [_program_instruction("channel_close_request", d["program"], (
        _meta(signer, True), _meta(d["config"]), _meta(d["escrow"]),
        _meta(channel, False, True), _meta(d["event"]), _meta(d["program"]),
    ))]


def compact_u16(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFF:
        raise ValueError("compact-u16 value must be between 0 and 65535")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


@dataclass(frozen=True)
class CompiledMessage:
    data: bytes
    account_keys: tuple[str, ...]
    required_signers: tuple[str, ...]


def compile_message(instructions: Sequence[Instruction], fee_payer: object,
                    recent_blockhash: object) -> CompiledMessage:
    payer = pubkey_text(fee_payer, "fee payer")
    blockhash = pubkey_bytes(recent_blockhash, "recent blockhash")
    privileges: dict[str, list[bool | int]] = {payer: [True, True, -1]}
    order = 0
    for instruction in instructions:
        for account in instruction.accounts:
            if account.pubkey not in privileges:
                privileges[account.pubkey] = [account.is_signer, account.is_writable, order]
                order += 1
            else:
                privileges[account.pubkey][0] = bool(privileges[account.pubkey][0]) or account.is_signer
                privileges[account.pubkey][1] = bool(privileges[account.pubkey][1]) or account.is_writable
        if instruction.program_id not in privileges:
            privileges[instruction.program_id] = [False, False, order]
            order += 1
    if len(privileges) > 256:
        raise ValueError("legacy Solana messages may contain at most 256 accounts")

    def category(item: tuple[str, list[bool | int]]) -> tuple[int, bytes]:
        key, (signer, writable, first) = item
        if key == payer:
            return -1, b""
        # Solana's legacy Message compiler accumulates metas in a BTreeMap, so
        # keys inside each privilege class are ordered by their raw bytes.
        del first
        return (0 if signer and writable else 1 if signer else 2 if writable else 3), pubkey_bytes(key)

    keys = tuple(key for key, _value in sorted(privileges.items(), key=category))
    index = {key: position for position, key in enumerate(keys)}
    required = tuple(key for key in keys if bool(privileges[key][0]))
    readonly_signed = sum(not bool(privileges[key][1]) for key in required)
    readonly_unsigned = sum(
        not bool(privileges[key][0]) and not bool(privileges[key][1]) for key in keys)
    header = bytes([len(required), readonly_signed, readonly_unsigned])
    body = bytearray(header + compact_u16(len(keys)) + b"".join(map(pubkey_bytes, keys)) + blockhash)
    body.extend(compact_u16(len(instructions)))
    for instruction in instructions:
        body.append(index[instruction.program_id])
        body.extend(compact_u16(len(instruction.accounts)))
        body.extend(bytes(index[account.pubkey] for account in instruction.accounts))
        body.extend(compact_u16(len(instruction.data)))
        body.extend(instruction.data)
    return CompiledMessage(bytes(body), keys, required)


@dataclass(frozen=True)
class SignedTransaction:
    data: bytes
    message: bytes
    signatures: tuple[bytes, ...]
    signature: str


class TransactionDefiniteFailure(VoucherError):
    """The RPC proved that a transaction was rejected or failed on chain."""


def validate_signature_status(status: object) -> dict | None:
    """Validate the status fields whose absence would change payment meaning."""
    if status is None:
        return None
    if not isinstance(status, dict) or "err" not in status:
        raise VoucherError("RPC returned malformed signature status")
    confirmation = status.get("confirmationStatus")
    if confirmation not in {"processed", "confirmed", "finalized"}:
        raise VoucherError("RPC returned malformed signature confirmation status")
    return status


def sign_transaction(instructions: Sequence[Instruction], fee_payer: object,
                     recent_blockhash: object, signer_seeds: Mapping[object, bytes]) -> SignedTransaction:
    message = compile_message(instructions, fee_payer, recent_blockhash)
    seeds = {pubkey_text(key): bytes(value) for key, value in signer_seeds.items()}
    signatures = []
    for signer in message.required_signers:
        try:
            seed = seeds[signer]
        except KeyError as exc:
            raise ValueError(f"missing transaction signer: {signer}") from exc
        if len(seed) != 32:
            raise ValueError(f"signer seed for {signer} must contain 32 bytes")
        private = _private_key(seed)
        actual = b58encode(private.public_key().public_bytes_raw())
        if actual != signer:
            raise ValueError(f"signer seed does not match public key {signer}")
        signatures.append(private.sign(message.data))
    raw = compact_u16(len(signatures)) + b"".join(signatures) + message.data
    return SignedTransaction(raw, message.data, tuple(signatures), b58encode(signatures[0]))


class PayRpc(JsonRpcChannelState):
    """The voucher reader's hardened RPC transport, extended for transactions."""

    def ensure_cluster(self) -> str:
        self._ensure_cluster()
        genesis = self._cluster_genesis
        if not isinstance(genesis, str):
            raise VoucherError("Solana RPC returned malformed getGenesisHash response")
        return genesis

    def cluster_label(self) -> str:
        genesis = self.ensure_cluster()
        return "devnet" if genesis == DEVNET_GENESIS_HASH else "local"

    def chain_time(self) -> int:
        """Return finalized cluster time, never the caller's wall clock."""
        self._ensure_cluster()
        slot = self._rpc("getSlot", [{"commitment": "finalized"}])
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise VoucherError("RPC returned an invalid finalized slot")
        timestamp = self._rpc("getBlockTime", [slot])
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            raise VoucherError("RPC returned an invalid finalized block time")
        return timestamp

    def account_info(self, address: object, *, commitment: str = "confirmed") -> dict | None:
        self._ensure_cluster()
        key = pubkey_text(address, "account address")
        result = self._rpc("getAccountInfo", [
            key, {"commitment": str(commitment), "encoding": "base64"},
        ])
        if not isinstance(result, dict) or not isinstance(result.get("context"), dict):
            raise VoucherError("RPC returned a malformed account result")
        value = result.get("value")
        if value is None:
            return None
        if not isinstance(value, dict) or not isinstance(value.get("owner"), str):
            raise VoucherError("RPC returned a malformed account value")
        encoded = value.get("data")
        if not isinstance(encoded, list) or len(encoded) != 2 or encoded[1] != "base64":
            raise VoucherError("RPC returned an unsupported account encoding")
        try:
            raw = base64.b64decode(encoded[0], validate=True)
        except (ValueError, TypeError) as exc:
            raise VoucherError("RPC returned invalid base64 account data") from exc
        if value.get("space") != len(raw):
            raise VoucherError("RPC account data length differs from space")
        return {**value, "data": raw}

    def _delegates_filtered(self, program_id: object, *, offset: int,
                            value: object) -> list[dict]:
        """Return strictly validated Delegate accounts matching one pubkey field."""
        self._ensure_cluster()
        program = pubkey_text(program_id, "program id")
        wanted = pubkey_text(value, "delegate filter key")
        result = self._rpc("getProgramAccounts", [program, {
            "commitment": "finalized", "encoding": "base64",
            "filters": [
                {"dataSize": DELEGATE_ACCOUNT_SPACE},
                {"memcmp": {"offset": offset, "bytes": wanted}},
            ],
        }])
        if not isinstance(result, list):
            raise VoucherError("RPC returned a malformed getProgramAccounts result")
        rows = []
        seen = set()
        for item in result:
            if (not isinstance(item, dict) or set(item) != {"pubkey", "account"}
                    or not isinstance(item.get("pubkey"), str)
                    or not isinstance(item.get("account"), dict)):
                raise VoucherError("RPC returned a malformed Delegate account entry")
            address = pubkey_text(item["pubkey"], "Delegate account address")
            account = item["account"]
            required = {"data", "executable", "lamports", "owner", "rentEpoch", "space"}
            if (set(account) != required or account.get("owner") != program
                    or account.get("space") != DELEGATE_ACCOUNT_SPACE
                    or not isinstance(account.get("executable"), bool)
                    or account["executable"]
                    or isinstance(account.get("lamports"), bool)
                    or not isinstance(account.get("lamports"), int)
                    or account["lamports"] < 0
                    or isinstance(account.get("rentEpoch"), bool)
                    or not isinstance(account.get("rentEpoch"), int)
                    or account["rentEpoch"] < 0):
                raise VoucherError("RPC returned an invalid Delegate account entry")
            encoded = account.get("data")
            if not isinstance(encoded, list) or len(encoded) != 2 or encoded[1] != "base64":
                raise VoucherError("RPC returned an unsupported Delegate account encoding")
            try:
                raw = base64.b64decode(encoded[0], validate=True)
                decoded = decode_delegate(raw)
            except (TypeError, ValueError) as exc:
                raise VoucherError(f"RPC returned an invalid Delegate account: {exc}") from exc
            field = "escrow" if offset == DELEGATE_ESCROW_OFFSET else "key"
            if decoded[field] != wanted:
                raise VoucherError("RPC returned a Delegate outside the requested filter")
            expected = delegate_pda(decoded["escrow"], decoded["key"], program)[0]
            if address != expected or address in seen:
                raise VoucherError("RPC returned a non-canonical or duplicate Delegate account")
            seen.add(address)
            rows.append({"address": address, **decoded})
        return rows

    def delegates_for_key(self, key: object, program_id: object) -> list[dict]:
        return self._delegates_filtered(
            program_id, offset=DELEGATE_KEY_OFFSET, value=key)

    def delegates_for_escrow(self, escrow: object, program_id: object) -> list[dict]:
        return self._delegates_filtered(
            program_id, offset=DELEGATE_ESCROW_OFFSET, value=escrow)

    def balance(self, address: object, *, commitment: str = "confirmed") -> int:
        self._ensure_cluster()
        result = self._rpc(
            "getBalance", [pubkey_text(address), {"commitment": str(commitment)}])
        value = result.get("value") if isinstance(result, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VoucherError("RPC returned an invalid lamport balance")
        return value

    def latest_blockhash_info(self, *, commitment: str = "confirmed") -> dict:
        self._ensure_cluster()
        result = self._rpc("getLatestBlockhash", [{"commitment": str(commitment)}])
        value = result.get("value") if isinstance(result, dict) else None
        blockhash = value.get("blockhash") if isinstance(value, dict) else None
        last_valid = value.get("lastValidBlockHeight") if isinstance(value, dict) else None
        if not isinstance(blockhash, str) or len(pubkey_bytes(blockhash, "recent blockhash")) != 32:
            raise VoucherError("RPC returned an invalid recent blockhash")
        if isinstance(last_valid, bool) or not isinstance(last_valid, int) or last_valid < 0:
            raise VoucherError("RPC returned an invalid lastValidBlockHeight")
        return {"blockhash": blockhash, "lastValidBlockHeight": last_valid}

    def latest_blockhash(self, *, commitment: str = "confirmed") -> str:
        return self.latest_blockhash_info(commitment=commitment)["blockhash"]

    def block_height(self, *, commitment: str = "finalized") -> int:
        self._ensure_cluster()
        height = self._rpc("getBlockHeight", [{"commitment": str(commitment)}])
        if isinstance(height, bool) or not isinstance(height, int) or height < 0:
            raise VoucherError("RPC returned an invalid block height")
        return height

    def signature_status(self, signature: str) -> dict | None:
        self._ensure_cluster()
        result = self._rpc(
            "getSignatureStatuses", [[str(signature)], {"searchTransactionHistory": True}])
        values = result.get("value") if isinstance(result, dict) else None
        if not isinstance(values, list) or len(values) != 1:
            raise VoucherError("RPC returned malformed signature status")
        return validate_signature_status(values[0])

    def rent_exemption(self, space: int, *, commitment: str = "confirmed") -> int:
        self._ensure_cluster()
        result = self._rpc("getMinimumBalanceForRentExemption", [
            int(space), {"commitment": str(commitment)},
        ])
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise VoucherError("RPC returned an invalid rent exemption")
        return result

    def fee_for_message(self, message: bytes, *, commitment: str = "confirmed") -> int:
        self._ensure_cluster()
        result = self._rpc("getFeeForMessage", [
            base64.b64encode(message).decode("ascii"), {"commitment": str(commitment)},
        ])
        value = result.get("value") if isinstance(result, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VoucherError("RPC returned an invalid transaction fee")
        return value

    def send_and_confirm(self, transaction: SignedTransaction, *, timeout: float = 60.0,
                         on_sent: Callable[[str], None] | None = None) -> str:
        self._ensure_cluster()
        try:
            returned = self._rpc("sendTransaction", [
                base64.b64encode(transaction.data).decode("ascii"),
                {"encoding": "base64", "preflightCommitment": "confirmed",
                 "skipPreflight": False, "maxRetries": 0},
            ])
        except VoucherError as exc:
            if "Solana RPC sendTransaction was refused" in str(exc):
                raise TransactionDefiniteFailure(str(exc)) from exc
            raise
        if returned != transaction.signature:
            raise VoucherError("RPC returned a different transaction signature")
        if on_sent is not None:
            on_sent(returned)
        deadline = time.monotonic() + float(timeout)
        last = None
        while time.monotonic() < deadline:
            last = self.signature_status(returned)
            if last is not None:
                if last["err"] is not None:
                    raise TransactionDefiniteFailure(
                        f"transaction failed: {last['err']!r}")
                if last["confirmationStatus"] == "finalized":
                    return returned
            time.sleep(0.15)
        raise VoucherError(
            f"transaction final state is unknown; signature {returned}; "
            f"last status={last!r}")


_ESCROW_DISC = hashlib.sha256(b"account:Escrow").digest()[:8]
_DELEGATE_DISC = hashlib.sha256(b"account:Delegate").digest()[:8]
DELEGATE_ACCOUNT_SPACE = 115
DELEGATE_ESCROW_OFFSET = 8
DELEGATE_KEY_OFFSET = 40


def decode_escrow(data: bytes) -> dict:
    raw = bytes(data)
    if len(raw) != 163 or raw[:8] != _ESCROW_DISC:
        raise ValueError("data is not an Escrow account")
    offset = 8
    owner = b58encode(raw[offset:offset + 32]); offset += 32
    vault = b58encode(raw[offset:offset + 32]); offset += 32
    committed, = struct.unpack_from("<Q", raw, offset); offset += 8
    open_channels, open_tasks = struct.unpack_from("<II", raw, offset); offset += 8
    channel_seq, task_seq, generation = struct.unpack_from("<QQQ", raw, offset); offset += 24
    tag = raw[offset]; offset += 1
    pending = None
    if tag == 1:
        amount, withdraw_at = struct.unpack_from("<Qq", raw, offset); offset += 16
        pending = {"amount": amount, "withdraw_at": withdraw_at}
    elif tag != 0:
        raise ValueError("invalid pending withdrawal option")
    rent_payer = b58encode(raw[offset:offset + 32]); offset += 32
    bump, vault_bump = raw[offset:offset + 2]
    # Anchor serializes Option::None more narrowly than Option::Some and does
    # not clear the fixed-size account tail when a pending withdrawal is
    # executed.  The discriminator, exact account size, and option tag above
    # remain authoritative; bytes after the active None layout are stale.
    return {"owner": owner, "vault": vault, "committed": committed,
            "open_channels": open_channels, "open_tasks": open_tasks,
            "channel_seq": channel_seq, "task_seq": task_seq, "generation": generation,
            "pending_withdraw": pending, "rent_payer": rent_payer,
            "bump": bump, "vault_bump": vault_bump}


def decode_delegate(data: bytes) -> dict:
    raw = bytes(data)
    if len(raw) != DELEGATE_ACCOUNT_SPACE or raw[:8] != _DELEGATE_DISC:
        raise ValueError("data is not a Delegate account")
    offset = 8
    escrow = b58encode(raw[offset:offset + 32]); offset += 32
    key = b58encode(raw[offset:offset + 32]); offset += 32
    generation, cap_total, spent = struct.unpack_from("<QQQ", raw, offset); offset += 24
    expires_at, = struct.unpack_from("<q", raw, offset); offset += 8
    flags = raw[offset]; offset += 1
    open_channels, open_tasks = struct.unpack_from("<II", raw, offset); offset += 8
    disabled = bool(raw[offset]); bump = raw[offset + 1]
    return {"escrow": escrow, "key": key, "generation": generation,
            "cap_total": cap_total, "spent": spent, "expires_at": expires_at,
            "flags": flags, "open_channels": open_channels, "open_tasks": open_tasks,
            "disabled": disabled, "bump": bump}


def token_amount(data: bytes) -> int:
    raw = bytes(data)
    if len(raw) != 165:
        raise ValueError("classic SPL token account must contain 165 bytes")
    return struct.unpack_from("<Q", raw, 64)[0]


def explorer_link(signature: str, cluster: str) -> str:
    if cluster == "devnet":
        return f"https://explorer.solana.com/tx/{signature}?cluster=devnet"
    return "local"


def configured_rpc_url(environ: Mapping[str, str]) -> str:
    value = environ.get("NAK_PAY_RPC", "").strip()
    if value:
        return value
    return environ.get("NAKSHATRA_PAY_RPC_URL", "").strip()


def configured_pubkey(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required for on-chain payment commands")
    try:
        return pubkey_text(value, name)
    except ValueError as exc:
        raise SystemExit(f"{name} is invalid: {exc}") from exc
