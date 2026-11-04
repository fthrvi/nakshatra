"""Pure legacy-Solana decoder and policy for the broker-held pay wallet.

This module is deliberately stdlib-only and self-contained.  It is intended to
be vendored byte-for-byte into the approval broker; it never opens files,
sockets, subprocesses, or reads environment variables.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final


SYSTEM_PROGRAM_ID: Final = "11111111111111111111111111111111"
TOKEN_PROGRAM_ID: Final = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM_ID: Final = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
UPGRADEABLE_LOADER_ID: Final = "BPFLoaderUpgradeab1e11111111111111111111111"
ED25519_PROGRAM_ID: Final = "Ed25519SigVerify111111111111111111111111111"
INSTRUCTIONS_SYSVAR_ID: Final = "Sysvar1nstructions1111111111111111111111111"
DEVNET_GENESIS_HASH: Final = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
NAKSHATRA_PAY_PROGRAM_ID: Final = "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo"
DEVNET_USDC_MINT: Final = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"

U64_MAX: Final = 2**64 - 1
I64_MIN: Final = -(2**63)
I64_MAX: Final = 2**63 - 1
NAK_WALLET_MAX_DELEGATE_MICRO_USDC: Final = 5_000_000
NAK_WALLET_MAX_DELEGATE_TTL_S: Final = 30 * 86400
NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC: Final = 1_000_000
NAK_WALLET_MAX_STARTER_MICRO_USDC: Final = 1_000_000
NAK_WALLET_MAX_STARTER_TTL_S: Final = 30 * 86400
NAK_WALLET_MAX_WITHDRAW_MICRO_USDC: Final = 5_000_000
NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS: Final = 10_000
NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS: Final = 1_700_000
NAK_WALLET_MAX_ATA_RENT_LAMPORTS: Final = 2_040_000
NAK_WALLET_MAX_SELF_SOL_LAMPORTS: Final = 30_000_000
NAK_WALLET_MAX_STARTER_SOL_LAMPORTS: Final = 10_000_000

DELEGATE_SET_DISCRIMINATOR: Final = bytes.fromhex("8b10e667fea614a5")
WITHDRAW_REQUEST_DISCRIMINATOR: Final = bytes.fromhex("f6f379c6f277d27a")
WITHDRAW_EXECUTE_DISCRIMINATOR: Final = bytes.fromhex("ac5dcf14503aa8f1")
DELEGATE_ACCOUNT_SPACE: Final = 115
ATA_ACCOUNT_SPACE: Final = 165
CONFIG_ACCOUNT_SPACE: Final = 208
DELEGATE_FLAGS_SELF: Final = 0x03
DELEGATE_FLAG_OPEN_CHANNEL: Final = 0x01

_PDA_MARKER = b"ProgramDerivedAddress"
_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {char: index for index, char in enumerate(_B58_ALPHABET)}
_ESCROW_DISC = hashlib.sha256(b"account:Escrow").digest()[:8]
_DELEGATE_DISC = hashlib.sha256(b"account:Delegate").digest()[:8]
_CONFIG_DISC = hashlib.sha256(b"account:Config").digest()[:8]

# A recipient or state role must never alias an executable/runtime account.
# Keep this broader than the currently allowed instruction programs so a role
# cannot become privileged if a builder adds one of these accounts later.
_PROTECTED_RUNTIME_IDS: Final = frozenset({
    SYSTEM_PROGRAM_ID, TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID,
    UPGRADEABLE_LOADER_ID, ED25519_PROGRAM_ID, INSTRUCTIONS_SYSVAR_ID,
    "ComputeBudget111111111111111111111111111111",
    "KeccakSecp256k11111111111111111111111111111",
    "Secp256r1SigVerify1111111111111111111111111",
    "AddressLookupTab1e1111111111111111111111111",
    "NativeLoader1111111111111111111111111111111",
    "BPFLoader1111111111111111111111111111111111",
    "BPFLoader2111111111111111111111111111111111",
    "LoaderV411111111111111111111111111111111111",
    "Config1111111111111111111111111111111111111",
    "Feature111111111111111111111111111111111111",
    "1nc1nerator11111111111111111111111111111111",
    "Stake11111111111111111111111111111111111111",
    "StakeConfig11111111111111111111111111111111",
    "Vote111111111111111111111111111111111111111",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
    "ZkTokenProof1111111111111111111111111111111",
    "ZkE1Gama1Proof11111111111111111111111111111",
    "Sysvar1111111111111111111111111111111111111",
    "SysvarC1ock11111111111111111111111111111111",
    "SysvarEpochRewards1111111111111111111111111",
    "SysvarEpochSchedu1e111111111111111111111111",
    "SysvarFees111111111111111111111111111111111",
    "SysvarRecentB1ockHashes11111111111111111111",
    "SysvarRent111111111111111111111111111111111",
    "SysvarRewards111111111111111111111111111111",
    "SysvarS1otHashes111111111111111111111111111",
    "SysvarS1otHistory11111111111111111111111111",
    "SysvarStakeHistory1111111111111111111111111",
    "SysvarLastRestartS1ot1111111111111111111111",
})


REFUSAL_CODES: Final = frozenset({
    "account_index", "account_order", "account_template", "ata_state",
    "blockhash", "cap_ceiling", "cluster_state", "delegate_state",
    "config_state", "duplicate_account", "escrow_state", "expiry_ceiling", "fee_ceiling",
    "fee_payer", "header", "instruction_template", "malformed_message",
    "mint", "noncanonical_shortvec", "policy", "privilege_template",
    "program_hash", "program_state", "programdata", "rent_ceiling",
    "signer_set", "sol_ceiling", "starter_cap_ceiling",
    "starter_remaining_ceiling", "trailing_bytes", "unknown_instruction",
    "unknown_program", "upgrade_authority", "versioned_message",
    "withdraw_ceiling", "withdraw_provenance", "withdraw_state", "wrong_genesis",
})


class PolicyRefusal(Exception):
    """A stable, closed-code refusal safe to expose to an outer broker."""

    def __init__(self, code: str):
        if code not in REFUSAL_CODES:
            raise ValueError("unknown policy refusal code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WalletPolicy:
    wallet_id: str
    node_id: str
    programdata_address: str
    program_sha256: str
    upgrade_authority: str
    program_id: str = NAKSHATRA_PAY_PROGRAM_ID
    mint_id: str = DEVNET_USDC_MINT
    genesis_hash: str = DEVNET_GENESIS_HASH
    max_delegate_micro_usdc: int = NAK_WALLET_MAX_DELEGATE_MICRO_USDC
    max_delegate_ttl_s: int = NAK_WALLET_MAX_DELEGATE_TTL_S
    max_starter_cap_total_micro_usdc: int = NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC
    max_starter_micro_usdc: int = NAK_WALLET_MAX_STARTER_MICRO_USDC
    max_starter_ttl_s: int = NAK_WALLET_MAX_STARTER_TTL_S
    max_withdraw_micro_usdc: int = NAK_WALLET_MAX_WITHDRAW_MICRO_USDC
    max_network_fee_lamports: int = NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS
    max_delegate_rent_lamports: int = NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS
    max_ata_rent_lamports: int = NAK_WALLET_MAX_ATA_RENT_LAMPORTS
    max_self_sol_lamports: int = NAK_WALLET_MAX_SELF_SOL_LAMPORTS
    max_starter_sol_lamports: int = NAK_WALLET_MAX_STARTER_SOL_LAMPORTS


@dataclass(frozen=True)
class ApprovedWithdrawRequest:
    """A broker-durable approval for the exact pending state on one escrow.

    ``Escrow::pending_withdraw`` exposes no sequence, so the strongest identity
    available on chain is ``(escrow_id, amount_micro_usdc, withdraw_at)``.
    Wallet import must require no pending withdrawal: a request predating import
    has no broker approval and execution is deliberately refused.

    This tuple is NOT unique on its own: two equal-amount requests landing in
    the same Unix second produce the same pending state. The broker journal
    record that supplies it MUST also retain the originating message digest,
    transaction signature, and escrow generation. Records may be created only
    from reconciled, byte-ceiling-checked withdrawal requests, and a competing
    signed attempt must prevent approval until it is reconciled. Wallet import
    must refuse a pending withdrawal because it has no broker provenance.

    Current closed templates cannot recreate an escrow, so generation is
    journal evidence rather than part of this lookup tuple. If recreation is
    ever admitted, bind escrow generation in this type and in the lookup before
    signing any execute. The broker must also prohibit competing signed attempts
    rather than treating an equal tuple as proof of origin.
    """

    escrow_id: str
    amount_micro_usdc: int
    withdraw_at: int


@dataclass(frozen=True)
class AccountView:
    address: str
    owner: str
    lamports: int
    data: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", bytes(self.data))


@dataclass(frozen=True)
class ClusterView:
    genesis_hash: str
    chain_time: int
    valid_recent_blockhashes: tuple[str, ...]
    network_fee_lamports: int
    program_executable: bool
    program_owner: str
    programdata_address: str
    programdata_owner: str
    program_sha256: str
    upgrade_authority: str
    delegate_rent_exempt_lamports: int
    ata_rent_exempt_lamports: int
    withdraw_delay_seconds: int
    accounts: tuple[AccountView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_recent_blockhashes",
                           tuple(self.valid_recent_blockhashes))
        object.__setattr__(self, "accounts", tuple(self.accounts))


@dataclass(frozen=True)
class PlanAccount:
    pubkey: str
    is_signer: bool
    is_writable: bool

    def to_dict(self) -> dict[str, object]:
        return {"pubkey": self.pubkey, "signer": self.is_signer,
                "writable": self.is_writable}


@dataclass(frozen=True)
class PlanInstruction:
    name: str
    program_id: str
    accounts: tuple[PlanAccount, ...]
    data_hex: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "program_id": self.program_id,
                "accounts": [account.to_dict() for account in self.accounts],
                "data_hex": self.data_hex}


@dataclass(frozen=True)
class PlanField:
    name: str
    value: str | int | bool

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class ApprovalPlan:
    raw_message: bytes
    message_sha256: str
    operation: str
    wallet_id: str
    recent_blockhash: str
    instructions: tuple[PlanInstruction, ...]
    rendered_fields: tuple[PlanField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_message", bytes(self.raw_message))

    def to_dict(self) -> dict[str, object]:
        return {
            "message_sha256": self.message_sha256,
            "operation": self.operation,
            "wallet_id": self.wallet_id,
            "recent_blockhash": self.recent_blockhash,
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "rendered_fields": [field.to_dict() for field in self.rendered_fields],
        }


@dataclass(frozen=True)
class _DecodedInstruction:
    program_id: str
    account_indexes: tuple[int, ...]
    data: bytes


@dataclass(frozen=True)
class _Message:
    keys: tuple[str, ...]
    signer: tuple[bool, ...]
    writable: tuple[bool, ...]
    blockhash: str
    instructions: tuple[_DecodedInstruction, ...]


@dataclass(frozen=True)
class _ExpectedInstruction:
    name: str
    program_id: str
    accounts: tuple[tuple[str, bool, bool], ...]
    data: bytes


def _refuse(code: str) -> None:
    raise PolicyRefusal(code)


def _b58decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("base58 text required")
    number = 0
    for char in value:
        number = number * 58 + _B58_INDEX[char]
    raw = (number.to_bytes((number.bit_length() + 7) // 8, "big")
           if number else b"")
    return b"\0" * (len(value) - len(value.lstrip("1"))) + raw


def _b58encode(raw: bytes) -> str:
    value = bytes(raw)
    number = int.from_bytes(value, "big")
    chars = ""
    while number:
        number, digit = divmod(number, 58)
        chars = _B58_ALPHABET[digit] + chars
    return "1" * (len(value) - len(value.lstrip(b"\0"))) + chars


def _pubkey(value: str) -> bytes:
    raw = _b58decode(value)
    if len(raw) != 32 or _b58encode(raw) != value:
        raise ValueError("non-canonical public key")
    return raw


def _is_ed25519_point(encoded: bytes) -> bool:
    if len(encoded) != 32:
        return False
    value = int.from_bytes(encoded, "little")
    y = (value & ((1 << 255) - 1)) % _P
    y2 = y * y % _P
    denominator = (_D * y2 + 1) % _P
    if denominator == 0:
        return False
    x2 = (y2 - 1) * pow(denominator, _P - 2, _P) % _P
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _I % _P
    return (x * x - x2) % _P == 0


def _create_program_address(seeds: tuple[bytes, ...], program_id: str) -> bytes:
    if len(seeds) > 16 or any(len(seed) > 32 for seed in seeds):
        raise ValueError("invalid PDA seeds")
    candidate = hashlib.sha256(
        b"".join(seeds) + _pubkey(program_id) + _PDA_MARKER).digest()
    if _is_ed25519_point(candidate):
        raise ValueError("on curve")
    return candidate


def _find_program_address_with_bump(
        seeds: tuple[bytes, ...], program_id: str) -> tuple[str, int]:
    if len(seeds) >= 16:
        raise ValueError("too many PDA seeds")
    for bump in range(255, -1, -1):
        try:
            return (_b58encode(_create_program_address(
                (*seeds, bytes([bump])), program_id)), bump)
        except ValueError as exc:
            if str(exc) != "on curve":
                raise
    raise ValueError("no PDA bump")


def _find_program_address(seeds: tuple[bytes, ...], program_id: str) -> str:
    return _find_program_address_with_bump(seeds, program_id)[0]


def _derive(policy: WalletPolicy) -> dict[str, str]:
    wallet = _pubkey(policy.wallet_id)
    program = policy.program_id
    config = _find_program_address((b"config",), program)
    escrow = _find_program_address((b"escrow", _pubkey(config), wallet), program)
    return {
        "config": config,
        "fee_vault": _find_program_address(
            (b"fee_vault", _pubkey(config)), program),
        "escrow": escrow,
        "vault": _find_program_address((b"vault", _pubkey(escrow)), program),
        "event": _find_program_address((b"__event_authority",), program),
        "ata": _find_program_address(
            (wallet, _pubkey(TOKEN_PROGRAM_ID), _pubkey(policy.mint_id)),
            ASSOCIATED_TOKEN_PROGRAM_ID),
        "programdata": _find_program_address((_pubkey(program),), UPGRADEABLE_LOADER_ID),
    }


def _delegate_pda(escrow: str, key: str, program: str) -> str:
    return _find_program_address(
        (b"delegate", _pubkey(escrow), _pubkey(key)), program)


def _shortvec(raw: bytes, offset: int) -> tuple[int, int]:
    start = offset
    value = 0
    shift = 0
    while True:
        if offset >= len(raw) or offset - start >= 3:
            _refuse("malformed_message")
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7f) << shift
        if not byte & 0x80:
            break
        shift += 7
    if value > 0xffff:
        _refuse("malformed_message")
    canonical = bytearray()
    remainder = value
    while True:
        byte = remainder & 0x7f
        remainder >>= 7
        canonical.append(byte | (0x80 if remainder else 0))
        if not remainder:
            break
    if raw[start:offset] != bytes(canonical):
        _refuse("noncanonical_shortvec")
    return value, offset


def _take(raw: bytes, offset: int, count: int) -> tuple[bytes, int]:
    end = offset + count
    if count < 0 or end > len(raw):
        _refuse("malformed_message")
    return raw[offset:end], end


def _decode_message(raw_message: bytes) -> _Message:
    try:
        raw = bytes(raw_message)
    except (TypeError, ValueError):
        _refuse("malformed_message")
    if not raw:
        _refuse("malformed_message")
    if raw[0] & 0x80:
        _refuse("versioned_message")
    if len(raw) < 3:
        _refuse("malformed_message")
    required, readonly_signed, readonly_unsigned = raw[:3]
    offset = 3
    key_count, offset = _shortvec(raw, offset)
    if not key_count or key_count > 256:
        _refuse("header")
    key_blob, offset = _take(raw, offset, key_count * 32)
    keys = tuple(_b58encode(key_blob[index:index + 32])
                 for index in range(0, len(key_blob), 32))
    if len(set(keys)) != len(keys):
        _refuse("duplicate_account")
    if (required == 0 or required > key_count or readonly_signed > required
            or readonly_unsigned > key_count - required):
        _refuse("header")
    signer = tuple(index < required for index in range(key_count))
    writable = tuple(
        (index < required - readonly_signed) if index < required
        else (index < key_count - readonly_unsigned)
        for index in range(key_count))
    blockhash_raw, offset = _take(raw, offset, 32)
    instruction_count, offset = _shortvec(raw, offset)
    instructions = []
    for _unused in range(instruction_count):
        program_raw, offset = _take(raw, offset, 1)
        account_count, offset = _shortvec(raw, offset)
        account_raw, offset = _take(raw, offset, account_count)
        data_count, offset = _shortvec(raw, offset)
        data, offset = _take(raw, offset, data_count)
        indexes = tuple(account_raw)
        if program_raw[0] >= key_count or any(index >= key_count for index in indexes):
            _refuse("account_index")
        instructions.append(_DecodedInstruction(
            keys[program_raw[0]], indexes, data))
    if offset != len(raw):
        _refuse("trailing_bytes")
    if not instructions:
        _refuse("instruction_template")
    return _Message(keys, signer, writable, _b58encode(blockhash_raw),
                    tuple(instructions))


def _account_map(view: ClusterView) -> dict[str, AccountView]:
    result: dict[str, AccountView] = {}
    for account in view.accounts:
        if not isinstance(account, AccountView):
            _refuse("cluster_state")
        try:
            _pubkey(account.address)
            _pubkey(account.owner)
        except (KeyError, TypeError, ValueError):
            _refuse("cluster_state")
        if (isinstance(account.lamports, bool) or not isinstance(account.lamports, int)
                or not 0 <= account.lamports <= U64_MAX):
            _refuse("cluster_state")
        if account.address in result:
            _refuse("cluster_state")
        result[account.address] = account
    return result


def _decode_escrow(data: bytes) -> dict[str, object]:
    raw = bytes(data)
    if len(raw) != 163 or raw[:8] != _ESCROW_DISC:
        _refuse("escrow_state")
    offset = 8
    owner = _b58encode(raw[offset:offset + 32]); offset += 32
    vault = _b58encode(raw[offset:offset + 32]); offset += 32
    committed, = struct.unpack_from("<Q", raw, offset); offset += 8
    open_channels, open_tasks = struct.unpack_from("<II", raw, offset); offset += 8
    channel_seq, task_seq, generation = struct.unpack_from("<QQQ", raw, offset); offset += 24
    tag = raw[offset]; offset += 1
    pending = None
    if tag == 1:
        amount, withdraw_at = struct.unpack_from("<Qq", raw, offset)
        offset += 16
        pending = (amount, withdraw_at)
    elif tag != 0:
        _refuse("escrow_state")
    rent_payer = _b58encode(raw[offset:offset + 32]); offset += 32
    bump, vault_bump = raw[offset:offset + 2]
    return {"owner": owner, "vault": vault, "committed": committed,
            "open_channels": open_channels, "open_tasks": open_tasks,
            "channel_seq": channel_seq, "task_seq": task_seq,
            "generation": generation, "pending": pending, "rent_payer": rent_payer,
            "bump": bump, "vault_bump": vault_bump}


def _decode_config(data: bytes) -> dict[str, object]:
    """Decode Config's fixed allocation using its variable Borsh prefix."""
    raw = bytes(data)
    if len(raw) != CONFIG_ACCOUNT_SPACE or raw[:8] != _CONFIG_DISC:
        _refuse("config_state")
    offset = 8
    authority = _b58encode(raw[offset:offset + 32]); offset += 32
    mint = _b58encode(raw[offset:offset + 32]); offset += 32
    fee_wallet = _b58encode(raw[offset:offset + 32]); offset += 32
    fee_bps, withdraw_delay_s, close_grace_s, config_delay_s = struct.unpack_from(
        "<Hqqq", raw, offset)
    offset += 26

    pending = None
    tag = raw[offset]; offset += 1
    if tag == 1:
        if offset + 42 > len(raw):
            _refuse("config_state")
        pending_fee_bps, = struct.unpack_from("<H", raw, offset); offset += 2
        pending_fee_wallet = _b58encode(raw[offset:offset + 32]); offset += 32
        effective_at, = struct.unpack_from("<q", raw, offset); offset += 8
        pending = (pending_fee_bps, pending_fee_wallet, effective_at)
    elif tag != 0:
        _refuse("config_state")

    pending_authority = None
    if offset >= len(raw):
        _refuse("config_state")
    tag = raw[offset]; offset += 1
    if tag == 1:
        if offset + 32 > len(raw):
            _refuse("config_state")
        pending_authority = _b58encode(raw[offset:offset + 32]); offset += 32
    elif tag != 0:
        _refuse("config_state")
    if offset + 2 > len(raw):
        _refuse("config_state")
    bump, fee_vault_bump = raw[offset:offset + 2]
    return {
        "authority": authority, "mint": mint, "fee_wallet": fee_wallet,
        "fee_bps": fee_bps, "withdraw_delay_s": withdraw_delay_s,
        "close_grace_s": close_grace_s, "config_delay_s": config_delay_s,
        "pending": pending, "pending_authority": pending_authority,
        "bump": bump, "fee_vault_bump": fee_vault_bump,
    }


def _decode_delegate(data: bytes) -> dict[str, object]:
    raw = bytes(data)
    if len(raw) != DELEGATE_ACCOUNT_SPACE or raw[:8] != _DELEGATE_DISC:
        _refuse("delegate_state")
    offset = 8
    escrow = _b58encode(raw[offset:offset + 32]); offset += 32
    key = _b58encode(raw[offset:offset + 32]); offset += 32
    generation, cap_total, spent = struct.unpack_from("<QQQ", raw, offset); offset += 24
    expires_at, = struct.unpack_from("<q", raw, offset); offset += 8
    flags = raw[offset]; offset += 1
    open_channels, open_tasks = struct.unpack_from("<II", raw, offset); offset += 8
    disabled = raw[offset]
    if disabled not in (0, 1):
        _refuse("delegate_state")
    bump = raw[offset + 1]
    return {"escrow": escrow, "key": key, "generation": generation,
            "cap_total": cap_total, "spent": spent, "expires_at": expires_at,
            "flags": flags, "open_channels": open_channels,
            "open_tasks": open_tasks, "disabled": bool(disabled), "bump": bump}


def _decode_spl_token_account(data: bytes) -> dict[str, object]:
    """Mirror classic SPL Token ``Account::unpack`` for fields we consume."""
    raw = bytes(data)
    if len(raw) != ATA_ACCOUNT_SPACE:
        _refuse("ata_state")
    # Classic SPL Account layout has three COption tags. Pack::unpack accepts
    # only 0/1 and rejects an uninitialized or unknown account-state byte.
    if any(struct.unpack_from("<I", raw, offset)[0] not in (0, 1)
           for offset in (72, 109, 129)) or raw[108] not in (1, 2):
        _refuse("ata_state")
    return {"mint": _b58encode(raw[:32]), "owner": _b58encode(raw[32:64])}


def _validate_policy(policy: WalletPolicy, view: ClusterView) -> dict[str, str]:
    try:
        for key in (policy.wallet_id, policy.node_id, policy.program_id, policy.mint_id,
                    policy.programdata_address, policy.upgrade_authority):
            _pubkey(key)
        if (len(policy.program_sha256) != 64
                or any(char not in "0123456789abcdef" for char in policy.program_sha256)):
            raise ValueError("invalid hash")
        ceilings = (
            (policy.max_delegate_micro_usdc, NAK_WALLET_MAX_DELEGATE_MICRO_USDC),
            (policy.max_delegate_ttl_s, NAK_WALLET_MAX_DELEGATE_TTL_S),
            (policy.max_starter_cap_total_micro_usdc,
             NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC),
            (policy.max_starter_micro_usdc, NAK_WALLET_MAX_STARTER_MICRO_USDC),
            (policy.max_starter_ttl_s, NAK_WALLET_MAX_STARTER_TTL_S),
            (policy.max_withdraw_micro_usdc, NAK_WALLET_MAX_WITHDRAW_MICRO_USDC),
            (policy.max_network_fee_lamports, NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS),
            (policy.max_delegate_rent_lamports, NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS),
            (policy.max_ata_rent_lamports, NAK_WALLET_MAX_ATA_RENT_LAMPORTS),
            (policy.max_self_sol_lamports, NAK_WALLET_MAX_SELF_SOL_LAMPORTS),
            (policy.max_starter_sol_lamports, NAK_WALLET_MAX_STARTER_SOL_LAMPORTS),
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 or value > hard_max or value > U64_MAX
               for value, hard_max in ceilings):
            raise ValueError("invalid ceiling")
    except (KeyError, TypeError, ValueError):
        _refuse("policy")
    if policy.program_id != NAKSHATRA_PAY_PROGRAM_ID:
        _refuse("policy")
    if policy.mint_id != DEVNET_USDC_MINT:
        _refuse("mint")
    try:
        derived = _derive(policy)
    except (KeyError, TypeError, ValueError):
        _refuse("policy")
    if policy.genesis_hash != DEVNET_GENESIS_HASH or view.genesis_hash != policy.genesis_hash:
        _refuse("wrong_genesis")
    if derived["programdata"] != policy.programdata_address:
        _refuse("programdata")
    if (not view.program_executable or view.program_owner != UPGRADEABLE_LOADER_ID
            or view.programdata_owner != UPGRADEABLE_LOADER_ID):
        _refuse("program_state")
    if view.programdata_address != policy.programdata_address:
        _refuse("programdata")
    if view.program_sha256 != policy.program_sha256:
        _refuse("program_hash")
    if view.upgrade_authority != policy.upgrade_authority:
        _refuse("upgrade_authority")
    if (isinstance(view.chain_time, bool) or not isinstance(view.chain_time, int)
            or not I64_MIN <= view.chain_time <= I64_MAX):
        _refuse("cluster_state")
    for value in (view.network_fee_lamports, view.delegate_rent_exempt_lamports,
                  view.ata_rent_exempt_lamports):
        if (isinstance(value, bool) or not isinstance(value, int)
                or not 0 <= value <= U64_MAX):
            _refuse("cluster_state")
    if (isinstance(view.withdraw_delay_seconds, bool)
            or not isinstance(view.withdraw_delay_seconds, int)
            or not I64_MIN <= view.withdraw_delay_seconds <= I64_MAX):
        _refuse("cluster_state")
    if view.network_fee_lamports > policy.max_network_fee_lamports:
        _refuse("fee_ceiling")
    return derived


def _existing_config(policy: WalletPolicy, view: ClusterView,
                     derived: dict[str, str],
                     accounts: dict[str, AccountView]) -> dict[str, object]:
    account = accounts.get(derived["config"])
    if account is None or account.owner != policy.program_id:
        _refuse("config_state")
    config = _decode_config(account.data)
    _address, config_bump = _find_program_address_with_bump(
        (b"config",), policy.program_id)
    _fee_vault, fee_vault_bump = _find_program_address_with_bump(
        (b"fee_vault", _pubkey(derived["config"])), policy.program_id)
    if (config["bump"] != config_bump
            or config["fee_vault_bump"] != fee_vault_bump):
        _refuse("config_state")
    if config["mint"] != DEVNET_USDC_MINT or config["mint"] != policy.mint_id:
        _refuse("mint")
    if config["withdraw_delay_s"] != view.withdraw_delay_seconds:
        _refuse("config_state")
    return config


def _existing_escrow(policy: WalletPolicy, view: ClusterView,
                     derived: dict[str, str], accounts: dict[str, AccountView]) -> dict[str, object]:
    account = accounts.get(derived["escrow"])
    if account is None or account.owner != policy.program_id:
        _refuse("escrow_state")
    escrow = _decode_escrow(account.data)
    if escrow["owner"] != policy.wallet_id or escrow["vault"] != derived["vault"]:
        _refuse("escrow_state")
    config = derived["config"]
    _escrow_address, escrow_bump = _find_program_address_with_bump(
        (b"escrow", _pubkey(config), _pubkey(policy.wallet_id)), policy.program_id)
    _vault_address, vault_bump = _find_program_address_with_bump(
        (b"vault", _pubkey(derived["escrow"])), policy.program_id)
    if escrow["bump"] != escrow_bump or escrow["vault_bump"] != vault_bump:
        _refuse("escrow_state")
    return escrow


def _creation_state(address: str, expected_owner: str, rent_minimum: int,
                    rent_ceiling: int, accounts: dict[str, AccountView],
                    kind: str) -> tuple[int, int, dict[str, object] | None]:
    account = accounts.get(address)
    state = None
    actual_shortfall = rent_minimum
    if account is None:
        pass
    elif account.owner == expected_owner:
        if kind == "delegate":
            state = _decode_delegate(account.data)
        else:
            state = _decode_spl_token_account(account.data)
        actual_shortfall = 0
    elif account.owner == SYSTEM_PROGRAM_ID and account.data == b"":
        actual_shortfall = max(0, rent_minimum - account.lamports)
    else:
        _refuse("delegate_state" if kind == "delegate" else "ata_state")
    if account is not None and (isinstance(account.lamports, bool)
                                or not isinstance(account.lamports, int)
                                or account.lamports < 0):
        _refuse("cluster_state")
    # Both accepted instructions can create the account. Existing state and
    # PDA dust can disappear before the signed message lands, so show today's
    # actual debit separately while bounding the full worst-case minimum.
    worst_case_rent = rent_minimum
    if worst_case_rent > rent_ceiling:
        _refuse("rent_ceiling")
    return actual_shortfall, worst_case_rent, state


def _expected_instructions(message: _Message, policy: WalletPolicy,
                           derived: dict[str, str]) -> tuple[str, tuple[_ExpectedInstruction, ...], dict[str, object]]:
    decoded: list[tuple[str, dict[str, object], bytes]] = []
    for instruction in message.instructions:
        data = instruction.data
        if instruction.program_id == policy.program_id:
            if data.startswith(DELEGATE_SET_DISCRIMINATOR):
                if len(data) != 57:
                    _refuse("unknown_instruction")
                decoded.append(("delegate_set", {
                    "delegate_key": _b58encode(data[8:40]),
                    "cap_total": struct.unpack_from("<Q", data, 40)[0],
                    "expires_at": struct.unpack_from("<q", data, 48)[0],
                    "flags": data[56],
                }, data))
            elif data.startswith(WITHDRAW_REQUEST_DISCRIMINATOR):
                if len(data) != 16:
                    _refuse("unknown_instruction")
                decoded.append(("withdraw_request", {
                    "amount": struct.unpack_from("<Q", data, 8)[0]}, data))
            elif data == WITHDRAW_EXECUTE_DISCRIMINATOR:
                decoded.append(("withdraw_execute", {}, data))
            else:
                _refuse("unknown_instruction")
        elif instruction.program_id == SYSTEM_PROGRAM_ID:
            if len(data) != 12 or data[:4] != struct.pack("<I", 2):
                _refuse("unknown_instruction")
            decoded.append(("system_transfer", {
                "lamports": struct.unpack_from("<Q", data, 4)[0]}, data))
        elif instruction.program_id == ASSOCIATED_TOKEN_PROGRAM_ID:
            if data != b"\x01":
                _refuse("unknown_instruction")
            decoded.append(("ata_create_idempotent", {}, data))
        else:
            _refuse("unknown_program")

    names = tuple(item[0] for item in decoded)
    delegate_item = next((item for item in decoded if item[0] == "delegate_set"), None)
    key = str(delegate_item[1]["delegate_key"]) if delegate_item else ""
    if names == ("withdraw_request",):
        operation = "withdraw_request"
    elif names == ("ata_create_idempotent", "withdraw_execute"):
        operation = "withdraw_execute"
    elif delegate_item is not None:
        operation = "self_delegate" if key == policy.node_id else "starter_grant"
    else:
        operation = "self_top_up"

    allowed_by_operation = {
        "self_delegate": {
            ("delegate_set",), ("system_transfer", "delegate_set"),
        },
        "starter_grant": {
            ("delegate_set",), ("delegate_set", "system_transfer"),
        },
        "self_top_up": {("system_transfer",)},
        "withdraw_request": {("withdraw_request",)},
        "withdraw_execute": {("ata_create_idempotent", "withdraw_execute")},
    }
    if names not in allowed_by_operation[operation]:
        _refuse("instruction_template")

    if operation == "self_top_up":
        destination = policy.node_id
    elif delegate_item is not None:
        destination = key
    else:
        destination = ""
    delegate = (_delegate_pda(derived["escrow"], key, policy.program_id)
                if delegate_item is not None else "")

    expected = []
    for name, values, data in decoded:
        if name == "delegate_set":
            specs = (
                (policy.wallet_id, True, True), (derived["config"], False, False),
                (derived["escrow"], False, False), (delegate, False, True),
                (SYSTEM_PROGRAM_ID, False, False), (derived["event"], False, False),
                (policy.program_id, False, False),
            )
        elif name == "system_transfer":
            specs = ((policy.wallet_id, True, True), (destination, False, True))
        elif name == "withdraw_request":
            specs = (
                (policy.wallet_id, True, False), (derived["config"], False, False),
                (derived["escrow"], False, True), (derived["vault"], False, False),
                (derived["event"], False, False), (policy.program_id, False, False),
            )
        elif name == "ata_create_idempotent":
            specs = (
                (policy.wallet_id, True, True), (derived["ata"], False, True),
                (policy.wallet_id, False, False), (policy.mint_id, False, False),
                (SYSTEM_PROGRAM_ID, False, False), (TOKEN_PROGRAM_ID, False, False),
            )
        else:
            specs = (
                (policy.wallet_id, True, False), (derived["config"], False, False),
                (policy.mint_id, False, False), (derived["escrow"], False, True),
                (derived["vault"], False, True), (derived["ata"], False, True),
                (TOKEN_PROGRAM_ID, False, False), (derived["event"], False, False),
                (policy.program_id, False, False),
            )
        expected.append(_ExpectedInstruction(name,
                        (SYSTEM_PROGRAM_ID if name == "system_transfer" else
                         ASSOCIATED_TOKEN_PROGRAM_ID if name == "ata_create_idempotent" else
                         policy.program_id), specs, data))
    details: dict[str, object] = {"decoded": tuple(decoded), "delegate": delegate,
                                  "delegate_key": key, "destination": destination}
    return operation, tuple(expected), details


def _validate_role_separation(roles: dict[str, str]) -> None:
    """Reject semantic-role aliases before Solana privilege aggregation."""
    protected = _PROTECTED_RUNTIME_IDS | {NAKSHATRA_PAY_PROGRAM_ID}
    if any(address in protected for address in roles.values()):
        _refuse("account_template")
    addresses = list(roles.values())
    if len(set(addresses)) != len(addresses):
        _refuse("account_template")


def _validate_compilation(message: _Message, policy: WalletPolicy,
                          expected: tuple[_ExpectedInstruction, ...]) -> None:
    if message.keys[0] != policy.wallet_id:
        _refuse("fee_payer")
    signers = tuple(key for key, signed in zip(message.keys, message.signer) if signed)
    if signers != (policy.wallet_id,):
        _refuse("signer_set")
    metas: dict[str, list[bool]] = {policy.wallet_id: [True, True]}
    for instruction in expected:
        for key, signer, writable in instruction.accounts:
            current = metas.setdefault(key, [False, False])
            current[0] = current[0] or signer
            current[1] = current[1] or writable
        metas.setdefault(instruction.program_id, [False, False])

    def category(key: str) -> tuple[int, bytes]:
        signer, writable = metas[key]
        return (0 if signer and writable else 1 if signer else
                2 if writable else 3), _pubkey(key)

    ordered = (policy.wallet_id,) + tuple(sorted(
        (key for key in metas if key != policy.wallet_id), key=category))
    if message.keys != ordered:
        _refuse("account_order")
    for index, key in enumerate(message.keys):
        if (message.signer[index], message.writable[index]) != tuple(metas[key]):
            _refuse("privilege_template")
    if len(expected) != len(message.instructions):
        _refuse("instruction_template")
    for actual, wanted in zip(message.instructions, expected):
        if actual.program_id != wanted.program_id or actual.data != wanted.data:
            _refuse("instruction_template")
        actual_keys = tuple(message.keys[index] for index in actual.account_indexes)
        wanted_keys = tuple(item[0] for item in wanted.accounts)
        if actual_keys != wanted_keys:
            _refuse("account_template")


def _money(value: int) -> str:
    return f"{value // 1_000_000}.{value % 1_000_000:06d}"


def _sol(value: int) -> str:
    return f"{value // 1_000_000_000}.{value % 1_000_000_000:09d}"


def decode_and_check(
        raw_message: bytes, policy: WalletPolicy, cluster_view: ClusterView,
        approved_withdraw_requests: frozenset[ApprovedWithdrawRequest] = frozenset(),
) -> ApprovalPlan:
    """Decode one exact legacy message and return its immutable approval plan.

    All expected policy failures raise :class:`PolicyRefusal` with a code from
    :data:`REFUSAL_CODES`.  The returned ``raw_message`` is the original bytes,
    never a reconstruction.  Withdraw execution additionally requires an exact
    match in ``approved_withdraw_requests``, supplied from the broker's durable
    store.  Wallet import must require no pending withdrawal; an older pending
    request cannot acquire provenance merely by appearing in an RPC view.
    """
    if not isinstance(policy, WalletPolicy) or not isinstance(cluster_view, ClusterView):
        _refuse("policy")
    message = _decode_message(raw_message)
    derived = _validate_policy(policy, cluster_view)
    if message.blockhash not in cluster_view.valid_recent_blockhashes:
        _refuse("blockhash")
    accounts = _account_map(cluster_view)
    _existing_config(policy, cluster_view, derived, accounts)
    escrow = _existing_escrow(policy, cluster_view, derived, accounts)
    operation, expected, details = _expected_instructions(message, policy, derived)
    roles = {
        "wallet": policy.wallet_id,
        "node": policy.node_id,
        "config": derived["config"],
        "escrow": derived["escrow"],
        "vault": derived["vault"],
        "ata": derived["ata"],
        "mint": policy.mint_id,
        "fee_vault": derived["fee_vault"],
        "event_authority": derived["event"],
    }
    if details["delegate"]:
        roles["delegate_pda"] = str(details["delegate"])
    if operation == "starter_grant":
        roles["recipient"] = str(details["delegate_key"])
    _validate_role_separation(roles)
    _validate_compilation(message, policy, expected)

    fields: list[PlanField] = [
        PlanField("wallet", policy.wallet_id), PlanField("node", policy.node_id),
        PlanField("escrow", derived["escrow"]),
        PlanField("network_fee_lamports", cluster_view.network_fee_lamports),
        PlanField("network_fee_sol", _sol(cluster_view.network_fee_lamports)),
        PlanField("network_fee_ceiling_lamports", policy.max_network_fee_lamports),
    ]
    decoded = details["decoded"]
    assert isinstance(decoded, tuple)
    by_name = {name: values for name, values, _data in decoded}

    if operation in {"self_delegate", "starter_grant"}:
        values = by_name["delegate_set"]
        cap = int(values["cap_total"])
        expiry = int(values["expires_at"])
        flags = int(values["flags"])
        key = str(values["delegate_key"])
        is_self = operation == "self_delegate"
        if is_self:
            if key != policy.node_id or flags != DELEGATE_FLAGS_SELF:
                _refuse("instruction_template")
            if cap > policy.max_delegate_micro_usdc:
                _refuse("cap_ceiling")
            ttl = policy.max_delegate_ttl_s
        else:
            if key == policy.node_id or flags != DELEGATE_FLAG_OPEN_CHANNEL:
                _refuse("instruction_template")
            if cap > policy.max_starter_cap_total_micro_usdc:
                _refuse("starter_cap_ceiling")
            # F1 is byte-only: RPC ``spent`` may never make an encoded cap
            # acceptable under an independently tightened remaining ceiling.
            if cap > policy.max_starter_micro_usdc:
                _refuse("starter_remaining_ceiling")
            ttl = policy.max_starter_ttl_s
        if expiry <= cluster_view.chain_time or expiry - cluster_view.chain_time > ttl:
            _refuse("expiry_ceiling")
        delegate = str(details["delegate"])
        rent_shortfall, rent_worst_case, state = _creation_state(
            delegate, policy.program_id, cluster_view.delegate_rent_exempt_lamports,
            policy.max_delegate_rent_lamports, accounts, "delegate")
        remaining = cap
        pre_spent = 0
        pre_cap = 0
        older_generation_reset = False
        if state is not None:
            if state["escrow"] != derived["escrow"] or state["key"] != key:
                _refuse("delegate_state")
            _delegate_address, delegate_bump = _find_program_address_with_bump(
                (b"delegate", _pubkey(derived["escrow"]), _pubkey(key)),
                policy.program_id)
            if state["bump"] != delegate_bump:
                _refuse("delegate_state")
            generation = int(state["generation"])
            escrow_generation = int(escrow["generation"])
            pre_spent = int(state["spent"])
            pre_cap = int(state["cap_total"])
            if (int(state["flags"]) & ~DELEGATE_FLAGS_SELF
                    or pre_spent > pre_cap or generation > escrow_generation):
                _refuse("delegate_state")
            # Mirrors nakshatra_pay instructions.rs:641-688: same-generation
            # state must be enabled and preserve spent; an older generation
            # resets spent only after proving it has no open work. The encoded
            # cap must cover the effective spent in either transition.
            if generation == escrow_generation:
                if bool(state["disabled"]) or cap < pre_spent:
                    _refuse("delegate_state")
                remaining = cap - pre_spent
            else:
                if int(state["open_channels"]) != 0 or int(state["open_tasks"]) != 0:
                    _refuse("delegate_state")
                older_generation_reset = True
        if not is_self and remaining > policy.max_starter_micro_usdc:
            _refuse("starter_remaining_ceiling")
        fields.extend((
            PlanField("delegate", delegate), PlanField("delegate_key", key),
            PlanField("cap_total_micro_usdc", cap),
            PlanField("cap_total_usdc", _money(cap)), PlanField("expires_at", expiry),
            PlanField("flags", flags),
            PlanField("delegate_rent_shortfall_lamports", rent_shortfall),
            PlanField("delegate_rent_worst_case_lamports", rent_worst_case),
            PlanField("delegate_rent_ceiling_lamports", policy.max_delegate_rent_lamports),
        ))
        fields.extend((PlanField("pre_cap_total_micro_usdc", pre_cap),
                       PlanField("pre_spent_micro_usdc", pre_spent),
                       PlanField("older_generation_reset", older_generation_reset)))
        if is_self:
            fields.append(PlanField("cap_total_ceiling_micro_usdc",
                                    policy.max_delegate_micro_usdc))
        else:
            fields.extend((PlanField("resulting_remaining_micro_usdc", remaining),
                           PlanField("starter_cap_total_ceiling_micro_usdc",
                                     policy.max_starter_cap_total_micro_usdc),
                           PlanField("starter_remaining_ceiling_micro_usdc",
                                     policy.max_starter_micro_usdc)))

    transfer = by_name.get("system_transfer")
    if transfer is not None:
        lamports = int(transfer["lamports"])
        ceiling = (policy.max_self_sol_lamports if operation != "starter_grant"
                   else policy.max_starter_sol_lamports)
        if lamports > ceiling:
            _refuse("sol_ceiling")
        fields.extend((PlanField("sol_destination", str(details["destination"])),
                       PlanField("sol_lamports", lamports),
                       PlanField("sol_amount", _sol(lamports)),
                       PlanField("sol_ceiling_lamports", ceiling)))

    if operation == "withdraw_request":
        amount = int(by_name["withdraw_request"]["amount"])
        if amount > policy.max_withdraw_micro_usdc:
            _refuse("withdraw_ceiling")
        if escrow["pending"] is not None:
            _refuse("withdraw_state")
        earliest_execute_time = (cluster_view.chain_time
                                 + cluster_view.withdraw_delay_seconds)
        if not I64_MIN <= earliest_execute_time <= I64_MAX:
            _refuse("cluster_state")
        fields.extend((PlanField("vault", derived["vault"]),
                       PlanField("amount_micro_usdc", amount),
                       PlanField("amount_usdc", _money(amount)),
                       PlanField("withdraw_ceiling_micro_usdc",
                                 policy.max_withdraw_micro_usdc),
                       PlanField("earliest_execute_time", earliest_execute_time)))

    if operation == "withdraw_execute":
        pending = escrow["pending"]
        if pending is None:
            _refuse("withdraw_state")
        amount, withdraw_at = pending
        if int(amount) > policy.max_withdraw_micro_usdc:
            _refuse("withdraw_ceiling")
        if not isinstance(approved_withdraw_requests, frozenset):
            _refuse("withdraw_provenance")
        for approved in approved_withdraw_requests:
            if not isinstance(approved, ApprovedWithdrawRequest):
                _refuse("withdraw_provenance")
            try:
                _pubkey(approved.escrow_id)
            except (KeyError, TypeError, ValueError):
                _refuse("withdraw_provenance")
            if (isinstance(approved.amount_micro_usdc, bool)
                    or not isinstance(approved.amount_micro_usdc, int)
                    or not 0 <= approved.amount_micro_usdc <= U64_MAX
                    or isinstance(approved.withdraw_at, bool)
                    or not isinstance(approved.withdraw_at, int)
                    or not I64_MIN <= approved.withdraw_at <= I64_MAX):
                _refuse("withdraw_provenance")
        approved_pending = ApprovedWithdrawRequest(
            derived["escrow"], int(amount), int(withdraw_at))
        if approved_pending not in approved_withdraw_requests:
            _refuse("withdraw_provenance")
        if cluster_view.chain_time < int(withdraw_at):
            _refuse("withdraw_state")
        rent_shortfall, rent_worst_case, state = _creation_state(
            derived["ata"], TOKEN_PROGRAM_ID, cluster_view.ata_rent_exempt_lamports,
            policy.max_ata_rent_lamports,
            accounts, "ata")
        if state is not None and (state["mint"] != policy.mint_id
                                  or state["owner"] != policy.wallet_id):
            _refuse("ata_state")
        fields.extend((PlanField("mint", policy.mint_id),
                       PlanField("vault", derived["vault"]),
                       PlanField("destination_ata", derived["ata"]),
                       PlanField("pending_amount_micro_usdc", int(amount)),
                       PlanField("pending_amount_usdc", _money(int(amount))),
                       PlanField("withdraw_ceiling_micro_usdc",
                                 policy.max_withdraw_micro_usdc),
                       PlanField("withdraw_at", int(withdraw_at)),
                       PlanField("ata_rent_shortfall_lamports", rent_shortfall),
                       PlanField("ata_rent_worst_case_lamports", rent_worst_case),
                       PlanField("ata_rent_ceiling_lamports", policy.max_ata_rent_lamports)))

    plan_instructions = tuple(PlanInstruction(
        wanted.name, wanted.program_id,
        tuple(PlanAccount(message.keys[index], message.signer[index],
                          message.writable[index])
              for index in actual.account_indexes),
        actual.data.hex())
        for actual, wanted in zip(message.instructions, expected))
    return ApprovalPlan(bytes(raw_message), hashlib.sha256(bytes(raw_message)).hexdigest(),
                        operation, policy.wallet_id, message.blockhash,
                        plan_instructions, tuple(fields))
