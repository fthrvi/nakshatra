#!/usr/bin/env python3
"""Regenerate frozen pay-wallet-policy fixtures from the real builders.

All keys are deterministic throwaway test keys.  No key is read from HOME and
no private seed is written to a fixture.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import asdict, replace
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak, nakd, pay_autochannel, pay_chain  # noqa: E402
from network import pay_wallet_policy as wallet_policy  # noqa: E402


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "pay_wallet_policy"
PROGRAM = wallet_policy.NAKSHATRA_PAY_PROGRAM_ID
MINT = wallet_policy.DEVNET_USDC_MINT
LOADER = wallet_policy.UPGRADEABLE_LOADER_ID
BLOCKHASH = pay_chain.b58encode(bytes(range(32, 64)))
STALE_BLOCKHASH = pay_chain.b58encode(bytes(range(31, 63)))
MAINNET_GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
CHAIN_TIME = 2_000_000_000
SELF_CAP = pay_autochannel.DEFAULT_LIMITS["cap_total"]
SELF_DAYS = pay_autochannel.DEFAULT_LIMITS["days"]
SELF_TOP_UP = nak.NODE_SOL_TARGET
STARTER_DAYS = nakd.STARTER_DELEGATE_DAYS
STARTER_TOP_UP = nakd.STARTER_NODE_SOL


def _address(seed_byte: int) -> str:
    private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes([seed_byte]) * 32)
    return pay_chain.b58encode(private.public_key().public_bytes_raw())


WALLET = _address(11)
NODE = _address(12)
STARTER = _address(13)
AUTHORITY = _address(14)
UNKNOWN_PROGRAM = _address(15)
WRONG_DESTINATION = _address(16)
PROGRAMDATA = pay_chain.find_program_address(
    [pay_chain.pubkey_bytes(PROGRAM)], LOADER)[0]
PROGRAM_HASH = hashlib.sha256(b"throwaway reviewed nakshatra_pay program").hexdigest()


def _escrow_data(*, generation: int = 7, pending: tuple[int, int] | None = None) -> bytes:
    derived = pay_chain._derived(WALLET, PROGRAM)
    escrow_bump = pay_chain.escrow_pda(WALLET, PROGRAM)[1]
    vault_bump = pay_chain.vault_pda(derived["escrow"], PROGRAM)[1]
    raw = bytearray(163)
    raw[:8] = hashlib.sha256(b"account:Escrow").digest()[:8]
    offset = 8
    raw[offset:offset + 32] = pay_chain.pubkey_bytes(WALLET); offset += 32
    raw[offset:offset + 32] = pay_chain.pubkey_bytes(derived["vault"]); offset += 32
    struct.pack_into("<QIIQQQ", raw, offset, 0, 0, 0, 3, 4, generation)
    offset += 40
    if pending is None:
        raw[offset] = 0; offset += 1
    else:
        raw[offset] = 1; offset += 1
        struct.pack_into("<Qq", raw, offset, *pending); offset += 16
    raw[offset:offset + 32] = pay_chain.pubkey_bytes(WALLET); offset += 32
    raw[offset:offset + 2] = bytes((escrow_bump, vault_bump))
    return bytes(raw)


def _delegate_data(key: str, *, generation: int = 7, cap: int = 500_000,
                   spent: int = 0, expires: int = CHAIN_TIME + 1000,
                   flags: int = 1, open_channels: int = 0,
                   open_tasks: int = 0, disabled: bool = False) -> bytes:
    escrow = pay_chain.escrow_pda(WALLET, PROGRAM)[0]
    delegate_bump = pay_chain.delegate_pda(escrow, key, PROGRAM)[1]
    raw = bytearray(pay_chain.DELEGATE_ACCOUNT_SPACE)
    raw[:8] = hashlib.sha256(b"account:Delegate").digest()[:8]
    offset = 8
    raw[offset:offset + 32] = pay_chain.pubkey_bytes(escrow); offset += 32
    raw[offset:offset + 32] = pay_chain.pubkey_bytes(key); offset += 32
    struct.pack_into("<QQQqBII", raw, offset, generation, cap, spent, expires,
                     flags, open_channels, open_tasks)
    offset += 41
    raw[offset] = int(disabled)
    raw[offset + 1] = delegate_bump
    return bytes(raw)


def _ata_data(*, mint: str = MINT, owner: str = WALLET, initialized: int = 1) -> bytes:
    raw = bytearray(165)
    raw[:32] = pay_chain.pubkey_bytes(mint)
    raw[32:64] = pay_chain.pubkey_bytes(owner)
    raw[108] = initialized
    return bytes(raw)


def base_policy() -> wallet_policy.WalletPolicy:
    return wallet_policy.WalletPolicy(
        wallet_id=WALLET, node_id=NODE, programdata_address=PROGRAMDATA,
        program_sha256=PROGRAM_HASH, upgrade_authority=AUTHORITY)


def base_view(*, pending: tuple[int, int] | None = None,
              extra_accounts: tuple[wallet_policy.AccountView, ...] = ()) -> wallet_policy.ClusterView:
    escrow = pay_chain.escrow_pda(WALLET, PROGRAM)[0]
    return wallet_policy.ClusterView(
        genesis_hash=wallet_policy.DEVNET_GENESIS_HASH, chain_time=CHAIN_TIME,
        valid_recent_blockhashes=(BLOCKHASH,), network_fee_lamports=5000,
        program_executable=True, program_owner=LOADER,
        programdata_address=PROGRAMDATA, programdata_owner=LOADER,
        program_sha256=PROGRAM_HASH, upgrade_authority=AUTHORITY,
        delegate_rent_exempt_lamports=1_691_280,
        ata_rent_exempt_lamports=2_039_280, withdraw_delay_seconds=86_400,
        accounts=(wallet_policy.AccountView(
            escrow, PROGRAM, 1_691_280, _escrow_data(pending=pending)), *extra_accounts))


def _compile(instructions: list[pay_chain.Instruction]) -> bytes:
    return pay_chain.compile_message(instructions, WALLET, BLOCKHASH).data


def golden_messages() -> dict[str, tuple[bytes, wallet_policy.ClusterView]]:
    # Exact constructor calls/order from nak._pay_ensure_locked,
    # nakd.NakDaemon._chain_starter, and nak's withdraw branch.  The self cap
    # comes from pay_autochannel.DEFAULT_LIMITS, as the callers do.
    self_set = pay_chain.delegate_set(
        WALLET, NODE, SELF_CAP, CHAIN_TIME + SELF_DAYS * 86400,
        pay_chain.DELEGATE_FLAGS_ALL, PROGRAM)
    starter_set = pay_chain.delegate_set(
        WALLET, STARTER, 1_000_000, CHAIN_TIME + STARTER_DAYS * 86400,
        pay_chain.DELEGATE_OPEN_CHANNEL, PROGRAM)
    return {
        "self_delegate": (_compile(self_set), base_view()),
        "self_transfer_delegate": (_compile([
            pay_chain.system_transfer(WALLET, NODE, SELF_TOP_UP), *self_set]), base_view()),
        "self_top_up": (_compile([
            pay_chain.system_transfer(WALLET, NODE, SELF_TOP_UP)]), base_view()),
        "starter_delegate": (_compile(starter_set), base_view()),
        "starter_top_up": (_compile([
            *starter_set, pay_chain.system_transfer(WALLET, STARTER, STARTER_TOP_UP)]), base_view()),
        "withdraw_request": (_compile(
            pay_chain.escrow_withdraw_request(WALLET, 2_500_000, PROGRAM)), base_view()),
        "withdraw_execute": (_compile(
            pay_chain.escrow_withdraw_execute(WALLET, MINT, PROGRAM)),
            base_view(pending=(2_500_000, CHAIN_TIME - 1))),
    }


def _serialize_policy(policy: wallet_policy.WalletPolicy) -> dict[str, object]:
    return asdict(policy)


def _serialize_view(view: wallet_policy.ClusterView) -> dict[str, object]:
    result = asdict(view)
    result["valid_recent_blockhashes"] = list(view.valid_recent_blockhashes)
    result["accounts"] = [
        {"address": account.address, "owner": account.owner,
         "lamports": account.lamports, "data_hex": account.data.hex()}
        for account in view.accounts]
    return result


def golden_fixtures() -> dict[str, dict[str, object]]:
    policy = base_policy()
    result = {}
    for name, (message, view) in golden_messages().items():
        result[name] = {
            "name": name, "raw_message_hex": message.hex(),
            "policy": _serialize_policy(policy), "cluster_view": _serialize_view(view),
            "expected_plan": wallet_policy.decode_and_check(message, policy, view).to_dict(),
        }
    return result


def _replace_account(view: wallet_policy.ClusterView, account: wallet_policy.AccountView
                     ) -> wallet_policy.ClusterView:
    return replace(view, accounts=tuple(
        item for item in view.accounts if item.address != account.address) + (account,))


def hostile_fixtures() -> list[dict[str, object]]:
    policy = base_policy()
    messages = golden_messages()
    self_raw, normal = messages["self_delegate"]
    starter_raw, _ = messages["starter_delegate"]
    execute_raw, execute_view = messages["withdraw_execute"]
    request_raw, _ = messages["withdraw_request"]
    self_ix = pay_chain.delegate_set(
        WALLET, NODE, SELF_CAP, CHAIN_TIME + SELF_DAYS * 86400,
        pay_chain.DELEGATE_FLAGS_ALL, PROGRAM)[0]
    starter_ix = pay_chain.delegate_set(
        WALLET, STARTER, 1_000_000, CHAIN_TIME + STARTER_DAYS * 86400,
        pay_chain.DELEGATE_OPEN_CHANNEL, PROGRAM)[0]

    cases: list[tuple[str, bytes, wallet_policy.WalletPolicy,
                      wallet_policy.ClusterView, str]] = []

    def add(name: str, raw: bytes = self_raw, *, p: wallet_policy.WalletPolicy = policy,
            view: wallet_policy.ClusterView = normal, code: str) -> None:
        cases.append((name, raw, p, view, code))

    add("extra_instruction", _compile([self_ix, self_ix]), code="instruction_template")
    unknown = pay_chain.Instruction(UNKNOWN_PROGRAM, self_ix.data, self_ix.accounts)
    add("unknown_program", _compile([unknown]), code="unknown_program")
    unknown_op = pay_chain.Instruction(PROGRAM, b"\0" * 8, self_ix.accounts)
    add("known_program_unknown_opcode", _compile([unknown_op]), code="unknown_instruction")
    swapped = pay_chain.Instruction(PROGRAM, self_ix.data,
                                    (self_ix.accounts[1], self_ix.accounts[0], *self_ix.accounts[2:]))
    add("swapped_instruction_accounts", _compile([swapped]), code="account_template")
    writable = bytearray(self_raw); writable[2] -= 1
    add("writable_bit_change", bytes(writable), code="privilege_template")
    wrong_payer = bytearray(self_raw)
    wrong_payer[4:36], wrong_payer[36:68] = wrong_payer[36:68], wrong_payer[4:36]
    add("wrong_fee_payer", bytes(wrong_payer), code="fee_payer")
    extra_signer = bytearray(self_raw); extra_signer[0] = 2
    add("extra_signer", bytes(extra_signer), code="signer_set")
    wrong_delegate = replace(self_ix, accounts=(
        *self_ix.accounts[:3], pay_chain.AccountMeta(WRONG_DESTINATION, False, True),
        *self_ix.accounts[4:]))
    add("wrong_delegate_pda", _compile([wrong_delegate]), code="account_order")
    execute_ixs = pay_chain.escrow_withdraw_execute(WALLET, MINT, PROGRAM)
    wrong_ata_ix = replace(execute_ixs[0], accounts=(
        execute_ixs[0].accounts[0], pay_chain.AccountMeta(WRONG_DESTINATION, False, True),
        *execute_ixs[0].accounts[2:]))
    add("wrong_ata", _compile([wrong_ata_ix, execute_ixs[1]]), view=execute_view,
        code="account_order")
    wrong_mint = replace(execute_ixs[1], accounts=(
        *execute_ixs[1].accounts[:2], pay_chain.AccountMeta(WRONG_DESTINATION),
        *execute_ixs[1].accounts[3:]))
    add("wrong_mint", _compile([execute_ixs[0], wrong_mint]), view=execute_view,
        code="account_order")
    transfer_wrong = _compile([pay_chain.system_transfer(WALLET, WRONG_DESTINATION, 1)])
    add("transfer_wrong_destination", transfer_wrong, code="account_order")
    noncanonical = self_raw[:3] + bytes([self_raw[3] | 0x80, 0]) + self_raw[4:]
    add("noncanonical_shortvec", noncanonical, code="noncanonical_shortvec")
    add("trailing_byte", self_raw + b"\0", code="trailing_bytes")
    versioned = bytes([self_raw[0] | 0x80]) + self_raw[1:]
    add("versioned_message", versioned, code="versioned_message")
    add("mainnet_genesis", view=replace(normal, genesis_hash=MAINNET_GENESIS),
        code="wrong_genesis")
    add("invalid_blockhash", view=replace(normal, valid_recent_blockhashes=(STALE_BLOCKHASH,)),
        code="blockhash")
    add("stale_blockhash", view=replace(normal, valid_recent_blockhashes=()), code="blockhash")
    over_cap = _compile(pay_chain.delegate_set(
        WALLET, NODE, 5_000_001, CHAIN_TIME + 100,
        pay_chain.DELEGATE_FLAGS_ALL, PROGRAM))
    add("self_cap_over_ceiling", over_cap, code="cap_ceiling")
    over_expiry = _compile(pay_chain.delegate_set(
        WALLET, NODE, 1, CHAIN_TIME + 30 * 86400 + 1,
        pay_chain.DELEGATE_FLAGS_ALL, PROGRAM))
    add("expiry_over_ceiling", over_expiry, code="expiry_ceiling")
    add("starter_remaining_over_ceiling", starter_raw,
        p=replace(policy, max_starter_micro_usdc=999_999),
        code="starter_remaining_ceiling")
    over_withdraw = _compile(pay_chain.escrow_withdraw_request(
        WALLET, 5_000_001, PROGRAM))
    add("withdraw_over_ceiling", over_withdraw, code="withdraw_ceiling")
    add("pending_execute_over_ceiling", execute_raw,
        view=base_view(pending=(5_000_001, CHAIN_TIME - 1)), code="withdraw_ceiling")
    over_sol = _compile([pay_chain.system_transfer(
        WALLET, NODE, policy.max_self_sol_lamports + 1)])
    add("sol_over_ceiling", over_sol, code="sol_ceiling")
    add("network_fee_over_ceiling", view=replace(
        normal, network_fee_lamports=policy.max_network_fee_lamports + 1),
        code="fee_ceiling")
    add("delegate_rent_over_ceiling", view=replace(
        normal, delegate_rent_exempt_lamports=policy.max_delegate_rent_lamports + 1),
        code="rent_ceiling")
    add("ata_rent_over_ceiling", execute_raw, view=replace(
        execute_view, ata_rent_exempt_lamports=policy.max_ata_rent_lamports + 1),
        code="rent_ceiling")
    add("programdata_address_mismatch", view=replace(
        normal, programdata_address=WRONG_DESTINATION), code="programdata")
    add("program_hash_mismatch", view=replace(
        normal, program_sha256="00" * 32), code="program_hash")
    add("upgrade_authority_mismatch", view=replace(
        normal, upgrade_authority=WRONG_DESTINATION), code="upgrade_authority")

    # F1: a view claiming almost all of an over-ceiling cap was spent must not widen policy.
    f1_raw = _compile(pay_chain.delegate_set(
        WALLET, STARTER, 1_000_001, CHAIN_TIME + 100,
        pay_chain.DELEGATE_OPEN_CHANNEL, PROGRAM))
    starter_address = pay_chain.delegate_pda(
        pay_chain.escrow_pda(WALLET, PROGRAM)[0], STARTER, PROGRAM)[0]
    f1_account = wallet_policy.AccountView(
        starter_address, PROGRAM, 1_691_280,
        _delegate_data(STARTER, cap=1_000_001, spent=1_000_000))
    add("f1_spent_cap_minus_one", f1_raw, view=_replace_account(normal, f1_account),
        code="starter_cap_ceiling")

    delegate_address = pay_chain.delegate_pda(
        pay_chain.escrow_pda(WALLET, PROGRAM)[0], NODE, PROGRAM)[0]
    dust_nonzero = wallet_policy.AccountView(
        delegate_address, pay_chain.SYSTEM_PROGRAM_ID, 1, b"x")
    add("dusted_delegate_nonzero_data", view=_replace_account(normal, dust_nonzero),
        code="delegate_state")
    wrong_owner = wallet_policy.AccountView(
        delegate_address, pay_chain.TOKEN_PROGRAM_ID, 0, b"")
    add("delegate_wrong_owner", view=_replace_account(normal, wrong_owner),
        code="delegate_state")
    ata = pay_chain.associated_token_address(WALLET, MINT)[0]
    ata_bad = wallet_policy.AccountView(ata, pay_chain.SYSTEM_PROGRAM_ID, 1, b"x")
    add("dusted_ata_nonzero_data", execute_raw,
        view=_replace_account(execute_view, ata_bad), code="ata_state")
    ata_wrong = wallet_policy.AccountView(ata, PROGRAM, 0, b"")
    add("ata_wrong_owner", execute_raw, view=_replace_account(execute_view, ata_wrong),
        code="ata_state")

    # Starter state only tightens the byte-level ceiling.
    def starter_state(name: str, **kwargs: object) -> None:
        account = wallet_policy.AccountView(
            starter_address, PROGRAM, 1_691_280, _delegate_data(STARTER, **kwargs))
        add(name, starter_raw, view=_replace_account(normal, account), code="delegate_state")

    starter_state("starter_newer_generation", generation=8)
    starter_state("starter_same_generation_disabled", disabled=True)
    starter_state("starter_open_channel", open_channels=1)
    starter_state("starter_open_task", open_tasks=1)

    result = []
    for name, raw, item_policy, view, code in cases:
        result.append({
            "name": name, "raw_message_hex": raw.hex(),
            "policy": _serialize_policy(item_policy),
            "cluster_view": _serialize_view(view), "expected_code": code,
        })
    return result


def write_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIXTURE_DIR.glob("*.json"):
        path.unlink()
    for name, fixture in golden_fixtures().items():
        (FIXTURE_DIR / f"golden-{name}.json").write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (FIXTURE_DIR / "hostile.json").write_text(
        json.dumps(hostile_fixtures(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_fixtures()
