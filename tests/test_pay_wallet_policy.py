"""Golden and hostile tests for the broker-wallet decoder/policy boundary."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from network import pay_chain
from network import pay_wallet_policy as policy_module


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "pay_wallet_policy"
GENERATOR_PATH = ROOT / "tests" / "generate_pay_wallet_policy_fixtures.py"
_SPEC = importlib.util.spec_from_file_location("pay_wallet_fixture_generator", GENERATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
fixture_generator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fixture_generator
_SPEC.loader.exec_module(fixture_generator)


def _policy(value: dict[str, object]) -> policy_module.WalletPolicy:
    return policy_module.WalletPolicy(**value)


def _view(value: dict[str, object]) -> policy_module.ClusterView:
    fields = dict(value)
    fields["valid_recent_blockhashes"] = tuple(fields["valid_recent_blockhashes"])
    fields["accounts"] = tuple(policy_module.AccountView(
        address=item["address"], owner=item["owner"], lamports=item["lamports"],
        data=bytes.fromhex(item["data_hex"])) for item in fields["accounts"])
    return policy_module.ClusterView(**fields)


def _approved(value: list[dict[str, object]]) -> frozenset[
        policy_module.ApprovedWithdrawRequest]:
    return frozenset(policy_module.ApprovedWithdrawRequest(**item) for item in value)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


GOLDEN_PATHS = sorted(FIXTURES.glob("golden-*.json"))
HOSTILE = _load(FIXTURES / "hostile.json")


@pytest.mark.parametrize("path", GOLDEN_PATHS, ids=lambda path: path.stem)
def test_frozen_golden_message_decodes_to_exact_plan(path: Path):
    fixture = _load(path)
    raw = bytes.fromhex(fixture["raw_message_hex"])
    plan = policy_module.decode_and_check(
        raw, _policy(fixture["policy"]), _view(fixture["cluster_view"]),
        _approved(fixture["approved_withdraw_requests"]))
    assert plan.raw_message is raw
    assert plan.to_dict() == fixture["expected_plan"]
    assert plan.message_sha256 == fixture["expected_plan"]["message_sha256"]


def test_frozen_goldens_still_match_real_pay_builders():
    generated = fixture_generator.golden_fixtures()
    assert set(generated) == {path.stem.removeprefix("golden-") for path in GOLDEN_PATHS}
    for name, expected in generated.items():
        frozen = _load(FIXTURES / f"golden-{name}.json")
        assert frozen["raw_message_hex"] == expected["raw_message_hex"]
        assert frozen["expected_plan"] == expected["expected_plan"]


def test_frozen_hostiles_still_match_one_mutation_generator():
    assert HOSTILE == fixture_generator.hostile_fixtures()
    goldens = fixture_generator.golden_fixtures()
    inputs = ("raw_message_hex", "policy", "cluster_view",
              "approved_withdraw_requests")
    for hostile in HOSTILE:
        golden = goldens[hostile["golden"]]
        changed = [name for name in inputs if hostile[name] != golden[name]]
        assert len(changed) == 1, (hostile["name"], changed)


@pytest.mark.parametrize("fixture", HOSTILE, ids=lambda fixture: fixture["name"])
def test_hostile_fixture_is_refused_with_closed_code(fixture: dict[str, object]):
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        policy_module.decode_and_check(
            bytes.fromhex(fixture["raw_message_hex"]),
            _policy(fixture["policy"]), _view(fixture["cluster_view"]),
            _approved(fixture["approved_withdraw_requests"]))
    assert refused.value.code == fixture["expected_code"]
    assert refused.value.code in policy_module.REFUSAL_CODES
    assert str(refused.value) == refused.value.code


def test_policy_and_cluster_inputs_are_immutable():
    fixture = _load(GOLDEN_PATHS[0])
    wallet_policy = _policy(fixture["policy"])
    cluster_view = _view(fixture["cluster_view"])
    with pytest.raises(FrozenInstanceError):
        wallet_policy.wallet_id = fixture_generator.NODE
    with pytest.raises(FrozenInstanceError):
        cluster_view.network_fee_lamports = 1
    with pytest.raises(FrozenInstanceError):
        cluster_view.accounts[0].lamports = 1


def test_q3_defaults_match_current_self_and_starter_builders():
    wallet_policy = fixture_generator.base_policy()
    assert wallet_policy.max_delegate_micro_usdc == fixture_generator.SELF_CAP
    assert wallet_policy.max_delegate_ttl_s == fixture_generator.SELF_DAYS * 86400
    assert wallet_policy.max_self_sol_lamports == fixture_generator.nak.NODE_SOL_TARGET
    assert wallet_policy.max_starter_ttl_s == fixture_generator.STARTER_DAYS * 86400
    assert wallet_policy.max_starter_sol_lamports == fixture_generator.STARTER_TOP_UP
    assert wallet_policy.max_starter_cap_total_micro_usdc == 1_000_000
    assert wallet_policy.max_withdraw_micro_usdc == 5_000_000


def test_hard_maxima_are_the_reviewed_q3_values_and_fit_u64():
    expected = {
        "NAK_WALLET_MAX_DELEGATE_MICRO_USDC": 5_000_000,
        "NAK_WALLET_MAX_DELEGATE_TTL_S": 30 * 86400,
        "NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC": 1_000_000,
        "NAK_WALLET_MAX_STARTER_MICRO_USDC": 1_000_000,
        "NAK_WALLET_MAX_STARTER_TTL_S": 30 * 86400,
        "NAK_WALLET_MAX_WITHDRAW_MICRO_USDC": 5_000_000,
        "NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS": 10_000,
        "NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS": 1_700_000,
        "NAK_WALLET_MAX_ATA_RENT_LAMPORTS": 2_040_000,
        "NAK_WALLET_MAX_SELF_SOL_LAMPORTS": 30_000_000,
        "NAK_WALLET_MAX_STARTER_SOL_LAMPORTS": 10_000_000,
    }
    assert policy_module.U64_MAX == 2**64 - 1
    for name, value in expected.items():
        assert getattr(policy_module, name) == value
        assert 0 < value <= policy_module.U64_MAX


@pytest.mark.parametrize("field,hard_max", [
    ("max_delegate_micro_usdc", policy_module.NAK_WALLET_MAX_DELEGATE_MICRO_USDC),
    ("max_delegate_ttl_s", policy_module.NAK_WALLET_MAX_DELEGATE_TTL_S),
    ("max_starter_cap_total_micro_usdc",
     policy_module.NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC),
    ("max_starter_micro_usdc", policy_module.NAK_WALLET_MAX_STARTER_MICRO_USDC),
    ("max_starter_ttl_s", policy_module.NAK_WALLET_MAX_STARTER_TTL_S),
    ("max_withdraw_micro_usdc", policy_module.NAK_WALLET_MAX_WITHDRAW_MICRO_USDC),
    ("max_network_fee_lamports", policy_module.NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS),
    ("max_delegate_rent_lamports", policy_module.NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS),
    ("max_ata_rent_lamports", policy_module.NAK_WALLET_MAX_ATA_RENT_LAMPORTS),
    ("max_self_sol_lamports", policy_module.NAK_WALLET_MAX_SELF_SOL_LAMPORTS),
    ("max_starter_sol_lamports", policy_module.NAK_WALLET_MAX_STARTER_SOL_LAMPORTS),
])
def test_policy_can_only_tighten_hard_maxima(field: str, hard_max: int):
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    widened = replace(fixture_generator.base_policy(), **{field: hard_max + 1})
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, view, widened)
    assert refused.value.code == "policy"


def test_pda_and_ata_derivation_matches_pay_chain_vectors():
    wallet_policy = fixture_generator.base_policy()
    derived = policy_module._derive(wallet_policy)
    expected = pay_chain._derived(wallet_policy.wallet_id, wallet_policy.program_id)
    assert derived["config"] == expected["config"]
    assert derived["escrow"] == expected["escrow"]
    assert derived["vault"] == expected["vault"]
    assert derived["event"] == expected["event"]
    assert derived["ata"] == pay_chain.associated_token_address(
        wallet_policy.wallet_id, wallet_policy.mint_id)[0]
    assert derived["programdata"] == pay_chain.find_program_address(
        [pay_chain.pubkey_bytes(wallet_policy.program_id)],
        policy_module.UPGRADEABLE_LOADER_ID)[0]
    owner = pay_chain.b58encode(bytes(range(32)))
    assert policy_module._find_program_address(
        (b"config",), fixture_generator.PROGRAM) == (
            "8Q1BNKohnutkCsWhPSKVydM5LBNbzVom8Zss5hNsa7r8")
    assert policy_module._find_program_address(
        (b"escrow", pay_chain.pubkey_bytes(pay_chain.config_pda(
            fixture_generator.PROGRAM)[0]), pay_chain.pubkey_bytes(owner)),
        fixture_generator.PROGRAM) == "4VP7HBng8QazeGroEKNS1jqR5JWofEeSzs2swagzgNqA"


def _decode(raw: bytes, view: policy_module.ClusterView,
            wallet_policy: policy_module.WalletPolicy | None = None,
            approved: frozenset[policy_module.ApprovedWithdrawRequest] = frozenset(),
            ) -> policy_module.ApprovalPlan:
    return policy_module.decode_and_check(
        raw, wallet_policy or fixture_generator.base_policy(), view, approved)


def _delegate_view(key: str, **state: object) -> policy_module.ClusterView:
    wallet_policy = fixture_generator.base_policy()
    derived = policy_module._derive(wallet_policy)
    address = pay_chain.delegate_pda(
        derived["escrow"], key, fixture_generator.PROGRAM)[0]
    account = policy_module.AccountView(
        address, fixture_generator.PROGRAM, 1_691_280,
        fixture_generator._delegate_data(key, **state))
    return fixture_generator._replace_account(fixture_generator.base_view(), account)


def _fields(plan: policy_module.ApprovalPlan) -> dict[str, str | int | bool]:
    return {field.name: field.value for field in plan.rendered_fields}


def test_p1_config_mint_must_be_pinned_before_usdc_rendering():
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    derived = policy_module._derive(fixture_generator.base_policy())
    config = (
        hashlib.sha256(b"account:Config").digest()[:8]
        + pay_chain.pubkey_bytes(fixture_generator.AUTHORITY)
        + pay_chain.pubkey_bytes(fixture_generator.WRONG_DESTINATION)
        + pay_chain.pubkey_bytes(fixture_generator.AUTHORITY)
        + struct.pack("<Hqqq", 300, 21_600, 86_400, 172_800)
        + b"\0\0"
        + bytes((pay_chain.config_pda(fixture_generator.PROGRAM)[1],
                 pay_chain.fee_vault_pda(fixture_generator.PROGRAM)[1]))
    ).ljust(208, b"\0")
    changed = fixture_generator._replace_account(view, policy_module.AccountView(
        derived["config"], fixture_generator.PROGRAM, 5_000_000, config))
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, changed)
    assert refused.value.code == "mint"


def test_p1_config_account_must_have_canonical_owner_layout_and_bumps():
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    derived = policy_module._derive(fixture_generator.base_policy())
    original = next(account for account in view.accounts
                    if account.address == derived["config"])
    bad_tag = bytearray(original.data)
    bad_tag[130] = 2
    bad_bump = bytearray(original.data)
    bad_bump[132] ^= 1
    cases = (
        fixture_generator._replace_account(view, replace(
            original, owner=pay_chain.SYSTEM_PROGRAM_ID)),
        fixture_generator._replace_account(view, replace(original, data=bytes(bad_tag))),
        fixture_generator._replace_account(view, replace(original, data=bytes(bad_bump))),
        replace(view, accounts=tuple(
            account for account in view.accounts if account.address != derived["config"])),
    )
    for changed in cases:
        with pytest.raises(policy_module.PolicyRefusal) as refused:
            _decode(raw, changed)
        assert refused.value.code == "config_state"


def test_p1_starter_cap_is_byte_bounded_by_both_ceilings():
    raw, _view = fixture_generator.golden_messages()["starter_delegate"]
    tight = replace(fixture_generator.base_policy(), max_starter_micro_usdc=1)
    view = _delegate_view(
        fixture_generator.STARTER, generation=7, cap=1_000_000, spent=999_999)
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, view, tight)
    assert refused.value.code == "starter_remaining_ceiling"


@pytest.mark.parametrize(
    "recipient",
    [fixture_generator.PROGRAM, pay_chain.SYSTEM_PROGRAM_ID,
     pay_chain.TOKEN_PROGRAM_ID, pay_chain.ASSOCIATED_TOKEN_PROGRAM_ID,
     pay_chain.ED25519_PROGRAM_ID, pay_chain.INSTRUCTIONS_SYSVAR_ID,
     pay_chain.escrow_pda(fixture_generator.WALLET, fixture_generator.PROGRAM)[0]],
    ids=["payment-program", "system-program", "token-program", "ata-program",
         "ed25519-program", "instructions-sysvar", "escrow-role"],
)
def test_p2_transfer_recipient_cannot_alias_protected_or_role_account(recipient: str):
    raw = fixture_generator._compile([
        *pay_chain.delegate_set(
            fixture_generator.WALLET, recipient, 1,
            fixture_generator.CHAIN_TIME + 1, pay_chain.DELEGATE_OPEN_CHANNEL,
            fixture_generator.PROGRAM),
        pay_chain.system_transfer(fixture_generator.WALLET, recipient, 1),
    ])
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, fixture_generator.base_view())
    assert refused.value.code == "account_template"


def test_p2_self_top_up_node_cannot_alias_escrow():
    wallet_policy = fixture_generator.base_policy()
    escrow = policy_module._derive(wallet_policy)["escrow"]
    wallet_policy = replace(wallet_policy, node_id=escrow)
    raw = fixture_generator._compile([
        pay_chain.system_transfer(fixture_generator.WALLET, escrow, 1)])
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, fixture_generator.base_view(), wallet_policy)
    assert refused.value.code == "account_template"


@pytest.mark.parametrize(
    "role",
    ["vault", "ata", "mint", "fee_vault"],
)
def test_p2_starter_recipient_cannot_alias_canonical_role(role: str):
    wallet_policy = fixture_generator.base_policy()
    derived = policy_module._derive(wallet_policy)
    recipient = wallet_policy.mint_id if role == "mint" else derived[role]
    raw = fixture_generator._compile(pay_chain.delegate_set(
        fixture_generator.WALLET, recipient, 1,
        fixture_generator.CHAIN_TIME + 1, pay_chain.DELEGATE_OPEN_CHANNEL,
        fixture_generator.PROGRAM))
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, fixture_generator.base_view(), wallet_policy)
    assert refused.value.code == "account_template"


@pytest.mark.parametrize(
    "golden,expected_extra_roles",
    [
        ("self_delegate", {"delegate_pda"}),
        ("starter_delegate", {"delegate_pda", "recipient"}),
        ("self_top_up", set()),
        ("withdraw_request", set()),
        ("withdraw_execute", set()),
    ],
    ids=["self-delegate", "starter-grant", "self-top-up",
         "withdraw-request", "withdraw-execute"],
)
def test_p2_every_template_checks_the_complete_role_set(
        monkeypatch: pytest.MonkeyPatch, golden: str,
        expected_extra_roles: set[str]):
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(
        policy_module, "_validate_role_separation",
        lambda roles: captured.append(dict(roles)))
    raw, view = fixture_generator.golden_messages()[golden]
    approved = (fixture_generator.approved_withdraw_requests(view)
                if golden == "withdraw_execute" else frozenset())
    _decode(raw, view, approved=approved)
    assert len(captured) == 1
    assert set(captured[0]) == {
        "wallet", "node", "config", "escrow", "vault", "ata", "mint",
        "fee_vault", "event_authority", *expected_extra_roles,
    }


@pytest.mark.parametrize("offset", [72, 109, 129], ids=["delegate", "is-native", "close-authority"])
def test_p2_classic_spl_account_rejects_invalid_coption_tags(offset: int):
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    derived = policy_module._derive(fixture_generator.base_policy())
    data = bytearray(fixture_generator._ata_data())
    struct.pack_into("<I", data, offset, 2)
    changed = fixture_generator._replace_account(view, policy_module.AccountView(
        derived["ata"], pay_chain.TOKEN_PROGRAM_ID, 2_039_280, bytes(data)))
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, changed, approved=fixture_generator.approved_withdraw_requests(changed))
    assert refused.value.code == "ata_state"


@pytest.mark.parametrize("state", [0, 3], ids=["uninitialized", "unknown"])
def test_p2_classic_spl_account_requires_initialized_state(state: int):
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    derived = policy_module._derive(fixture_generator.base_policy())
    data = bytearray(fixture_generator._ata_data())
    data[108] = state
    changed = fixture_generator._replace_account(view, policy_module.AccountView(
        derived["ata"], pay_chain.TOKEN_PROGRAM_ID, 2_039_280, bytes(data)))
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, changed, approved=fixture_generator.approved_withdraw_requests(changed))
    assert refused.value.code == "ata_state"


@pytest.mark.parametrize(
    "state",
    [
        {"generation": 6, "open_channels": 1},
        {"generation": 6, "open_tasks": 1},
        {"generation": 7, "disabled": True},
        {"generation": 7, "cap": 6_000_000, "spent": 6_000_000},
        {"generation": 7, "cap": 4_000_000, "spent": 4_500_000},
        {"generation": 7, "flags": 255},
    ],
    ids=["older-open-channel", "older-open-task", "same-disabled",
         "encoded-cap-below-spent", "spent-above-pre-cap", "unknown-flags"],
)
def test_p2_self_delegate_enforces_program_transition_preconditions(
        state: dict[str, object]):
    raw, _view = fixture_generator.golden_messages()["self_delegate"]
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, _delegate_view(fixture_generator.NODE, **state))
    assert refused.value.code == "delegate_state"


@pytest.mark.parametrize(
    "change",
    [{"withdraw_delay_seconds": policy_module.U64_MAX},
     {"chain_time": policy_module.I64_MAX}],
    ids=["delay-not-i64", "checked-add-overflow"],
)
def test_p2_withdraw_time_is_checked_i64_arithmetic(change: dict[str, int]):
    raw, view = fixture_generator.golden_messages()["withdraw_request"]
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, replace(view, **change))
    assert refused.value.code == "cluster_state"


def test_p3_older_starter_displays_real_prestate_and_reset():
    raw, _view = fixture_generator.golden_messages()["starter_delegate"]
    plan = _decode(raw, _delegate_view(
        fixture_generator.STARTER, generation=6, cap=1_000_000, spent=999_999))
    fields = _fields(plan)
    assert fields["pre_cap_total_micro_usdc"] == 1_000_000
    assert fields["pre_spent_micro_usdc"] == 999_999
    assert fields["older_generation_reset"] is True
    assert fields["resulting_remaining_micro_usdc"] == 1_000_000


def test_p3_older_self_delegate_displays_real_prestate_and_reset():
    raw, _view = fixture_generator.golden_messages()["self_delegate"]
    plan = _decode(raw, _delegate_view(
        fixture_generator.NODE, generation=6, cap=1_000_000,
        spent=999_999, disabled=True))
    fields = _fields(plan)
    assert fields["pre_cap_total_micro_usdc"] == 1_000_000
    assert fields["pre_spent_micro_usdc"] == 999_999
    assert fields["older_generation_reset"] is True


def test_p3_rent_displays_actual_shortfall_and_worst_case():
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    derived = policy_module._derive(fixture_generator.base_policy())
    initialized = fixture_generator._replace_account(view, policy_module.AccountView(
        derived["ata"], pay_chain.TOKEN_PROGRAM_ID, 2_039_280,
        fixture_generator._ata_data()))
    fields = _fields(_decode(
        raw, initialized,
        approved=fixture_generator.approved_withdraw_requests(initialized)))
    assert fields["ata_rent_shortfall_lamports"] == 0
    assert fields["ata_rent_worst_case_lamports"] == 2_039_280

    dusted = fixture_generator._replace_account(view, policy_module.AccountView(
        derived["ata"], pay_chain.SYSTEM_PROGRAM_ID, 1_000_000, b""))
    fields = _fields(_decode(
        raw, dusted, approved=fixture_generator.approved_withdraw_requests(dusted)))
    assert fields["ata_rent_shortfall_lamports"] == 1_039_280
    assert fields["ata_rent_worst_case_lamports"] == 2_039_280


def test_p3_self_and_execute_plans_display_applicable_money_ceilings():
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    assert _fields(_decode(raw, view))["cap_total_ceiling_micro_usdc"] == 5_000_000
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    assert _fields(_decode(
        raw, view,
        approved=fixture_generator.approved_withdraw_requests(view))
    )["withdraw_ceiling_micro_usdc"] == 5_000_000


def test_withdraw_provenance_contract_is_documented():
    contract = policy_module.ApprovedWithdrawRequest.__doc__ or ""
    for phrase in ("message digest", "signature", "escrow generation",
                   "reconciled", "competing signed attempts", "pending withdrawal"):
        assert phrase in contract
    note = (ROOT / "docs" / "design" / "2026-11-04-broker-pay-wallet.md").read_text(
        encoding="utf-8")
    for phrase in ("originating message digest", "transaction signature",
                   "escrow generation", "competing signed attempts"):
        assert phrase in note


@pytest.mark.parametrize(
    "lamports",
    [0, 400_000, 1_691_280, 9_000_000],
)
def test_dusted_delegate_pda_still_renders_worst_case_rent(lamports: int):
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    derived = pay_chain._derived(fixture_generator.WALLET, fixture_generator.PROGRAM)
    delegate = pay_chain.delegate_pda(
        derived["escrow"], fixture_generator.NODE, fixture_generator.PROGRAM)[0]
    account = policy_module.AccountView(
        delegate, pay_chain.SYSTEM_PROGRAM_ID, lamports, b"")
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["delegate_rent_worst_case_lamports"] == 1_691_280


def test_permissionless_closable_stale_delegate_renders_worst_case_rent():
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    derived = pay_chain._derived(fixture_generator.WALLET, fixture_generator.PROGRAM)
    delegate = pay_chain.delegate_pda(
        derived["escrow"], fixture_generator.NODE, fixture_generator.PROGRAM)[0]
    account = policy_module.AccountView(
        delegate, fixture_generator.PROGRAM, 1,
        fixture_generator._delegate_data(
            fixture_generator.NODE, generation=6, flags=pay_chain.DELEGATE_FLAGS_ALL))
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["delegate_rent_worst_case_lamports"] == 1_691_280


@pytest.mark.parametrize(
    "lamports",
    [0, 1_000_000, 2_039_280, 9_000_000],
)
def test_dusted_ata_still_renders_worst_case_rent(lamports: int):
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    ata = pay_chain.associated_token_address(
        fixture_generator.WALLET, fixture_generator.MINT)[0]
    account = policy_module.AccountView(
        ata, pay_chain.SYSTEM_PROGRAM_ID, lamports, b"")
    changed_view = fixture_generator._replace_account(view, account)
    plan = _decode(raw, changed_view, approved=
                   fixture_generator.approved_withdraw_requests(changed_view))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["ata_rent_worst_case_lamports"] == 2_039_280


def test_initialized_canonical_ata_still_renders_worst_case_rent():
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    ata = pay_chain.associated_token_address(
        fixture_generator.WALLET, fixture_generator.MINT)[0]
    account = policy_module.AccountView(
        ata, pay_chain.TOKEN_PROGRAM_ID, 1,
        fixture_generator._ata_data())
    changed_view = fixture_generator._replace_account(view, account)
    plan = _decode(raw, changed_view, approved=
                   fixture_generator.approved_withdraw_requests(changed_view))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["ata_rent_worst_case_lamports"] == 2_039_280


def test_withdraw_execute_requires_exact_durable_provenance():
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    exact = fixture_generator.approved_withdraw_requests(view)
    assert _decode(raw, view, approved=exact).operation == "withdraw_execute"
    approved = next(iter(exact))
    mismatches = (
        replace(approved, escrow_id=fixture_generator.WRONG_DESTINATION),
        replace(approved, amount_micro_usdc=approved.amount_micro_usdc + 1),
        replace(approved, withdraw_at=approved.withdraw_at + 1),
    )
    for mismatch in mismatches:
        with pytest.raises(policy_module.PolicyRefusal) as refused:
            _decode(raw, view, approved=frozenset((mismatch,)))
        assert refused.value.code == "withdraw_provenance"


def test_required_isolated_hostile_fixtures_are_frozen():
    names = {fixture["name"] for fixture in HOSTILE}
    assert {
        "wrong_fee_payer", "duplicate_keys", "instruction_index_aliasing",
        "readonly_signed_account", "anchor_trailing_argument_bytes",
        "ata_create_non_idempotent", "token_2022_program",
        "system_transfer_with_seed", "system_allocate", "system_assign",
        "hard_ceiling_widening", "execute_without_provenance",
        "self_reversed_instruction_order", "starter_reversed_instruction_order",
        "wrong_policy_mint",
    } <= names


@pytest.mark.parametrize(
    "generation,spent,disabled,remaining",
    [(7, 250_000, False, 750_000), (6, 999_999, False, 1_000_000),
     (6, 999_999, True, 1_000_000)],
)
def test_starter_state_can_only_tighten(generation: int, spent: int,
                                        disabled: bool, remaining: int):
    raw, view = fixture_generator.golden_messages()["starter_delegate"]
    escrow = pay_chain.escrow_pda(fixture_generator.WALLET, fixture_generator.PROGRAM)[0]
    delegate = pay_chain.delegate_pda(
        escrow, fixture_generator.STARTER, fixture_generator.PROGRAM)[0]
    account = policy_module.AccountView(
        delegate, fixture_generator.PROGRAM, 1,
        fixture_generator._delegate_data(
            fixture_generator.STARTER, generation=generation,
            cap=max(500_000, spent), spent=spent, disabled=disabled))
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["resulting_remaining_micro_usdc"] == remaining


@pytest.mark.parametrize(
    "change,code",
    [({"program_executable": False}, "program_state"),
     ({"program_owner": pay_chain.SYSTEM_PROGRAM_ID}, "program_state"),
     ({"programdata_owner": pay_chain.SYSTEM_PROGRAM_ID}, "program_state")],
)
def test_program_loader_state_fails_closed(change: dict[str, object], code: str):
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, replace(view, **change))
    assert refused.value.code == code


def test_existing_escrow_is_required_even_for_transfer_only():
    raw, view = fixture_generator.golden_messages()["self_top_up"]
    escrow = pay_chain.escrow_pda(
        fixture_generator.WALLET, fixture_generator.PROGRAM)[0]
    missing = replace(view, accounts=tuple(
        account for account in view.accounts if account.address != escrow))
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, missing)
    assert refused.value.code == "escrow_state"


@pytest.mark.parametrize("instructions", [
    lambda: pay_chain.escrow_init(
        fixture_generator.WALLET, fixture_generator.MINT, fixture_generator.PROGRAM),
    lambda: pay_chain.escrow_deposit(
        fixture_generator.WALLET, fixture_generator.MINT, 1, fixture_generator.PROGRAM),
    lambda: pay_chain.delegate_revoke(
        fixture_generator.WALLET, fixture_generator.NODE, fixture_generator.PROGRAM),
    lambda: pay_chain.channel_close_request(
        fixture_generator.WALLET, fixture_generator.WALLET, 7, 3,
        fixture_generator.PROGRAM),
], ids=["escrow_init", "deposit", "delegate_revoke", "channel_close"])
def test_q1_excluded_program_operations_stay_closed(instructions):
    raw = pay_chain.compile_message(
        instructions(), fixture_generator.WALLET, fixture_generator.BLOCKHASH).data
    view = fixture_generator.base_view()
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        _decode(raw, view)
    assert refused.value.code == "unknown_instruction"


def test_refusal_code_set_is_closed():
    with pytest.raises(ValueError, match="unknown policy refusal code"):
        policy_module.PolicyRefusal("made_up")


def test_module_imports_only_python_stdlib_and_is_isolated(tmp_path: Path):
    path = ROOT / "scripts" / "network" / "pay_wallet_policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported == {"__future__", "dataclasses", "hashlib", "struct", "typing"}
    env = {"HOME": str(tmp_path), "PYTHONPATH": "", "PATH": os.environ["PATH"]}
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import importlib.util;"
         f"s=importlib.util.spec_from_file_location('isolated_policy',{str(path)!r});"
         "m=importlib.util.module_from_spec(s);import sys;sys.modules[s.name]=m;"
         "s.loader.exec_module(m);print(m.NAKSHATRA_PAY_PROGRAM_ID)"],
        check=True, capture_output=True, text=True, env=env)
    assert result.stdout.strip() == policy_module.NAKSHATRA_PAY_PROGRAM_ID
