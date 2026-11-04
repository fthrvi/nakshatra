"""Golden and hostile tests for the broker-wallet decoder/policy boundary."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
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


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


GOLDEN_PATHS = sorted(FIXTURES.glob("golden-*.json"))
HOSTILE = _load(FIXTURES / "hostile.json")


@pytest.mark.parametrize("path", GOLDEN_PATHS, ids=lambda path: path.stem)
def test_frozen_golden_message_decodes_to_exact_plan(path: Path):
    fixture = _load(path)
    raw = bytes.fromhex(fixture["raw_message_hex"])
    plan = policy_module.decode_and_check(
        raw, _policy(fixture["policy"]), _view(fixture["cluster_view"]))
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


@pytest.mark.parametrize("fixture", HOSTILE, ids=lambda fixture: fixture["name"])
def test_hostile_fixture_is_refused_with_closed_code(fixture: dict[str, object]):
    with pytest.raises(policy_module.PolicyRefusal) as refused:
        policy_module.decode_and_check(
            bytes.fromhex(fixture["raw_message_hex"]),
            _policy(fixture["policy"]), _view(fixture["cluster_view"]))
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
            wallet_policy: policy_module.WalletPolicy | None = None
            ) -> policy_module.ApprovalPlan:
    return policy_module.decode_and_check(
        raw, wallet_policy or fixture_generator.base_policy(), view)


@pytest.mark.parametrize(
    "lamports,expected_shortfall",
    [(0, 1_691_280), (400_000, 1_291_280), (1_691_280, 0), (9_000_000, 0)],
)
def test_dusted_delegate_pda_credits_prefunding(lamports: int, expected_shortfall: int):
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    derived = pay_chain._derived(fixture_generator.WALLET, fixture_generator.PROGRAM)
    delegate = pay_chain.delegate_pda(
        derived["escrow"], fixture_generator.NODE, fixture_generator.PROGRAM)[0]
    account = policy_module.AccountView(
        delegate, pay_chain.SYSTEM_PROGRAM_ID, lamports, b"")
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["delegate_rent_shortfall_lamports"] == expected_shortfall


def test_initialized_delegate_has_zero_rent_shortfall():
    raw, view = fixture_generator.golden_messages()["self_delegate"]
    derived = pay_chain._derived(fixture_generator.WALLET, fixture_generator.PROGRAM)
    delegate = pay_chain.delegate_pda(
        derived["escrow"], fixture_generator.NODE, fixture_generator.PROGRAM)[0]
    account = policy_module.AccountView(
        delegate, fixture_generator.PROGRAM, 1,
        fixture_generator._delegate_data(
            fixture_generator.NODE, flags=pay_chain.DELEGATE_FLAGS_ALL))
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["delegate_rent_shortfall_lamports"] == 0


@pytest.mark.parametrize(
    "lamports,expected_shortfall",
    [(0, 2_039_280), (1_000_000, 1_039_280), (2_039_280, 0), (9_000_000, 0)],
)
def test_dusted_ata_credits_prefunding(lamports: int, expected_shortfall: int):
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    ata = pay_chain.associated_token_address(
        fixture_generator.WALLET, fixture_generator.MINT)[0]
    account = policy_module.AccountView(
        ata, pay_chain.SYSTEM_PROGRAM_ID, lamports, b"")
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["ata_rent_shortfall_lamports"] == expected_shortfall


def test_initialized_canonical_ata_has_zero_rent_shortfall():
    raw, view = fixture_generator.golden_messages()["withdraw_execute"]
    ata = pay_chain.associated_token_address(
        fixture_generator.WALLET, fixture_generator.MINT)[0]
    account = policy_module.AccountView(
        ata, pay_chain.TOKEN_PROGRAM_ID, 1,
        fixture_generator._ata_data())
    plan = _decode(raw, fixture_generator._replace_account(view, account))
    fields = {field.name: field.value for field in plan.rendered_fields}
    assert fields["ata_rent_shortfall_lamports"] == 0


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
            spent=spent, disabled=disabled))
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
    missing = replace(view, accounts=())
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
