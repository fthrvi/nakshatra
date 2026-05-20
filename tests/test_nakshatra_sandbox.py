"""Tests for scripts/nakshatra_sandbox.py — runtime introspection +
SandboxSpec compliance validator (Phase G of the worker hardening
sprint, 2026-05-19).

Tests pure-Python logic only; the actual /proc + /sys/fs/cgroup reads
happen at collect_runtime_facts() boundaries that we don't exercise
here (those are platform-dependent and CI would need a Linux container
to test meaningfully).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_sandbox as sb  # noqa: E402


# ── RuntimeFacts: dataclass shape ────────────────────────────────────

def test_runtime_facts_default_all_unknown():
    f = sb.RuntimeFacts()
    assert f.in_container is None
    assert f.cpu_threads_limit is None
    assert f.ram_limit_gb is None
    assert f.can_ptrace is None


def test_collect_runtime_facts_returns_facts():
    """Smoke check that collect_runtime_facts() doesn't crash on the
    test host. The actual values depend on the platform — non-Linux
    hosts get the "non-Linux host" note."""
    facts = sb.collect_runtime_facts()
    assert isinstance(facts, sb.RuntimeFacts)
    assert facts.os_kernel  # non-empty


# ── ComplianceReport helpers ─────────────────────────────────────────

def test_compliance_report_empty_is_compliant():
    r = sb.SandboxComplianceReport()
    assert r.is_fully_compliant is True
    assert r.is_mode_c_compliant() is True


def test_compliance_report_with_non_compliance():
    r = sb.SandboxComplianceReport()
    r.checks.append(sb.CheckResult(
        "in_container", "true", "false", "non_compliant"))
    assert r.has_non_compliance is True
    assert r.is_fully_compliant is False
    assert r.is_mode_c_compliant() is False


def test_compliance_report_with_unknown_only_is_compliant():
    """Unknown checks (non-Linux host) don't fail is_mode_c_compliant
    on their own — but is_fully_compliant requires *all* compliant."""
    r = sb.SandboxComplianceReport()
    r.checks.append(sb.CheckResult(
        "in_container", "true", "unknown", "unknown"))
    assert r.has_non_compliance is False
    assert r.is_fully_compliant is False  # unknown ≠ compliant
    assert r.is_mode_c_compliant() is True  # but it doesn't block Mode C


def test_compliance_report_spec_says_not_mode_c_compatible():
    """mode_c_compatible=False in the SandboxSpec (e.g. mps_cooperative
    peer) → is_mode_c_compliant returns False regardless of runtime."""
    r = sb.SandboxComplianceReport(mode_c_compatible_per_spec=False)
    assert r.is_mode_c_compliant() is False


# ── validate_against_runtime: container presence ─────────────────────

def _fake_facts(**overrides) -> sb.RuntimeFacts:
    f = sb.RuntimeFacts(os_kernel="Linux")
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


def test_validate_runtime_in_container_compliant():
    spec = {"mode_c_compatible": True}
    facts = _fake_facts(in_container=True, container_runtime="docker")
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["in_container"].status == "compliant"


def test_validate_runtime_in_container_non_compliant():
    spec = {"mode_c_compatible": True}
    facts = _fake_facts(in_container=False)
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["in_container"].status == "non_compliant"
    assert r.is_mode_c_compliant() is False


def test_validate_runtime_in_container_unknown_on_non_linux():
    spec = {"mode_c_compatible": True}
    facts = _fake_facts(in_container=None)  # non-Linux signal
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["in_container"].status == "unknown"


# ── CPU + RAM limits ─────────────────────────────────────────────────

def test_validate_cpu_within_limit_compliant():
    spec = {"cpu_threads_limit": 8, "mode_c_compatible": True}
    facts = _fake_facts(in_container=True, cpu_threads_limit=8)
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["cpu_threads_limit"].status == "compliant"


def test_validate_cpu_over_limit_non_compliant():
    spec = {"cpu_threads_limit": 4, "mode_c_compatible": True}
    facts = _fake_facts(in_container=True, cpu_threads_limit=16)
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["cpu_threads_limit"].status == "non_compliant"


def test_validate_ram_within_limit_compliant():
    spec = {"ram_limit_gb": 32.0, "mode_c_compatible": True}
    facts = _fake_facts(in_container=True, ram_limit_gb=32.0)
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["ram_limit_gb"].status == "compliant"


def test_validate_ram_over_limit_non_compliant():
    spec = {"ram_limit_gb": 16.0, "mode_c_compatible": True}
    facts = _fake_facts(in_container=True, ram_limit_gb=128.0)
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["ram_limit_gb"].status == "non_compliant"


# ── seccomp ──────────────────────────────────────────────────────────

def test_validate_seccomp_filter_mode_compliant():
    spec = {"seccomp_profile": "fabric-default-v1", "mode_c_compatible": True}
    facts = _fake_facts(in_container=True, seccomp_mode="filter")
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["seccomp_profile"].status == "compliant"


def test_validate_seccomp_disabled_non_compliant():
    spec = {"seccomp_profile": "fabric-default-v1", "mode_c_compatible": True}
    facts = _fake_facts(in_container=True, seccomp_mode="disabled")
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["seccomp_profile"].status == "non_compliant"


def test_validate_seccomp_unset_in_spec_passes():
    spec = {"mode_c_compatible": True}  # no seccomp_profile field
    facts = _fake_facts(in_container=True, seccomp_mode="disabled")
    r = sb.validate_against_runtime(spec, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["seccomp_profile"].status == "compliant"


# ── CAP_SYS_PTRACE ───────────────────────────────────────────────────

def test_validate_cap_ptrace_denied_compliant():
    facts = _fake_facts(in_container=True, can_ptrace=False)
    r = sb.validate_against_runtime({}, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["cap_sys_ptrace"].status == "compliant"


def test_validate_cap_ptrace_permitted_non_compliant():
    facts = _fake_facts(in_container=True, can_ptrace=True)
    r = sb.validate_against_runtime({}, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["cap_sys_ptrace"].status == "non_compliant"


# ── Network namespace ────────────────────────────────────────────────

def test_validate_network_namespace_isolated_compliant():
    facts = _fake_facts(in_container=True, network_namespaced=True)
    r = sb.validate_against_runtime({}, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["network_namespace"].status == "compliant"


def test_validate_network_namespace_shared_non_compliant():
    facts = _fake_facts(in_container=True, network_namespaced=False)
    r = sb.validate_against_runtime({}, facts)
    by_field = {c.field_name: c for c in r.checks}
    assert by_field["network_namespace"].status == "non_compliant"


# ── compliance_summary_for_peer_body ─────────────────────────────────

def test_compliance_summary_shape():
    r = sb.SandboxComplianceReport()
    r.checks.append(sb.CheckResult(
        "in_container", "true", "false", "non_compliant"))
    r.checks.append(sb.CheckResult(
        "cpu_threads_limit", "8", "8", "compliant"))
    summary = sb.compliance_summary_for_peer_body(r)
    assert summary["is_mode_c_compliant"] is False
    assert summary["is_fully_compliant"] is False
    assert "in_container" in summary["non_compliant_fields"]
    assert "cpu_threads_limit" not in summary["non_compliant_fields"]


def test_full_compliant_pipeline_smoke():
    """End-to-end: well-formed spec + compliant runtime → reports
    all-green and Mode-C compliant."""
    spec = {
        "seccomp_profile": "fabric-default-v1",
        "cpu_threads_limit": 8,
        "ram_limit_gb": 32.0,
        "mode_c_compatible": True,
    }
    facts = _fake_facts(
        in_container=True, container_runtime="docker",
        cpu_threads_limit=8, ram_limit_gb=32.0,
        seccomp_mode="filter", can_ptrace=False, network_namespaced=True,
    )
    r = sb.validate_against_runtime(spec, facts)
    assert r.is_mode_c_compliant() is True
    assert r.is_fully_compliant is True


def test_format_human_includes_all_checks():
    """Smoke check on the human-readable report."""
    facts = _fake_facts(in_container=True, cpu_threads_limit=8,
                         seccomp_mode="filter", can_ptrace=False,
                         network_namespaced=True)
    r = sb.validate_against_runtime({}, facts)
    txt = r.format_human()
    assert "in_container" in txt
    assert "Mode-C compliant" in txt
