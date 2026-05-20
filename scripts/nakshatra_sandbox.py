"""Worker-side sandbox compliance verifier (Phase G of the worker
hardening sprint, 2026-05-19).

Sthambha emits a ``SandboxSpec`` at ``/join`` per network-fabric.md §11.1
listing the OS sandbox the worker should be running inside (cgroup
limits, seccomp profile, egress allowlist, layer-cache mount). The full
multi-tenant defense expected workers to spawn their own containers,
which is a multi-week effort and a privileged code path.

This module takes a cleaner split:

  - **Operator** owns the orchestration: Docker / Podman / systemd-nspawn
    / k8s — anything that produces the right runtime environment. See
    ``docs/SANDBOX-EXAMPLES/`` for reference templates.
  - **Worker** verifies it is in fact inside that environment. Reads
    /proc + /sys/fs/cgroup, compares against the spec, refuses to serve
    Mode-C inference if not compliant.

The split moves seccomp-profile authoring, cgroup-write, iptables
egress, and root-level container spawn out of the worker. The worker
stays a regular Python process — read-only introspection only.

For non-Linux hosts (Mac Silicon, Windows), this module reports
``unknown`` for everything; the worker logs and continues if Mode A/B,
refuses if Mode C — operators on those platforms must run the worker
inside a Linux VM / Docker Desktop to get Mode-C compliance.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── RuntimeFacts: read-only introspection of the worker's environment ─

@dataclass
class RuntimeFacts:
    """A snapshot of what the worker can determine about its own
    runtime sandbox. Every field is best-effort — None means "couldn't
    determine on this platform" (the validator treats that as ``unknown``
    rather than a violation)."""
    os_kernel: str = ""
    in_container: Optional[bool] = None
    container_runtime: str = ""           # docker / podman / nspawn / k8s / ""
    cpu_threads_limit: Optional[int] = None    # cgroup cpu quota → effective threads
    ram_limit_gb: Optional[float] = None       # cgroup memory.max
    seccomp_mode: str = ""                # disabled / strict / filter / unknown
    can_ptrace: Optional[bool] = None     # CapSysPtrace from /proc/self/status
    network_namespaced: Optional[bool] = None  # heuristic on /proc/self/ns/net
    # Raw paths/values for the operator's troubleshooting.
    cgroup_v2_root: str = ""
    facts_collection_errors: list = field(default_factory=list)


def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.readline().strip()
    except OSError:
        return None


def _detect_container_runtime() -> tuple[Optional[bool], str]:
    """Returns (in_container, runtime_name). Multi-signal heuristic —
    no single test is reliable across Docker / Podman / systemd-nspawn
    / k8s."""
    # /.dockerenv is Docker's marker; Podman sometimes drops it too.
    if Path("/.dockerenv").exists():
        return True, "docker"
    # /run/.containerenv is Podman's modern marker.
    if Path("/run/.containerenv").exists():
        return True, "podman"
    # systemd-nspawn sets a specific env var.
    if os.environ.get("container") == "systemd-nspawn":
        return True, "systemd-nspawn"
    # Kubernetes injects this env var by convention.
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True, "kubernetes"
    # /proc/1/cgroup containing docker/kubepods/lxc/etc.
    cg = _read_first_line("/proc/1/cgroup")
    if cg:
        for marker, name in (("docker", "docker"), ("kubepods", "kubernetes"),
                              ("lxc", "lxc"), ("podman", "podman")):
            if marker in cg:
                return True, name
    return False, ""


def _read_cpu_limit() -> Optional[int]:
    """Effective CPU thread budget from cgroup v2 cpu.max, or v1 cpu.cfs_*.
    Returns None when no limit is set or the cgroup files aren't
    readable. On a host with cpu.max=200000 100000 (= 2 cores), returns 2."""
    # cgroup v2
    raw = _read_first_line("/sys/fs/cgroup/cpu.max")
    if raw and raw != "max ...":
        parts = raw.split()
        if len(parts) >= 2 and parts[0] != "max":
            try:
                quota = int(parts[0]); period = int(parts[1])
                if period > 0:
                    return max(1, quota // period)
            except ValueError:
                pass
    # cgroup v1
    quota = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period:
        try:
            q = int(quota); p = int(period)
            if q > 0 and p > 0:
                return max(1, q // p)
        except ValueError:
            pass
    return None


def _read_ram_limit_gb() -> Optional[float]:
    """RAM cap from cgroup v2 memory.max or v1 memory.limit_in_bytes."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read_first_line(path)
        if raw and raw != "max":
            try:
                n = int(raw)
                # Linux often returns absurdly large limit_in_bytes (~2^63)
                # for "no limit"; treat anything > 1 EiB as no-limit.
                if n > 0 and n < (1 << 60):
                    return n / (1024 ** 3)
            except ValueError:
                pass
    return None


def _read_seccomp_mode() -> str:
    """/proc/self/status Seccomp: line. 0=disabled, 1=strict, 2=filter."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("Seccomp:"):
                    val = line.split()[1]
                    return {"0": "disabled", "1": "strict",
                            "2": "filter"}.get(val, "unknown")
    except OSError:
        pass
    return "unknown"


def _check_ptrace_capability() -> Optional[bool]:
    """Inspect CapEff (effective capabilities) for CAP_SYS_PTRACE.
    Returns True iff ptrace is permitted in the current namespace."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    cap = int(line.split()[1], 16)
                    # CAP_SYS_PTRACE = bit 19
                    return bool(cap & (1 << 19))
    except OSError:
        pass
    return None


def _check_network_namespace() -> Optional[bool]:
    """Compare /proc/self/ns/net inode to /proc/1/ns/net. If different,
    we're in a separate network namespace from PID 1 — strong signal
    of a container-managed netns."""
    try:
        self_ns = os.readlink("/proc/self/ns/net")
        init_ns = os.readlink("/proc/1/ns/net")
        return self_ns != init_ns
    except OSError:
        return None


def collect_runtime_facts() -> RuntimeFacts:
    """One-shot snapshot. Cheap (a handful of /proc reads). Call at
    startup; cache the result for the lifetime of the worker."""
    facts = RuntimeFacts(os_kernel=platform.system())
    if platform.system() != "Linux":
        # Non-Linux: containers / cgroups / seccomp are different stories
        # (Docker Desktop runs a Linux VM you can introspect from inside
        # if the worker is inside that VM, but not from the macOS host).
        facts.facts_collection_errors.append(
            f"non-Linux host ({platform.system()}); cgroup/seccomp "
            f"introspection not available — run the worker inside a Linux "
            f"container/VM for Mode C"
        )
        return facts
    try:
        in_c, runtime = _detect_container_runtime()
        facts.in_container = in_c
        facts.container_runtime = runtime
    except Exception as e:
        facts.facts_collection_errors.append(f"container detect: {e}")
    try:
        facts.cpu_threads_limit = _read_cpu_limit()
    except Exception as e:
        facts.facts_collection_errors.append(f"cpu limit: {e}")
    try:
        facts.ram_limit_gb = _read_ram_limit_gb()
    except Exception as e:
        facts.facts_collection_errors.append(f"ram limit: {e}")
    try:
        facts.seccomp_mode = _read_seccomp_mode()
    except Exception as e:
        facts.facts_collection_errors.append(f"seccomp: {e}")
    try:
        cap = _check_ptrace_capability()
        facts.can_ptrace = cap
    except Exception as e:
        facts.facts_collection_errors.append(f"capability: {e}")
    try:
        facts.network_namespaced = _check_network_namespace()
    except Exception as e:
        facts.facts_collection_errors.append(f"netns: {e}")
    return facts


# ── SandboxSpec parsing + ComplianceReport ───────────────────────────

@dataclass
class CheckResult:
    """One row of the compliance table: which spec field, what the
    operator should expect, what we actually observed, and whether
    that's good enough."""
    field_name: str
    expected: str
    actual: str
    status: str  # "compliant" | "non_compliant" | "unknown"
    detail: str = ""


@dataclass
class SandboxComplianceReport:
    """A structured verdict per SandboxSpec field. The worker uses the
    summary helpers (is_compliant, is_mode_c_compliant) to gate startup
    in Mode C. Operators read the full table to debug their orchestration."""
    checks: list[CheckResult] = field(default_factory=list)
    mode_c_compatible_per_spec: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def is_fully_compliant(self) -> bool:
        return all(c.status == "compliant" for c in self.checks)

    @property
    def has_non_compliance(self) -> bool:
        return any(c.status == "non_compliant" for c in self.checks)

    def is_mode_c_compliant(self) -> bool:
        """The Mode-C go/no-go gate. Failure modes that block Mode-C
        inference: any non_compliant row, OR the spec itself declares
        mode_c_compatible=False (mps_cooperative case from §11.2)."""
        return self.mode_c_compatible_per_spec and not self.has_non_compliance

    def to_dict(self) -> dict:
        return {
            "is_fully_compliant": self.is_fully_compliant,
            "is_mode_c_compliant": self.is_mode_c_compliant(),
            "mode_c_compatible_per_spec": self.mode_c_compatible_per_spec,
            "checks": [
                {"field": c.field_name, "expected": c.expected,
                 "actual": c.actual, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
            "notes": list(self.notes),
        }

    def format_human(self) -> str:
        lines = ["Sandbox compliance report:"]
        for c in self.checks:
            tag = {"compliant": "  ✓", "non_compliant": "  ✗",
                   "unknown": "  ?"}[c.status]
            lines.append(f"{tag} {c.field_name}: expected={c.expected!r} "
                         f"actual={c.actual!r}"
                         + (f" — {c.detail}" if c.detail else ""))
        lines.append(
            f"Mode-C compliant: {self.is_mode_c_compliant()} "
            f"(spec mode_c_compatible={self.mode_c_compatible_per_spec})"
        )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def validate_against_runtime(spec: dict,
                              facts: Optional[RuntimeFacts] = None,
                              tolerance: float = 0.10
                              ) -> SandboxComplianceReport:
    """Check a SandboxSpec dict against the worker's RuntimeFacts.

    ``spec`` is the ``sandbox`` block from a /join response; see
    sthambha/core.py SandboxSpec for the shape.

    ``tolerance`` (default 10%) is how much slack we give numeric
    bounds — a worker that has 7.5 cores when the spec asked for 8 is
    compliant; 4 cores when 8 was asked is not.

    Returns a report; never raises. Worker decides how to act on it.
    """
    if facts is None:
        facts = collect_runtime_facts()
    report = SandboxComplianceReport()
    report.mode_c_compatible_per_spec = bool(
        spec.get("mode_c_compatible", True))

    # Container presence — the node-pila check.
    expected_in_container = True
    actual_in_container = facts.in_container
    if actual_in_container is True:
        report.checks.append(CheckResult(
            "in_container", "true",
            f"true ({facts.container_runtime or 'unknown runtime'})",
            "compliant",
        ))
    elif actual_in_container is False:
        report.checks.append(CheckResult(
            "in_container", "true", "false",
            "non_compliant",
            detail=("worker runs on the bare host; container orchestration "
                    "is required for Mode C (see docs/SANDBOX-EXAMPLES/)"),
        ))
    else:
        report.checks.append(CheckResult(
            "in_container", "true", "unknown",
            "unknown",
            detail=("non-Linux host or /proc unreadable; cannot determine "
                    "container status from here"),
        ))

    # CPU threads — within tolerance below the cap is fine; way under
    # is fine too (operator gave us less than the budget). Over the
    # spec's limit means the cgroup didn't constrain us.
    spec_cpu = int(spec.get("cpu_threads_limit", 0) or 0)
    actual_cpu = facts.cpu_threads_limit
    if spec_cpu <= 0:
        report.checks.append(CheckResult(
            "cpu_threads_limit", "<unset>", "<unset>", "compliant",
            detail="spec did not declare a CPU limit",
        ))
    elif actual_cpu is None:
        report.checks.append(CheckResult(
            "cpu_threads_limit", str(spec_cpu), "unknown",
            "unknown",
            detail="cgroup cpu.max not readable",
        ))
    else:
        upper = spec_cpu * (1.0 + tolerance)
        if actual_cpu <= upper:
            report.checks.append(CheckResult(
                "cpu_threads_limit", str(spec_cpu), str(actual_cpu),
                "compliant",
            ))
        else:
            report.checks.append(CheckResult(
                "cpu_threads_limit", f"≤{spec_cpu}", str(actual_cpu),
                "non_compliant",
                detail=(f"runtime sees {actual_cpu} CPU threads available "
                        f"but spec capped at {spec_cpu}; "
                        f"cgroup limit not enforced"),
            ))

    # RAM — same shape as CPU.
    spec_ram = float(spec.get("ram_limit_gb", 0) or 0)
    actual_ram = facts.ram_limit_gb
    if spec_ram <= 0:
        report.checks.append(CheckResult(
            "ram_limit_gb", "<unset>", "<unset>", "compliant",
            detail="spec did not declare a RAM limit",
        ))
    elif actual_ram is None:
        report.checks.append(CheckResult(
            "ram_limit_gb", f"{spec_ram:.1f}", "unknown",
            "unknown", detail="cgroup memory.max not readable",
        ))
    else:
        upper = spec_ram * (1.0 + tolerance)
        if actual_ram <= upper:
            report.checks.append(CheckResult(
                "ram_limit_gb", f"{spec_ram:.1f}", f"{actual_ram:.1f}",
                "compliant",
            ))
        else:
            report.checks.append(CheckResult(
                "ram_limit_gb", f"≤{spec_ram:.1f}", f"{actual_ram:.1f}",
                "non_compliant",
                detail=("cgroup memory.max not enforcing the spec's cap"),
            ))

    # seccomp — spec asks for a profile name. We can't tell which
    # *profile* is loaded, only that filter mode is on. Mode "filter"
    # is compliant; "strict" is overly restrictive but acceptable;
    # "disabled" is not compliant for Mode C.
    spec_seccomp = str(spec.get("seccomp_profile", "") or "")
    actual_seccomp = facts.seccomp_mode
    if not spec_seccomp:
        report.checks.append(CheckResult(
            "seccomp_profile", "<unset>", "<unset>", "compliant",
        ))
    elif actual_seccomp == "filter":
        report.checks.append(CheckResult(
            "seccomp_profile", spec_seccomp,
            f"filter (specific profile not verifiable from userspace)",
            "compliant",
            detail=("worker is running under a seccomp filter; can't tell "
                    "which profile, but the filter exists"),
        ))
    elif actual_seccomp == "strict":
        report.checks.append(CheckResult(
            "seccomp_profile", spec_seccomp, "strict",
            "compliant",
            detail="strict mode is even tighter than the requested filter",
        ))
    elif actual_seccomp == "disabled":
        report.checks.append(CheckResult(
            "seccomp_profile", spec_seccomp, "disabled",
            "non_compliant",
            detail=("worker process has no seccomp filter — ptrace + "
                    "process_vm_readv not blocked"),
        ))
    else:
        report.checks.append(CheckResult(
            "seccomp_profile", spec_seccomp, actual_seccomp,
            "unknown", detail="/proc/self/status Seccomp: unreadable",
        ))

    # Capability check — CAP_SYS_PTRACE is the single most dangerous
    # capability for a multi-tenant worker; explicit row.
    actual_ptrace = facts.can_ptrace
    if actual_ptrace is True:
        report.checks.append(CheckResult(
            "cap_sys_ptrace", "denied", "permitted",
            "non_compliant",
            detail="CAP_SYS_PTRACE in CapEff — worker can ptrace co-tenants",
        ))
    elif actual_ptrace is False:
        report.checks.append(CheckResult(
            "cap_sys_ptrace", "denied", "denied", "compliant",
        ))
    else:
        report.checks.append(CheckResult(
            "cap_sys_ptrace", "denied", "unknown", "unknown",
            detail="/proc/self/status CapEff: unreadable",
        ))

    # Network namespace — non-default netns means egress is being
    # filtered by the operator's orchestration layer. We don't
    # introspect the allowlist itself (would need root + iptables-save);
    # we just confirm the worker is in a non-default netns.
    actual_netns = facts.network_namespaced
    if actual_netns is True:
        report.checks.append(CheckResult(
            "network_namespace", "isolated", "isolated", "compliant",
        ))
    elif actual_netns is False:
        report.checks.append(CheckResult(
            "network_namespace", "isolated", "shared with PID 1",
            "non_compliant",
            detail=("worker shares PID 1's network namespace; egress "
                    "controls (allowed_egress) cannot be enforced from "
                    "inside the worker"),
        ))
    else:
        report.checks.append(CheckResult(
            "network_namespace", "isolated", "unknown", "unknown",
            detail="/proc/self/ns/net not readable",
        ))

    for err in facts.facts_collection_errors:
        report.notes.append(err)

    return report


# ── Compliance summary for the /peer body (G5 hook) ──────────────────

# Phase I8: soft remote attestation. Worker computes a hash of its
# observed runtime fingerprint (cgroup state, seccomp mode, capability
# set, namespace inodes, container marker) and signs H(nonce || fingerprint)
# with its Ed25519 key. The whole /peer body is already signed at the
# auth layer; this is an INNER attestation binding the worker's identity
# to its claimed runtime. Not TPM-grade — an attacker who patched the
# worker can lie about the fingerprint. The protocol shape is forward-
# compatible with hardware attestation (drop in a TPM-quote signer to
# replace the Ed25519 path).

def build_runtime_fingerprint_hash(facts: Optional[RuntimeFacts] = None) -> str:
    """Hash a stable representation of the worker's runtime. Same
    inputs produce the same hash, so operators can compare across
    reboots to detect drift (the attestation log on the pillar records
    each observation and audit-flags fingerprint changes)."""
    if facts is None:
        facts = collect_runtime_facts()
    canonical = "|".join((
        f"os={facts.os_kernel}",
        f"in_container={facts.in_container}",
        f"runtime={facts.container_runtime}",
        f"cpu_limit={facts.cpu_threads_limit}",
        f"ram_limit_gb={facts.ram_limit_gb}",
        f"seccomp={facts.seccomp_mode}",
        f"ptrace={facts.can_ptrace}",
        f"netns={facts.network_namespaced}",
    ))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Phase J2: attestation blob shape version. Bump when adding fields the
# pillar can't safely ignore (e.g. a future TEE-quote bytes alongside
# the fingerprint hash). Pillar accepts known versions and audits
# unknown versions for visibility — see observe_attestation.
ATTESTATION_VERSION = 1


def build_attestation_blob(nonce_hex: str,
                            facts: Optional[RuntimeFacts] = None
                            ) -> dict:
    """The ``attestation`` field the worker includes in /peer body.
    nonce_hex was issued by the pillar on the previous /peer response;
    on first contact it's empty (the pillar accepts an empty nonce as
    "no attestation yet" and returns a fresh nonce the worker uses on
    the next heartbeat).

    Phase J2: ``attestation_version`` lets the pillar tell post-J2
    workers apart from pre-J2 (legacy implicit-v0). Future TEE quote
    extensions bump the version."""
    return {
        "attestation_version": ATTESTATION_VERSION,
        "nonce_hex": nonce_hex or "",
        "fingerprint_hash": build_runtime_fingerprint_hash(facts),
    }


# Phase I5: report version on the wire shape. Bump when adding fields
# the pillar's Phase-G planner gate cannot ignore (the gate today reads
# is_mode_c_compliant only; any future filter additions go through a
# version-aware codepath). Workers always emit the current version;
# pillars accept any version they recognize and treat unknown versions
# as "report present but undecodable" → same effect as legacy empty.
REPORT_VERSION = 1


def compliance_summary_for_peer_body(report: SandboxComplianceReport) -> dict:
    """The compact summary the worker includes in /peer registration
    so the pillar can surface ``sandbox_compliance`` on PeerStatus.
    The planner's Mode-C filter refuses non-compliant peers.

    We deliberately don't include the full per-field detail in /peer
    bodies — that would balloon every heartbeat. Operators query the
    worker directly (or check its logs) for the verbose report.

    Phase I5: includes ``report_version: int`` so future shape evolution
    is detectable. Pillars treat unknown versions as legacy-equivalent.
    """
    non_compliant_fields = [
        c.field_name for c in report.checks if c.status == "non_compliant"
    ]
    return {
        "report_version": REPORT_VERSION,
        "is_mode_c_compliant": report.is_mode_c_compliant(),
        "is_fully_compliant": report.is_fully_compliant,
        "non_compliant_fields": non_compliant_fields,
    }
