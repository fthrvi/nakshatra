"""A milestone guard: pins that the REAL `--code` dispatcher cannot pass preflight today.

⚠️⚠️ WHY THIS TEST EXISTS. The scoping review of `scripts/join/` (2026-09-13, full analysis
in nakshatra memory `reference_join_package_scoping_and_verdict.md`) found the package
overclaimed a working onboarding flow: `admit` can be satisfied (see
`test_join_production_path.py`), but nothing past it can, because no code in this package
produces the box/package facts phases 2+ require. This test drives the REAL dispatcher
(`join.observe.observe`, not a fixture) through `admit` then `preflight`, stubbing only the
one network call (`act.fetch_join_info`, same style as `test_join_production_path.py`), and
asserts `preflight` fails with EXACTLY the 7 "not observed" problems that fact-gap produces
today.

⚠️⚠️ THIS TEST IS EXPECTED TO BREAK THE DAY SOMEONE IMPLEMENTS REAL PREFLIGHT OBSERVATION
(disk-slice sizing, port planning, VRAM/layer/ctx capacity facts). Breaking it is a GOOD SIGN,
not a regression — it means the gap this test pins has actually been closed. Per this repo's
own guidance on self-certifying checks (a check must be keyed on the substance it certifies,
not survive past it hollow), whoever wires real preflight facts must consciously update or
delete this test rather than have it silently keep "passing" for the wrong reason. Do not
loosen the assertion (e.g. to "at least these problems") to make it pass again without
addressing why it broke — read the diff that broke it first.
"""
from __future__ import annotations

from join import act, observe as obs
from join.phase_admit import admit
from join.phase_preflight import preflight
from joincode import encode_join

EXPECTED_NOT_OBSERVED = {
    "slice_bytes", "port", "vram_bytes", "n_layers",
    "model_bytes_per_layer", "ctx_tokens", "kv_bytes_per_token",
}


def test_real_dispatcher_admits_then_fails_preflight_on_missing_facts(monkeypatch):
    # Only the network call is stubbed — everything else (join-code decode, admit's own
    # decision, preflight's real local observation via nvidia-smi/statvfs/ss, and preflight's
    # own decision) runs for real, exactly as it would on a real machine.
    monkeypatch.setattr(act, "fetch_join_info", lambda coordinator, timeout=15: {
        "package_url": "https://cdn.example.com/m.gguf",
        "coordinator_version": "1.1.0",
        "node_id": "n-test",
    })

    code = encode_join("https://hub.example", "tok", expires_at=0)
    facts = {"join_code": code}

    # Phase 1: admit — this DOES pass on the real path (test_join_production_path.py already
    # pins this). Confirmed here too, so a future admit regression fails loudly at the right
    # phase instead of being misread as this test's gap widening.
    facts.update(obs.observe("admit", facts))
    ok, why, upd = admit(facts)
    assert ok, f"admit unexpectedly failed: {why}"
    facts.update(upd)

    # Phase 2: preflight — the real dispatcher gathers accel/free_bytes/listeners from the
    # actual machine (no stub needed, no GPU/network required for these three), but nothing
    # in this package ever produces slice_bytes, port, vram_bytes, n_layers,
    # model_bytes_per_layer, ctx_tokens or kv_bytes_per_token. This IS the gap.
    facts.update(obs.observe("preflight", facts))
    ok, problems, _ = preflight(facts)

    assert not ok, (
        "preflight PASSED on the real dispatcher — if real observation facts were wired up, "
        "this test is now measuring something real and should be updated/removed, not "
        "loosened. See the module docstring."
    )

    not_observed = {
        p[: -len(" not observed")] for p in problems if p.endswith(" not observed")
    }
    assert not_observed == EXPECTED_NOT_OBSERVED, (
        f"the set of unobserved preflight facts changed: {not_observed!r} vs expected "
        f"{EXPECTED_NOT_OBSERVED!r} — someone touched observe('preflight') or "
        f"phase_preflight's NEEDED tuple. Read what changed before updating this test: it "
        f"exists to make exactly that change loud."
    )
