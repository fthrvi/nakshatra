# Operator junction — deploy runbook

The admission-gated multi-tenant operator junction (`junction.py`): a blind rendezvous relay
(`relay.py` `RendezvousRelay`, pairs by `rendezvous_id`, forwards ciphertext, **never IP-routes**) with
a **default-deny admission door** in front. This is what makes the Nakshatra network safe for a *second
operator* to connect at the web — and because it never IP-routes, the per-operator-overlay /
`10.42.0.0/24` address-collision problem (the audit 🔴) does not exist here at all.

## Live deployment (2026-06-15) — Vultr VPS `45.63.109.137`
Additive service; does NOT touch `wg-egress` / PNL / the home mesh.
```
/opt/nakshatra-junction/{junction.py, relay.py, admission.py, peers.tsv, models.tsv}
systemd: nakshatra-junction.service  (DynamicUser, ProtectSystem=strict, port 9778)
ufw: allow 9778/tcp
Environment: NAKSHATRA_ADMISSION_DIR=/opt/nakshatra-junction  NAKSHATRA_JUNCTION_PORT=9778
```
- **admission roster** = `/opt/nakshatra-junction/peers.tsv` (operator-curated; PUBLIC keys + tiers +
  tenant, no secrets). A peer NOT listed → **default-deny** (verified live: an unlisted operator's
  connection is closed, logged `JUNCTION deny: peer not in admission registry`).
- **verify ready:** `NAKSHATRA_JUNCTION_PORT=9778 python3 trisul/infra/nakshatra-junction/readiness_check.py`
  → all 6 invariants green = **READY for a 2nd operator**.

## How an operator connects (peer side)
```python
from transport import junction, admission   # admission.sign_admission to build the request
req = admission.sign_admission(my_priv_hex, model, operator="me")
sock = junction.connect_admitted("45.63.109.137", 9778, rendezvous_id, req)
# admitted → `sock` is paired to the partner; run the identity handshake + encrypted data plane over it.
# denied → the junction closes the socket (no channel).
```

## Onboard an operator
Add their Ed25519 pubkey to `peers.tsv` at a trust tier (`stranger`/`known`/`trusted`/`self`) + a tenant
tag, sync to the VPS, restart. The tier gates which models may tensor-split across them
(`admission.eligible_split_peers` — a peer that runs your layers sees your activations).

## Roll back
`systemctl disable --now nakshatra-junction && ufw delete allow 9778/tcp` — fully removes the door.
