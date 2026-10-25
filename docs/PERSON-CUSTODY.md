# Person custody selection

Nakshatra reads `.nakshatra/net/person-custody` once at the start of each human act and carries
that decision through preflight and signing. The marker contains either `file` or `broker` and
must be a regular, effective-user-owned file that is not group/world writable. Broker mode also
requires an equally protected `person-broker.json` and a regular effective-user-owned `0600`
bearer-token file. Broker HTTP redirects are refused.

For backward compatibility, a missing custody marker means `file`. This does not restore or
regenerate authority: after a broker import removes `.sthambha/person-TEST.key`, the missing-marker
fallback fails closed because file custody has no private key with which to sign. Restore the
correct broker marker/configuration (or perform an explicit authority recovery) instead of
creating a replacement key.

Short-lived person proofs use a 90-second broker TTL/approval deadline: nakd accepts them for 120
seconds and the client reserves 30 seconds for signing verification and local delivery. After the
broker returns signatures, the client checks the embedded timestamp again and refuses before any
local write or nakd/network send if the proof is already stale.
