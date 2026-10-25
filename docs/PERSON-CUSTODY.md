# Person custody selection

Nakshatra pins `.nakshatra/net/person-custody` at the start of each human act and re-reads it at
every signing boundary. If it changes, the act is refused before a file key is loaded or a broker
sign request is sent. The marker contains either `file` or `broker` and must be a regular,
effective-user-owned file that is not group/world writable. Broker mode also requires an equally
protected `person-broker.json` and a regular effective-user-owned `0600` bearer-token file. Broker
HTTP redirects are refused.

For backward compatibility, a missing custody marker means `file`. This does not restore or
regenerate authority: after a broker import removes `.sthambha/person-TEST.key`, the missing-marker
fallback fails closed because file custody has no private key with which to sign. Restore the
correct broker marker/configuration (or perform an explicit authority recovery) instead of
creating a replacement key.

Short-lived person proofs use a 90-second broker TTL/approval deadline: nakd accepts them for 120
seconds and the client reserves 30 seconds for signing verification and local delivery. After the
broker returns signatures, the client checks the embedded timestamp again and refuses before any
local write or nakd/network send if the proof is already stale.

Worker and consumer invites retain the established onboarding order: the onboarding server stages
and returns its one-time code first, then that exact code is included in the person-signed invite.
If approval or any later step fails, Nakshatra uses the existing server cancel route and closes any
possibly registered local invite. The onboarding server may retain a durable cancelled/audit row;
it must not leave the stage redeemable. A locally cancelled invite may likewise remain as consumed
audit evidence but is not returned by the open-invites path.
