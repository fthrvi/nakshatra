This package is a narrow fork of
`github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay` at v0.48.0.

Nakshatra adds `WithAdmission`, whose lifecycle callbacks carry one stable
attempt token. The upstream metrics interface deliberately has no circuit
identity and therefore cannot safely drive security accounting under
concurrency. All circuit-relay protocol behavior remains upstream-derived.
