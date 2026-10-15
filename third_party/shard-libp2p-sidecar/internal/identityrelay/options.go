package identityrelay

import (
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// Admission is an identity-bearing relay lifecycle hook. Every callback for a
// circuit carries the same opaque attempt token.
type Admission interface {
	AllowReserve(peer.ID, multiaddr.Multiaddr) bool
	AdmitConnect(src peer.ID, srcAddr multiaddr.Multiaddr, dest peer.ID) (uint64, bool)
	ConnectEstablished(uint64)
	ConnectFailed(uint64)
	ConnectClosed(uint64, time.Duration)
}

type Option func(*Relay) error

// WithResources is a Relay option that sets specific relay resources for the relay.
func WithResources(rc Resources) Option {
	return func(r *Relay) error {
		r.rc = rc
		return nil
	}
}

// WithLimit is a Relay option that sets only the relayed connection limits for the relay.
func WithLimit(limit *RelayLimit) Option {
	return func(r *Relay) error {
		r.rc.Limit = limit
		return nil
	}
}

// Reservation address function used to promote addresses to connected nodes
type ReservationAddressFilterFunc func(addr multiaddr.Multiaddr) (include bool)

// Overrides the default reservation address filter.
// This will permit the relay let the client know it have access to non public addresses too.
func WithReservationAddressFilter(filter ReservationAddressFilterFunc) (option Option) {
	return func(r *Relay) (err error) {
		r.reservationAddrFilter = filter
		return nil
	}
}

// WithInfiniteLimits is a Relay option that disables limits.
func WithInfiniteLimits() Option {
	return func(r *Relay) error {
		r.rc.Limit = nil
		return nil
	}
}

// WithACL is a Relay option that supplies an ACLFilter for access control.
func WithACL(acl ACLFilter) Option {
	return func(r *Relay) error {
		r.acl = acl
		return nil
	}
}

// WithAdmission installs identity-bound circuit admission and accounting.
func WithAdmission(admission Admission) Option {
	return func(r *Relay) error {
		r.admission = admission
		return nil
	}
}

// WithMetricsTracer is a Relay option that supplies a MetricsTracer for metrics
func WithMetricsTracer(mt MetricsTracer) Option {
	return func(r *Relay) error {
		r.metricsTracer = mt
		return nil
	}
}
