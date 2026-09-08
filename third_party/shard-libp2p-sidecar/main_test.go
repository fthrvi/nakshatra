package main

import (
	"context"
	"errors"
	"testing"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

func TestOpenActivationStreamAllowsLimitedRelay(t *testing.T) {
	wantPeer := peer.ID("relay-target")
	wantErr := errors.New("sentinel")
	called := false

	stream, err := openActivationStream(
		context.Background(),
		wantPeer,
		func(ctx context.Context, gotPeer peer.ID, protocols ...protocol.ID) (network.Stream, error) {
			called = true
			if gotPeer != wantPeer {
				t.Fatalf("peer = %q, want %q", gotPeer, wantPeer)
			}
			if len(protocols) != 1 || protocols[0] != protocol.ID(activationProto) {
				t.Fatalf("protocols = %q, want [%q]", protocols, activationProto)
			}
			allowed, reason := network.GetAllowLimitedConn(ctx)
			if !allowed {
				t.Fatal("activation stream did not allow a limited relay connection")
			}
			if reason != activationRelayReason {
				t.Fatalf("limited-connection reason = %q, want %q", reason, activationRelayReason)
			}
			return nil, wantErr
		},
	)

	if !called {
		t.Fatal("stream opener was not called")
	}
	if stream != nil {
		t.Fatalf("stream = %v, want nil", stream)
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}
