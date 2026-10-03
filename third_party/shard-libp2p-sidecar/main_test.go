package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
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

func TestDialListenMustBeLoopback(t *testing.T) {
	for _, good := range []string{"127.0.0.1:51831", "[::1]:51831"} {
		if err := validateDialListen(good); err != nil {
			t.Fatalf("%s: %v", good, err)
		}
	}
	for _, bad := range []string{"0.0.0.0:51831", "[::]:51831", "192.168.1.2:51831", "localhost:51831", "51831"} {
		if err := validateDialListen(bad); err == nil {
			t.Fatalf("accepted non-literal/non-loopback bind %q", bad)
		}
	}
}

func TestPeerIDFixedVector(t *testing.T) {
	pub, err := crypto.UnmarshalPublicKey(append([]byte{0x08, 0x01, 0x12, 0x20}, []byte{
		0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
		0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a,
	}...))
	if err != nil {
		t.Fatal(err)
	}
	id, err := peer.IDFromPublicKey(pub)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := id.String(), "12D3KooWQK1wnefoLrcVHbbnf5tLzbopUd3K3bFAoJpA7YJgL5pV"; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

func TestDialProtocolRejectsBadAndOverlongLines(t *testing.T) {
	for _, line := range []string{"DIAL not-a-peer-id\n", "NOPE x\n", "DIAL " + strings.Repeat("x", maxDialLine) + "\n"} {
		server, client := net.Pipe()
		go handleDialConn(nil, server, nil, time.Second)
		go func() { _, _ = io.WriteString(client, line) }()
		got, err := bufio.NewReader(client).ReadString('\n')
		client.Close()
		if err != nil || !strings.HasPrefix(got, "ERR ") {
			t.Fatalf("line %q: got %q, %v", line[:min(len(line), 40)], got, err)
		}
	}
}

func TestRelayCircuitAddressesAreAddedForPeerOnlyDial(t *testing.T) {
	privA, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	privRelay, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	privTarget, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	a, err := newHost(privA, "/ip4/127.0.0.1/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	relay, err := newHost(privRelay, "/ip4/127.0.0.1/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer relay.Close()
	target, err := peer.IDFromPrivateKey(privTarget)
	if err != nil {
		t.Fatal(err)
	}
	addRelayCircuitAddrs(a, target, []peer.AddrInfo{{ID: relay.ID(), Addrs: relay.Addrs()}})
	addrs := a.Peerstore().Addrs(target)
	if len(addrs) == 0 || !strings.Contains(addrs[0].String(), "/p2p/"+relay.ID().String()+"/p2p-circuit") {
		t.Fatalf("missing relay circuit address for %s: %v", target, addrs)
	}
}

func TestDialListenerDirectRoundTrip(t *testing.T) {
	privA, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	privB, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	a, err := newHost(privA, "/ip4/127.0.0.1/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := newHost(privB, "/ip4/127.0.0.1/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	a.Peerstore().AddAddrs(b.ID(), b.Addrs(), peerstore.PermanentAddrTTL)

	echo, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer echo.Close()
	go func() {
		for {
			c, err := echo.Accept()
			if err != nil {
				return
			}
			go func() { defer c.Close(); _, _ = io.Copy(c, c) }()
		}
	}()
	runInbound(b, echo.Addr().String())

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go serveDialListener(a, ln, nil, 5*time.Second)
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := fmt.Fprintf(c, "DIAL %s\n", b.ID()); err != nil {
		t.Fatal(err)
	}
	r := bufio.NewReader(c)
	if line, err := r.ReadString('\n'); err != nil || line != "OK direct\n" {
		t.Fatalf("reply %q, %v", line, err)
	}
	want := []byte("nakshatra over a direct libp2p stream")
	if _, err := c.Write(want); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(want))
	if _, err := io.ReadFull(r, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("round trip: got %q, want %q", got, want)
	}
	if directConn(a, b.ID()) == nil {
		t.Fatal("the stream was not carried by a direct connection")
	}
}
