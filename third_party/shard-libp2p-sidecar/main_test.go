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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	"github.com/libp2p/go-libp2p/core/protocol"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	"github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"
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

func TestNakdUsesDistinctDirectOnlyProtocol(t *testing.T) {
	if nakdProto == activationProto {
		t.Fatal("nakd must not share the relay-capable activation protocol")
	}
	if directPathAllowed(true, "/ip4/127.0.0.1/tcp/1") {
		t.Fatal("nakd accepted a limited relay connection")
	}
	if directPathAllowed(false, "/ip4/1.2.3.4/tcp/1/p2p-circuit") {
		t.Fatal("nakd accepted a circuit multiaddr")
	}
}

func TestDialSocketIsUserOnly(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nakshatra")
	path := filepath.Join(dir, "p2p.sock")
	ln, err := listenDialSocket(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	if got := mustMode(t, dir); got != 0o700 {
		t.Fatalf("directory mode = %o, want 700", got)
	}
	if got := mustMode(t, path); got != 0o600 {
		t.Fatalf("socket mode = %o, want 600", got)
	}
	if _, err := listenDialSocket("relative.sock"); err == nil {
		t.Fatal("accepted a relative socket path")
	}
	bad := filepath.Join(t.TempDir(), "not-a-socket")
	if err := os.WriteFile(bad, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := listenDialSocket(bad); err == nil {
		t.Fatal("replaced a non-socket path")
	}
}

func mustMode(t *testing.T, path string) os.FileMode {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Mode().Perm()
}

func TestNakdInboundRequiresDirectConnection(t *testing.T) {
	for _, tc := range []struct {
		limited bool
		addr    string
		want    bool
	}{{false, "/ip4/127.0.0.1/tcp/1", true},
		{true, "/ip4/127.0.0.1/tcp/1", false},
		{false, "/ip4/1.2.3.4/tcp/1/p2p-circuit", false}} {
		if got := directPathAllowed(tc.limited, tc.addr); got != tc.want {
			t.Fatalf("directPathAllowed(%v, %q) = %v, want %v", tc.limited, tc.addr, got, tc.want)
		}
	}
}

func TestWildcardIPv6ListenAddresses(t *testing.T) {
	got := wildcardIPv6ListenAddrs("/ip4/0.0.0.0/tcp/29600", true)
	want := []string{"/ip6/::/tcp/29600", "/ip6/::/udp/29600/quic-v1"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i].String() != want[i] {
			t.Fatalf("address %d = %s, want %s", i, got[i], want[i])
		}
	}
	if got := wildcardIPv6ListenAddrs("/ip4/127.0.0.1/tcp/29600", true); len(got) != 0 {
		t.Fatalf("loopback listener unexpectedly mirrored to IPv6: %v", got)
	}
}

func TestIPv6ListenFailureDoesNotAbortHost(t *testing.T) {
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	var attempted []string
	h, err := newHost(priv, "/ip4/0.0.0.0/tcp/0", natOpts{
		quic: true,
		ipv6Listen: func(addr multiaddr.Multiaddr) error {
			attempted = append(attempted, addr.String())
			return errors.New("IPv6 unavailable")
		},
	})
	if err != nil {
		t.Fatalf("IPv6 failure aborted IPv4 host startup: %v", err)
	}
	defer h.Close()
	if len(attempted) != 2 {
		t.Fatalf("IPv6 attempts = %v, want TCP and QUIC", attempted)
	}
}

func TestPreferredDirectDialAddresses(t *testing.T) {
	input := []multiaddr.Multiaddr{
		multiaddr.StringCast("/ip4/8.8.8.8/tcp/1"),
		multiaddr.StringCast("/ip6/2606:4700:4700::1111/tcp/1"),
		multiaddr.StringCast("/ip4/192.168.1.20/tcp/1"),
		multiaddr.StringCast("/ip6/fd00::20/tcp/1"),
		multiaddr.StringCast("/ip4/127.0.0.1/tcp/1"),
		multiaddr.StringCast("/ip4/169.254.1.2/tcp/1"),
		multiaddr.StringCast("/ip6/fe80::1/tcp/1"),
		multiaddr.StringCast("/ip4/1.2.3.4/tcp/1/p2p-circuit"),
	}
	got := preferredDirectDialAddrs(input)
	want := []string{
		"/ip4/192.168.1.20/tcp/1",
		"/ip6/fd00::20/tcp/1",
		"/ip6/2606:4700:4700::1111/tcp/1",
		"/ip4/8.8.8.8/tcp/1",
	}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i].String() != want[i] {
			t.Fatalf("address %d = %s, want %s (all: %v)", i, got[i], want[i], got)
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
	b, err := newHost(privB, "/ip4/0.0.0.0/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	bAddrs := preferredDirectDialAddrs(b.Addrs())
	if len(bAddrs) == 0 {
		t.Skip("test host has no non-loopback LAN address")
	}
	a.Peerstore().AddAddrs(b.ID(), bAddrs, peerstore.PermanentAddrTTL)

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
	runNakdInbound(b, echo.Addr().String())

	sockPath := filepath.Join(t.TempDir(), "nakshatra", "p2p.sock")
	ln, err := listenDialSocket(sockPath)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go serveDialListener(a, ln, nil, 5*time.Second)
	c, err := net.Dial("unix", sockPath)
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

func TestWaitDirectUsesAddressesLearnedThroughRelay(t *testing.T) {
	newKey := func() crypto.PrivKey {
		t.Helper()
		priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		return priv
	}
	relay, err := newHost(newKey(), "/ip4/0.0.0.0/tcp/0", natOpts{relayService: true})
	if err != nil {
		t.Fatal(err)
	}
	defer relay.Close()
	// Disable DCUtR in this test so only waitDirect's new Identify-driven force dial
	// can create the direct connection.
	a, err := newHost(newKey(), "/ip4/0.0.0.0/tcp/0", natOpts{disableHolePunching: true})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := newHost(newKey(), "/ip4/0.0.0.0/tcp/0", natOpts{disableHolePunching: true})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if len(preferredDirectDialAddrs(b.Addrs())) == 0 {
		t.Skip("test host has no non-loopback LAN address")
	}

	relayAddrs := preferredDirectDialAddrs(relay.Addrs())
	if len(relayAddrs) == 0 {
		t.Skip("test host has no non-loopback address for its in-process relay")
	}
	relayInfo := peer.AddrInfo{ID: relay.ID(), Addrs: relayAddrs[:1]}
	reserveCtx, reserveCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer reserveCancel()
	if err := b.Connect(reserveCtx, relayInfo); err != nil {
		t.Fatalf("B connect relay: %v", err)
	}
	if _, err := relayclient.Reserve(reserveCtx, b, relayInfo); err != nil {
		t.Fatalf("B reserve relay: %v", err)
	}
	if got := a.Peerstore().Addrs(b.ID()); len(got) != 0 {
		t.Fatalf("A unexpectedly knew B's direct addresses up front: %v", got)
	}

	directCtx, directCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer directCancel()
	if err := waitDirect(directCtx, a, b.ID(), []peer.AddrInfo{relayInfo}); err != nil {
		t.Fatal(err)
	}
	conn := directConn(a, b.ID())
	if conn == nil {
		t.Fatal("relay identify did not lead to a direct connection")
	}
	remote := conn.RemoteMultiaddr()
	if manet.IsIPLoopback(remote) || manet.IsIP6LinkLocal(remote) {
		t.Fatalf("direct dial used forbidden peer address %s", remote)
	}
}
