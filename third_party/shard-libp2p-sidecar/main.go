// Command sidecar is shard's libp2p transport daemon.
//
// One sidecar runs alongside the Python engine on every node. It owns the node's
// cryptographic identity (an ed25519 keypair -> libp2p PeerId) and carries the engine's
// inter-stage connections to/from its ring neighbours over authenticated, encrypted
// libp2p streams. It runs as a transparent TCP<->libp2p tunnel: the engine keeps its
// plain socket code and just talks to localhost; the sidecar does the network. This is
// what replaces the shared-SHARD_PSK TCP wire (phase0/wire.py).
//
// Per the boundary law (docs/INTEGRATION.md): pure engine plumbing. It knows nothing
// about $ZERO, accounts, payments, or the orchestrator — only peers and bytes.
//
// Modes:
//
//	-inbound HOST:PORT            tunnel: dial the local engine for each inbound stream
//	-nakd-inbound HOST:PORT       direct-only tunnel: dial nakd for each nakd stream
//	-forward LOCAL=PEER_MULTIADDR tunnel: listen LOCAL, carry each conn to PEER (repeatable)
//	-peer PEER_MULTIADDR          self-test: round-trip one frame to a listener (connectivity check)
//	(none)                        self-test listener: echo one frame back
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/connmgr"
	"github.com/libp2p/go-libp2p/core/control"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	"github.com/libp2p/go-libp2p/core/protocol"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	"github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"
)

// activationProto is the stream protocol carrying inter-stage traffic (and the self-test).
const (
	activationProto       = "/shard/activation/1.0.0"
	nakdProto             = "/nakshatra/nakd/1.0.0"
	activationRelayReason = "shard activation stream"
	forceDirectReason     = "nakd direct path"
	maxDialLine           = 512
	directDialRetries     = 3
	maxDirectDialAddrs    = 8
	maxConcurrentDials    = 16
	directDialRetryDelay  = 250 * time.Millisecond
	directDialAttemptMax  = 3 * time.Second
)

// openActivationStream opts into circuit-relay connections. go-libp2p marks
// relayed connections limited; without this context option Swarm.NewStream waits
// for a direct-path upgrade until the caller's deadline expires.
func openActivationStream(
	ctx context.Context,
	p peer.ID,
	open func(context.Context, peer.ID, ...protocol.ID) (network.Stream, error),
) (network.Stream, error) {
	ctx = network.WithAllowLimitedConn(ctx, activationRelayReason)
	return open(ctx, p, activationProto)
}

// stringList is a repeatable string flag (used for -forward).
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// loadOrCreateKey returns a stable node identity, persisting it to keyPath so a node
// keeps the same PeerId across restarts. This per-node key is what replaces the shared
// SHARD_PSK: a node proves who it is by holding this key, not by knowing a secret.
func loadOrCreateKey(keyPath string) (crypto.PrivKey, error) {
	if keyPath != "" {
		if b, err := os.ReadFile(keyPath); err == nil {
			return crypto.UnmarshalPrivateKey(b)
		}
	}
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		return nil, err
	}
	if keyPath != "" {
		b, err := crypto.MarshalPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		if err := os.WriteFile(keyPath, b, 0o600); err != nil {
			return nil, err
		}
	}
	return priv, nil
}

// writeFrame / readFrame: a 4-byte big-endian length prefix + payload (the self-test wire).
func writeFrame(w io.Writer, b []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func readFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	b := make([]byte, binary.BigEndian.Uint32(hdr[:]))
	_, err := io.ReadFull(r, b)
	return b, err
}

type natOpts struct {
	quic                bool
	relayService        bool
	disableHolePunching bool
	announce            string
	staticRelays        []peer.AddrInfo
	dialPolicy          *outboundDialPolicy
	// ipv6Listen is a test seam for hosts where binding IPv6 fails. Production uses
	// h.Network().Listen. IPv6 is deliberately added after the IPv4 host starts so a
	// kernel or container without IPv6 cannot take the sidecar down.
	ipv6Listen func(multiaddr.Multiaddr) error
}

// outboundDialPolicy is the host-wide SSRF boundary for peer-supplied addresses.
// Identify and Identify Push can replace peerstore entries at any time, so filtering
// the peerstore is only hygiene; the connection gater is the enforcement point.
// Peers named by an operator flag are exempt so explicitly configured loopback test,
// forward, and relay endpoints continue to work exactly as supplied.
type outboundDialPolicy struct {
	explicitPeers       map[peer.ID]struct{}
	explicitRelayRoutes []string
	dialMu              sync.RWMutex
	dialCandidates      map[peer.ID]map[string]struct{}
	dialLocks           [32]sync.Mutex
}

var _ connmgr.ConnectionGater = (*outboundDialPolicy)(nil)

func newOutboundDialPolicy(relays []peer.AddrInfo, forwards []string, selfTestPeer string) (*outboundDialPolicy, error) {
	p := &outboundDialPolicy{
		explicitPeers:  make(map[peer.ID]struct{}),
		dialCandidates: make(map[peer.ID]map[string]struct{}),
	}
	for _, relay := range relays {
		p.explicitPeers[relay.ID] = struct{}{}
		for _, addr := range relay.Addrs {
			route := addr.Encapsulate(multiaddr.StringCast("/p2p/" + relay.ID.String()))
			route = route.Encapsulate(multiaddr.StringCast("/p2p-circuit"))
			p.explicitRelayRoutes = append(p.explicitRelayRoutes, route.String())
		}
	}
	for _, forward := range forwards {
		parts := strings.SplitN(forward, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("bad -forward %q (want localAddr=peerMultiaddr)", forward)
		}
		if err := p.addExplicitPeer(parts[1]); err != nil {
			return nil, fmt.Errorf("bad -forward %q: %w", forward, err)
		}
	}
	if selfTestPeer != "" {
		if err := p.addExplicitPeer(selfTestPeer); err != nil {
			return nil, fmt.Errorf("bad -peer: %w", err)
		}
	}
	return p, nil
}

func (p *outboundDialPolicy) addExplicitPeer(value string) error {
	addr, err := multiaddr.NewMultiaddr(value)
	if err != nil {
		return err
	}
	info, err := peer.AddrInfoFromP2pAddr(addr)
	if err != nil {
		return err
	}
	p.explicitPeers[info.ID] = struct{}{}
	return nil
}

func (p *outboundDialPolicy) InterceptPeerDial(id peer.ID) bool {
	// Peer identity alone has no network location to classify. Every actual outbound
	// address is checked below after libp2p has resolved it.
	return id != ""
}

func (p *outboundDialPolicy) InterceptAddrDial(id peer.ID, addr multiaddr.Multiaddr) bool {
	value := addr.String()
	p.dialMu.RLock()
	candidates, bounded := p.dialCandidates[id]
	_, selected := candidates[value]
	p.dialMu.RUnlock()
	if bounded {
		// Entries are installed only from the safe direct selector or from routes
		// derived from operator-configured relays.
		return selected
	}
	for _, route := range p.explicitRelayRoutes {
		if value == route || strings.HasPrefix(value, route+"/") {
			return true
		}
	}
	if _, ok := p.explicitPeers[id]; ok {
		return true
	}
	ip, err := manet.ToIP(addr)
	if err != nil {
		return false
	}
	if !safeOutboundIP(ip) {
		return false
	}
	return true
}

// withDialCandidates constrains the swarm itself, which otherwise re-reads every
// peerstore address even when Host.Connect receives a shorter AddrInfo. Dials to the
// same peer are serialized while the gater exposes exactly this attempt's candidates.
func (p *outboundDialPolicy) withDialCandidates(id peer.ID, addrs []multiaddr.Multiaddr, dial func() error) error {
	var shard byte
	for _, b := range []byte(id) {
		shard ^= b
	}
	lock := &p.dialLocks[int(shard)%len(p.dialLocks)]
	lock.Lock()
	defer lock.Unlock()

	selected := make(map[string]struct{}, len(addrs))
	for _, addr := range addrs {
		selected[addr.String()] = struct{}{}
	}
	p.dialMu.Lock()
	p.dialCandidates[id] = selected
	p.dialMu.Unlock()
	defer func() {
		p.dialMu.Lock()
		delete(p.dialCandidates, id)
		p.dialMu.Unlock()
	}()
	return dial()
}

func safeOutboundIP(ip net.IP) bool {
	return ip.IsGlobalUnicast() && !ip.IsLoopback() && !ip.IsUnspecified() &&
		!ip.IsLinkLocalUnicast() && !ip.IsLinkLocalMulticast() && !ip.IsMulticast()
}

func (p *outboundDialPolicy) InterceptAccept(network.ConnMultiaddrs) bool { return true }

func (p *outboundDialPolicy) InterceptSecured(network.Direction, peer.ID, network.ConnMultiaddrs) bool {
	return true
}

func (p *outboundDialPolicy) InterceptUpgraded(network.Conn) (bool, control.DisconnectReason) {
	return true, 0
}

// tcpToQuic derives a QUIC listen addr from a TCP one: /ip4/x/tcp/P -> /ip4/x/udp/P/quic-v1.
func tcpToQuic(maddr string) string {
	i := strings.Index(maddr, "/tcp/")
	if i < 0 {
		return ""
	}
	port := maddr[i+len("/tcp/"):]
	if j := strings.Index(port, "/"); j >= 0 {
		port = port[:j]
	}
	return maddr[:i] + "/udp/" + port + "/quic-v1"
}

// wildcardIPv6ListenAddrs mirrors an IPv4 wildcard TCP listener onto IPv6. A fixed
// port stays fixed; port zero deliberately remains zero so the kernel can choose it.
func wildcardIPv6ListenAddrs(listen string, quic bool) []multiaddr.Multiaddr {
	addr, err := multiaddr.NewMultiaddr(listen)
	if err != nil {
		return nil
	}
	ip4, err := addr.ValueForProtocol(multiaddr.P_IP4)
	if err != nil || ip4 != "0.0.0.0" {
		return nil
	}
	port, err := addr.ValueForProtocol(multiaddr.P_TCP)
	if err != nil {
		return nil
	}
	strings := []string{"/ip6/::/tcp/" + port}
	if quic {
		strings = append(strings, "/ip6/::/udp/"+port+"/quic-v1")
	}
	out := make([]multiaddr.Multiaddr, 0, len(strings))
	for _, s := range strings {
		out = append(out, multiaddr.StringCast(s))
	}
	return out
}

func newHost(priv crypto.PrivKey, listen string, n natOpts) (host.Host, error) {
	// libp2p defaults give Noise/TLS encryption + a stream muxer; every link is
	// authenticated to the peer's key. The NAT stack (DCUtR hole-punching + circuit
	// relay) lets home GPUs behind NAT join; QUIC (udp) hole-punches more reliably.
	listens := []string{listen}
	if n.quic {
		if q := tcpToQuic(listen); q != "" {
			listens = append(listens, q)
		}
	}
	opts := []libp2p.Option{
		libp2p.Identity(priv),
		libp2p.ListenAddrStrings(listens...),
	}
	dialPolicy := n.dialPolicy
	if dialPolicy == nil {
		dialPolicy, _ = newOutboundDialPolicy(n.staticRelays, nil, "")
	}
	opts = append(opts, libp2p.ConnectionGater(dialPolicy))
	if !n.disableHolePunching {
		opts = append(opts, libp2p.EnableHolePunching()) // DCUtR: punch a direct hole between two NAT'd peers
	}
	if n.announce != "" {
		// libp2p only sees container-internal addrs behind Vast's port mapping; advertise
		// the real public addr so reservations/circuit addrs others get are actually dialable.
		ann, err := multiaddr.NewMultiaddr(n.announce)
		if err != nil {
			return nil, err
		}
		opts = append(opts, libp2p.AddrsFactory(func(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
			return append([]multiaddr.Multiaddr{ann}, addrs...)
		}))
	}
	if n.relayService {
		// be a public relay (circuit-relay-v2) + an AutoNAT server (so NAT'd peers can
		// learn their reachability + observed address from us). Force public reachability
		// so the hop service activates immediately.
		opts = append(opts, libp2p.EnableRelayService(), libp2p.ForceReachabilityPublic(), libp2p.EnableNATService())
	}
	// NAT'd nodes reserve on relays explicitly (in main) and let AutoNAT + the
	// observed-address manager (from several observer peers) determine reachability — so
	// DCUtR can hole-punch when the NAT is cone-type. Forcing private here is wrong: it
	// leaves holepunch with no public address to offer ("waiting for a public address").
	h, err := libp2p.New(opts...)
	if err != nil {
		return nil, err
	}
	listenIPv6 := n.ipv6Listen
	if listenIPv6 == nil {
		listenIPv6 = func(addr multiaddr.Multiaddr) error {
			return h.Network().Listen(addr)
		}
	}
	for _, addr := range wildcardIPv6ListenAddrs(listen, n.quic) {
		if err := listenIPv6(addr); err != nil {
			log.Printf("IPv6 listen %s: %v (continuing)", addr, err)
		}
	}
	return h, nil
}

// fullAddrs returns this host's dialable /p2p multiaddrs (addr + /p2p/<peerid>).
func fullAddrs(h host.Host) []string {
	p2p := multiaddr.StringCast("/p2p/" + h.ID().String())
	out := make([]string, 0, len(h.Addrs()))
	for _, a := range h.Addrs() {
		out = append(out, a.Encapsulate(p2p).String())
	}
	return out
}

// halfLife returns a time roughly halfway to expiration. No floor: a relay that grants a
// short TTL means exactly that little time exists, and a floor here would wait PAST the
// expiration it's meant to beat — the wrong failure mode this patch exists to fix in the
// first place. Reserve()'s own network round-trip (and the 30s retry backoff on failure)
// already bound how often this can actually fire.
func halfLife(expiration time.Time) time.Time {
	half := time.Until(expiration) / 2
	if half < 0 {
		half = 0
	}
	return time.Now().Add(half)
}

// renewRelayReservation keeps a circuit-relay-v2 reservation on relay alive for the life of h.
// PATCH (2026-09-07, nakshatra): upstream's relayclient.Reserve() reserves ONCE and is never
// called again by anything in this file — a reservation silently expires (observed ~1hr TTL)
// after which the relay drops this node with no error logged anywhere; only the NEXT dial
// attempt through it fails, downstream, as NO_RESERVATION. This loop re-reserves at half the
// granted TTL and retries every 30s on failure (a relay that's briefly unreachable should not
// permanently strand this node) — see NAKSHATRA_INTEGRATION.md for why this departs from
// upstream's "vendored verbatim" claim.
func renewRelayReservation(h host.Host, relay peer.AddrInfo, next time.Time) {
	for {
		if wait := time.Until(next); wait > 0 {
			time.Sleep(wait)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if err := h.Connect(ctx, relay); err != nil {
			cancel()
			log.Printf("relay renew %s: connect: %v (retrying in 30s)", relay.ID, err)
			next = time.Now().Add(30 * time.Second)
			continue
		}
		res, err := relayclient.Reserve(ctx, h, relay)
		cancel()
		if err != nil {
			log.Printf("relay renew %s: reserve: %v (retrying in 30s)", relay.ID, err)
			next = time.Now().Add(30 * time.Second)
			continue
		}
		h.ConnManager().Protect(relay.ID, "relay")
		log.Printf("RENEWED relay slot on %s (expires %s)", relay.ID, res.Expiration)
		next = halfLife(res.Expiration)
	}
}

func main() {
	keyPath := flag.String("key", "", "path to persist the node key (keeps PeerId stable)")
	peerAddr := flag.String("peer", "", "self-test: dial this /p2p multiaddr and round-trip a frame")
	addrFile := flag.String("addrfile", "", "write this host's dial multiaddr here (for scripting)")
	listenAddr := flag.String("listen", "/ip4/0.0.0.0/tcp/0", "libp2p listen multiaddr; pin the port for cross-box reach, e.g. /ip4/0.0.0.0/tcp/29600")
	inbound := flag.String("inbound", "", "tunnel: dial this local engine addr (host:port) for each inbound libp2p stream")
	nakdInbound := flag.String("nakd-inbound", "", "direct-only tunnel: dial this local nakd addr for each nakd libp2p stream")
	dialListen := flag.String("dial-listen", "", "listen on this user-only UNIX socket for DIAL <peerid> requests that require a direct libp2p path")
	directWait := flag.Duration("direct-wait", 15*time.Second, "maximum time to wait for a direct (non-relayed) connection")
	var forwards stringList
	flag.Var(&forwards, "forward", "tunnel: localAddr=peerMultiaddr — listen localAddr, carry each conn to the peer (repeatable)")
	size := flag.Int("size", 1<<20, "self-test frame size in bytes (default 1 MiB)")
	relaySvc := flag.Bool("relay", false, "run as a circuit-relay-v2 server (public rendezvous for NAT'd nodes)")
	relaysCSV := flag.String("relays", "", "comma-separated relay /p2p multiaddrs to use when behind NAT")
	useQuic := flag.Bool("quic", false, "also listen on QUIC (udp) — better hole-punching + lossy links")
	announce := flag.String("announce", "", "advertise this public multiaddr ahead of auto-detected ones (e.g. /ip4/PUBIP/tcp/PORT)")
	prove := flag.String("prove", "", "identity binding: sign this challenge with the node key; print PEERID + SIG")
	verify := flag.String("verify", "", "identity binding: verify a proof 'peerid,nonce,b64sig' -> OK/FAIL (reference for c0mpute)")
	flag.Parse()
	if *dialListen != "" {
		if *directWait <= 0 {
			log.Fatalf("direct-wait must be positive")
		}
	}

	// Identity-binding verify: prove a PeerId controls its key, from (peerid, nonce, sig)
	// alone — no node key needed. This is the check c0mpute runs (ported to TS) before it
	// records PeerId <-> account. ed25519 PeerIds embed the public key, so the verifier
	// needs nothing but the proof.
	if *verify != "" {
		p := strings.SplitN(*verify, ",", 3)
		if len(p) != 3 {
			log.Fatalf("verify wants 'peerid,nonce,b64sig'")
		}
		pid, err := peer.Decode(p[0])
		if err != nil {
			log.Fatalf("peerid: %v", err)
		}
		pub, err := pid.ExtractPublicKey()
		if err != nil {
			log.Fatalf("extract pubkey from peerid: %v", err)
		}
		sig, err := base64.StdEncoding.DecodeString(p[2])
		if err != nil {
			log.Fatalf("sig: %v", err)
		}
		ok, _ := pub.Verify([]byte(p[1]), sig)
		fmt.Printf("VERIFY %v\n", ok)
		return
	}

	priv, err := loadOrCreateKey(*keyPath)
	if err != nil {
		log.Fatalf("key: %v", err)
	}

	// Identity-binding proof: sign a challenge nonce with the node key. The node-agent
	// sends {PEERID, SIG} to c0mpute alongside its cwt_ token; c0mpute verifies the sig
	// against the PeerId and records PeerId <-> account. Pure crypto — knows nothing of c0mpute.
	if *prove != "" {
		pid, err := peer.IDFromPublicKey(priv.GetPublic())
		if err != nil {
			log.Fatalf("peerid: %v", err)
		}
		sig, err := priv.Sign([]byte(*prove))
		if err != nil {
			log.Fatalf("sign: %v", err)
		}
		fmt.Printf("PEERID %s\nSIG %s\n", pid, base64.StdEncoding.EncodeToString(sig))
		return
	}
	var staticRelays []peer.AddrInfo
	for _, s := range strings.Split(*relaysCSV, ",") {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		ma, err := multiaddr.NewMultiaddr(s)
		if err != nil {
			log.Fatalf("bad -relays entry %q: %v", s, err)
		}
		ai, err := peer.AddrInfoFromP2pAddr(ma)
		if err != nil {
			log.Fatalf("bad -relays entry %q: %v", s, err)
		}
		staticRelays = append(staticRelays, *ai)
	}
	dialPolicy, err := newOutboundDialPolicy(staticRelays, forwards, *peerAddr)
	if err != nil {
		log.Fatal(err)
	}
	h, err := newHost(priv, *listenAddr, natOpts{quic: *useQuic, relayService: *relaySvc, announce: *announce, staticRelays: staticRelays, dialPolicy: dialPolicy})
	if err != nil {
		log.Fatalf("host: %v", err)
	}
	defer h.Close()
	log.Printf("peer id: %s", h.ID())
	addrs := fullAddrs(h)
	for _, a := range addrs {
		fmt.Printf("ADDR %s\n", a)
	}
	if *addrFile != "" {
		pick := addrs[0]
		for _, a := range addrs {
			if strings.Contains(a, "127.0.0.1") {
				pick = a
				break
			}
		}
		if err := os.WriteFile(*addrFile, []byte(pick), 0o644); err != nil {
			log.Fatalf("addrfile: %v", err)
		}
	}

	go monitorConns(h) // log RELAY vs DIRECT connections so we can watch DCUtR upgrade

	// NAT'd node: explicitly reserve a slot on each relay and keep the connection
	// protected, so the relay can forward inbound connections to us. Clear errors,
	// no autorelay guesswork.
	for _, relay := range staticRelays {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if err := h.Connect(ctx, relay); err != nil {
			log.Printf("relay connect %s: %v", relay.ID, err)
			cancel()
			go renewRelayReservation(h, relay, time.Now().Add(30*time.Second))
			continue
		}
		res, err := relayclient.Reserve(ctx, h, relay)
		cancel()
		if err != nil {
			log.Printf("relay reserve %s: %v", relay.ID, err)
			go renewRelayReservation(h, relay, time.Now().Add(30*time.Second))
			continue
		}
		h.ConnManager().Protect(relay.ID, "relay")
		log.Printf("RESERVED relay slot on %s (expires %s)", relay.ID, res.Expiration)
		go renewRelayReservation(h, relay, halfLife(res.Expiration))
	}

	// Tunnel mode: a transparent TCP<->libp2p bridge. The engine keeps its own socket
	// code and just talks to localhost; the sidecar carries each connection to/from the
	// right ring neighbour over libp2p. This is what replaces wire.py's TCP.
	if *inbound != "" || *nakdInbound != "" || *dialListen != "" || len(forwards) > 0 {
		if *inbound != "" {
			runInbound(h, *inbound)
		}
		if *nakdInbound != "" {
			runNakdInbound(h, *nakdInbound)
		}
		if *dialListen != "" {
			ln, err := listenDialSocket(*dialListen)
			if err != nil {
				log.Fatalf("dial-listen %s: %v", *dialListen, err)
			}
			go serveDialListener(h, ln, staticRelays, *directWait, dialPolicy)
		}
		for _, f := range forwards {
			pp := strings.SplitN(f, "=", 2)
			if len(pp) != 2 {
				log.Fatalf("bad -forward %q (want localAddr=peerMultiaddr)", f)
			}
			go runForward(h, pp[0], pp[1])
		}
		log.Printf("tunnel up (inbound=%q nakd-inbound=%q dial-listen=%q forwards=%v)", *inbound, *nakdInbound, *dialListen, []string(forwards))
		select {}
	}

	// Self-test dialer: connect by multiaddr, round-trip a frame, verify it byte-for-byte.
	// libp2p's Noise handshake guarantees the peer holds the key in the multiaddr.
	if *peerAddr != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		maddr, err := multiaddr.NewMultiaddr(*peerAddr)
		if err != nil {
			log.Fatalf("bad -peer: %v", err)
		}
		info, err := peer.AddrInfoFromP2pAddr(maddr)
		if err != nil {
			log.Fatalf("bad -peer: %v", err)
		}
		if err := h.Connect(ctx, *info); err != nil {
			log.Fatalf("connect: %v", err)
		}
		s, err := openActivationStream(ctx, info.ID, h.NewStream)
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		defer s.Close()
		log.Printf("connected to %s (key-authenticated)", s.Conn().RemotePeer())

		blob := make([]byte, *size)
		if _, err := rand.Read(blob); err != nil {
			log.Fatalf("rand: %v", err)
		}
		start := time.Now()
		if err := writeFrame(s, blob); err != nil {
			log.Fatalf("send: %v", err)
		}
		got, err := readFrame(s)
		if err != nil {
			log.Fatalf("recv: %v", err)
		}
		rtt := time.Since(start)
		if !bytes.Equal(got, blob) {
			log.Fatalf("ROUND-TRIP MISMATCH: sent %d bytes, got %d", len(blob), len(got))
		}
		fmt.Printf("ROUND-TRIP OK: %d bytes echoed by %s in %v\n", len(blob), s.Conn().RemotePeer(), rtt)
		return
	}

	// Self-test listener: echo any frame back to the sender (connectivity check).
	h.SetStreamHandler(activationProto, func(s network.Stream) {
		defer s.Close()
		b, err := readFrame(s)
		if err != nil {
			log.Printf("recv: %v", err)
			return
		}
		log.Printf("recv %d bytes from %s", len(b), s.Conn().RemotePeer())
		if err := writeFrame(s, b); err != nil {
			log.Printf("send: %v", err)
		}
	})
	log.Printf("listening; start a second sidecar with -peer <one of the ADDR lines above>")
	select {} // serve until killed
}

// openStream dials a peer by multiaddr and opens an activation stream. libp2p's
// Noise handshake guarantees the peer holds the key named in the multiaddr.
func openStream(h host.Host, peerAddr string) (network.Stream, error) {
	maddr, err := multiaddr.NewMultiaddr(peerAddr)
	if err != nil {
		return nil, err
	}
	info, err := peer.AddrInfoFromP2pAddr(maddr)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := h.Connect(ctx, *info); err != nil {
		return nil, err
	}
	return openActivationStream(ctx, info.ID, h.NewStream)
}

// listenDialSocket keeps the peer-dial primitive scoped to this Unix uid. The containing directory
// is private, must be owned by us, and the socket itself is mode 0600. Refuse pre-existing non-socket
// paths rather than unlinking an arbitrary file selected through configuration.
func listenDialSocket(path string) (net.Listener, error) {
	if path == "" || !filepath.IsAbs(path) {
		return nil, fmt.Errorf("want an absolute UNIX socket path")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("create socket directory: %w", err)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return nil, fmt.Errorf("inspect socket directory: %w", err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return nil, fmt.Errorf("socket directory is not a real directory")
	}
	if int(stat.Uid) != os.Getuid() {
		return nil, fmt.Errorf("socket directory is owned by uid %d, not us", stat.Uid)
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return nil, fmt.Errorf("make socket directory private: %w", err)
	}
	if old, err := os.Lstat(path); err == nil {
		oldStat, owned := old.Sys().(*syscall.Stat_t)
		if old.Mode()&os.ModeSocket == 0 || !owned || int(oldStat.Uid) != os.Getuid() {
			return nil, fmt.Errorf("refusing to replace non-owned/non-socket path")
		}
		if err := os.Remove(path); err != nil {
			return nil, fmt.Errorf("remove stale socket: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("inspect socket path: %w", err)
	}
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		ln.Close()
		_ = os.Remove(path)
		return nil, fmt.Errorf("make socket private: %w", err)
	}
	return ln, nil
}

func parseDialLine(line []byte) (peer.ID, error) {
	if len(line) > maxDialLine {
		return "", fmt.Errorf("DIAL line too long")
	}
	line = bytes.TrimSuffix(line, []byte("\n"))
	line = bytes.TrimSuffix(line, []byte("\r"))
	if !bytes.HasPrefix(line, []byte("DIAL ")) || len(line) == len("DIAL ") {
		return "", fmt.Errorf("want DIAL <peerid>")
	}
	value := string(line[len("DIAL "):])
	if strings.ContainsAny(value, " \t\r\n") {
		return "", fmt.Errorf("want one peer id")
	}
	p, err := peer.Decode(value)
	if err != nil {
		return "", fmt.Errorf("bad peer id: %w", err)
	}
	return p, nil
}

func writeDialError(w io.Writer, err error) {
	reason := strings.NewReplacer("\r", " ", "\n", " ").Replace(err.Error())
	if len(reason) > 300 {
		reason = reason[:300]
	}
	_, _ = fmt.Fprintf(w, "ERR %s\n", reason)
}

func directPathAllowed(limited bool, remoteAddr string) bool {
	return !limited && !strings.Contains(remoteAddr, "p2p-circuit")
}

func isDirectConn(c network.Conn) bool {
	return directPathAllowed(c.Stat().Limited, c.RemoteMultiaddr().String())
}

func directConn(h host.Host, p peer.ID) network.Conn {
	for _, c := range h.Network().ConnsToPeer(p) {
		if isDirectConn(c) {
			return c
		}
	}
	return nil
}

// preferredDirectDialAddrs selects the peer's safe, non-relay IP endpoints and puts
// same-LAN addresses first, then globally routable IPv6, then other public IPs. DNS
// names are intentionally omitted: without resolving them first we could not uphold
// the rule that this path never dials a peer-supplied loopback or link-local endpoint.
func preferredDirectDialAddrs(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
	out := make([]multiaddr.Multiaddr, 0, len(addrs))
	for _, addr := range addrs {
		if !safeDirectDialAddr(addr) {
			continue
		}
		out = append(out, addr)
	}
	sort.SliceStable(out, func(i, j int) bool {
		pi, pj := directDialPriority(out[i]), directDialPriority(out[j])
		if pi != pj {
			return pi < pj
		}
		return out[i].String() < out[j].String()
	})
	if len(out) > maxDirectDialAddrs {
		out = out[:maxDirectDialAddrs]
	}
	return out
}

func safeDirectDialAddr(addr multiaddr.Multiaddr) bool {
	if _, err := addr.ValueForProtocol(multiaddr.P_CIRCUIT); err == nil {
		return false
	}
	ip, err := manet.ToIP(addr)
	return err == nil && safeOutboundIP(ip)
}

// removeUnsafeDirectDialAddrs makes the safety filter effective inside go-libp2p too:
// Host.Connect absorbs the supplied AddrInfo but the swarm ultimately reads every address
// already in the peerstore. Circuit addresses stay for rendezvous; unsafe direct endpoints do not.
func removeUnsafeDirectDialAddrs(h host.Host, p peer.ID) {
	for _, addr := range h.Peerstore().Addrs(p) {
		if _, err := addr.ValueForProtocol(multiaddr.P_CIRCUIT); err == nil {
			continue
		}
		if !safeDirectDialAddr(addr) {
			h.Peerstore().SetAddr(p, addr, 0)
		}
	}
}

func directDialPriority(addr multiaddr.Multiaddr) int {
	if manet.IsPrivateAddr(addr) {
		return 0
	}
	if _, err := addr.ValueForProtocol(multiaddr.P_IP6); err == nil && manet.IsPublicAddr(addr) {
		return 1
	}
	return 2
}

func directPathKind(addr multiaddr.Multiaddr) string {
	if manet.IsPrivateAddr(addr) {
		return "lan"
	}
	if _, err := addr.ValueForProtocol(multiaddr.P_IP6); err == nil && manet.IsPublicAddr(addr) {
		return "ipv6"
	}
	return "punched"
}

// retryDirectDials complements DCUtR. Once the relay connection's Identify exchange has
// populated the peerstore, explicitly try those advertised LAN/global-IPv6 addresses.
// WithForceDirectDial prevents an existing limited relay connection from satisfying Connect.
func retryDirectDialAttempts(
	ctx context.Context,
	p peer.ID,
	maxAttempts int,
	directConnected func() bool,
	addresses func() []multiaddr.Multiaddr,
	connect func(context.Context, peer.AddrInfo) error,
) {
	for attempt := 0; attempt < maxAttempts; {
		if directConnected() || ctx.Err() != nil {
			return
		}
		addrs := preferredDirectDialAddrs(addresses())
		if len(addrs) == 0 {
			if !waitForDirectRetry(ctx) {
				return
			}
			continue
		}
		attemptTimeout := directDialAttemptMax
		if deadline, ok := ctx.Deadline(); ok {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return
			}
			share := remaining / time.Duration(maxAttempts-attempt)
			if share < attemptTimeout {
				attemptTimeout = share
			}
		}
		attemptCtx, cancel := context.WithTimeout(ctx, attemptTimeout)
		_ = connect(network.WithForceDirectDial(attemptCtx, forceDirectReason), peer.AddrInfo{ID: p, Addrs: addrs})
		cancel()
		attempt++
		if attempt == maxAttempts || directConnected() {
			return
		}
		if !waitForDirectRetry(ctx) {
			return
		}
	}
}

func waitForDirectRetry(ctx context.Context) bool {
	timer := time.NewTimer(directDialRetryDelay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func retryDirectDials(ctx context.Context, h host.Host, p peer.ID, maxAttempts int, policy *outboundDialPolicy) {
	connect := h.Connect
	if policy != nil {
		connect = func(ctx context.Context, info peer.AddrInfo) error {
			return policy.withDialCandidates(info.ID, info.Addrs, func() error {
				return h.Connect(ctx, info)
			})
		}
	}
	retryDirectDialAttempts(
		ctx,
		p,
		maxAttempts,
		func() bool { return directConn(h, p) != nil },
		func() []multiaddr.Multiaddr { return h.Peerstore().Addrs(p) },
		connect,
	)
}

// addRelayCircuitAddrs teaches the host how to reach p through each configured relay. The circuit is
// rendezvous only: waitDirect refuses it for the user's stream and waits for DCUtR to produce a
// separate non-relayed connection.
func addRelayCircuitAddrs(h host.Host, p peer.ID, relays []peer.AddrInfo) []multiaddr.Multiaddr {
	var added []multiaddr.Multiaddr
	for _, relay := range relays {
		for _, addr := range relay.Addrs {
			full := addr.Encapsulate(multiaddr.StringCast("/p2p/" + relay.ID.String()))
			full = full.Encapsulate(multiaddr.StringCast("/p2p-circuit/p2p/" + p.String()))
			info, err := peer.AddrInfoFromP2pAddr(full)
			if err == nil {
				h.Peerstore().AddAddrs(p, info.Addrs, peerstore.TempAddrTTL)
				added = append(added, info.Addrs...)
			}
		}
	}
	return added
}

func waitDirect(ctx context.Context, h host.Host, p peer.ID, relays []peer.AddrInfo, policy *outboundDialPolicy) error {
	if directConn(h, p) != nil {
		return nil
	}
	circuitAddrs := addRelayCircuitAddrs(h, p, relays)
	removeUnsafeDirectDialAddrs(h, p)
	dialCtx, cancelDials := context.WithCancel(ctx)
	defer cancelDials()
	directAttempts := directDialRetries
	if len(circuitAddrs) > 0 {
		directAttempts-- // the bounded relay rendezvous is this DIAL request's first attempt
		// Only routes derived from operator-configured relays enter this bounded first
		// attempt. The remaining two attempts can use newly identified direct addresses.
		go func() {
			attemptCtx, cancel := context.WithTimeout(dialCtx, directDialAttemptMax)
			defer cancel()
			connect := func() error { return h.Connect(attemptCtx, peer.AddrInfo{ID: p, Addrs: circuitAddrs}) }
			var err error
			if policy != nil {
				err = policy.withDialCandidates(p, circuitAddrs, connect)
			} else {
				err = connect()
			}
			if err != nil && dialCtx.Err() == nil {
				log.Printf("relay rendezvous %s: %v", p, err)
			}
		}()
	}
	go retryDirectDials(dialCtx, h, p, directAttempts, policy)
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if directConn(h, p) != nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("no direct connection before timeout")
		case <-tick.C:
		}
	}
}

func openDirectNakdStream(ctx context.Context, h host.Host, p peer.ID, relays []peer.AddrInfo, policy *outboundDialPolicy) (network.Stream, error) {
	if err := waitDirect(ctx, h, p, relays, policy); err != nil {
		return nil, err
	}
	// Unlike openActivationStream, this intentionally does NOT call WithAllowLimitedConn. A relay
	// circuit is a limited/transient connection and may rendezvous DCUtR, but may never carry nakd data.
	s, err := h.NewStream(ctx, p, nakdProto)
	if err != nil {
		return nil, err
	}
	if !isDirectConn(s.Conn()) {
		s.Reset()
		return nil, fmt.Errorf("libp2p selected a relayed connection")
	}
	return s, nil
}

func handleDialConn(h host.Host, c net.Conn, relays []peer.AddrInfo, directWait time.Duration, policy *outboundDialPolicy) {
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	reader := bufio.NewReaderSize(c, maxDialLine+1)
	line, err := reader.ReadSlice('\n')
	if err == bufio.ErrBufferFull || len(line) > maxDialLine {
		writeDialError(c, fmt.Errorf("DIAL line too long"))
		c.Close()
		return
	}
	if err != nil {
		writeDialError(c, fmt.Errorf("read DIAL line: %w", err))
		c.Close()
		return
	}
	p, err := parseDialLine(line)
	if err != nil {
		writeDialError(c, err)
		c.Close()
		return
	}
	_ = c.SetDeadline(time.Time{})
	ctx, cancel := context.WithTimeout(context.Background(), directWait)
	defer cancel()
	s, err := openDirectNakdStream(ctx, h, p, relays, policy)
	if err != nil {
		writeDialError(c, err)
		c.Close()
		return
	}
	if _, err := io.WriteString(c, "OK direct\n"); err != nil {
		s.Reset()
		c.Close()
		return
	}
	log.Printf("DIAL %s: DIRECT via %s (%s)", p, s.Conn().RemoteMultiaddr(), directPathKind(s.Conn().RemoteMultiaddr()))
	pipe(c, s)
}

func serveDialListener(h host.Host, ln net.Listener, relays []peer.AddrInfo, directWait time.Duration, policy *outboundDialPolicy) {
	log.Printf("dial listener on %s (direct wait %s)", ln.Addr(), directWait)
	slots := make(chan struct{}, maxConcurrentDials)
	for {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		dispatchDialConn(c, slots, func(c net.Conn) {
			handleDialConn(h, c, relays, directWait, policy)
		})
	}
}

func dispatchDialConn(c net.Conn, slots chan struct{}, handle func(net.Conn)) {
	select {
	case slots <- struct{}{}:
		go func() {
			defer func() { <-slots }()
			handle(c)
		}()
	default:
		writeDialError(c, fmt.Errorf("busy"))
		_ = c.Close()
	}
}

// monitorConns logs each new connection and whether it's via a relay or DIRECT — so we
// can watch DCUtR upgrade a relay rendezvous into a direct hole-punched link.
func monitorConns(h host.Host) {
	seen := map[string]bool{}
	for {
		time.Sleep(5 * time.Second)
		for _, c := range h.Network().Conns() {
			a := c.RemoteMultiaddr().String()
			kind := "DIRECT"
			if strings.Contains(a, "p2p-circuit") {
				kind = "RELAY"
			}
			key := c.RemotePeer().String() + "|" + a
			if !seen[key] {
				seen[key] = true
				log.Printf("CONN %s via %s [%s]", c.RemotePeer(), a, kind)
			}
		}
	}
}

// pipe copies bytes bidirectionally between two streams until either side closes.
func pipe(a, b io.ReadWriteCloser) {
	done := make(chan struct{}, 2)
	cp := func(dst io.Writer, src io.Reader) { io.Copy(dst, src); done <- struct{}{} }
	go cp(a, b)
	go cp(b, a)
	<-done
	a.Close()
	b.Close()
}

// runForward listens on a local TCP addr; each accepted connection is carried to the
// peer over a fresh libp2p stream — so the engine dials localhost and reaches the peer.
func runForward(h host.Host, listenAddr, peerMaddr string) {
	// pre-establish the connection so DCUtR can upgrade relay->direct BEFORE data flows
	// (otherwise the engine's first stream lands on the slow relay connection).
	if ma, err := multiaddr.NewMultiaddr(peerMaddr); err == nil {
		if ai, err := peer.AddrInfoFromP2pAddr(ma); err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			if err := h.Connect(ctx, *ai); err != nil {
				log.Printf("forward pre-connect %s: %v", ai.ID, err)
			}
			cancel()
		}
	}
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("forward listen %s: %v", listenAddr, err)
	}
	log.Printf("forward %s -> %s", listenAddr, peerMaddr)
	for {
		c, err := ln.Accept()
		if err != nil {
			log.Printf("forward accept: %v", err)
			return
		}
		go func() {
			s, err := openStream(h, peerMaddr)
			if err != nil {
				log.Printf("forward dial: %v", err)
				c.Close()
				return
			}
			pipe(c, s)
		}()
	}
}

// runInbound pipes each inbound libp2p stream to a fresh connection to the local
// engine — so the engine accepts on localhost, fed by its ring neighbours.
func runInbound(h host.Host, engineAddr string) {
	h.SetStreamHandler(activationProto, func(s network.Stream) {
		c, err := net.Dial("tcp", engineAddr)
		if err != nil {
			log.Printf("inbound -> engine %s: %v", engineAddr, err)
			s.Reset()
			return
		}
		pipe(s, c)
	})
}

// runNakdInbound is deliberately separate from the activation tunnel. Inference activation streams
// are allowed to fall back to circuit relay; nakd already has its own relay fallback, so carrying its
// bytes over a limited circuit would waste the rendezvous relay's tight data allowance.
func runNakdInbound(h host.Host, nakdAddr string) {
	h.SetStreamHandler(nakdProto, func(s network.Stream) {
		if !isDirectConn(s.Conn()) {
			log.Printf("refused inbound nakd stream over limited/relayed connection from %s", s.Conn().RemotePeer())
			_ = s.Reset()
			return
		}
		c, err := net.Dial("tcp", nakdAddr)
		if err != nil {
			log.Printf("inbound -> nakd %s: %v", nakdAddr, err)
			s.Reset()
			return
		}
		pipe(s, c)
	})
}
