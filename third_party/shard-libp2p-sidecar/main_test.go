package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	"github.com/libp2p/go-libp2p/core/protocol"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	"github.com/libp2p/go-libp2p/p2p/protocol/identify"
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

func TestDirectActivationInboundIsOptIn(t *testing.T) {
	tests := []struct {
		name                               string
		requireDirect, allowRelay, limited bool
		addr                               string
		want                               bool
	}{
		{"legacy relay", false, false, true, "/ip4/1.2.3.4/tcp/1/p2p-circuit", false},
		{"legacy circuit", false, false, false, "/ip4/1.2.3.4/tcp/1/p2p-circuit", false},
		{"legacy direct", false, false, false, "/ip4/1.2.3.4/tcp/1", true},
		{"strict limited", true, false, true, "/ip4/1.2.3.4/tcp/1", false},
		{"strict circuit", true, false, false, "/ip4/1.2.3.4/tcp/1/p2p-circuit", false},
		{"strict direct", true, false, false, "/ip4/1.2.3.4/tcp/1", true},
		{"opted limited", true, true, true, "/ip4/1.2.3.4/tcp/1", true},
		{"opted circuit", true, true, false, "/ip4/1.2.3.4/tcp/1/p2p-circuit", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := activationPathAllowed(tc.requireDirect, tc.allowRelay, tc.limited, tc.addr); got != tc.want {
				t.Fatalf("activationPathAllowed(%v, %v, %v, %q) = %v, want %v", tc.requireDirect, tc.allowRelay, tc.limited, tc.addr, got, tc.want)
			}
		})
	}
}

func TestRelayResourcesAreExplicitAndCapped(t *testing.T) {
	cfg := defaultRelayServerConfig()
	resources, guard, err := cfg.resources()
	if err != nil {
		t.Fatal(err)
	}
	if resources.Limit == nil || resources.Limit.Data != 64<<20 || resources.Limit.Duration != 6*time.Hour {
		t.Fatalf("relay limit = %+v", resources.Limit)
	}
	if resources.MaxCircuits != 2 {
		t.Fatalf("per-peer circuits = %d, want 2", resources.MaxCircuits)
	}
	if guard.maxCircuits != 64 || guard.dailyBytes != 20<<30 || guard.peerDailyBytes != 2<<30 {
		t.Fatalf("guard limits = total %d global %d peer %d", guard.maxCircuits, guard.dailyBytes, guard.peerDailyBytes)
	}
	cfg = relayServerConfig{
		circuitBytes: 12345, circuitDuration: 42 * time.Second,
		maxCircuits: 7, circuitsPerPeer: 2, dailyBytes: 20 * 12345,
		peerDailyBytes: 6 * 12345,
	}
	resources, guard, err = cfg.resources()
	if err != nil {
		t.Fatal(err)
	}
	if resources.Limit.Data != 12345 || resources.Limit.Duration != 42*time.Second ||
		resources.MaxCircuits != 2 || guard.maxCircuits != 7 || guard.circuitCharge != 2*12345 {
		t.Fatalf("custom flag resources = %+v, guard total=%d charge=%d", resources, guard.maxCircuits, guard.circuitCharge)
	}

	cfg.circuitBytes = 0
	if _, _, err := cfg.resources(); err == nil {
		t.Fatal("zero circuit byte cap was accepted")
	}
	cfg = defaultRelayServerConfig()
	cfg.circuitsPerPeer = cfg.maxCircuits + 1
	if _, _, err := cfg.resources(); err == nil {
		t.Fatal("per-peer cap above total cap was accepted")
	}
}

func TestRelayAllowFileIsMandatory(t *testing.T) {
	if err := validateRelayFlags(true, ""); err == nil || !strings.Contains(err.Error(), "requires -relay-allow-file") {
		t.Fatalf("missing allow file error = %v", err)
	}
	if err := validateRelayFlags(true, "/var/lib/relay.allow"); err != nil {
		t.Fatalf("configured allow file was refused: %v", err)
	}
}

func TestProductionBinaryHasNoRelayOpenFlag(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "nakshatra-sidecar")
	build := exec.Command("go", "build", "-buildvcs=false", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build production binary: %v\n%s", err, out)
	}
	cmd := exec.Command(bin, "-relay-open")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "flag provided but not defined: -relay-open") {
		t.Fatalf("production -relay-open result err=%v output=%q", err, out)
	}
	cmd = exec.Command(bin, "-relay")
	out, err = cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "-relay requires -relay-allow-file") {
		t.Fatalf("production relay without allow file err=%v output=%q", err, out)
	}
}

func testRelayGuard(t *testing.T, cfg relayServerConfig) *relayGuard {
	t.Helper()
	g, err := newRelayGuard(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestRelayChargesOnlySuccessfulEstablishedCircuits(t *testing.T) {
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes = 100
	cfg.dailyBytes = 1000
	cfg.peerDailyBytes = 1000
	g := testRelayGuard(t, cfg)
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	src, dest := testPeerID(t), testPeerID(t)
	token, ok := g.AdmitConnect(src, addr, dest)
	if !ok {
		t.Fatal("attempt was refused")
	}
	if g.active != 0 || len(g.state.Days) != 1 || g.state.Days[g.state.MaxDay].GlobalBytes != 0 {
		t.Fatal("pending attempt consumed a slot or budget")
	}
	g.ConnectFailed(token)
	if g.active != 0 || g.state.Days[g.state.MaxDay].GlobalBytes != 0 {
		t.Fatal("failed attempt consumed a slot or budget")
	}
	token, ok = g.AdmitConnect(src, addr, dest)
	if !ok {
		t.Fatal("attempt after failure was refused")
	}
	g.ConnectEstablished(token)
	usage := g.state.Days[g.state.MaxDay]
	if g.active != 1 || usage.GlobalBytes != 200 || usage.Peers[src.String()].Bytes != 200 || usage.Peers[dest.String()].Bytes != 200 {
		t.Fatalf("successful charge: active=%d usage=%+v", g.active, usage)
	}
}

func TestRelayBudgetPersistsAcrossRestartAndClockRollback(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "relay-state.json")
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes, cfg.dailyBytes, cfg.peerDailyBytes = 100, 400, 200
	cfg.statePath = statePath
	now := time.Date(2026, 10, 16, 1, 0, 0, 0, time.UTC)
	src, dest := testPeerID(t), testPeerID(t)
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	g := testRelayGuard(t, cfg)
	g.now = func() time.Time { return now }
	token, ok := g.AdmitConnect(src, addr, dest)
	if !ok {
		t.Fatal("first circuit refused")
	}
	g.ConnectEstablished(token)
	if mode := mustMode(t, statePath); mode != 0o600 {
		t.Fatalf("state mode = %o, want 600", mode)
	}

	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
	restarted := testRelayGuard(t, cfg)
	restarted.now = func() time.Time { return now }
	if _, ok := restarted.AdmitConnect(src, addr, dest); ok {
		t.Fatal("restart reset per-peer budget")
	}
	now = now.Add(-24 * time.Hour)
	if err := restarted.Close(); err != nil {
		t.Fatal(err)
	}
	rolledBack := testRelayGuard(t, cfg)
	rolledBack.now = func() time.Time { return now }
	defer rolledBack.Close()
	if _, ok := rolledBack.AdmitConnect(src, addr, dest); ok {
		t.Fatal("clock rollback granted a fresh day")
	}
}

func TestRelayStateExclusiveLockAndFileSecurity(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "relay-state.json")
	if err := os.WriteFile(statePath, []byte(`{"max_day":"","days":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := defaultRelayServerConfig()
	cfg.statePath = statePath
	first := testRelayGuard(t, cfg)
	if _, err := newRelayGuard(cfg); err == nil || !strings.Contains(err.Error(), "already locked") {
		t.Fatalf("second relay state lock error = %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(statePath, 0o620); err != nil {
		t.Fatal(err)
	}
	if _, err := newRelayGuard(cfg); err == nil || !strings.Contains(err.Error(), "group/world writable") {
		t.Fatalf("insecure state mode error = %v", err)
	}
	if err := os.Chmod(statePath, 0o600); err != nil {
		t.Fatal(err)
	}
	symlinkPath := filepath.Join(dir, "state-link.json")
	if err := os.Symlink(statePath, symlinkPath); err != nil {
		t.Fatal(err)
	}
	cfg.statePath = symlinkPath
	if _, err := newRelayGuard(cfg); err == nil {
		t.Fatal("symlink relay state was accepted")
	}
}

type relayStateOwnerInfo struct {
	os.FileInfo
	stat syscall.Stat_t
}

func (i relayStateOwnerInfo) Sys() any { return &i.stat }

func TestRelayStateRequiresRunningUIDOwnership(t *testing.T) {
	path := filepath.Join(t.TempDir(), "relay-state.json")
	if err := os.WriteFile(path, []byte(`{"max_day":"","days":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	stat := *info.Sys().(*syscall.Stat_t)
	stat.Uid = uint32(os.Geteuid() + 1)
	err = validateRelayStateFile(relayStateOwnerInfo{FileInfo: info, stat: stat}, path)
	if err == nil || !strings.Contains(err.Error(), "want running uid") {
		t.Fatalf("foreign-owned state error = %v", err)
	}
}

func TestRelayStateCorruptionFailsClosed(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "relay-state.json")
	if err := os.WriteFile(statePath, []byte("not json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := defaultRelayServerConfig()
	cfg.statePath = statePath
	if _, err := newRelayGuard(cfg); err == nil || !strings.Contains(err.Error(), "invalid character") {
		t.Fatalf("corrupt state error = %v", err)
	}
}

func TestRelayGlobalDailyBudget(t *testing.T) {
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes, cfg.dailyBytes, cfg.peerDailyBytes = 100, 200, 1000
	g := testRelayGuard(t, cfg)
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	token, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t))
	if !ok {
		t.Fatal("first circuit refused")
	}
	g.ConnectEstablished(token)
	if _, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t)); ok {
		t.Fatal("global daily byte cap was not enforced")
	}
}

func TestRelayUTCDayRolloverChargesActiveCircuitAgainstNewDay(t *testing.T) {
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes, cfg.dailyBytes, cfg.peerDailyBytes = 100, 200, 200
	g := testRelayGuard(t, cfg)
	now := time.Date(2026, 10, 15, 23, 59, 59, 0, time.UTC)
	g.now = func() time.Time { return now }
	src, dest := testPeerID(t), testPeerID(t)
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	token, ok := g.AdmitConnect(src, addr, dest)
	if !ok {
		t.Fatal("pre-midnight circuit refused")
	}
	g.ConnectEstablished(token)

	now = now.Add(2 * time.Second)
	if _, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t)); ok {
		t.Fatal("new-day admission ignored the still-active circuit's remaining cap")
	}
	newDay := now.Format("2006-01-02")
	usage := g.state.Days[newDay]
	if usage == nil || usage.GlobalBytes != g.circuitCharge ||
		usage.Peers[src.String()].Bytes != g.circuitCharge ||
		usage.Peers[dest.String()].Bytes != g.circuitCharge {
		t.Fatalf("new-day active-circuit charge = %+v", usage)
	}
	g.ConnectClosed(token, 0)
	if _, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t)); ok {
		t.Fatal("closing the rolled-over circuit refunded the conservative new-day charge")
	}
}

func TestRelayUTCDayRolloverChargesCircuitBeforeClosingIt(t *testing.T) {
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes, cfg.dailyBytes, cfg.peerDailyBytes = 100, 200, 200
	g := testRelayGuard(t, cfg)
	now := time.Date(2026, 10, 15, 23, 59, 59, 0, time.UTC)
	g.now = func() time.Time { return now }
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	token, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t))
	if !ok {
		t.Fatal("pre-midnight circuit refused")
	}
	g.ConnectEstablished(token)
	now = now.Add(2 * time.Second)
	g.ConnectClosed(token, 0)
	if usage := g.state.Days[now.Format("2006-01-02")]; usage == nil || usage.GlobalBytes != g.circuitCharge {
		t.Fatalf("circuit close omitted new-day charge: %+v", usage)
	}
	if _, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t)); ok {
		t.Fatal("circuit close before admission restored the new-day budget")
	}
}

func TestRelayAllowFileAcceptsPeerIDsAndRosterPublicKeysAndReloads(t *testing.T) {
	privA, pubA, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	_ = privA
	idA, err := peer.IDFromPublicKey(pubA)
	if err != nil {
		t.Fatal(err)
	}
	privB, pubB, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	_ = privB
	idB, err := peer.IDFromPublicKey(pubB)
	if err != nil {
		t.Fatal(err)
	}
	rawB, err := pubB.Raw()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "relay.allow")
	if err := os.WriteFile(path, []byte(idA.String()+"\n"+fmt.Sprintf("%x\n", rawB)), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := defaultRelayServerConfig()
	cfg.allowFile = path
	g := testRelayGuard(t, cfg)
	allowNow := time.Now().Add(defaultRelayAllowStatEvery)
	g.allowlist.now = func() time.Time { return allowNow }
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	token, admitted := g.AdmitConnect(idA, addr, idB)
	if !g.AllowReserve(idA, addr) || !g.AllowReserve(idB, addr) || !admitted {
		t.Fatal("listed peer id/public key was refused")
	}
	g.ConnectFailed(token)

	other := testPeerID(t)
	_, admitted = g.AdmitConnect(idA, addr, other)
	if g.AllowReserve(other, addr) || admitted {
		t.Fatal("unlisted relay identity was allowed")
	}
	if err := os.WriteFile(path, []byte(other.String()+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
	allowNow = allowNow.Add(defaultRelayAllowStatEvery)
	if !g.AllowReserve(other, addr) || g.AllowReserve(idA, addr) {
		t.Fatal("allow file mtime reload did not replace identities")
	}
}

func TestRelayAllowFileFailsClosedWhenMissingOrMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.allow")
	cfg := defaultRelayServerConfig()
	cfg.allowFile = path
	g := testRelayGuard(t, cfg)
	allowNow := time.Now().Add(defaultRelayAllowStatEvery)
	g.allowlist.now = func() time.Time { return allowNow }
	p := testPeerID(t)
	addr := multiaddr.StringCast("/ip4/192.0.2.1/tcp/1")
	if g.AllowReserve(p, addr) {
		t.Fatal("missing allow file did not fail closed")
	}
	if err := os.WriteFile(path, []byte("not-an-identity\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	allowNow = allowNow.Add(defaultRelayAllowStatEvery)
	if g.AllowReserve(p, addr) {
		t.Fatal("malformed allow file did not fail closed")
	}
	if err := os.WriteFile(path, []byte(p.String()+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	allowNow = allowNow.Add(defaultRelayAllowStatEvery)
	if !g.AllowReserve(p, addr) {
		t.Fatal("valid replacement allow file was not reloaded")
	}
	if err := os.Chmod(path, 0); err != nil {
		t.Fatal(err)
	}
	allowNow = allowNow.Add(defaultRelayAllowStatEvery)
	if g.AllowReserve(p, addr) {
		t.Fatal("unreadable allow file did not fail closed")
	}
}

func TestRelayAllowFileMalformedLineIsLogged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "relay.allow")
	if err := os.WriteFile(path, []byte("# comment\n\nnot-an-identity\n"), 0o640); err != nil {
		t.Fatal(err)
	}
	var logs bytes.Buffer
	old := log.Writer()
	log.SetOutput(&logs)
	defer log.SetOutput(old)
	if (&relayAllowlist{path: path}).contains(testPeerID(t)) {
		t.Fatal("malformed allow file was accepted")
	}
	if !strings.Contains(logs.String(), "line 3") {
		t.Fatalf("malformed line log = %q", logs.String())
	}
}

func TestRelayAllowFileSecureLoadingAndContentHashReload(t *testing.T) {
	idA, idB := testPeerID(t), testPeerID(t)
	if len(idA.String()) != len(idB.String()) {
		t.Fatal("test peer IDs unexpectedly differ in length")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "relay.allow")
	bodyA := []byte("# generated by root sync job\r\n\r\n" + idA.String() + "\r\n")
	if err := os.WriteFile(path, bodyA, 0o640); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	a := &relayAllowlist{path: path, now: func() time.Time { return now }}
	if !a.contains(idA) {
		t.Fatal("0640 allow file with comments and CRLF was refused")
	}
	bodyB := []byte("# generated by root sync job\r\n\r\n" + idB.String() + "\r\n")
	if len(bodyA) != len(bodyB) {
		t.Fatal("test replacement changed size")
	}
	if err := os.WriteFile(path, bodyB, 0o640); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	now = now.Add(defaultRelayAllowHashEvery)
	if !a.contains(idB) || a.contains(idA) {
		t.Fatal("same-size/same-mtime content replacement was not detected")
	}
	symlinkPath := filepath.Join(dir, "relay-link.allow")
	if err := os.Symlink(path, symlinkPath); err != nil {
		t.Fatal(err)
	}
	if (&relayAllowlist{path: symlinkPath}).contains(idB) {
		t.Fatal("symlink allow file was accepted")
	}
}

func TestRelayAllowFileChecksAreCachedAndRateLimitRunsFirst(t *testing.T) {
	src, dest := testPeerID(t), testPeerID(t)
	path := filepath.Join(t.TempDir(), "relay.allow")
	if err := os.WriteFile(path, []byte(src.String()+"\n"+dest.String()+"\n"), 0o640); err != nil {
		t.Fatal(err)
	}
	cfg := defaultRelayServerConfig()
	cfg.allowFile = path
	g := testRelayGuard(t, cfg)
	rateNow := time.Now()
	allowNow := rateNow
	g.now = func() time.Time { return rateNow }
	g.allowlist.now = func() time.Time { return allowNow }
	addr := multiaddr.StringCast("/ip4/192.0.2.90/tcp/1")

	// Exhaust the cheap bucket with failed lifecycle outcomes while the cached allowlist is valid.
	for i := 0; i < int(defaultRelayRateBurst); i++ {
		token, ok := g.AdmitConnect(src, addr, dest)
		if !ok {
			t.Fatalf("attempt %d refused early", i+1)
		}
		g.ConnectFailed(token)
	}
	// Change the file after the hash interval. If allow-file work happened before the limiter,
	// this attempt would reload the replacement; the denied attempt must leave the cache untouched.
	if err := os.WriteFile(path, []byte("not-an-identity\n"), 0o640); err != nil {
		t.Fatal(err)
	}
	allowNow = allowNow.Add(defaultRelayAllowHashEvery)
	if _, ok := g.AdmitConnect(src, addr, dest); ok {
		t.Fatal("rate-exhausted source was admitted")
	}
	if !g.allowlist.loaded {
		t.Fatal("rate-limited attempt reached and invalidated the allow-file cache")
	}
	if g.allowlist.lastHash.Equal(allowNow) {
		t.Fatal("rate-limited attempt performed allow-file hash work")
	}
}

func TestRelayAllowFileBoundsFailClosed(t *testing.T) {
	id := testPeerID(t).String()
	path := filepath.Join(t.TempDir(), "relay.allow")
	var entries strings.Builder
	for i := 0; i <= maxRelayAllowEntries; i++ {
		entries.WriteString(id)
		entries.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(entries.String()), 0o640); err != nil {
		t.Fatal(err)
	}
	if (&relayAllowlist{path: path}).contains(peer.ID(id)) {
		t.Fatal("allow file above entry bound was accepted")
	}
	if err := os.WriteFile(path, bytes.Repeat([]byte{'#'}, maxRelayAllowFileBytes+1), 0o640); err != nil {
		t.Fatal(err)
	}
	if (&relayAllowlist{path: path}).contains(peer.ID(id)) {
		t.Fatal("allow file above byte bound was accepted")
	}
}

func TestRelayAdmissionConcurrentInterleavedOutcomes(t *testing.T) {
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes = 1
	cfg.maxCircuits = 64
	cfg.circuitsPerPeer = 2
	cfg.dailyBytes = 128
	cfg.peerDailyBytes = 4
	g := testRelayGuard(t, cfg)
	type admitted struct {
		token     uint64
		src, dest peer.ID
	}
	var mu sync.Mutex
	var got []admitted
	var wg sync.WaitGroup
	for i := 0; i < 160; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			src, dest := testPeerID(t), testPeerID(t)
			addr := multiaddr.StringCast(fmt.Sprintf("/ip4/198.51.%d.%d/tcp/1", i/250, i%250+1))
			if token, ok := g.AdmitConnect(src, addr, dest); ok {
				mu.Lock()
				got = append(got, admitted{token: token, src: src, dest: dest})
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(got) != cfg.maxCircuits {
		t.Fatalf("admitted %d attempts, want global cap %d", len(got), cfg.maxCircuits)
	}
	for i, attempt := range got {
		wg.Add(1)
		go func(i int, attempt admitted) {
			defer wg.Done()
			if i%2 == 0 {
				g.ConnectEstablished(attempt.token)
			} else {
				g.ConnectFailed(attempt.token)
			}
		}(i, attempt)
	}
	wg.Wait()
	g.mu.Lock()
	usage := g.state.Days[g.state.MaxDay]
	if g.active != 32 || len(g.attempts) != 32 || usage.GlobalBytes != 64 {
		t.Fatalf("after interleaving: active=%d attempts=%d bytes=%d", g.active, len(g.attempts), usage.GlobalBytes)
	}
	for i, attempt := range got {
		_, srcCharged := usage.Peers[attempt.src.String()]
		_, destCharged := usage.Peers[attempt.dest.String()]
		if (i%2 == 0) != (srcCharged && destCharged) {
			t.Fatalf("attempt %d charge attribution src=%v dest=%v", i, srcCharged, destCharged)
		}
	}
	g.mu.Unlock()
	for i, attempt := range got {
		if i%2 == 0 {
			wg.Add(1)
			go func(token uint64) { defer wg.Done(); g.ConnectClosed(token, 0) }(attempt.token)
		}
	}
	wg.Wait()
	if g.active != 0 || len(g.attempts) != 0 {
		t.Fatalf("close leak: active=%d attempts=%d", g.active, len(g.attempts))
	}
}

func TestRelayAdmissionConcurrentPerPeerAndBudgetBounds(t *testing.T) {
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes = 5
	cfg.maxCircuits = 64
	cfg.circuitsPerPeer = 2
	cfg.dailyBytes = 20
	cfg.peerDailyBytes = 20
	g := testRelayGuard(t, cfg)
	src := testPeerID(t)
	var tokensMu sync.Mutex
	var tokens []uint64
	var wg sync.WaitGroup
	for i := 0; i < 80; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			addr := multiaddr.StringCast(fmt.Sprintf("/ip4/203.0.113.%d/tcp/1", i%250+1))
			if token, ok := g.AdmitConnect(src, addr, testPeerID(t)); ok {
				tokensMu.Lock()
				tokens = append(tokens, token)
				tokensMu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(tokens) != 2 {
		t.Fatalf("per-peer concurrent admissions = %d, want 2", len(tokens))
	}
	for _, token := range tokens {
		g.ConnectEstablished(token)
	}
	usage := g.state.Days[g.state.MaxDay]
	if usage.GlobalBytes != cfg.dailyBytes || usage.Peers[src.String()].Bytes != cfg.peerDailyBytes {
		t.Fatalf("budget bounds: global=%d peer=%d", usage.GlobalBytes, usage.Peers[src.String()].Bytes)
	}
}

func TestRelayConnectAttemptsAreRateLimitedPerAuthenticatedIdentityAndIP(t *testing.T) {
	g := testRelayGuard(t, defaultRelayServerConfig())
	addr := multiaddr.StringCast("/ip4/192.0.2.44/tcp/1")
	src := testPeerID(t)
	for i := 0; i < int(defaultRelayRateBurst); i++ {
		token, ok := g.AdmitConnect(src, addr, testPeerID(t))
		if !ok {
			t.Fatalf("attempt %d was rate-limited early", i+1)
		}
		g.ConnectFailed(token)
	}
	if _, ok := g.AdmitConnect(src, addr, testPeerID(t)); ok {
		t.Fatal("identity/IP burst limit was not enforced")
	}
	if token, ok := g.AdmitConnect(testPeerID(t), addr, testPeerID(t)); !ok {
		t.Fatal("a distinct authenticated identity sharing the IP was rate-limited")
	} else {
		g.ConnectFailed(token)
	}
	otherAddr := multiaddr.StringCast("/ip4/192.0.2.45/tcp/1")
	if token, ok := g.AdmitConnect(src, otherAddr, testPeerID(t)); !ok {
		t.Fatal("the same identity from a distinct IP did not receive its own bucket")
	} else {
		g.ConnectFailed(token)
	}
}

func TestRelayRateSourcesEvictLRUInsteadOfFailingClosed(t *testing.T) {
	g := testRelayGuard(t, defaultRelayServerConfig())
	src, dest := testPeerID(t), testPeerID(t)
	for i := 0; i < maxRelayRateSources; i++ {
		addr := multiaddr.StringCast(fmt.Sprintf("/ip4/198.18.%d.%d/tcp/1", i/256, i%256))
		token, ok := g.AdmitConnect(src, addr, dest)
		if !ok {
			t.Fatalf("source %d refused before LRU capacity", i)
		}
		g.ConnectFailed(token)
	}
	legitimate := testPeerID(t)
	token, ok := g.AdmitConnect(legitimate, multiaddr.StringCast("/ip4/203.0.113.99/tcp/1"), testPeerID(t))
	if !ok {
		t.Fatal("new legitimate source was refused at rate-source capacity")
	}
	g.ConnectFailed(token)
	g.rateMu.Lock()
	defer g.rateMu.Unlock()
	if len(g.rate) != maxRelayRateSources || g.rateLRU.Len() != maxRelayRateSources {
		t.Fatalf("bounded rate LRU map=%d list=%d want=%d", len(g.rate), g.rateLRU.Len(), maxRelayRateSources)
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

func TestPreferredDirectDialAddressesAreCapped(t *testing.T) {
	var input []multiaddr.Multiaddr
	for i := 1; i <= 24; i++ {
		input = append(input, multiaddr.StringCast(fmt.Sprintf("/ip4/10.0.0.%d/tcp/1", i)))
	}
	got := preferredDirectDialAddrs(input)
	if len(got) != maxDirectDialAddrs {
		t.Fatalf("candidate count = %d, want %d: %v", len(got), maxDirectDialAddrs, got)
	}
}

func TestOutboundDialPolicy(t *testing.T) {
	newPeer := func() peer.ID {
		t.Helper()
		priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		id, err := peer.IDFromPrivateKey(priv)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	relayID, forwardID, otherID := newPeer(), newPeer(), newPeer()
	relay := peer.AddrInfo{ID: relayID, Addrs: []multiaddr.Multiaddr{multiaddr.StringCast("/ip4/127.0.0.1/tcp/4001")}}
	forward := "127.0.0.1:1=/ip4/127.0.0.1/tcp/4002/p2p/" + forwardID.String()
	policy, err := newOutboundDialPolicy([]peer.AddrInfo{relay}, []string{forward}, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, addr := range []string{
		"/ip4/127.0.0.1/tcp/1",
		"/ip4/0.0.0.0/tcp/1",
		"/ip4/169.254.1.1/tcp/1",
		"/ip4/224.0.0.1/udp/1/quic-v1",
		"/ip6/::/tcp/1",
		"/ip6/::1/tcp/1",
		"/ip6/fe80::1/tcp/1",
		"/ip6/ff02::1/udp/1/quic-v1",
	} {
		if policy.InterceptAddrDial(otherID, multiaddr.StringCast(addr)) {
			t.Errorf("allowed unsafe address %s", addr)
		}
	}
	for _, addr := range []string{
		"/ip4/192.168.1.10/tcp/1",
		"/ip6/fd00::10/tcp/1",
		"/ip4/8.8.8.8/tcp/1",
		"/ip6/2606:4700:4700::1111/tcp/1",
	} {
		if !policy.InterceptAddrDial(otherID, multiaddr.StringCast(addr)) {
			t.Errorf("blocked allowed address %s", addr)
		}
	}
	if !policy.InterceptAddrDial(forwardID, multiaddr.StringCast("/ip4/127.0.0.1/tcp/4002")) {
		t.Fatal("blocked operator-configured forward target")
	}
	relayRoute := multiaddr.StringCast("/ip4/127.0.0.1/tcp/4001/p2p/" + relayID.String() + "/p2p-circuit")
	if !policy.InterceptAddrDial(otherID, relayRoute) {
		t.Fatal("blocked route through operator-configured relay")
	}
	var all []multiaddr.Multiaddr
	for i := 1; i <= 20; i++ {
		all = append(all, multiaddr.StringCast(fmt.Sprintf("/ip4/10.2.0.%d/tcp/1", i)))
	}
	selected := preferredDirectDialAddrs(all)
	allowed := 0
	if err := policy.withDialCandidates(otherID, selected, func() error {
		for _, addr := range all {
			if policy.InterceptAddrDial(otherID, addr) {
				allowed++
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if allowed != maxDirectDialAddrs {
		t.Fatalf("gater allowed %d direct candidates, want %d", allowed, maxDirectDialAddrs)
	}
}

func TestDirectDialAttemptsAreCapped(t *testing.T) {
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	id, err := peer.IDFromPrivateKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	var candidates []multiaddr.Multiaddr
	for i := 1; i <= 20; i++ {
		candidates = append(candidates, multiaddr.StringCast(fmt.Sprintf("/ip4/10.1.0.%d/tcp/1", i)))
	}
	var attempts atomic.Int32
	retryDirectDialAttempts(
		context.Background(),
		id,
		directDialRetries,
		func() bool { return false },
		func() []multiaddr.Multiaddr { return candidates },
		func(_ context.Context, info peer.AddrInfo) error {
			attempts.Add(1)
			if len(info.Addrs) != maxDirectDialAddrs {
				t.Fatalf("attempt got %d candidates, want %d", len(info.Addrs), maxDirectDialAddrs)
			}
			return errors.New("unreachable")
		},
	)
	if got := attempts.Load(); got != directDialRetries {
		t.Fatalf("attempts = %d, want %d", got, directDialRetries)
	}
}

func TestDialHandlerConcurrencyCap(t *testing.T) {
	slots := make(chan struct{}, maxConcurrentDials)
	release := make(chan struct{})
	started := make(chan struct{}, maxConcurrentDials)
	clients := make([]net.Conn, 0, maxConcurrentDials)
	for i := 0; i < maxConcurrentDials; i++ {
		server, client := net.Pipe()
		clients = append(clients, client)
		dispatchDialConn(server, slots, func(c net.Conn) {
			started <- struct{}{}
			<-release
			_ = c.Close()
		})
	}
	for i := 0; i < maxConcurrentDials; i++ {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("handler did not start")
		}
	}
	extraServer, extraClient := net.Pipe()
	dispatched := make(chan struct{})
	go func() {
		dispatchDialConn(extraServer, slots, func(net.Conn) { t.Error("busy handler was started") })
		close(dispatched)
	}()
	_ = extraClient.SetReadDeadline(time.Now().Add(time.Second))
	line, err := bufio.NewReader(extraClient).ReadString('\n')
	if err != nil || line != "ERR busy\n" {
		t.Fatalf("busy reply = %q, %v", line, err)
	}
	<-dispatched
	_ = extraClient.Close()
	close(release)
	for _, client := range clients {
		_ = client.Close()
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

func TestControlProtocolRejectsBadAndOverlongLines(t *testing.T) {
	for _, line := range []string{"DIAL not-a-peer-id\n", "FORWARD not-a-peer-id\n", "NOPE x\n", "DIAL " + strings.Repeat("x", maxDialLine) + "\n"} {
		server, client := net.Pipe()
		go handleDialConn(nil, server, nil, time.Second, nil)
		go func() { _, _ = io.WriteString(client, line) }()
		got, err := bufio.NewReader(client).ReadString('\n')
		client.Close()
		if err != nil || !strings.HasPrefix(got, "ERR ") {
			t.Fatalf("line %q: got %q, %v", line[:min(len(line), 40)], got, err)
		}
	}
}

func testPeerID(t *testing.T) peer.ID {
	t.Helper()
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	id, err := peer.IDFromPrivateKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

type gatedAcceptListener struct {
	*net.TCPListener
	accepted chan struct{}
	release  chan struct{}
	once     sync.Once
}

func (l *gatedAcceptListener) Accept() (net.Conn, error) {
	c, err := l.TCPListener.Accept()
	if err == nil {
		l.once.Do(func() { close(l.accepted) })
		<-l.release
	}
	return c, err
}

func TestDynamicForwardsAreCachedAndLRUEvictIdle(t *testing.T) {
	m := newForwardManager(nil, nil, time.Second, nil, false)
	defer m.close()
	clock := time.Unix(1, 0)
	m.now = func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	}
	ids := make([]peer.ID, maxDynamicForwards+1)
	addrs := make(map[peer.ID]string)
	for i := 0; i < maxDynamicForwards; i++ {
		ids[i] = testPeerID(t)
		addr, err := m.get(ids[i])
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(addr, "127.0.0.1:") {
			t.Fatalf("forward address = %q, want loopback", addr)
		}
		addrs[ids[i]] = addr
	}
	if got, err := m.get(ids[0]); err != nil || got != addrs[ids[0]] {
		t.Fatalf("cached forward = %q, %v; want %q", got, err, addrs[ids[0]])
	}
	ids[maxDynamicForwards] = testPeerID(t)
	if _, err := m.get(ids[maxDynamicForwards]); err != nil {
		t.Fatal(err)
	}
	m.mu.Lock()
	_, keptRecent := m.entries[ids[0]]
	_, evictedOldest := m.entries[ids[1]]
	count := len(m.entries)
	m.mu.Unlock()
	if !keptRecent || evictedOldest || count != maxDynamicForwards {
		t.Fatalf("LRU state: kept recent=%v, oldest present=%v, count=%d", keptRecent, evictedOldest, count)
	}
	if c, err := net.DialTimeout("tcp", addrs[ids[1]], 100*time.Millisecond); err == nil {
		_ = c.Close()
		t.Fatal("evicted forward listener still accepted connections")
	}
}

func TestDynamicForwardAcceptIsCountedBeforeEviction(t *testing.T) {
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHost(priv, "/ip4/127.0.0.1/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	m := newForwardManager(h, nil, 2*time.Second, nil, false)
	defer m.close()
	clock := time.Unix(1, 0)
	m.now = func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	}
	var gated *gatedAcceptListener
	listenCalls := 0
	m.listen = func(_, _ string) (net.Listener, error) {
		ln, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.ParseIP("127.0.0.1")})
		if err != nil {
			return nil, err
		}
		listenCalls++
		if listenCalls == 1 {
			gated = &gatedAcceptListener{
				TCPListener: ln,
				accepted:    make(chan struct{}),
				release:     make(chan struct{}),
			}
			return gated, nil
		}
		return ln, nil
	}

	ids := make([]peer.ID, maxDynamicForwards+1)
	var firstAddr string
	for i := 0; i < maxDynamicForwards; i++ {
		ids[i] = testPeerID(t)
		addr, err := m.get(ids[i])
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			firstAddr = addr
		}
	}
	client, err := net.Dial("tcp", firstAddr)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	select {
	case <-gated.accepted:
	case <-time.After(time.Second):
		t.Fatal("listener did not accept the test connection")
	}

	ids[maxDynamicForwards] = testPeerID(t)
	evicted := make(chan error, 1)
	go func() {
		_, err := m.get(ids[maxDynamicForwards])
		evicted <- err
	}()
	select {
	case err := <-evicted:
		t.Fatalf("eviction completed before accepted connection was counted: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	close(gated.release)
	select {
	case err := <-evicted:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("eviction did not resume after Accept completed")
	}

	m.mu.Lock()
	first := m.entries[ids[0]]
	active := 0
	if first != nil {
		active = first.active
	}
	_, evictedIdle := m.entries[ids[1]]
	m.mu.Unlock()
	if first == nil || active != 1 {
		t.Fatalf("accepted forward was evicted or not counted: entry=%v active=%d", first != nil, active)
	}
	if evictedIdle {
		t.Fatal("oldest idle forward was not evicted after accepted forward became busy")
	}
	if err := client.SetReadDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Read(make([]byte, 1)); err == nil {
		t.Fatal("unexpected data from forward")
	} else if netErr, ok := err.(net.Error); !ok || !netErr.Timeout() {
		t.Fatalf("accepted client was reset during eviction: %v", err)
	}
}

func TestDynamicForwardLimitDoesNotEvictBusyEntries(t *testing.T) {
	m := newForwardManager(nil, nil, time.Second, nil, false)
	defer m.close()
	for i := 0; i < maxDynamicForwards; i++ {
		id := testPeerID(t)
		if _, err := m.get(id); err != nil {
			t.Fatal(err)
		}
		m.mu.Lock()
		m.entries[id].active = 1
		m.mu.Unlock()
	}
	if _, err := m.get(testPeerID(t)); err == nil || !strings.Contains(err.Error(), "busy") {
		t.Fatalf("full busy cache error = %v", err)
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

func TestIdentifyPushCannotBypassOutboundDialPolicy(t *testing.T) {
	newKey := func() crypto.PrivKey {
		t.Helper()
		priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		return priv
	}
	relay, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{relayService: true})
	if err != nil {
		t.Fatal(err)
	}
	defer relay.Close()
	relayInfo := peer.AddrInfo{ID: relay.ID(), Addrs: relay.Addrs()}
	policyA, err := newOutboundDialPolicy([]peer.AddrInfo{relayInfo}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	policyB, err := newOutboundDialPolicy([]peer.AddrInfo{relayInfo}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	a, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{disableHolePunching: true, dialPolicy: policyA})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{disableHolePunching: true, dialPolicy: policyB})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := b.Connect(ctx, relayInfo); err != nil {
		t.Fatalf("B connect relay: %v", err)
	}
	if _, err := relayclient.Reserve(ctx, b, relayInfo); err != nil {
		t.Fatalf("B reserve relay: %v", err)
	}
	circuitAddrs := addRelayCircuitAddrs(a, b.ID(), []peer.AddrInfo{relayInfo})
	if err := a.Connect(ctx, peer.AddrInfo{ID: b.ID(), Addrs: circuitAddrs}); err != nil {
		t.Fatalf("A connect B through relay: %v", err)
	}
	conn := a.Network().ConnsToPeer(b.ID())[0]
	ids := a.(interface{ IDService() identify.IDService }).IDService()
	select {
	case <-ids.IdentifyWait(conn):
	case <-ctx.Done():
		t.Fatal("initial identify timed out")
	}

	before := make(map[string]struct{})
	for _, addr := range b.Addrs() {
		before[addr.String()] = struct{}{}
	}
	if err := b.Network().Listen(multiaddr.StringCast("/ip4/127.0.0.1/tcp/0")); err != nil {
		t.Fatal(err)
	}
	var pushed multiaddr.Multiaddr
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, addr := range a.Peerstore().Addrs(b.ID()) {
			if !manet.IsIPLoopback(addr) {
				continue
			}
			if _, existed := before[addr.String()]; !existed {
				pushed = addr
				break
			}
		}
		if pushed != nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if pushed == nil {
		t.Fatalf("loopback address was not received through Identify Push; peerstore: %v", a.Peerstore().Addrs(b.ID()))
	}

	if err := a.Network().ClosePeer(b.ID()); err != nil {
		t.Fatal(err)
	}
	a.Peerstore().ClearAddrs(b.ID())
	a.Peerstore().AddAddr(b.ID(), pushed, peerstore.PermanentAddrTTL)
	dialCtx, dialCancel := context.WithTimeout(context.Background(), time.Second)
	defer dialCancel()
	err = a.Connect(network.WithForceDirectDial(dialCtx, forceDirectReason), peer.AddrInfo{ID: b.ID(), Addrs: []multiaddr.Multiaddr{pushed}})
	if err == nil {
		t.Fatalf("connected to peer-supplied loopback address %s", pushed)
	}
	if directConn(a, b.ID()) != nil {
		t.Fatalf("peer-supplied loopback address %s produced a direct connection", pushed)
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
	go serveDialListener(a, ln, nil, 5*time.Second, nil, false)
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

func controlRequest(t *testing.T, socket, request string) string {
	t.Helper()
	c, err := net.Dial("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := io.WriteString(c, request+"\n"); err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(c).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	return line
}

func TestForwardListenerDirectActivationRoundTripAndCache(t *testing.T) {
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
	runInboundWithPolicy(b, echo.Addr().String(), true, false)

	sockPath := filepath.Join(t.TempDir(), "nakshatra", "p2p.sock")
	ln, err := listenDialSocket(sockPath)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go serveDialListener(a, ln, nil, 5*time.Second, nil, false)
	request := "FORWARD " + b.ID().String()
	line := controlRequest(t, sockPath, request)
	if !strings.HasPrefix(line, "OK 127.0.0.1:") {
		t.Fatalf("reply = %q", line)
	}
	addr := strings.TrimSpace(strings.TrimPrefix(line, "OK "))
	if again := controlRequest(t, sockPath, request); again != line {
		t.Fatalf("cached reply = %q, want %q", again, line)
	}

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	want := []byte("inference activation over a direct libp2p stream")
	if _, err := c.Write(want); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(want))
	if _, err := io.ReadFull(c, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("round trip: got %q, want %q", got, want)
	}
	if directConn(a, b.ID()) == nil {
		t.Fatal("activation stream was not carried by a direct connection")
	}
}

func TestForwardRequestFailsWithoutDirectPath(t *testing.T) {
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHost(priv, "/ip4/127.0.0.1/tcp/0", natOpts{})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	sockPath := filepath.Join(t.TempDir(), "nakshatra", "p2p.sock")
	ln, err := listenDialSocket(sockPath)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go serveDialListener(h, ln, nil, 50*time.Millisecond, nil, false)
	line := controlRequest(t, sockPath, "FORWARD "+testPeerID(t).String())
	if !strings.HasPrefix(line, "ERR no direct connection") {
		t.Fatalf("reply = %q", line)
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
	if err := waitDirect(directCtx, a, b.ID(), []peer.AddrInfo{relayInfo}, nil); err != nil {
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

func relayOnlyTestHosts(t *testing.T, circuitBytes int64) (host.Host, host.Host, peer.AddrInfo) {
	t.Helper()
	newKey := func() crypto.PrivKey {
		priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		return priv
	}
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes = circuitBytes
	cfg.peerDailyBytes = 32 * circuitBytes
	relay, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{relayService: true, relayConfig: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = relay.Close() })
	relayInfo := peer.AddrInfo{ID: relay.ID(), Addrs: relay.Addrs()}
	policyA, err := newOutboundDialPolicy([]peer.AddrInfo{relayInfo}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	policyB, err := newOutboundDialPolicy([]peer.AddrInfo{relayInfo}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	a, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{
		disableHolePunching: true, staticRelays: []peer.AddrInfo{relayInfo}, dialPolicy: policyA,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{
		disableHolePunching: true, staticRelays: []peer.AddrInfo{relayInfo}, dialPolicy: policyB,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := b.Connect(ctx, relayInfo); err != nil {
		t.Fatalf("worker connect relay: %v", err)
	}
	if _, err := relayclient.Reserve(ctx, b, relayInfo); err != nil {
		t.Fatalf("worker reserve relay: %v", err)
	}
	circuitAddrs := addRelayCircuitAddrs(a, b.ID(), []peer.AddrInfo{relayInfo})
	if err := policyA.withDialCandidates(b.ID(), circuitAddrs, func() error {
		return a.Connect(ctx, peer.AddrInfo{ID: b.ID(), Addrs: circuitAddrs})
	}); err != nil {
		t.Fatalf("dial relay circuit: %v", err)
	}
	if directConn(a, b.ID()) != nil || relayedConn(a, b.ID()) == nil {
		t.Fatal("test peers did not connect exclusively through the relay")
	}
	return a, b, relayInfo
}

func TestRelayDestinationWithoutReservationConsumesNoBudget(t *testing.T) {
	newKey := func() crypto.PrivKey {
		priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		return priv
	}
	cfg := defaultRelayServerConfig()
	cfg.circuitBytes = 100
	cfg.dailyBytes = 1000
	cfg.peerDailyBytes = 1000
	relay, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{relayService: true, relayConfig: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	defer relay.Close()
	relayWithGuard, ok := relay.(*relayHost)
	if !ok {
		t.Fatal("relay host does not expose identity-bound admission")
	}
	relayInfo := peer.AddrInfo{ID: relay.ID(), Addrs: relay.Addrs()}
	policy, err := newOutboundDialPolicy([]peer.AddrInfo{relayInfo}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	src, err := newHost(newKey(), "/ip4/127.0.0.1/tcp/0", natOpts{
		disableHolePunching: true, staticRelays: []peer.AddrInfo{relayInfo}, dialPolicy: policy,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	destKey := newKey()
	destID, err := peer.IDFromPrivateKey(destKey)
	if err != nil {
		t.Fatal(err)
	}
	circuitAddrs := addRelayCircuitAddrs(src, destID, []peer.AddrInfo{relayInfo})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := policy.withDialCandidates(destID, circuitAddrs, func() error {
		return src.Connect(ctx, peer.AddrInfo{ID: destID, Addrs: circuitAddrs})
	}); err == nil {
		t.Fatal("relay connected to a destination without a reservation")
	}
	g := relayWithGuard.guard
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.attempts) != 0 || g.active != 0 {
		t.Fatalf("unreserved destination reached admission: attempts=%d active=%d", len(g.attempts), g.active)
	}
	for _, usage := range g.state.Days {
		if usage.GlobalBytes != 0 || len(usage.Peers) != 0 {
			t.Fatalf("unreserved destination consumed budget: %+v", usage)
		}
	}
}

func TestRelayedActivationRequiresBothEndsToOptIn(t *testing.T) {
	for _, tc := range []struct {
		name, dialerOpt, workerOpt string
		want                       bool
	}{
		{"neither", "off", "off", false},
		{"dialer only", "on", "off", false},
		{"worker only", "off", "on", false},
		{"both", "on", "on", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b, relayInfo := relayOnlyTestHosts(t, 1<<20)
			payload := []byte("activation over an encrypted relay circuit")
			workerAllows := tc.workerOpt == "on"
			b.SetStreamHandler(activationProto, func(s network.Stream) {
				if !activationPathAllowed(true, workerAllows, s.Conn().Stat().Limited, s.Conn().RemoteMultiaddr().String()) {
					_ = s.Reset()
					return
				}
				defer s.Close()
				buf := make([]byte, len(payload))
				if _, err := io.ReadFull(s, buf); err == nil {
					_, _ = s.Write(buf)
				}
			})

			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			dialerAllows := tc.dialerOpt == "on"
			_, err := waitActivationPath(ctx, a, b.ID(), []peer.AddrInfo{relayInfo}, nil, dialerAllows)
			cancel()
			if err != nil {
				if tc.want {
					t.Fatalf("activation path: %v", err)
				}
				return
			}
			streamCtx, streamCancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer streamCancel()
			s, err := openActivationStream(streamCtx, b.ID(), a.NewStream)
			if err == nil {
				_, err = s.Write(payload)
			}
			got := make([]byte, len(payload))
			if err == nil {
				_, err = io.ReadFull(s, got)
			}
			if s != nil {
				_ = s.Close()
			}
			succeeded := err == nil && bytes.Equal(got, payload)
			if succeeded != tc.want {
				t.Fatalf("relayed round trip succeeded=%v, want %v (err=%v)", succeeded, tc.want, err)
			}
		})
	}
}

func TestStaticForwardProductionListenerRequiresDialerRelayOptIn(t *testing.T) {
	a, b, relayInfo := relayOnlyTestHosts(t, 1<<20)
	engine, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = engine.Close() })
	go func() {
		for {
			conn, err := engine.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				_, _ = io.Copy(conn, conn)
			}()
		}
	}()
	// The worker explicitly accepts relay activation. This test must therefore fail solely because
	// the real static -forward listener's dialer did not opt into relay fallback.
	runInboundWithPolicy(b, engine.Addr().String(), true, true)
	relayConn := relayedConn(a, b.ID())
	if relayConn == nil {
		t.Fatal("test setup has no relay-only connection")
	}
	peerAddr := relayConn.RemoteMultiaddr().Encapsulate(
		multiaddr.StringCast("/p2p/" + b.ID().String())).String()
	forward, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go serveForwardListener(a, forward, peerAddr, []peer.AddrInfo{relayInfo}, nil,
		100*time.Millisecond, false)
	t.Cleanup(func() { _ = forward.Close() })

	client, err := net.Dial("tcp", forward.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err := client.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Write([]byte("must-not-cross-relay")); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 1)
	if _, err := client.Read(buf); err == nil {
		t.Fatal("static -forward carried activation over relay without -relay-activation-fallback")
	}
}

func TestRelayCircuitByteCapCutsActivation(t *testing.T) {
	const capBytes = int64(32 << 10)
	a, b, _ := relayOnlyTestHosts(t, capBytes)
	received := make(chan int64, 1)
	b.SetStreamHandler(activationProto, func(s network.Stream) {
		defer s.Close()
		n, _ := io.Copy(io.Discard, s)
		received <- n
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s, err := openActivationStream(ctx, b.ID(), a.NewStream)
	if err != nil {
		t.Fatal(err)
	}
	payload := bytes.Repeat([]byte{0xa5}, int(4*capBytes))
	_, _ = s.Write(payload)
	_ = s.CloseWrite()
	select {
	case n := <-received:
		if n >= int64(len(payload)) || n > capBytes {
			t.Fatalf("relay delivered %d bytes from a %d-byte write with cap %d", n, len(payload), capBytes)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for capped relay stream to close")
	}
	_ = s.Close()
}
