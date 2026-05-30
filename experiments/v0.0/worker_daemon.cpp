// Nakshatra worker daemon (M5).
//
// Long-lived process spawned by the Python gRPC worker. Loads a sub-GGUF at
// startup, then reads framed binary messages from stdin and writes framed
// responses to stdout. One message = one llama_decode call.
//
// Wire format (all little-endian):
//   request:  u32 cmd | u32 n_tokens | u32 start_pos | u32 flags | u32 payload_bytes | bytes payload
//   response: u32 status | u32 payload_bytes | bytes payload
//
// 2026-05-30 fabric C++ Phase A.3 — alternate shm transport. Same wire
// envelope as stdio; the request/response bytes ride two SPSC shared-memory
// rings instead of stdin/stdout. Activated when both --fabric-shm-req <path>
// and --fabric-shm-resp <path> are passed on argv (after the positional
// args). Stdio path stays the default; existing callers unchanged.
//
// Flags:
//   bit 0 (0x1) = keep_kv. If set, daemon skips llama_memory_clear before
//                 decode (streaming generation). Tokens are placed in KV
//                 cache at positions [start_pos, start_pos + n_tokens).
//                 If unset, daemon clears KV first (single-shot prefill).
//
// Commands:
//   1 = TOKEN_DECODE  payload = int32 token_ids[n_tokens]
//                     response payload = float32 hidden[n_tokens * n_embd]
//                                                         (worker 0/middle)
//                                       OR int32 top_token (last worker —
//                                       caller can detect by payload size)
//   2 = EMBD_DECODE   payload = float32 hidden[n_tokens * n_embd]
//                     response payload = same shape as TOKEN_DECODE response
//   3 = INFO          payload empty
//                     response payload = int32 n_layer_start, n_layer_end,
//                                              n_embd, has_token_embd,
//                                              has_lm_head, n_vocab
//
// Status codes: 0 = ok, 1 = decode error, 2 = wire format error, 3 = arch
// not supported.

#include "common.h"
#include "llama.h"

#include "shm_ring.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

// ── Transport: stdio (default) or shm (--fabric-shm-{req,resp}) ────

// When non-null, daemon uses shm transport instead of stdio. Owned by
// main() — globals here are fine because there's exactly one daemon
// process per worker and the transport choice is set once at startup.
static nakshatra::fabric::ShmRing* g_shm_req = nullptr;
static nakshatra::fabric::ShmRing* g_shm_resp = nullptr;

static int read_all_fd(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t total = 0;
    while (total < n) {
        ssize_t r = read(fd, p + total, n - total);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        total += (size_t)r;
    }
    return 0;
}

static int write_all_fd(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    size_t total = 0;
    while (total < n) {
        ssize_t w = write(fd, p + total, n - total);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        total += (size_t)w;
    }
    return 0;
}

// Read one full request frame. In stdio mode: 5 u32 header reads
// followed by a payload read. In shm mode: one framed read off the
// request ring. Returns 0 on success, -1 on EOF / error.
static int read_request(uint32_t& cmd, uint32_t& n_tokens,
                          uint32_t& start_pos, uint32_t& flags,
                          std::vector<uint8_t>& payload) {
    if (g_shm_req) {
        std::vector<uint8_t> frame;
        try {
            g_shm_req->read_message_blocking(frame);
        } catch (const nakshatra::fabric::ShmRingError& e) {
            fprintf(stderr, "[daemon] shm req error: %s\n", e.what());
            return -1;
        }
        if (frame.size() < 20) {
            fprintf(stderr, "[daemon] shm request frame too short: %zu\n", frame.size());
            return -1;
        }
        std::memcpy(&cmd,       frame.data() + 0,  4);
        std::memcpy(&n_tokens,  frame.data() + 4,  4);
        std::memcpy(&start_pos, frame.data() + 8,  4);
        std::memcpy(&flags,     frame.data() + 12, 4);
        uint32_t payload_bytes;
        std::memcpy(&payload_bytes, frame.data() + 16, 4);
        if (frame.size() != 20 + (size_t)payload_bytes) {
            fprintf(stderr, "[daemon] shm request payload size mismatch: "
                            "declared %u, frame has %zu\n",
                    payload_bytes, frame.size() - 20);
            return -1;
        }
        payload.assign(frame.begin() + 20, frame.end());
        return 0;
    }
    // Stdio path — unchanged from the pre-A.3 daemon.
    uint32_t payload_bytes;
    if (read_all_fd(0, &cmd,           4) != 0) return -1;
    if (read_all_fd(0, &n_tokens,      4) != 0) return -1;
    if (read_all_fd(0, &start_pos,     4) != 0) return -1;
    if (read_all_fd(0, &flags,         4) != 0) return -1;
    if (read_all_fd(0, &payload_bytes, 4) != 0) return -1;
    payload.resize(payload_bytes);
    if (payload_bytes > 0 &&
        read_all_fd(0, payload.data(), payload_bytes) != 0) {
        return -1;
    }
    return 0;
}

static void send_response(uint32_t status, const void* payload,
                            uint32_t payload_bytes) {
    if (g_shm_resp) {
        // Build the response frame (8-byte header + payload) and
        // ship it as ONE framed shm message. The ring's own length
        // prefix wraps the daemon's wire envelope; the wire bytes
        // stay byte-identical to the stdio path so a Python consumer
        // can parse them the same way.
        std::vector<uint8_t> frame(8 + payload_bytes);
        std::memcpy(frame.data() + 0, &status,        4);
        std::memcpy(frame.data() + 4, &payload_bytes, 4);
        if (payload_bytes > 0) {
            std::memcpy(frame.data() + 8, payload, payload_bytes);
        }
        g_shm_resp->write_message_blocking(frame.data(), frame.size());
        return;
    }
    // Stdio path — unchanged.
    write_all_fd(1, &status,        4);
    write_all_fd(1, &payload_bytes, 4);
    if (payload_bytes > 0) write_all_fd(1, payload, payload_bytes);
    fflush(stdout);
}

// 2026-05-30 Phase B — compile-time version stamping for cluster-wide
// version-lock. CMake passes -DNAKSHATRA_FABRIC_SHA + -D...BUILD_HOST
// at build time; daemon prints them on --version. scripts/smoke_
// daemon_version.py SSH's each cluster machine + asserts the SHAs
// match — closes the long-standing daemon-skew finding from 2026-05-21.
#ifndef NAKSHATRA_FABRIC_SHA
#define NAKSHATRA_FABRIC_SHA "unknown"
#endif
#ifndef NAKSHATRA_FABRIC_BUILD_HOST
#define NAKSHATRA_FABRIC_BUILD_HOST "unknown"
#endif

int main(int argc, char ** argv) {
    // --version short-circuits before any model loading / arg
    // validation, so the version-lock smoke can probe a binary
    // without needing a model file on hand.
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("nakshatra-fabric-worker\n");
        printf("  sha        %s\n", NAKSHATRA_FABRIC_SHA);
        printf("  built_on   %s\n", NAKSHATRA_FABRIC_BUILD_HOST);
        printf("  built_at   %s %s\n", __DATE__, __TIME__);
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: %s <sub_gguf_path> <mode: first|middle|last> [n_ctx] [n_threads] [n_gpu_layers] [--fabric-shm-req <path> --fabric-shm-resp <path>]\n", argv[0]);
        fprintf(stderr, "       %s --version\n", argv[0]);
        return 1;
    }
    std::string sub_gguf = argv[1];
    std::string mode_str = argv[2];
    bool mode_last  = (mode_str == "last");
    bool mode_first = (mode_str == "first");
    bool mode_mid   = (mode_str == "middle");
    if (!mode_last && !mode_first && !mode_mid) {
        fprintf(stderr, "[daemon] bad mode '%s' (expected first|middle|last)\n", mode_str.c_str());
        return 1;
    }
    int n_ctx        = argc > 3 ? atoi(argv[3]) : 256;
    int n_threads    = argc > 4 ? atoi(argv[4]) : 0;  // 0 = leave llama.cpp's default
    int n_gpu_layers = argc > 5 ? atoi(argv[5]) : 0;  // 0 = CPU only; 99 = all GPU

    // 2026-05-30 Phase A.3 — parse the optional shm transport flags.
    // They appear after the positional args. When both are present,
    // attach both rings + flip into shm mode. Absent → stdio (legacy).
    std::string shm_req_path, shm_resp_path;
    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--fabric-shm-req") == 0 && i + 1 < argc) {
            shm_req_path = argv[++i];
        } else if (strcmp(argv[i], "--fabric-shm-resp") == 0 && i + 1 < argc) {
            shm_resp_path = argv[++i];
        }
    }
    bool shm_mode = !shm_req_path.empty() && !shm_resp_path.empty();
    if (shm_mode) {
        try {
            // Heap-allocate so global pointers can outlive the local
            // attach() return values via move construction.
            g_shm_req  = new nakshatra::fabric::ShmRing(
                nakshatra::fabric::ShmRing::attach(shm_req_path));
            g_shm_resp = new nakshatra::fabric::ShmRing(
                nakshatra::fabric::ShmRing::attach(shm_resp_path));
            fprintf(stderr, "[daemon] shm transport attached: req=%s resp=%s "
                            "capacity=%zu\n",
                    shm_req_path.c_str(), shm_resp_path.c_str(),
                    g_shm_req->capacity());
        } catch (const std::exception& e) {
            fprintf(stderr, "[daemon] shm attach failed: %s\n", e.what());
            return 4;
        }
    }

    common_init();
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model* model = llama_model_load_from_file(sub_gguf.c_str(), mp);
    if (!model) {
        fprintf(stderr, "[daemon] failed to load model: %s\n", sub_gguf.c_str());
        return 2;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = n_ctx;
    cp.embeddings = true;
    if (n_threads > 0) {
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "[daemon] failed to init context\n");
        return 3;
    }
    int n_embd  = llama_model_n_embd(model);
    int n_layer = llama_model_n_layer(model);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // The Nakshatra fields live on the model — reach in via llama.cpp's pubic
    // accessors where possible; for partial-load metadata we expose the
    // model-level KVs we wrote in partial_gguf.py via direct GGUF reads.
    // For M5 simplicity we rely on the patched loader having populated
    // layers correctly, and we trust the cluster config rather than
    // round-tripping the metadata. The INFO response surfaces what we know.

    fprintf(stderr, "[daemon] ready: %s n_embd=%d n_layer=%d n_vocab=%d n_ctx=%d n_gpu_layers=%d transport=%s\n",
            sub_gguf.c_str(), n_embd, n_layer, n_vocab, n_ctx, n_gpu_layers,
            shm_mode ? "shm" : "stdio");

    // Main message loop
    while (true) {
        uint32_t cmd, n_tokens, start_pos, flags;
        std::vector<uint8_t> payload;
        if (read_request(cmd, n_tokens, start_pos, flags, payload) != 0) break;
        uint32_t payload_bytes = (uint32_t)payload.size();

        const bool keep_kv = (flags & 0x1) != 0;

        if (cmd == 3) {
            // INFO
            int32_t info[6];
            info[0] = 0;          // n_layer_start  — not directly accessible from public API
            info[1] = n_layer;    // n_layer_end    — same
            info[2] = n_embd;
            info[3] = 1;          // has_token_embd — assume true; client validates via cluster yaml
            info[4] = 1;          // has_lm_head    — same
            info[5] = n_vocab;
            send_response(0, info, sizeof(info));
            continue;
        }

        int rc = -1;
        // Streaming KV reuse: skip the clear when caller asserts keep_kv.
        // First step in a session always sends keep_kv=false (cold prefill);
        // subsequent steps send keep_kv=true with start_pos = prefix_length
        // so the new tokens append to the existing KV cache.
        if (!keep_kv) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }

        if (cmd == 1) {
            // TOKEN_DECODE — use llama_batch_init so we can set explicit
            // positions (start_pos + i) instead of relying on the auto-zeroed
            // positions of llama_batch_get_one.
            if (payload_bytes != n_tokens * sizeof(int32_t)) {
                send_response(2, nullptr, 0); continue;
            }
            const int32_t* tok = (const int32_t*) payload.data();
            llama_batch batch = llama_batch_init(n_tokens, 0, 1);
            batch.n_tokens = n_tokens;
            for (uint32_t i = 0; i < n_tokens; ++i) {
                batch.token[i] = (llama_token) tok[i];
                batch.pos[i] = (int32_t)(start_pos + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
            }
            rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
        } else if (cmd == 2) {
            // EMBD_DECODE
            if (payload_bytes != n_tokens * (uint32_t)n_embd * sizeof(float)) {
                send_response(2, nullptr, 0); continue;
            }
            llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
            batch.n_tokens = n_tokens;
            memcpy(batch.embd, payload.data(), payload_bytes);
            for (uint32_t i = 0; i < n_tokens; ++i) {
                batch.pos[i] = (int32_t)(start_pos + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
            }
            rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
        } else {
            send_response(2, nullptr, 0);
            continue;
        }

        if (rc != 0) {
            send_response(1, nullptr, 0);
            continue;
        }

        // Caller specified mode at startup; respect it.
        // Response payload: first 4 bytes = result_type (0=hidden, 1=token).
        if (mode_last) {
            float* logits = llama_get_logits_ith(ctx, -1);
            if (!logits) { send_response(1, nullptr, 0); continue; }
            int top = 0; float top_v = logits[0];
            for (int i = 1; i < n_vocab; ++i) if (logits[i] > top_v) { top_v = logits[i]; top = i; }
            uint8_t buf[8];
            *(uint32_t*)(buf + 0) = 1;      // result_type = token
            *(int32_t *)(buf + 4) = top;
            send_response(0, buf, 8);
        } else {
            // first/middle: return hidden state
            float* hidden = llama_get_embeddings(ctx);
            if (!hidden) { send_response(1, nullptr, 0); continue; }
            size_t hbytes = (size_t)n_tokens * (size_t)n_embd * sizeof(float);
            std::vector<uint8_t> out(4 + hbytes);
            *(uint32_t*)out.data() = 0;     // result_type = hidden
            memcpy(out.data() + 4, hidden, hbytes);
            send_response(0, out.data(), (uint32_t)out.size());
        }
    }

    fprintf(stderr, "[daemon] eof, shutting down\n");
    if (g_shm_req)  { delete g_shm_req;  g_shm_req  = nullptr; }
    if (g_shm_resp) { delete g_shm_resp; g_shm_resp = nullptr; }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
