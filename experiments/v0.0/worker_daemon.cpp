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

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

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

static int read_u32(uint32_t* v) { return read_all_fd(0, v, 4); }
static int write_u32(uint32_t v) { return write_all_fd(1, &v, 4); }

static void send_response(uint32_t status, const void* payload, uint32_t payload_bytes) {
    write_u32(status);
    write_u32(payload_bytes);
    if (payload_bytes > 0) write_all_fd(1, payload, payload_bytes);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <sub_gguf_path> <mode: first|middle|last> [n_ctx]\n", argv[0]);
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
    int n_ctx = argc > 3 ? atoi(argv[3]) : 256;

    common_init();
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(sub_gguf.c_str(), mp);
    if (!model) {
        fprintf(stderr, "[daemon] failed to load model: %s\n", sub_gguf.c_str());
        return 2;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = n_ctx;
    cp.embeddings = true;
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

    fprintf(stderr, "[daemon] ready: %s n_embd=%d n_layer=%d n_vocab=%d n_ctx=%d\n",
            sub_gguf.c_str(), n_embd, n_layer, n_vocab, n_ctx);

    // Main message loop
    while (true) {
        uint32_t cmd, n_tokens, start_pos, flags, payload_bytes;
        if (read_u32(&cmd) != 0) break;
        if (read_u32(&n_tokens) != 0) break;
        if (read_u32(&start_pos) != 0) break;
        if (read_u32(&flags) != 0) break;
        if (read_u32(&payload_bytes) != 0) break;

        std::vector<uint8_t> payload(payload_bytes);
        if (payload_bytes > 0 && read_all_fd(0, payload.data(), payload_bytes) != 0) break;

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
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
