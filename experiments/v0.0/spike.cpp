// Phase 0b spike: ferry hidden state at a target tensor name between two
// processes via localhost TCP. Both processes load the full model; one
// sends its target-tensor data after capture, the other overwrites its own
// tensor with the received bytes mid-decode and continues to logits.
//
// Validates the orchestration protocol BEFORE paying v0.1's C++ cost on a
// patched llama_decode + partial-model loading.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

enum class SpikeMode { NONE, SEND, RECV, OBSERVE };

struct SpikeData {
    SpikeMode mode = SpikeMode::NONE;
    std::string target_name = "l_out-13";
    int port = 5555;
    int socket_fd = -1;
    int conn_fd = -1;
    bool fired = false;
    long long bytes_sent = 0;
    long long bytes_received = 0;
    bool ok = true;
};

static bool send_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, p + total, len - total);
        if (n < 0) { if (errno == EINTR) continue; fprintf(stderr, "[spike] send: %s\n", strerror(errno)); return false; }
        total += (size_t)n;
    }
    return true;
}

static bool recv_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, p + total, len - total);
        if (n < 0) { if (errno == EINTR) continue; fprintf(stderr, "[spike] recv: %s\n", strerror(errno)); return false; }
        if (n == 0) { fprintf(stderr, "[spike] recv eof at %zu/%zu\n", total, len); return false; }
        total += (size_t)n;
    }
    return true;
}

static uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static bool spike_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    SpikeData* sd = (SpikeData*)user_data;
    const char* name = ggml_get_name(t);
    if (!name || strcmp(name, sd->target_name.c_str()) != 0) return false;
    if (ask) return true;
    if (sd->fired) return false;
    sd->fired = true;

    size_t nbytes = ggml_nbytes(t);
    fprintf(stderr, "[spike] target '%s' fired: type=%d ne=[%lld,%lld,%lld,%lld] nbytes=%zu\n",
            name, (int)t->type,
            (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
            nbytes);

    if (sd->mode == SpikeMode::OBSERVE) {
        uint64_t h = fnv1a(t->data, nbytes);
        fprintf(stderr, "[spike] OBSERVE %zu bytes (fnv1a=0x%016llx) — no I/O\n", nbytes, (unsigned long long)h);
    } else if (sd->mode == SpikeMode::SEND) {
        uint64_t h = fnv1a(t->data, nbytes);
        uint32_t len = (uint32_t)nbytes;
        if (!send_all(sd->conn_fd, &len, sizeof(len))) { sd->ok = false; return false; }
        if (!send_all(sd->conn_fd, t->data, nbytes)) { sd->ok = false; return false; }
        sd->bytes_sent = (long long)nbytes;
        fprintf(stderr, "[spike] sent %zu bytes (fnv1a=0x%016llx)\n", nbytes, (unsigned long long)h);
    } else if (sd->mode == SpikeMode::RECV) {
        uint32_t len = 0;
        if (!recv_all(sd->conn_fd, &len, sizeof(len))) { sd->ok = false; return false; }
        if ((size_t)len != nbytes) { fprintf(stderr, "[spike] size mismatch peer=%u local=%zu\n", len, nbytes); sd->ok = false; return false; }
        uint64_t pre = fnv1a(t->data, nbytes);
        if (!recv_all(sd->conn_fd, t->data, nbytes)) { sd->ok = false; return false; }
        uint64_t post = fnv1a(t->data, nbytes);
        sd->bytes_received = (long long)nbytes;
        fprintf(stderr, "[spike] recv %zu bytes  pre_hash=0x%016llx  post_hash=0x%016llx  %s\n",
                nbytes, (unsigned long long)pre, (unsigned long long)post,
                pre == post ? "(byte-equal local vs remote)" : "(REMOTE OVERWROTE LOCAL)");
    }
    return false;
}

static int argmax_logits(llama_context* ctx, const llama_vocab* vocab) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    float* logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return -1;
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    return best;
}

int main(int argc, char ** argv) {
    SpikeData sd;
    std::vector<char*> clean;
    clean.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--spike-mode") && i+1 < argc) {
            std::string m = argv[++i];
            sd.mode = (m == "send") ? SpikeMode::SEND : (m == "recv") ? SpikeMode::RECV : (m == "observe") ? SpikeMode::OBSERVE : SpikeMode::NONE;
        } else if (!strcmp(argv[i], "--spike-port") && i+1 < argc) {
            sd.port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--spike-target") && i+1 < argc) {
            sd.target_name = argv[++i];
        } else if (!strcmp(argv[i], "--spike-none")) {
            sd.mode = SpikeMode::NONE;  // reference run: no socket, no intercept
        } else {
            clean.push_back(argv[i]);
        }
    }

    setlocale(LC_NUMERIC, "C");

    if (sd.mode == SpikeMode::OBSERVE) {
        fprintf(stderr, "[spike] OBSERVE mode (no socket), target=%s\n", sd.target_name.c_str());
    } else if (sd.mode == SpikeMode::SEND) {
        sd.conn_fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(sd.port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        for (int t = 0; ; ++t) {
            if (connect(sd.conn_fd, (sockaddr*)&a, sizeof(a)) == 0) break;
            if (t > 40) { fprintf(stderr, "[spike] connect timeout\n"); return 1; }
            usleep(250000);
        }
        fprintf(stderr, "[spike] SEND mode connected to localhost:%d, target=%s\n", sd.port, sd.target_name.c_str());
    } else if (sd.mode == SpikeMode::RECV) {
        sd.socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        int yes = 1; setsockopt(sd.socket_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(sd.port); a.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(sd.socket_fd, (sockaddr*)&a, sizeof(a)) < 0) { fprintf(stderr, "bind: %s\n", strerror(errno)); return 1; }
        listen(sd.socket_fd, 1);
        fprintf(stderr, "[spike] RECV mode listening on :%d, target=%s\n", sd.port, sd.target_name.c_str());
        sd.conn_fd = accept(sd.socket_fd, nullptr, nullptr);
        if (sd.conn_fd < 0) { fprintf(stderr, "accept: %s\n", strerror(errno)); return 1; }
        fprintf(stderr, "[spike] RECV accepted connection\n");
    } else {
        fprintf(stderr, "[spike] reference mode (no socket, no intercept)\n");
    }

    int new_argc = (int)clean.size();
    char** new_argv = clean.data();
    common_params params;
    if (!common_params_parse(new_argc, new_argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    if (sd.mode != SpikeMode::NONE) {
        params.cb_eval = spike_cb;
        params.cb_eval_user_data = &sd;
    }
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);
    if (tokens.empty()) { fprintf(stderr, "[spike] no tokens for prompt\n"); return 1; }
    fprintf(stderr, "[spike] tokenized to %zu tokens\n", tokens.size());

    int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    fprintf(stderr, "[spike] decode rc=%d  fired=%s  sent=%lld  recv=%lld  ok=%d\n",
            rc, sd.fired ? "yes" : "no", sd.bytes_sent, sd.bytes_received, sd.ok ? 1 : 0);

    int top = argmax_logits(ctx, vocab);
    char tokbuf[64] = {0};
    if (top >= 0) {
        int n = llama_token_to_piece(vocab, (llama_token)top, tokbuf, sizeof(tokbuf) - 1, 0, true);
        if (n < 0) n = 0;
        tokbuf[n] = '\0';
    }
    fprintf(stderr, "[spike] top-1 token id=%d str='%s'\n", top, tokbuf);
    fprintf(stdout, "TOPTOK %d %s\n", top, tokbuf);

    if (sd.conn_fd >= 0) close(sd.conn_fd);
    if (sd.socket_fd >= 0) close(sd.socket_fd);
    llama_backend_free();
    return sd.ok ? 0 : 2;
}
