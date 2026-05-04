// Step 2 of the v0.0 spike — does cb_eval permit MODIFYING tensor data
// mid-graph, or is it observation-only?
//
// Design:
//   * Single process. Loads Qwen3-0.6B, decodes the prompt once.
//   * Two modes selected by argv[1]:
//       reference: cb_eval observes only; nothing is touched.
//       perturb:   when l_out-13 arrives with ask=false (post-compute),
//                  overwrite its data with all zeros, return false to
//                  let decode continue.
//   * After llama_decode returns, take argmax of the logits and print
//     that token id.
//
// Pass criterion: argmax under `perturb` differs from argmax under
// `reference`. That proves the write at l_out-13 propagated through
// layers 14..27 + output_norm + lm_head and changed the prediction.
//
// Why zero out: it's the cheapest perturbation that's guaranteed to
// produce a meaningfully different downstream activation if the write
// took effect. (A small noise perturbation might get swallowed by
// later RMSNorms; zeros cannot.)

#include "llama.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum Mode { MODE_REFERENCE, MODE_PERTURB };
Mode  g_mode = MODE_REFERENCE;
const char * CUT_TENSOR = "l_out-13";

// Stats: how many times we saw the cut tensor in each callback phase.
int g_seen_ask = 0;
int g_seen_post = 0;
int g_zeroed = 0;
size_t g_zero_bytes = 0;

bool cb_eval(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    const char * name = t->name;
    if (!name || strcmp(name, CUT_TENSOR) != 0) {
        // Not our cut tensor. In reference mode, ignore everything.
        // In perturb mode, also ignore — we only modify the cut.
        return false; // false on ask=true → no post-compute callback for this tensor
    }

    if (ask) {
        g_seen_ask++;
        // We want the post-compute callback iff we are perturbing.
        return g_mode == MODE_PERTURB;
    }

    // ask=false: the cut tensor has just been computed.
    g_seen_post++;
    if (g_mode == MODE_PERTURB) {
        size_t nbytes = ggml_nbytes(t);
        if (t->data) {
            std::memset(t->data, 0, nbytes);
            g_zeroed++;
            g_zero_bytes += nbytes;
        } else {
            fprintf(stderr, "WARN: %s post-compute has t->data == nullptr (nbytes=%zu)\n",
                    name, nbytes);
        }
    }
    return false; // do not abort decode
}

llama_token argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return (llama_token)best;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s {reference|perturb} [model.gguf] [prompt]\n", argv[0]);
        return 2;
    }
    if      (strcmp(argv[1], "reference") == 0) g_mode = MODE_REFERENCE;
    else if (strcmp(argv[1], "perturb")   == 0) g_mode = MODE_PERTURB;
    else { fprintf(stderr, "unknown mode: %s\n", argv[1]); return 2; }

    const char * model_path = argc > 2 ? argv[2] : "/tmp/nakshatra-test/qwen3-0.6b-full.gguf";
    const char * prompt     = argc > 3 ? argv[3] : "The capital of France is";

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context create failed\n"); llama_model_free(model); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(64);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), (int)tokens.size(),
                           /*add_special=*/true, /*parse_special=*/false);
    if (n < 0) { fprintf(stderr, "tokenize failed (rc=%d)\n", n); return 1; }
    tokens.resize(n);

    fprintf(stderr, "[%s] prompt: %d tokens =", argv[1], n);
    for (auto t : tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { fprintf(stderr, "llama_decode failed (rc=%d)\n", rc); return 1; }

    int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, n - 1); // last position
    if (!logits) { fprintf(stderr, "llama_get_logits_ith returned null\n"); return 1; }
    llama_token argmax = argmax_logits(logits, n_vocab);

    char piece[256] = {0};
    int piece_len = llama_token_to_piece(vocab, argmax, piece, sizeof(piece) - 1, 0, true);
    if (piece_len < 0) piece_len = 0;
    piece[piece_len] = 0;

    fprintf(stderr, "[%s] cb stats: ask=%d post=%d zeroed=%d (%zu bytes)\n",
            argv[1], g_seen_ask, g_seen_post, g_zeroed, g_zero_bytes);
    // The summary line is on stdout so the test harness can grep it.
    printf("MODE=%s ARGMAX=%d PIECE=%.*s\n",
           argv[1], (int)argmax, piece_len, piece);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
