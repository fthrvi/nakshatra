// Step 3 of the v0.0 spike — two contexts in one process exchange the
// l_out-13 activation across a cb_eval boundary, no networking.
//
// Goal: prove the byte-level handoff mechanics work end-to-end before
// adding the network layer in Step 4. If context B's argmax token
// matches the single-context reference, the capture/restore byte path
// is correct (sizes, layout, dtype all match, the buffer copy is sound).
//
// Why this isn't trivially the same as Step 2:
//   Step 2 proved cb_eval-write affects downstream computation (zeroing
//   the cut tensor changed the predicted token).
//   Step 3 proves the *cross-context byte transfer* itself is correct.
//   A and B are running the same model on the same input, so A's bytes
//   for l_out-13 are exactly what B would compute on its own — meaning
//   if any part of our copy is wrong (size mismatch, dtype mismatch,
//   layout mismatch, dangling pointer) B's output will diverge.
//
// Layout:
//   1) Load Qwen3-0.6B once into a shared llama_model.
//   2) Create Context A with cb_capture; decode the prompt; expect a
//      non-zero rc because cb_capture aborts on l_out-13 (intentional —
//      saves layers 14..27 of compute on A).
//   3) Sanity check: capture happened.
//   4) Create Context B with cb_restore; decode the prompt; expect rc=0.
//   5) Read B's logits, take argmax, compare to expected 12095 (" Paris").

#include "llama.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char * CUT_TENSOR = "l_out-13";

struct CapturedActivation {
    bool   captured = false;
    int    dtype    = -1;
    int64_t ne[4]   = {0, 0, 0, 0};
    std::vector<uint8_t> bytes;
};

CapturedActivation g_act;

// Mode flag accessible in callbacks.
enum CaptureBehavior { CAP_ABORT, CAP_CONTINUE };
CaptureBehavior g_cap_behavior = CAP_ABORT;

// When true, cb_restore returns true on ask=true (requesting post-compute)
// but performs NO modification on ask=false. Tests whether merely registering
// interest in a tensor truncates the graph downstream.
bool g_passthrough_only = false;

int g_cb_capture_calls = 0;   // total cb_capture invocations
int g_cb_capture_cut_post = 0; // post-compute hits on cut tensor
int g_cb_restore_calls = 0;
int g_cb_restore_cut_post = 0;

// Context A's callback — copy out + (optionally) abort decode.
bool cb_capture(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    g_cb_capture_calls++;
    if (!t->name || strcmp(t->name, CUT_TENSOR) != 0) return false;
    if (ask) return true; // yes, notify post-compute

    // post-compute notification
    size_t nb = ggml_nbytes(t);
    if (!t->data) {
        fprintf(stderr, "cb_capture: t->data is null at %s\n", CUT_TENSOR);
        return true;
    }
    g_act.bytes.assign((const uint8_t *)t->data,
                       (const uint8_t *)t->data + nb);
    for (int i = 0; i < 4; i++) g_act.ne[i] = t->ne[i];
    g_act.dtype = t->type;
    g_act.captured = true;
    g_cb_capture_cut_post++;
    const float * f = (const float *)t->data;
    fprintf(stderr, "[capture] %s ne=[%lld,%lld,%lld,%lld] dtype=%s nbytes=%zu\n",
            t->name,
            (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3],
            ggml_type_name(t->type), nb);
    fprintf(stderr, "[capture] first 8 floats: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    // Behavior depends on flag: ABORT → return true; CONTINUE → return false.
    return g_cap_behavior == CAP_ABORT;
}

// Context B's callback — overwrite with captured bytes, continue.
bool cb_restore(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    g_cb_restore_calls++;
    if (!t->name || strcmp(t->name, CUT_TENSOR) != 0) return false;
    if (ask) return true; // yes, notify post-compute

    g_cb_restore_cut_post++;
    if (g_passthrough_only) {
        fprintf(stderr, "cb_restore: passthrough_only=true, NOT modifying t->data\n");
        return false;
    }
    if (!g_act.captured) {
        fprintf(stderr, "cb_restore: no captured activation available — leaving live bytes alone\n");
        return false; // continue decode normally; we just don't override
    }
    size_t nb = ggml_nbytes(t);
    if (nb != g_act.bytes.size()) {
        fprintf(stderr, "cb_restore: byte size mismatch (live=%zu, captured=%zu)\n",
                nb, g_act.bytes.size());
        return true;
    }
    if ((int)t->type != g_act.dtype) {
        fprintf(stderr, "cb_restore: dtype mismatch (live=%d, captured=%d)\n",
                (int)t->type, g_act.dtype);
        return true;
    }
    for (int i = 0; i < 4; i++) {
        if (t->ne[i] != g_act.ne[i]) {
            fprintf(stderr, "cb_restore: shape mismatch dim %d (live=%lld, captured=%lld)\n",
                    i, (long long)t->ne[i], (long long)g_act.ne[i]);
            return true;
        }
    }
    const float * live = (const float *)t->data;
    fprintf(stderr, "[restore-pre] live %s first 8 floats: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            t->name, live[0], live[1], live[2], live[3], live[4], live[5], live[6], live[7]);
    const float * cap = (const float *)g_act.bytes.data();
    fprintf(stderr, "[restore-cap] captured first 8 floats: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            cap[0], cap[1], cap[2], cap[3], cap[4], cap[5], cap[6], cap[7]);
    std::memcpy(t->data, g_act.bytes.data(), nb);
    const float * after = (const float *)t->data;
    fprintf(stderr, "[restore-post] live after memcpy: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            after[0], after[1], after[2], after[3], after[4], after[5], after[6], after[7]);
    fprintf(stderr, "[restore] overwrote %s with %zu bytes\n", t->name, nb);
    return false; // continue decode normally
}

llama_token argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float bestv = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > bestv) { bestv = logits[i]; best = i; }
    }
    return (llama_token)best;
}

} // namespace

int main(int argc, char ** argv) {
    // mode: "handoff" (default) — A captures, B restores
    //       "ctxb_only" — skip A, run B with cb_restore but g_act.captured=false
    //                     so cb_restore returns early (writes nothing). This isolates
    //                     "is ctxB itself broken?" from "is the handoff broken?"
    // Modes:
    //   handoff           — A captures + aborts; B restores; full pipeline.
    //   handoff_no_abort  — A captures but does NOT abort; runs full decode; B restores.
    //                       Tests whether ctxA's abort is what corrupts ctxB.
    //   ctxb_only         — skip A entirely; B runs with cb_restore (no override since
    //                       g_act.captured stays false). Validates ctxB-by-itself works.
    const char * mode = "handoff";
    int argv_off = 1;
    if (argc > 1 && (strcmp(argv[1], "handoff") == 0
                  || strcmp(argv[1], "handoff_no_abort") == 0
                  || strcmp(argv[1], "ctxb_only") == 0
                  || strcmp(argv[1], "passthrough") == 0)) {
        mode = argv[1];
        argv_off = 2;
    }
    const char * model_path = argc > argv_off ? argv[argv_off]   : "/tmp/nakshatra-test/qwen3-0.6b-full.gguf";
    const char * prompt     = argc > argv_off + 1 ? argv[argv_off + 1] : "The capital of France is";
    fprintf(stderr, "mode=%s\n", mode);
    g_cap_behavior = (strcmp(mode, "handoff_no_abort") == 0) ? CAP_CONTINUE : CAP_ABORT;
    g_passthrough_only = (strcmp(mode, "passthrough") == 0);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(64);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), (int)tokens.size(),
                           /*add_special=*/true, /*parse_special=*/false);
    if (n < 0) { fprintf(stderr, "tokenize failed (rc=%d)\n", n); return 1; }
    tokens.resize(n);

    fprintf(stderr, "prompt: %d tokens =", n);
    for (auto t : tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    // ---- Context A — capture (skipped in ctxb_only mode) ----
    if (strcmp(mode, "handoff") == 0 || strcmp(mode, "handoff_no_abort") == 0) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.cb_eval = cb_capture;

        llama_context * ctxA = llama_init_from_model(model, cp);
        if (!ctxA) { fprintf(stderr, "ctxA create failed\n"); return 1; }

        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        int rc = llama_decode(ctxA, batch);
        fprintf(stderr, "ctxA: llama_decode rc=%d (non-zero is expected — aborted on cut)\n", rc);
        if (!g_act.captured) {
            fprintf(stderr, "ctxA: capture did NOT fire — abort\n");
            llama_free(ctxA);
            return 1;
        }
        llama_free(ctxA);
    } else {
        fprintf(stderr, "ctxb_only mode: skipping ctxA\n");
    }

    // ---- Context B — restore ----
    llama_token argmax = -1;
    int rcB = -1;
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.cb_eval = cb_restore;

        llama_context * ctxB = llama_init_from_model(model, cp);
        if (!ctxB) { fprintf(stderr, "ctxB create failed\n"); return 1; }

        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        rcB = llama_decode(ctxB, batch);
        if (rcB != 0) {
            fprintf(stderr, "ctxB: llama_decode rc=%d (expected 0)\n", rcB);
            llama_free(ctxB);
            return 1;
        }

        int n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits_ith(ctxB, n - 1);
        if (!logits) { fprintf(stderr, "ctxB: get_logits_ith null\n"); return 1; }
        // Diagnostic: how does the logit distribution actually look?
        float lmin = logits[0], lmax = logits[0];
        double lsum = 0.0;
        int n_nan = 0, n_inf = 0;
        for (int i = 0; i < n_vocab; i++) {
            float v = logits[i];
            if (std::isnan(v)) n_nan++;
            else if (std::isinf(v)) n_inf++;
            else { if (v < lmin) lmin = v; if (v > lmax) lmax = v; lsum += v; }
        }
        fprintf(stderr, "ctxB logits: min=%.4f max=%.4f mean=%.4f n_nan=%d n_inf=%d (n_vocab=%d)\n",
                lmin, lmax, lsum / n_vocab, n_nan, n_inf, n_vocab);
        fprintf(stderr, "ctxB logits[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], logits[6], logits[7]);
        argmax = argmax_logits(logits, n_vocab);

        char piece[256] = {0};
        int piece_len = llama_token_to_piece(vocab, argmax, piece, sizeof(piece) - 1, 0, true);
        if (piece_len < 0) piece_len = 0;
        piece[piece_len] = 0;
        printf("MODE=%s ARGMAX=%d PIECE=%.*s rcB=%d cb_restore_calls=%d cb_restore_cut_post=%d\n",
               mode, (int)argmax, piece_len, piece, rcB,
               g_cb_restore_calls, g_cb_restore_cut_post);

        llama_free(ctxB);
    }
    fprintf(stderr, "totals: cb_capture_calls=%d cb_capture_cut_post=%d cb_restore_calls=%d cb_restore_cut_post=%d\n",
            g_cb_capture_calls, g_cb_capture_cut_post,
            g_cb_restore_calls, g_cb_restore_cut_post);

    llama_model_free(model);
    llama_backend_free();

    // Reference token is 12095 (" Paris") per Step 2 reference run.
    return (argmax == 12095) ? 0 : 3;
}
