// M4 step 4+5 chain test: validate the patched llama.cpp can run a two-worker
// chain in a single process.
//
//   Worker A: w0 sub-GGUF (layers [0,N1), has token_embd, no lm_head).
//     Input:  prompt tokens via batch.token.
//     Output: residual stream after layer N1-1, read via llama_get_embeddings.
//
//   Worker B: wlast sub-GGUF (layers [N1,N), has lm_head, has token_embd
//             for tied-embed fallback).
//     Input:  worker A's hidden state via batch.embd (no batch.token).
//     Output: logits via llama_get_logits_ith.
//
// We argmax the logits and compare to a single-machine reference top-1 token.
// Both should produce the same token if the chain is correct.

#include "common.h"
#include "llama.h"

#include <cassert>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct WorkerCtx {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
};

static WorkerCtx load_worker(const std::string & path, int n_ctx) {
    WorkerCtx w;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    w.model = llama_model_load_from_file(path.c_str(), mp);
    if (!w.model) {
        fprintf(stderr, "[chain] failed to load model: %s\n", path.c_str());
        exit(2);
    }
    auto cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = n_ctx;
    cp.embeddings = true;
    w.ctx = llama_init_from_model(w.model, cp);
    if (!w.ctx) {
        fprintf(stderr, "[chain] failed to init context for: %s\n", path.c_str());
        exit(3);
    }
    return w;
}

int main(int argc, char ** argv) {
    setlocale(LC_NUMERIC, "C");

    std::string w0_path    = argc > 1 ? argv[1] : "/tmp/cuts/w0_v2.gguf";
    std::string wlast_path = argc > 2 ? argv[2] : "/tmp/cuts/wlast_v2.gguf";
    std::string prompt     = argc > 3 ? argv[3] : "The capital of France is";

    common_init();
    llama_backend_init();

    fprintf(stderr, "[chain] loading worker A: %s\n", w0_path.c_str());
    WorkerCtx wa = load_worker(w0_path, 256);
    int n_embd = llama_model_n_embd(wa.model);
    fprintf(stderr, "[chain] worker A loaded; hidden_size=%d\n", n_embd);

    // Tokenize via worker A's vocab
    const llama_vocab * vocab_a = llama_model_get_vocab(wa.model);
    bool add_bos = llama_vocab_get_add_bos(vocab_a);
    std::vector<llama_token> tokens = common_tokenize(wa.ctx, prompt, add_bos);
    fprintf(stderr, "[chain] %zu tokens for prompt %s\n", tokens.size(), prompt.c_str());

    // Worker A: decode with tokens
    int rc = llama_decode(wa.ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    fprintf(stderr, "[chain] worker A decode rc=%d\n", rc);
    if (rc != 0) return 4;

    float * embd_a_all = llama_get_embeddings(wa.ctx);
    if (!embd_a_all) {
        fprintf(stderr, "[chain] worker A: llama_get_embeddings returned null\n");
        return 5;
    }
    fprintf(stderr, "[chain] worker A hidden[:4] = %.4f %.4f %.4f %.4f\n",
            embd_a_all[0], embd_a_all[1], embd_a_all[2], embd_a_all[3]);

    // Stash a copy because loading worker B may invalidate worker A's context buffer
    std::vector<float> hidden_in(tokens.size() * n_embd);
    memcpy(hidden_in.data(), embd_a_all, hidden_in.size() * sizeof(float));

    fprintf(stderr, "[chain] loading worker B: %s\n", wlast_path.c_str());
    WorkerCtx wb = load_worker(wlast_path, 256);
    int n_embd_b = llama_model_n_embd(wb.model);
    int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(wb.model));
    fprintf(stderr, "[chain] worker B loaded; hidden_size=%d vocab=%d\n", n_embd_b, n_vocab);
    if (n_embd_b != n_embd) {
        fprintf(stderr, "[chain] hidden_size mismatch: A=%d B=%d\n", n_embd, n_embd_b);
        return 6;
    }

    // Worker B: decode with hidden_in via batch.embd
    int n_tokens = (int)tokens.size();
    llama_batch batch_b = llama_batch_init(n_tokens, n_embd, 1);
    batch_b.n_tokens = n_tokens;
    memcpy(batch_b.embd, hidden_in.data(), hidden_in.size() * sizeof(float));
    for (int i = 0; i < n_tokens; ++i) {
        batch_b.pos[i] = i;
        batch_b.n_seq_id[i] = 1;
        batch_b.seq_id[i][0] = 0;
        batch_b.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    rc = llama_decode(wb.ctx, batch_b);
    fprintf(stderr, "[chain] worker B decode rc=%d\n", rc);
    if (rc != 0) {
        llama_batch_free(batch_b);
        return 7;
    }

    float * logits = llama_get_logits_ith(wb.ctx, -1);
    if (!logits) {
        fprintf(stderr, "[chain] worker B: llama_get_logits_ith returned null\n");
        llama_batch_free(batch_b);
        return 8;
    }
    int top = 0;
    float top_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > top_v) { top_v = logits[i]; top = i; }
    }

    char tokbuf[64] = {0};
    int n = llama_token_to_piece(llama_model_get_vocab(wb.model),
                                 (llama_token)top, tokbuf, sizeof(tokbuf) - 1, 0, true);
    if (n < 0) n = 0;
    tokbuf[n] = '\0';

    fprintf(stderr, "[chain] argmax token id=%d str='%s' logit=%.4f\n", top, tokbuf, top_v);
    fprintf(stdout, "TOPTOK_CHAIN %d %s\n", top, tokbuf);

    llama_batch_free(batch_b);
    llama_free(wb.ctx);
    llama_model_free(wb.model);
    llama_free(wa.ctx);
    llama_model_free(wa.model);
    llama_backend_free();
    return 0;
}
