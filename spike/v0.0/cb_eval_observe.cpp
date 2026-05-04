// Step 1 of the v0.0 spike — observe-only cb_eval
//
// Goal: identify the exact tensor name at the cut point we will use for the
// two-worker handoff (post-block-13 / pre-block-14 of Qwen3-0.6B). The cut
// point candidate name in llama.cpp's graph builder is typically `l_out-N`
// for layer N's output, but architectures vary — we let the model tell us
// what it actually emits rather than guessing.
//
// What this program does:
//   1. Loads the full Qwen3-0.6B GGUF.
//   2. Creates a llama_context with a cb_eval callback installed.
//   3. The callback is called by GGML once per tensor in the compute graph,
//      with `ask=true` first (asking whether to be notified post-compute).
//      We return `false` on `ask=true` so we don't slow decode down with a
//      second wave of callbacks; we just observe names + shapes + dtypes.
//   4. Runs one llama_decode on the prompt tokens. Every tensor in the
//      graph passes through the callback exactly once.
//   5. Prints each unique (name, shape, dtype) we saw. From that list we
//      pick the cut tensor for Steps 2+.

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

std::set<std::string> g_seen;

bool cb_eval(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    // ask=true: GGML is asking whether we want a post-compute callback.
    // ask=false: post-compute notification (only if we said yes on ask=true).
    // For observation we just want the name once; don't abort decode.
    if (!ask) return false;

    const char * raw_name = t->name;
    std::string name = (raw_name && raw_name[0]) ? raw_name : "(unnamed)";
    if (g_seen.insert(name).second) {
        // First time we've seen this name. Print its shape + dtype.
        printf("CB %-40s ne=[%lld,%lld,%lld,%lld] dtype=%s\n",
               name.c_str(),
               (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3],
               ggml_type_name(t->type));
    }
    return false; // don't request the post-compute notification
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = argc > 1
        ? argv[1]
        : "/tmp/nakshatra-test/qwen3-0.6b-full.gguf";
    const char * prompt = argc > 2
        ? argv[2]
        : "The capital of France is";

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU only — keeps the spike portable

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context create failed\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(64);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), (int)tokens.size(),
                           /*add_special=*/true, /*parse_special=*/false);
    if (n < 0) {
        fprintf(stderr, "tokenize failed (rc=%d)\n", n);
        return 1;
    }
    tokens.resize(n);

    fprintf(stderr, "prompt: %d tokens =", n);
    for (auto t : tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n\n--- cb_eval observations (one line per unique tensor) ---\n");

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "\nllama_decode failed (rc=%d)\n", rc);
        return 1;
    }

    fprintf(stderr, "\nllama_decode ok — saw %zu unique tensor names\n", g_seen.size());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
