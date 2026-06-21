"""eagle_speculative.py — EagleDraft as a speculative-decode DraftModel.

Plugs the trained EAGLE-3 head into the spec loop: given the prefix tokens, it
gets the target's hidden_states[0,1,2] from the first worker (cmd=5), runs the
head, and proposes K draft tokens — the same `propose(prefix_tokens, k)` interface
as scripts/speculative.py's GGUF DraftModel, so it's a drop-in. The worker
connection is injected as `cmd5_fn` (decouples from transport; testable).

Wire-up (client.py spec loop): build an EagleDraft with the SERVING-hidden head
(head_serving_hidden.pt) + a cmd5_fn that calls the first worker, use it as the
draft. The retrained head (train/serve-consistent) makes its proposals accepted.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/EAGLE")); sys.path.insert(0, os.path.expanduser("~"))


class EagleDraft:
    def __init__(self, head_ckpt, base, config_path, cmd5_fn):
        import eagle_draft
        self._ed = eagle_draft
        self.model, _, self.cfg = eagle_draft.load_head(head_ckpt, base, config_path)
        self.cmd5_fn = cmd5_fn   # (prefix_ids:list[int]) -> hidden3 flat [S*3*n_embd] or [S,3*n_embd]
        self.n_embd = self.cfg.hidden_size

    def propose(self, prefix_tokens, k):
        """Return k draft token ids (full vocab) for the continuation of prefix_tokens."""
        import torch
        S = len(prefix_tokens)
        hid = self.cmd5_fn(list(prefix_tokens))
        h = torch.as_tensor(hid, dtype=torch.float32).reshape(1, S, 3 * self.n_embd)
        ids = torch.tensor([list(prefix_tokens) + [prefix_tokens[-1]]])  # len S+1 (topK drops first)
        return self._ed.propose(self.model, h, ids, k)


def daemon_cmd5_fn(send_recv):
    """Build a cmd5_fn from a worker request function `send_recv(cmd,n,start,flags,payload)
    -> (status, bytes)` (the daemon protocol). Returns the first worker's hidden3."""
    import struct
    def fn(prefix_ids):
        S = len(prefix_ids)
        st, d = send_recv(5, S, 0, 0, struct.pack(f"<{S}i", *prefix_ids))
        assert st == 0 and struct.unpack("<I", d[:4])[0] == 3, "cmd=5 failed"
        n = (len(d) - 4) // 4
        return list(struct.unpack(f"<{n}f", d[4:]))
    return fn
