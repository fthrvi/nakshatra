"""
Unit tests for scripts/remote_client.py — the draft-side CLI for NKS_REMOTE_PROPOSALS.

No GPU, no model weights, no live HTTP server: `DraftModel`/`Llama` are replaced with fakes
(same trick tests/test_remote_proposals.py and tests/test_remote_verifier_backend.py use), so
these tests prove the WIRING — argument parsing, prompt resolution, tokenizer plumbing, and
that `run()` assembles `proposal_loop`'s call correctly — not model or network behavior
(that's remote_proposals.py's and remote_verifier_backend.py's own job, already covered).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import remote_client  # noqa: E402


# ---------------------------------------------------------------- fakes

class FakeTokenizer:
    """Stands in for Llama(vocab_only=True). A trivial byte-ish "tokenizer": splits on
    spaces, maps each distinct word to a stable id. Enough to prove tokenize -> proposal_loop
    -> detokenize round-trips through remote_client.run() correctly."""

    EOS = 999

    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = {}

    def _id_for(self, word):
        if word not in self._word_to_id:
            tid = 100 + len(self._word_to_id)
            self._word_to_id[word] = tid
            self._id_to_word[tid] = word
        return self._word_to_id[word]

    def tokenize(self, text_bytes, add_bos=True, special=True):
        words = text_bytes.decode("utf-8").split()
        ids = [1] if add_bos else []   # fake BOS
        ids += [self._id_for(w) for w in words]
        return ids

    def detokenize(self, ids):
        words = [self._id_to_word.get(t, f"<{t}>") for t in ids if t != 1]
        return " ".join(words).encode("utf-8")

    def token_eos(self):
        return self.EOS


class FakeDraftModel:
    """Stands in for speculative.DraftModel. Records every .propose() call so tests can
    assert remote_client wired prefix_tokens/k through unchanged, and returns a fixed
    canned proposal so the whole run() path is deterministic."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.calls = []
        self.closed = False

    def propose(self, prefix_tokens, k):
        self.calls.append((list(prefix_tokens), k))
        return [777] * k   # arbitrary fixed proposal, never accepted by the fake verifier below

    def close(self):
        self.closed = True


# ---------------------------------------------------------------- CLI parsing

def test_arg_parser_requires_draft_model_and_verifier():
    ap = remote_client.build_arg_parser()
    with pytest.raises(SystemExit):
        ap.parse_args(["--prompt", "hi"])


def test_arg_parser_requires_exactly_one_prompt_source():
    ap = remote_client.build_arg_parser()
    with pytest.raises(SystemExit):
        ap.parse_args(["--draft-model-path", "d.gguf", "--verifier-url", "h:1"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--draft-model-path", "d.gguf", "--verifier-url", "h:1",
                        "--prompt", "a", "--prompt-file", "b.txt"])


def test_arg_parser_defaults_match_documented_convention():
    ap = remote_client.build_arg_parser()
    args = ap.parse_args(["--draft-model-path", "d.gguf", "--verifier-url", "h:1",
                           "--prompt", "hi"])
    assert args.max_tokens == 128
    assert args.k == 4               # matches client.py's --draft-max default
    assert args.eos_id is None
    assert args.timeout == 5.0


def test_arg_parser_accepts_prompt_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("hello from a file")
    ap = remote_client.build_arg_parser()
    args = ap.parse_args(["--draft-model-path", "d.gguf", "--verifier-url", "h:1",
                          "--prompt-file", str(p)])
    assert remote_client.resolve_prompt(args) == "hello from a file"


# ---------------------------------------------------------------- URL normalization

@pytest.mark.parametrize("given,want", [
    ("10.42.0.5:8910", "10.42.0.5:8910"),
    ("http://10.42.0.5:8910", "10.42.0.5:8910"),
    ("https://10.42.0.5:8910/", "10.42.0.5:8910"),
])
def test_normalize_verifier_url(given, want):
    assert remote_client.normalize_verifier_url(given) == want


# ---------------------------------------------------------------- draft adapter

def test_make_draft_propose_passes_through_to_draft_model():
    fake = FakeDraftModel("d.gguf")
    propose = remote_client.make_draft_propose(fake)
    out = propose([1, 2, 3], 4)
    assert out == [777, 777, 777, 777]
    assert fake.calls == [([1, 2, 3], 4)]


# ---------------------------------------------------------------- run() wiring, end to end (fakes only)

def test_run_wires_tokenize_proposal_loop_and_detokenize(monkeypatch):
    tokenizer = FakeTokenizer()
    draft = FakeDraftModel("draft.gguf")

    monkeypatch.setattr(remote_client, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(remote_client, "DraftModel", lambda path: draft)

    submitted = []

    def fake_http_submit(peer, proposal, timeout=5.0):
        submitted.append((peer, list(proposal), timeout))
        # accept nothing, correct with a fixed word-id (deterministic, ends quickly)
        word_id = tokenizer._id_for("world")
        return 0, word_id, 0

    monkeypatch.setattr(remote_client, "http_submit", fake_http_submit)

    ap = remote_client.build_arg_parser()
    args = ap.parse_args([
        "--draft-model-path", "draft.gguf",
        "--verifier-url", "http://10.42.0.5:8910",
        "--prompt", "hello",
        "--max-tokens", "1",
        "--k", "2",
        "--timeout", "1.5",
    ])

    text = remote_client.run(args)

    # tokenizer.tokenize was called with the prompt, and propose() got the same prefix.
    # proposal_loop caps the request at `remaining` (max_tokens=1 here), so k=1 even
    # though --k=2 was passed — that's proposal_loop's own bounding, not this script's.
    assert draft.calls == [([1, tokenizer._id_for("hello")], 1)]
    # http_submit got the normalized peer, the draft's proposal, and the CLI timeout
    assert submitted == [("10.42.0.5:8910", [777], 1.5)]
    # detokenize sees exactly what proposal_loop committed (the single correction token)
    assert text == "world"
    assert draft.closed is True


def test_run_uses_tokenizer_eos_when_eos_id_not_given(monkeypatch):
    tokenizer = FakeTokenizer()
    draft = FakeDraftModel("draft.gguf")
    monkeypatch.setattr(remote_client, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(remote_client, "DraftModel", lambda path: draft)

    seen_eos_ids = {}

    import remote_proposals as rp
    real_loop = rp.proposal_loop

    def spying_loop(**kwargs):
        seen_eos_ids["eos_ids"] = kwargs["eos_ids"]
        return real_loop(**kwargs)

    monkeypatch.setattr(remote_client, "proposal_loop", spying_loop)
    monkeypatch.setattr(remote_client, "http_submit",
                        lambda peer, proposal, timeout=5.0: (0, tokenizer.EOS, 0))

    ap = remote_client.build_arg_parser()
    args = ap.parse_args(["--draft-model-path", "draft.gguf", "--verifier-url", "h:1",
                          "--prompt", "hi", "--max-tokens", "1"])
    remote_client.run(args)
    assert seen_eos_ids["eos_ids"] == {tokenizer.EOS}


def test_run_respects_explicit_eos_id_override(monkeypatch):
    tokenizer = FakeTokenizer()
    draft = FakeDraftModel("draft.gguf")
    monkeypatch.setattr(remote_client, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(remote_client, "DraftModel", lambda path: draft)

    seen_eos_ids = {}
    import remote_proposals as rp
    real_loop = rp.proposal_loop

    def spying_loop(**kwargs):
        seen_eos_ids["eos_ids"] = kwargs["eos_ids"]
        return real_loop(**kwargs)

    monkeypatch.setattr(remote_client, "proposal_loop", spying_loop)
    monkeypatch.setattr(remote_client, "http_submit",
                        lambda peer, proposal, timeout=5.0: (0, 42, 0))

    ap = remote_client.build_arg_parser()
    args = ap.parse_args(["--draft-model-path", "draft.gguf", "--verifier-url", "h:1",
                          "--prompt", "hi", "--max-tokens", "1", "--eos-id", "42"])
    remote_client.run(args)
    assert seen_eos_ids["eos_ids"] == {42}
