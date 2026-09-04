# Branch disposition — assessed 2026-09-04

Against `forgejo/main` @ `64a1b2c5`. Assessed **by content diff, not by commit ancestry**, and
that distinction is the whole finding.

## Verdicts

| branch | verdict | evidence |
|---|---|---|
| `perf/proposals-connection-reuse` | **LANDED** | genuinely new (`http.client` pooling absent from main), 0 conflict markers, 36/36 tests pass. Merged 2026-09-04. |
| `inference/run-receipts` | **close** | byte-identical to main; would now *conflict* with main's newer EAGLE block in `client.py` |
| `inference/edge-supervision` | **close** | byte-identical to main |
| `inference/rtt-topology-order` | **close** | byte-identical to main |
| `serve/proxy-backend` | **close** | its own work landed as `b56a22d`; the only live content it carries is inherited from connection-reuse |
| `discovery/nostr-relay-wireup` | **close** | already on main — the discovery gap was a config value, not missing code |

## ⚠️⚠️ The error this assessment was built on

The claim "five branches are unmerged" came from:

    git rev-list --count forgejo/main..<branch>

which counts **commits absent from main's ancestry**. Four of those branches' *content* is
already on main, landed under different SHAs:

    git diff forgejo/main <branch> -- <its files>    # 0 lines

**Commits-ahead is not content-missing.** An ancestry check was used to answer a content
question — the same class of defect this repo keeps rediscovering, and the reason to record it
here rather than only in a commit message.

## ⚠️ `BRANCHES.md` recorded a merge SHA that does not exist

Three of these were marked `merged→fcc012c`. That commit is **dangling** — not an ancestor of
`main`, nor of any of the three branches. It survives only inside an unrelated branch's commit
message. The ledger's *conclusion* was right and its *evidence* was fabricated.

That file is what sessions on other machines read to avoid editing the same code in parallel.
A stale-but-plausible SHA in it is worse than a blank field: a blank invites a check, a wrong
SHA invites trust. Worth repairing separately from any code change.

## What "close" means here

Nothing is lost. Each closed branch's content is verifiably present on `main` — confirmed by a
zero-line diff of its own files, not by reading the ledger. Deleting them removes duplicate
history, not work.

## Method, so this is reproducible

    git merge-tree $(git merge-base forgejo/main <ref>) forgejo/main <ref>   # conflicts?
    git diff forgejo/main <ref> -- <files the branch touches>                # content?
    git worktree add /tmp/wt <ref> && pytest <its tests>                     # green?

⚠️ Run the tests in a **temporary worktree**. This checkout is shared by roughly twenty
processes, and `git checkout` in it moves HEAD for every one of them.
