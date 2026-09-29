# LLM Serving Mastery — stable submission example

Use this repository as the known-good starting point for your own leaderboard submission. Create a
repository from it, replace the author and experiment notes, and keep the required CI check green.
The check validates the submission contract, immutable revisions, lockfile, shell syntax, and
consistency between `submission.yaml` and the serving command before a tag is created.

Copy these files into your own **public** repository. Change anything except the contract:

| Fixed by the contract | |
|---|---|
| `prepare.sh` | networked bootstrap: install the locked environment and fetch the immutable model revision; no judge seed or hidden prompts are present |
| `serve.sh` | starts an OpenAI-compatible server on `0.0.0.0:8000`; on the judge it resolves the prefetched immutable snapshot locally and starts without egress; WSL2 uses vLLM's V1 runner and native sampler because UVA and FlashInfer JIT are unavailable |
| model name | `submission` (`--served-model-name submission`) |
| model revision | immutable commit in both `submission.yaml` and the server's `--revision` argument |
| API | streaming `/v1/chat/completions` that honours `max_tokens` and `ignore_eos`, `/v1/models`; accepts requests of up to 4096 tokens (prompt + answer) |
| `uv.lock` | pinned environment: `uv lock` after every dependency change, commit it |
| `submission.yaml` | author, model, immutable model revision, stack |
| `JOURNAL.md` | one entry per round |
| `Dockerfile` | optional alternative; it must bake in dependencies/model artifacts because the scored container has no egress |

Submit a round: `git tag round-01 && git push origin round-01` before **Thursday 23:59 Moscow time (MSK, UTC+3)**.

## Submission calendar

| Round | Tag | Deadline (MSK) |
|---:|---|---|
| 01 | `round-01` | 8 October 2026, 23:59 |
| 02 | `round-02` | 15 October 2026, 23:59 |
| 03 | `round-03` | 22 October 2026, 23:59 |
| 04 | `round-04` | 29 October 2026, 23:59 |
| 05 | `round-05` | 5 November 2026, 23:59 |
| 06 | `round-06` | 12 November 2026, 23:59 |
| 07 | `round-07` | 19 November 2026, 23:59 |
| 08 | `round-08` | 26 November 2026, 23:59 |
| 09 | `round-09` | 3 December 2026, 23:59 |
| 10 | `round-10` | 10 December 2026, 23:59 |

The judge snapshots the tag one minute after each deadline. Moving a tag afterwards does not change
the submitted commit.

## Quick start (Colab or any Linux box with an NVIDIA GPU)

```bash
pip install uv                 # once
uv lock                        # after every change to pyproject.toml — commit uv.lock
bash serve.sh                  # starts the server on :8000
```

Self-check exactly like the judge: see `judge/README.md` in the course repository.

This template is the **reference implementation**. Round 01 uses the verified config-v5 RTX 5070 Ti
baseline published in `judge/reference/round-01.v5.result.json` and
`judge/thresholds/round-01.json`.
