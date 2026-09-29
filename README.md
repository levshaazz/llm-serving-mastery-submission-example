# LLM Serving Mastery — stable submission example

Use this public repository as the starting point for your own leaderboard submission and for the
course notebooks. Create a repository from it, replace the author and experiment notes, and keep
the required CI check green.
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
# In a second terminal, when /v1/models is ready:
python scripts/smoke_submission.py
```

The CI check verifies the public submission contract without a GPU. The smoke script checks model
metadata and streaming on your GPU. Hidden quality, canary, latency and speed measurements run on
the instructor's RTX 5070 Ti; your local result cannot predict a score there. A free Colab GPU is
not guaranteed and may differ from the T4 used in the published demonstration. Tuesday dry-run logs
show whether the current default branch starts on the judge GPU.

## Course notebooks

| Topic | Notebook | Open in Colab |
|---|---|---|
| 00 · live demonstration | [pipeline vs vLLM](seminars/00-live-demo-pipeline-vs-vllm.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/00-live-demo-pipeline-vs-vllm.ipynb) |
| 01 · single-request baseline | [serving baseline](seminars/01-serving-system-baseline.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/01-serving-system-baseline.ipynb) |
| 02 · GPU bottlenecks | [profiler lab](seminars/02-gpu-profiling-and-bottlenecks.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/02-gpu-profiling-and-bottlenecks.ipynb) |

The [course site](https://levshaazz.github.io/llm-serving-mastery/) has the syllabus, deadlines,
published thresholds, slides, required artifacts and acceptance checklists. Keep generated evidence
in your own repository; never commit access tokens, model caches or weights.

This template is the **reference implementation**. The [Round 01 leaderboard page](https://levshaazz.github.io/llm-serving-mastery/en/leaderboard/)
publishes the verified config-v5 RTX 5070 Ti bars.
