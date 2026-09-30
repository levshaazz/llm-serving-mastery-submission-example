# LLM Serving Mastery — stable submission example

Use this public repository as the starting point for your own leaderboard submission and for the
course notebooks. Create a repository from it, replace the author and experiment notes, and keep
the required CI check green.
The check validates the submission contract, immutable revisions, lockfile, shell syntax, and
consistency between `submission.yaml` and the serving command before a tag is created.

Copy these files into your own **private** repository (do not fork the public example if you need
private visibility). Invite **levshaazz** via **Settings → Collaborators → Add people** and confirm
the invitation is accepted. The judge needs read access; never send your own token. Only this generic
example remains public. Change anything except the contract:

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

First [register your private repository in the course sheet](https://docs.google.com/spreadsheets/d/1y6w_Ruf3Ofh5gBMFH3LUmgBmP6faaMUG/edit).
Use the repository URL, not your profile URL; `submission.yaml` author must match its GitHub owner.
Registration and green CI do not replace the round tag or accepted instructor access. Never put
credentials in the sheet or repository.

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

Do not move the tag after the deadline. Capture starts one minute later; the judge records the
actual capture time and commit SHA, then measures that frozen commit. Commit dates are not deadline
evidence. Round 01 capture begins 9 October 2026 at 00:00 MSK.

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
| 03 · quantization | [FP16/NF4 paired lab](seminars/03-quantization-tradeoffs.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/03-quantization-tradeoffs.ipynb) |
| 04 · vLLM API | [local server and SDK lab](seminars/04-vllm-openai-serving.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/04-vllm-openai-serving.ipynb) |

Topics 03–04 deliberately use small, bounded smoke experiments. All Topics 01–04 notebooks have
passed a sequential offline instructor GPU rehearsal on an RTX 5070 Ti. The notebooks themselves
contain no prefilled results; the reviewed instructor evidence and limitations are linked from the
[course provenance page](https://levshaazz.github.io/llm-serving-mastery/en/provenance/).
Set `LSM_MODEL_CACHE` to persistent storage if the runtime is ephemeral. Topics 02–04 default to
ignored `.cache/models`; Topic 03 and Topic 04 reuse the same pinned 0.5B snapshot, while Topic 02
also uses a pinned 1.5B snapshot. Topic 01 shares the same durable-cache behavior. Matching pinned
packages are reused without reinstalling on every notebook execution. Later runs load pinned models
locally rather than downloading for every fresh server process. Keep caches and local logs
out of Git; inspect evidence before publishing it.

The [course site](https://levshaazz.github.io/llm-serving-mastery/) has the syllabus, deadlines,
published thresholds, slides, required artifacts and acceptance checklists. Keep generated evidence
in your own repository; never commit access tokens, model caches or weights.

This template is the **reference implementation**. The [Round 01 leaderboard page](https://levshaazz.github.io/llm-serving-mastery/en/leaderboard/)
publishes the verified config-v5 RTX 5070 Ti bars.
