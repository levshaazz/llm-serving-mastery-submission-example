# LLM Serving Mastery — stable submission example

Topic 03 starts with your own grouped quantizer and real Qwen weights/activations,
then a calibration-only choice, held-out check and a separate FP16/NF4 GPU A/B.
Download the [complete lab ZIP](https://levshaazz.github.io/llm-serving-mastery/downloads/topic03-pilot.zip);
in Colab upload `topic03_pilot.py` and `topic03-real-layer.npz` from it. Local copies
also live in `teaching/` and `public/downloads/`. The older
`03-quantization-tradeoffs.ipynb` remains historical, not the required current route.
The current guided-investigation-v6 notebook passed a complete standard-RAM Colab
T4 rehearsal with explicit private instructor solutions on 6 October: all 27 code
cells in order, no errors, both profiles complete and zero post-worker GPU use.
The [dated v6 receipt](https://levshaazz.github.io/llm-serving-mastery/seminars/runs/2026-10-06-topic03-v6-colab/README.md)
binds the immutable archived v6 template. This prose-only follow-up preserves all
25 student code cells, their identifiers and metadata exactly; the whole-file hash
differs because explanations changed. Historical captures remain separate.
Learner validation is still pending, independently of technical execution.
The lab now compares grouping, calibration-selected clipping and supplied NF4
on the same weights/inputs before the separate whole-model GPU A/B.

The NPZ derives from Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Copyright 2024 Alibaba Cloud,
Apache License 2.0; the exact [upstream license](public/downloads/QWEN-LICENSE.txt) is included.
Modification: export of layer-12 MLP down-projection rows 0–63 as FP16 arrays,
plus recorded FP32 user-position activations. This is not a complete checkpoint.

Use this public repository as the starting point for your own leaderboard submission and for the
course notebooks. Create a repository from it, replace the author and experiment notes, and keep
the required CI check green.

**Start with [STUDENT_GUIDE.md](STUDENT_GUIDE.md)**: private copy including CI, access receipt,
your own CUDA GPU, evidence files, Tuesday readiness Issue, round tag and support.
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
uv lock --check                 # regenerate/commit only when dependencies change
bash serve.sh                  # starts the server on :8000
# In a second terminal, when /v1/models is ready:
python scripts/smoke_submission.py
```

The CI check verifies the public submission contract without a GPU. The smoke script checks model
metadata and streaming on your GPU. Hidden quality, canary, latency and speed measurements run on
the instructor's RTX 5070 Ti; your local result cannot predict a score there. A free Colab GPU is
not guaranteed. Students find their own CUDA GPU; instructor development slots are not promised.
The manual Tuesday startup/smoke result is reviewed and returned in your private **Course readiness**
Issue with its checked commit and time. It is not a quality/speed grade. General questions without
private code or data go to this repository's Issues; individual diagnostics and appeals go to your
private repository's Issue with **@levshaazz**. Never put raw logs or credentials in public Issues.

## Course notebooks

| Topic | Notebook | Open in Colab |
|---|---|---|
| 00 · live demonstration | [pipeline vs vLLM](seminars/00-live-demo-pipeline-vs-vllm.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/00-live-demo-pipeline-vs-vllm.ipynb) |
| 01 · single-request baseline | [serving baseline](seminars/01-serving-system-baseline.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/01-serving-system-baseline.ipynb) |
| 02 · GPU bottlenecks | [profiler lab](seminars/02-gpu-profiling-and-bottlenecks.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/02-gpu-profiling-and-bottlenecks.ipynb) |
| 03 · quantization | [One matrix, four bits: required investigation](seminars/03-quantization-pilot.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/03-quantization-pilot.ipynb) |
| 04 · vLLM API | [local server and SDK lab](seminars/04-vllm-openai-serving.ipynb) | [Open](https://colab.research.google.com/github/levshaazz/llm-serving-mastery-submission-example/blob/main/seminars/04-vllm-openai-serving.ipynb) |

Topics 03–04 deliberately use small, bounded smoke experiments. All Topics 01–04 notebooks have
passed sequential offline instructor GPU rehearsals on an RTX 5070 Ti. The notebooks have no executed
cell outputs; Topics 02–04 embed explicitly labeled recorded instructor evidence for CPU/offline
replay. These recordings are not evidence that you performed the required GPU lab. Evidence limits are on the
[course provenance page](https://levshaazz.github.io/llm-serving-mastery/en/provenance/).
Set `LSM_MODEL_CACHE` to persistent storage if the runtime is ephemeral. Topics 03–04 default to
ignored `.cache/models` (Topic 02 defaults to `~/.cache/lsm-models`); Topic 03 and Topic 04 reuse the same pinned 0.5B snapshot, while Topic 02
also uses a pinned 1.5B snapshot. Topic 01 shares the same durable-cache behavior. Matching pinned
packages are reused without reinstalling on every notebook execution. Later runs load pinned models
locally rather than downloading for every fresh server process. Keep caches and local logs
out of Git; inspect evidence before publishing it.

Topics 03–04 now start with three executable CPU/offline cells: a complete INT4 arithmetic example
or fragmented UTF-8/SSE parser, then recorded real model/API evidence. Their bounded v3 GPU workloads
are unchanged, with incremental checkpoints and owned-server teardown. The canonical helpers require
Linux/WSL, an idle CUDA GPU and at least 24 GiB available host RAM. Do not lower resource gates to force
a run. After your own successful run:

```bash
python scripts/validate_topic03.py evidence/topic-03-comparison.json
python scripts/validate_topic04.py evidence/topic-04-service.json
```

Validators check identity, raw counts/clocks, completion and artifact consistency—not semantic quality,
speed or a leaderboard score. Keep `topic-03-decision.md` and `topic-04-analysis.md` alongside the JSON
in `evidence/`. Preserve failed attempts and never publish raw local server logs.

The [course site](https://levshaazz.github.io/llm-serving-mastery/) has the syllabus, deadlines,
published thresholds, slides, required artifacts and acceptance checklists. Keep generated evidence
in your own repository; never commit access tokens, model caches or weights.

This template is the **reference implementation**. The [Round 01 leaderboard page](https://levshaazz.github.io/llm-serving-mastery/en/leaderboard/)
publishes the verified config-v5 RTX 5070 Ti bars.
