# From an empty repository to Round 01

You submit a **private** GitHub repository. The public example supplies the files and CPU checks;
the instructor's private course repository is not needed. The judge runs your tagged server, not
your Colab notebook. A notebook result is learning evidence, not a leaderboard score.

## 1. Make a private copy, including CI

Create a new **empty private repository** in your personal GitHub account, with no initial README.
Do not use Fork: a fork of this public example is not the private-copy workflow. Then run:

```bash
git clone https://github.com/levshaazz/llm-serving-mastery-submission-example.git my-serving
cd my-serving
git remote rename origin upstream
git remote add origin https://github.com/YOUR_LOGIN/YOUR_PRIVATE_REPO.git
git push -u origin main
```

Replace both placeholders with your own account and repository name. Authenticate normally with
GitHub; never put a token in the remote URL. These commands retain the example's history and its
`.github/workflows/` checks, and make your private repository the push destination. They do not
push any example tags. In `submission.yaml`, replace `author` with the owner's exact GitHub login.
Keep the immutable model revision unless you deliberately change the model. Commit your edits.

Open **Settings → Collaborators → Add people**, invite **levshaazz**, and wait for acceptance.
Then [register the repository URL in the course sheet](https://docs.google.com/spreadsheets/d/1y6w_Ruf3Ofh5gBMFH3LUmgBmP6faaMUG/edit).
Registration, an invitation, and a readable repository are three separate states.

## 2. Obtain an access receipt

Enable Issues in your private repository if necessary. Open one Issue titled **Course readiness**
and mention **@levshaazz**. Ask for judge access confirmation and include your default-branch commit
SHA (`git rev-parse HEAD`). The instructor replies **Access checked**, with the checked commit and
time, only after the judge can read the repository. An accepted invitation alone is not that receipt.

If the invitation is still pending and the private mention cannot reach the instructor, open a
general access-help Issue in the [public example](https://github.com/levshaazz/llm-serving-mastery-submission-example/issues).
Say that an invitation is pending; do not paste the private URL, solution, log, token or personal data.
The registration sheet already supplies the private URL to the instructor.

## 3. Complete the CPU checks

Use Python 3.12 and install `uv`. From the repository root:

```bash
python scripts/validate_submission.py --root .
python -m unittest discover -s tests -v
bash -n prepare.sh serve.sh
uv lock --check
```

Push your commit and inspect **Actions → Submission contract** for that exact commit. The local
commands remain usable if GitHub Actions is unavailable because of an account quota; report the
quota problem in the private readiness Issue rather than inventing a passing CI result.
Run `uv lock` and commit `uv.lock` only when changing dependencies. Do not refresh the lockfile for
every launch. CPU checks verify packaging and consistency, not model quality, speed or CUDA support.

## 4. Arrange your GPU and keep the two environments separate

You are responsible for finding a **CUDA GPU**. Colab does not guarantee a GPU or a particular model,
and the course does not provide instructor-machine development slots. CPU work can begin immediately,
but it does not replace a GPU smoke check. Report a blocker before the deadline; there is no automatic
extension or exemption. Instructor infrastructure incidents follow the published course-wide policy.

The Topic 01 notebook uses a small pinned model to teach measurement. The submission example uses a
different, larger reference model to meet judge quality gates. Keep the notebook and server package
environments separate; do not replace the submission model with the lab model just to copy its output.
Local GPU performance is not a prediction of your RTX 5070 Ti leaderboard score.

Run the notebook linked from the [Topic 01 hub](https://levshaazz.github.io/llm-serving-mastery/en/topics/01/).
Copy the resulting files from Colab to your private repository before the runtime is discarded:

```text
evidence/topic-01-baseline.json
evidence/topic-01-reflection.md
evidence/topic-01-boundary-trace.md
```

The trace names client, gateway, scheduler and worker boundaries. Mark events you did not measure
**not observed**; a diagram is not permission to invent internal timestamps. Save the completed
notebook without credentials or cached weights. Study Topic 02 and save its artifacts as listed on
its topic page. Keep `JOURNAL.md` as the record of changes, measurements and failed hypotheses.

Topics 03–04 begin with CPU-only worked examples and explicitly labeled recorded instructor
replays. Those cells need no network, weights or GPU; they do not count as your own experiment.
Their bounded CUDA routes require Linux/WSL, an idle GPU and at least 24 GiB available host RAM.
Keep the paired natural-EOS/forced-length evidence separate in Topic 03, and complete/partial/failed
streams separate in Topic 04. After your own successful run, validate the JSON before submission:

```bash
python scripts/validate_topic03.py evidence/topic-03-comparison.json
python scripts/validate_topic04.py evidence/topic-04-service.json
```

Include `evidence/topic-03-decision.md` and `evidence/topic-04-analysis.md` as directed in the notebooks.
Passing these CPU validators proves artifact consistency, not production quality, speed or a score.
Preserve failed attempts locally, stop on resource pressure, and never delete model caches to retry.

## 5. Run the GPU streaming smoke

On Linux or a suitable Colab environment, run `bash serve.sh` from your submission checkout.
The first launch may download the pinned model and dependencies; keep their caches for later runs.
In a second terminal, once `/v1/models` is ready, run:

```bash
python scripts/smoke_submission.py
```

Keep the exit status, commit SHA, GPU/runtime identity and a sanitized output in your journal.
The server must expose model name `submission`, support streaming and the declared generation
parameters. Stop it after testing. A local smoke is necessary but does not establish hidden quality,
load tolerance, canary consistency or the final grade. You do not need a judge seed: the instructor
keeps and applies the same protected seed to every submission within the round.

## 6. Read the Tuesday result

The instructor checks the **current default branch** each Tuesday before the Thursday deadline.
For Round 01 that is **6 October 2026**. Results are reviewed manually and returned in your private
**Course readiness** Issue. No exact hour is promised. These are future checks, not already earned
passes. If no result has arrived by Wednesday, request status in that Issue.

The reply distinguishes **access checked**, **startup/smoke passed**, **student fix needed**, and
**judge incident / not checked**. It identifies the checked SHA and gives a sanitized diagnostic.
Only `startup/smoke passed` demonstrates startup on the judge GPU for that commit. It does not run
quality or speed scoring. Later changes require their own checks. Raw server logs and private URLs
stay private; the instructor does not post them to the public portal.

## 7. Freeze and submit

Check the [canonical deadlines](https://levshaazz.github.io/llm-serving-mastery/en/schedule/).
Commit all source, lockfile, journal and required evidence. Confirm `git status` is clean, the intended
commit passed your checks, the judge has access, and your private remote is correct. Then:

```bash
git push origin main
git tag round-01
git push origin round-01
git ls-remote --tags origin refs/tags/round-01
```

The last command confirms that the tag exists on GitHub. Do not overwrite an existing tag blindly;
inspect it and ask in the private Issue if unsure. Do not move it after the deadline. Submission is
the tag, not a sheet entry, branch push, notebook upload or screenshot of green CI. The judge captures
the tag one minute after the published deadline and records the actual capture time and SHA. Commit
dates do not prove when a tag was pushed. A Tuesday rehearsal never replaces this official snapshot.

## 8. Scores, questions and appeals

Read all [quality, latency, length, error, canary and speed rules](https://levshaazz.github.io/llm-serving-mastery/en/leaderboard/)
before tuning. Green CI and a working server do not guarantee a nonzero mark. The published reference
is a starting implementation, not permission to special-case hidden tests or cache their answers.

General questions go to [public example Issues](https://github.com/levshaazz/llm-serving-mastery-submission-example/issues).
Individual diagnostics and appeals go to an Issue in **your private repository**, mentioning
**@levshaazz**. Include round, tag, SHA, published result and the specific discrepancy; never include
tokens or hidden judge material. Reviewed results are due within five calendar days after the
deadline; request review within three calendar days of publication. A judge error receives no zero
and is rerun under the course incident policy. See the
[syllabus](https://levshaazz.github.io/llm-serving-mastery/en/syllabus/) for the complete policy.
