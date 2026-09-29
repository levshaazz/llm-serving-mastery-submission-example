# Journal

One entry per round. Write it the day you run the experiment. This is the raw material of your defense.

## Round 01 — verified FP16 baseline
Hypothesis: plain vLLM provides a reproducible starting point before optimization.
Change: pinned Qwen2.5-3B-Instruct and vLLM 0.29.0 with a 4096-token context.
Colab (T4): run the self-check before copying this value into your own journal.
Judge: S = 2514.1 tok/s, p95 TTFT@64 = 0.5888 s, reference bar.
Kept? / Next: keep as the known-good baseline; students should change one variable at a time.
